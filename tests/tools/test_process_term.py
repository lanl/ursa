import asyncio
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

from ursa.tools.terminal.process import ProcessTerm

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Unix shell integration"
)


async def test_process_term_runs_command_captures_combined_output_and_exits(
    tmp_path,
):
    terminal = ProcessTerm(
        "proc1234", ["/bin/sh"], env={"URSA_TERM_TEST": "works"}, cwd=tmp_path
    )
    await terminal.start(
        "printf 'out:%s\\n' \"$URSA_TERM_TEST\"; printf 'err\\n' >&2; exit 7"
    )
    assert await terminal.wait() == 7
    assert await terminal.read() == "out:works\nerr\n"
    assert await terminal.is_alive() == {"exit_code": 7}
    await terminal.terminate()


async def test_process_term_supports_argument_list_and_tail_slicing(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start(["printf", "one\\ntwo\\nthree\\nfour\\n"])
    assert await terminal.wait() == 0
    assert await terminal.read(lines=2) == "three\nfour\n"
    assert await terminal.read(offset=1, lines=2) == "two\nthree\n"
    assert await terminal.read(offset=2) == "one\ntwo\n"
    assert await terminal.read(lines=0) == ""
    await terminal.terminate()


def test_process_command_formatting_and_shell_detection():
    assert ProcessTerm._format_command(["echo", "a b", "x'y"]) == (
        "echo 'a b' 'x'\"'\"'y'"
    )
    assert ProcessTerm._is_powershell([r"C:\Tools\PwSh.EXE"]) is True
    assert ProcessTerm._is_powershell(["/bin/bash"]) is False
    assert (
        ProcessTerm._powershell_command(["echo", "a'b", "$x; y"])
        == "& 'echo' 'a''b' '$x; y'"
    )
    assert ProcessTerm._powershell_command([]) == ""


async def test_process_start_builds_literal_powershell_argv(monkeypatch):
    captured = {}

    class FakeProcess:
        stdin = None
        pid = 123

        def poll(self):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        "ursa.tools.terminal.process.subprocess.Popen", fake_popen
    )
    terminal = ProcessTerm("proc1234", [r"C:\Tools\PwSh.EXE", "-NoLogo"])
    await terminal.start(["echo", "a'b", "$x; y"])
    assert captured["argv"] == [
        r"C:\Tools\PwSh.EXE",
        "-NoLogo",
        "-Command",
        "& 'echo' 'a''b' '$x; y'",
    ]
    assert captured["kwargs"]["stdin"] is subprocess.PIPE
    await terminal.terminate()


async def test_process_term_quotes_argument_list_as_one_command(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start(["printf", "%s", "argument with spaces"])
    assert await terminal.wait() == 0
    assert await terminal.read() == "argument with spaces"
    await terminal.terminate()


async def test_process_term_interactive_sends_and_lifecycle(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    with pytest.raises(RuntimeError, match="not been started"):
        await terminal.read()
    await terminal.start()
    assert await terminal.is_alive() == {"is_alive": True}
    await terminal.send_text("printf interactive")
    await terminal.send_keycode(10)
    await terminal.send_line("exit")
    assert await terminal.wait() == 0
    assert await terminal.read() == "interactive"
    with pytest.raises(BrokenPipeError, match="not running"):
        await terminal.send_bytes(b"again")
    await terminal.terminate()


async def test_process_term_validates_reads_and_unsupported_screen_operations(
    tmp_path,
):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start()
    with pytest.raises(ValueError, match="offset"):
        await terminal.read(offset=-1)
    with pytest.raises(ValueError, match="lines"):
        await terminal.read(lines=-1)
    with pytest.raises(TypeError, match="bytes"):
        await terminal.send_bytes("not bytes")
    with pytest.raises(NotImplementedError, match="resized"):
        await terminal.resize(30, 90)
    with pytest.raises(NotImplementedError, match="cursor"):
        await terminal.cursor()
    with pytest.raises(NotImplementedError, match="screen size"):
        await terminal.size()
    await terminal.terminate()


async def test_process_term_rejects_double_start(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start()
    with pytest.raises(RuntimeError, match="already started"):
        await terminal.start()
    await terminal.terminate()


async def test_process_term_terminate_stops_running_shell(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start()
    await terminal.terminate()
    state = await terminal.is_alive()
    assert state["exit_code"] != 0


async def test_process_term_termination_is_concurrent_idempotent_and_cleans_temp(
    tmp_path,
):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start("exit 0")
    await terminal.wait()
    output_path = terminal._output_path
    assert output_path is not None and output_path.exists()
    await asyncio.gather(terminal.terminate(), terminal.terminate())
    assert not output_path.exists()
    assert terminal._output_path is None


async def test_process_term_cancelled_terminate_still_finishes_cleanup(
    tmp_path, monkeypatch
):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start()
    output_path = terminal._output_path
    original = terminal._terminate_sync

    def delayed_terminate():
        import time

        time.sleep(0.02)
        original()

    monkeypatch.setattr(terminal, "_terminate_sync", delayed_terminate)
    task = asyncio.create_task(terminal.terminate())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert terminal._termination_task is not None
    await terminal._termination_task
    assert output_path is not None and not output_path.exists()


async def test_process_term_escalates_to_kill_after_terminate_timeout(
    monkeypatch,
):
    terminal = ProcessTerm("proc1234", ["fake-shell"])

    class FakeInput:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        stdin = FakeInput()

        def __init__(self):
            self.calls = []
            self.pid = 12345

        def poll(self):
            return None

        def terminate(self):
            self.calls.append("terminate")

        def wait(self, timeout=None):
            self.calls.append(("wait", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return -9

        def kill(self):
            self.calls.append("kill")

    process = FakeProcess()
    signals = []
    monkeypatch.setattr(
        "ursa.tools.terminal.process.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    times = iter((0.0, 3.0))
    monkeypatch.setattr(
        "ursa.tools.terminal.process.time",
        SimpleNamespace(monotonic=lambda: next(times), sleep=lambda _: None),
    )
    terminal._process = process
    await terminal.terminate()
    assert process.calls == [
        ("wait", 2),
        ("wait", None),
    ]
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert process.stdin.closed is True


async def test_process_term_accepts_repeated_writes(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/sh"], cwd=tmp_path)
    await terminal.start()
    for number in range(10):
        await terminal.send_line(f"printf '{number}\\n'")
    await terminal.send_line("exit")
    await terminal.wait()
    assert sorted((await terminal.read()).splitlines(), key=int) == [
        str(number) for number in range(10)
    ]
    await terminal.terminate()


async def _wait_for_output(terminal, pattern, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        match = pattern.search(await terminal.read())
        if match:
            return match
        await asyncio.sleep(0.01)
    pytest.fail(f"output did not match {pattern.pattern!r}")


async def _wait_until_process_gone(pid, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"process {pid} still exists")


async def test_process_terminate_kills_descendant_process_group(tmp_path):
    import re

    terminal = ProcessTerm("proc1234", ["/bin/bash"], cwd=tmp_path)
    await terminal.start()
    await terminal.send_line("sleep 30 & echo CHILD:$!")
    match = await _wait_for_output(terminal, re.compile(r"CHILD:(\d+)"))
    child_pid = int(match.group(1))
    assert terminal._process is not None
    assert os.getpgid(child_pid) == terminal._process.pid
    await asyncio.wait_for(terminal.terminate(), timeout=3.0)
    await _wait_until_process_gone(child_pid)


async def test_process_terminate_unblocks_large_pipe_send(tmp_path):
    terminal = ProcessTerm("proc1234", ["/bin/bash"], cwd=tmp_path)
    await terminal.start("sleep 30")
    send = asyncio.create_task(terminal.send_bytes(b"x" * (10 * 1024 * 1024)))
    await asyncio.sleep(0.05)
    assert not send.done()
    await asyncio.wait_for(terminal.terminate(), timeout=3.0)
    with pytest.raises(BrokenPipeError):
        await asyncio.wait_for(send, timeout=1.0)
