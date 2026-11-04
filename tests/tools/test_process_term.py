"""Representative ProcessTerm integration and unit contracts."""

import asyncio
import os
import sys

import pytest

from ursa.tools.terminal.process import ProcessTerm

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="Unix shell integration"
)


def term(name="proc1234", **kwargs):
    return ProcessTerm(name, ["/bin/sh"], **kwargs)


@pytest.mark.asyncio
@posix_only
async def test_command_capture_read_slicing_snapshot_and_bounded_ring(tmp_path):
    terminal = term(cwd=tmp_path, capture_max_bytes=64)
    await terminal.start("printf 'one\\ntwo\\nthree\\nfour\\n'; printf err >&2")
    assert await terminal.wait() == 0
    assert (await terminal.read(lines=2)).endswith("four\nerr")
    assert await terminal.is_alive() == {"exit_code": 0}
    snapshot = await terminal.render_snapshot()
    assert not snapshot.screen and snapshot.spans[0].text.endswith("err")
    await terminal.terminate()


@pytest.mark.asyncio
@posix_only
async def test_interactive_input_markers_and_lifecycle(tmp_path):
    terminal = term(cwd=tmp_path)
    await terminal.start()
    await terminal.send_line("printf BEFORE")
    await _wait_for(terminal, "BEFORE")
    marker = await terminal.output_marker()
    await terminal.send_line("printf AFTER; exit")
    assert await terminal.wait() == 0
    assert "AFTER" in await terminal.output_since(marker)
    with pytest.raises(BrokenPipeError):
        await terminal.send_bytes(b"again")
    await terminal.terminate()


@pytest.mark.asyncio
@posix_only
async def test_wait_does_not_depend_on_descendant_pipe_closure(tmp_path):
    terminal = term(cwd=tmp_path)
    await terminal.start("sleep 30 & printf ready")
    assert await asyncio.wait_for(terminal.wait(), timeout=2) == 0
    assert await terminal.read() == "ready"
    await terminal.terminate()


def test_command_formatting_and_powershell_start(monkeypatch):
    assert ProcessTerm._format_command(["echo", "a b", "x'y"]) == (
        "echo 'a b' 'x'\"'\"'y'"
    )
    assert ProcessTerm._powershell_command(["echo", "a'b"]) == "& 'echo' 'a''b'"
    captured = {}

    class FakeProcess:
        stdin = None
        pid = 1

        def poll(self):
            return 0

    monkeypatch.setattr(
        "ursa.tools.terminal.process.subprocess.Popen",
        lambda argv, **kwargs: captured.update(argv=argv, kwargs=kwargs)
        or FakeProcess(),
    )
    terminal = ProcessTerm("powersh", [r"C:\\pwsh.exe"])
    asyncio.run(terminal.start(["echo", "a'b"]))
    assert captured["argv"][-2:] == ["-Command", "& 'echo' 'a''b'"]


@posix_only
def test_group_signal_permission_falls_back_to_process(monkeypatch):
    class FakeProcess:
        pid = 1
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def denied(*_args):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    process = FakeProcess()
    ProcessTerm._stop_process(process)
    assert process.returncode == 0


@pytest.mark.parametrize("read_args", [{"offset": -1}, {"lines": -1}])
@pytest.mark.asyncio
@posix_only
async def test_validates_reads_and_input(tmp_path, read_args):
    terminal = term(cwd=tmp_path)
    await terminal.start("exit")
    with pytest.raises(ValueError):
        await terminal.read(**read_args)
    with pytest.raises(TypeError):
        await terminal.send_bytes("bad")
    await terminal.terminate()


@pytest.mark.asyncio
async def test_cross_platform_python_lifecycle(tmp_path):
    terminal = ProcessTerm("portable", [sys.executable], cwd=tmp_path)
    await terminal.start("print('portable')")
    assert await terminal.wait() == 0
    assert "portable" in await terminal.read()
    await terminal.terminate()


async def _wait_for(terminal, text):
    for _ in range(100):
        if text in await terminal.read():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"missing {text}")
