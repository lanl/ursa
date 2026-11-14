import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.chat_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from tests.event_fixtures import (
    FIXED_MONOTONIC_TIMESTAMP_NS,
)
from tests.event_fixtures import (
    fixed_monotonic_timestamp as fixed_monotonic_timestamp,
)
from tests.tools.utils import (
    invoke_with_event_recorder,
    invoke_with_parent_run,
    make_runtime,
)
from ursa.sandbox.base import _run_process_group
from ursa.sandbox.mac import MacSandbox
from ursa.tools.run_command_tool import SafetyAssessment, run_command


@pytest.fixture(autouse=True)
def isolate_command_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests cover command handling independently of host sandbox setup.
    monkeypatch.setenv("URSA_SBX", "0")
    monkeypatch.delenv("URSA_SAFETY_LEVEL", raising=False)


def _patch_safety_result(
    monkeypatch,
    chat_model: BaseChatModel,
    *,
    is_safe: bool,
    reason: str = "Evaluated in test",
):
    captured = {}

    def fake_with_structured_output(self, schema):
        captured["schema"] = schema

        def invoke(prompt):
            captured["prompt"] = prompt
            return {"is_safe": is_safe, "reason": reason}

        return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(
        chat_model.__class__,
        "with_structured_output",
        fake_with_structured_output,
    )
    return captured


@pytest.mark.parametrize("setting, expected", [(None, 600000), ("7200", 7200)])
def test_run_command_invokes_subprocess_in_workspace(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel, setting, expected
):
    monkeypatch.delenv("URSA_COMMAND_TIMEOUT_SECONDS", raising=False)
    if setting is not None:
        monkeypatch.setenv("URSA_COMMAND_TIMEOUT_SECONDS", setting)
    recorded = {}
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def fake_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return SimpleNamespace(stdout="output", stderr="", returncode=0)

    monkeypatch.setattr("ursa.sandbox.base._run_process_group", fake_run)
    monkeypatch.setattr("ursa.sandbox.linux.subprocess.run", fake_run)
    monkeypatch.setattr("ursa.sandbox.mac.subprocess.run", fake_run)

    result, recorder = invoke_with_event_recorder(
        run_command.func,
        "echo hi",
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            thread_id="run-thread",
            tool_call_id="run-call",
        ),
    )
    events = recorder.events

    assert result == "STDOUT:\noutput\nSTDERR:\n"
    assert Path(recorded["kwargs"]["cwd"]) == tmp_path
    assert recorded["kwargs"]["shell"] is True
    assert recorded["kwargs"]["timeout"] == expected
    assert events[0] == (
        "ursa_agent_progress",
        {
            "tool": "run_command",
            "tool_call_id": "run-call",
            "stage": "safety_check",
            "message": "Command passed safety check",
            "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
            "query": "echo hi",
            "safe": True,
            "reason": "Evaluated in test",
        },
    )
    assert events[1] == (
        "ursa_agent_progress",
        {
            "tool": "run_command",
            "tool_call_id": "run-call",
            "stage": "execute",
            "message": "Running command",
            "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
            "phase": "start",
            "query": "echo hi",
        },
    )
    event_name, payload = events[2]
    assert event_name == "ursa_agent_progress"
    assert payload["tool"] == "run_command"
    assert payload["tool_call_id"] == "run-call"
    assert payload["stage"] == "execute"
    assert payload["message"] == "Command finished"
    assert payload["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert payload["phase"] == "end"
    assert payload["query"] == "echo hi"
    assert payload["returncode"] == 0
    assert payload["stdout_chars"] == len("output")
    assert payload["stderr_chars"] == 0
    assert payload["stdout_truncated"] is False
    assert payload["stderr_truncated"] is False
    assert payload["artifacts"] == [
        {
            "content": "output",
            "mime_type": "text/plain",
            "metadata": {"title": "stdout"},
        }
    ]
    assert isinstance(payload["elapsed_ms"], float)


def test_run_command_truncates_output(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    long_stdout = "a" * 200
    long_stderr = "b" * 200
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    monkeypatch.setattr(
        "ursa.sandbox.base._run_process_group",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=long_stdout, stderr=long_stderr
        ),
    )
    monkeypatch.setattr(
        "ursa.sandbox.linux.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=long_stdout, stderr=long_stderr
        ),
    )
    monkeypatch.setattr(
        "ursa.sandbox.mac.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=long_stdout, stderr=long_stderr
        ),
    )

    result = invoke_with_parent_run(
        lambda config: run_command.func(
            "noop",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                limit=64,
                tool_call_id="truncate",
                thread_id="run-thread",
                config=config,
            ),
        )
    )

    stdout_part, stderr_part = result.split("STDERR:\n", maxsplit=1)
    stdout_body = stdout_part.replace("STDOUT:\n", "", 1).rstrip("\n")
    stderr_body = stderr_part

    assert "... [snipped" in stdout_body
    assert "... [snipped" in stderr_body
    assert len(stdout_body) < len(long_stdout)
    assert len(stderr_body) < len(long_stderr)


@pytest.mark.parametrize("setting", ["0", "-1", "invalid", "1.5", ""])
def test_run_command_rejects_invalid_timeout(
    monkeypatch, tmp_path, chat_model, setting
):
    monkeypatch.setenv("URSA_COMMAND_TIMEOUT_SECONDS", setting)
    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run",
        lambda *args, **kwargs: pytest.fail("Command must not execute"),
    )
    result = run_command.func("echo hi", make_runtime(tmp_path, llm=chat_model))
    assert "Invalid URSA_COMMAND_TIMEOUT_SECONDS" in result


def test_run_command_reports_timeout(monkeypatch, tmp_path, chat_model):
    monkeypatch.setenv("URSA_COMMAND_TIMEOUT_SECONDS", "300")
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def expire(query, **kwargs):
        raise subprocess.TimeoutExpired(query, kwargs["timeout"])

    monkeypatch.setattr("ursa.sandbox.base._run_process_group", expire)
    result = invoke_with_parent_run(
        lambda config: run_command.func(
            "sleep 999",
            make_runtime(tmp_path, llm=chat_model, config=config),
        )
    )
    assert "Command timed out after 300 seconds" in result


def test_run_command_handles_keyboard_interrupt(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr("ursa.sandbox.base._run_process_group", raise_interrupt)

    monkeypatch.setattr("ursa.sandbox.linux.subprocess.run", raise_interrupt)

    monkeypatch.setattr("ursa.sandbox.mac.subprocess.run", raise_interrupt)

    result = invoke_with_parent_run(
        lambda config: run_command.func(
            "sleep 1",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                tool_call_id="interrupt",
                thread_id="run-thread",
                config=config,
            ),
        )
    )

    assert "KeyboardInterrupt:" in result


def test_run_command_returns_message_for_unicode_input(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        thread_id="run-thread",
        tool_call_id="unicode",
    )
    monkeypatch.setattr(
        "ursa.sandbox.base._run_process_group",
        lambda *args, **kwargs: pytest.fail(
            "subprocess should not run for invalid input"
        ),
    )

    monkeypatch.setattr(
        "ursa.sandbox.linux.subprocess.run",
        lambda *args, **kwargs: pytest.fail(
            "subprocess should not run for invalid input"
        ),
    )

    monkeypatch.setattr(
        "ursa.sandbox.mac.subprocess.run",
        lambda *args, **kwargs: pytest.fail(
            "subprocess should not run for invalid input"
        ),
    )

    result = run_command.func("ls café", runtime)

    assert result.startswith("Invalid query:")
    assert "U+00E9" in result
    assert "corrected ASCII string" in result


def test_run_command_blocks_commands_that_fail_safety_check(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    # Force the default safety prompt level so safe_codes appear in the prompt
    # regardless of the caller's URSA_SAFETY_LEVEL environment variable.
    monkeypatch.delenv("URSA_SAFETY_LEVEL", raising=False)
    captured = _patch_safety_result(
        monkeypatch, chat_model, is_safe=False, reason="Not safe in test"
    )
    monkeypatch.setattr(
        "ursa.sandbox.base._run_process_group",
        lambda *args, **kwargs: pytest.fail(
            "subprocess.run should not be called for unsafe commands"
        ),
    )

    monkeypatch.setattr(
        "ursa.sandbox.linux.subprocess.run",
        lambda *args, **kwargs: pytest.fail(
            "subprocess.run should not be called for unsafe commands"
        ),
    )

    monkeypatch.setattr(
        "ursa.sandbox.mac.subprocess.run",
        lambda *args, **kwargs: pytest.fail(
            "subprocess.run should not be called for unsafe commands"
        ),
    )

    store = InMemoryStore()
    store.put(("workspace", "file_edit"), "script.py", {})
    store.put(("workspace", "safe_codes"), "python", {})
    store.put(("workspace", "safe_codes"), "julia", {})

    search_calls = []
    original_search = InMemoryStore.search

    def tracked_search(self, namespace_prefix, /, **kwargs):
        if self is store:
            search_calls.append((namespace_prefix, kwargs.get("limit")))
        return original_search(self, namespace_prefix, **kwargs)

    monkeypatch.setattr(InMemoryStore, "search", tracked_search)

    result, recorder = invoke_with_event_recorder(
        run_command.func,
        "rm -rf important_files",
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            store=store,
            tool_call_id="unsafe",
            thread_id="run-thread",
        ),
    )

    assert result.startswith(
        "[UNSAFE] That command `rm -rf important_files` was deemed unsafe"
    )
    assert captured["schema"] is SafetyAssessment
    assert "python" in captured["prompt"]
    assert "julia" in captured["prompt"]
    assert "script.py" in captured["prompt"]
    assert search_calls == [
        (("workspace", "file_edit"), 1000),
        (("workspace", "safe_codes"), 1000),
    ]
    assert recorder.events == [
        (
            "ursa_agent_progress",
            {
                "tool": "run_command",
                "tool_call_id": "unsafe",
                "stage": "safety_check",
                "message": "Command deemed unsafe",
                "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
                "query": "rm -rf important_files",
                "safe": False,
                "reason": "Not safe in test",
            },
        )
    ]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_no_sandbox_timeout_kills_command_process_group(monkeypatch, tmp_path):
    calls = {}

    class Process:
        pid = 4321
        returncode = -9

        def communicate(self, timeout):
            calls["timeout"] = timeout
            raise subprocess.TimeoutExpired("loop", timeout)

        def wait(self):
            calls["waited"] = True

        def kill(self):
            pytest.fail("POSIX group kill should be used")

    def popen(*args, **kwargs):
        calls["popen"] = (args, kwargs)
        return Process()

    monkeypatch.setattr("ursa.sandbox.base.subprocess.Popen", popen)
    monkeypatch.setattr(
        "ursa.sandbox.base.os.killpg",
        lambda pid, sig: calls.update(killpg=(pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        _run_process_group(
            "python loop.py",
            text=True,
            shell=True,
            capture_output=True,
            timeout=5,
            cwd=tmp_path,
        )

    assert calls["popen"][1]["start_new_session"] is True
    assert calls["killpg"] == (4321, signal.SIGKILL)
    assert calls["waited"] is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_no_sandbox_process_group_returns_completed_process(
    monkeypatch, tmp_path
):
    class Process:
        pid = 4321
        returncode = 0

        def communicate(self, timeout):
            assert timeout == 5
            return "hello", ""

    monkeypatch.setattr(
        "ursa.sandbox.base.subprocess.Popen", lambda *a, **k: Process()
    )
    result = _run_process_group(
        "printf hello",
        text=True,
        shell=True,
        capture_output=True,
        timeout=5,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == "hello"
    assert result.stderr == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows subprocess behavior")
def test_no_sandbox_windows_uses_subprocess_run(monkeypatch, tmp_path):
    captured = {}
    expected = SimpleNamespace(stdout="ok", stderr="", returncode=0)

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr("ursa.sandbox.base.subprocess.run", run)
    result = _run_process_group(
        "echo hello",
        text=True,
        shell=True,
        capture_output=True,
        timeout=5,
        cwd=tmp_path,
    )

    assert result is expected
    assert captured["command"] == "echo hello"
    assert captured["kwargs"]["check"] is False
    assert captured["kwargs"]["timeout"] == 5


def test_mac_sandbox_uses_process_group_runner(monkeypatch, tmp_path):
    captured = {}
    expected = SimpleNamespace(stdout="ok", stderr="", returncode=0)

    def run_group(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(MacSandbox, "sbx_path", "/usr/bin/sandbox-exec")
    monkeypatch.setattr("ursa.sandbox.mac._run_process_group", run_group)
    sandbox = MacSandbox(
        cwd=tmp_path,
        extra_writable_paths=[],
        timeout=7,
    )

    result = sandbox.execute("python loop.py")

    assert result is expected
    assert captured["command"][-3:] == [sandbox.shell, "-c", "python loop.py"]
    assert captured["kwargs"] == {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "shell": False,
        "capture_output": True,
        "timeout": 7,
        "cwd": str(tmp_path.resolve()),
    }


def test_run_command_replaces_undecodable_output(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)
    (tmp_path / "data.bin").write_bytes(b"ok\xba\xffend")

    result = invoke_with_parent_run(
        lambda config: run_command.func(
            "cat data.bin",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                tool_call_id="binary",
                thread_id="run-thread",
                config=config,
            ),
        )
    )

    assert result.startswith("STDOUT:\nok��end")


def test_run_command_reports_subprocess_errors(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def raise_error(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr("ursa.sandbox.base._run_process_group", raise_error)

    result = invoke_with_parent_run(
        lambda config: run_command.func(
            "ls",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                tool_call_id="oserror",
                thread_id="run-thread",
                config=config,
            ),
        )
    )

    assert "OSError: boom" in result
