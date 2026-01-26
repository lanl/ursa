from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.tools.utils import make_runtime
from ursa.tools.run_command_tool import run_command
from ursa.util.types import AsciiStr


def test_run_command_invokes_subprocess_in_workspace(
    monkeypatch, tmp_path: Path
):
    recorded = {}

    def fake_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return SimpleNamespace(stdout="output", stderr="")

    monkeypatch.setattr("ursa.tools.run_command_tool.subprocess.run", fake_run)

    result = run_command.func(
        "echo hi",
        runtime=make_runtime(
            tmp_path, thread_id="run-thread", tool_call_id="run-call"
        ),
    )

    assert result == "STDOUT:\noutput\nSTDERR:\n"
    assert recorded["kwargs"]["cwd"] == tmp_path
    assert recorded["kwargs"]["shell"] is True


def test_run_command_truncates_output(monkeypatch, tmp_path: Path):
    long_stdout = "a" * 200
    long_stderr = "b" * 200

    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=long_stdout, stderr=long_stderr
        ),
    )

    result = run_command.func(
        "noop",
        runtime=make_runtime(
            tmp_path, limit=64, tool_call_id="truncate", thread_id="run-thread"
        ),
    )

    stdout_part, stderr_part = result.split("STDERR:\n", maxsplit=1)
    stdout_body = stdout_part.replace("STDOUT:\n", "", 1).rstrip("\n")
    stderr_body = stderr_part

    assert "... [snipped" in stdout_body
    assert "... [snipped" in stderr_body
    assert len(stdout_body) < len(long_stdout)
    assert len(stderr_body) < len(long_stderr)


def test_run_command_handles_keyboard_interrupt(monkeypatch, tmp_path: Path):
    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run", raise_interrupt
    )

    result = run_command.func(
        "sleep 1",
        runtime=make_runtime(
            tmp_path, tool_call_id="interrupt", thread_id="run-thread"
        ),
    )

    assert "KeyboardInterrupt:" in result


def test_run_command_rejects_unicode_input(tmp_path: Path):
    runtime = make_runtime(
        tmp_path, thread_id="run-thread", tool_call_id="unicode"
    )

    with pytest.raises(ValidationError):
        run_command.invoke({"query": "ls café", "runtime": runtime})


def test_run_command_schema_has_regex_constraint():
    field = run_command.args_schema.model_fields["query"]
    assert field.annotation is str
    constraints = [meta for meta in field.metadata if hasattr(meta, "pattern")]
    assert constraints
    ascii_constraints = [
        meta for meta in AsciiStr.__metadata__ if hasattr(meta, "pattern")
    ]
    assert ascii_constraints
    assert constraints[0].pattern == ascii_constraints[0].pattern
