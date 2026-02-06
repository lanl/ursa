from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.chat_models import BaseChatModel
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from tests.tools.utils import make_runtime
from ursa.tools.run_command_tool import SafetyAssessment, run_command
from ursa.util.types import AsciiStr


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

        class Invoker:
            def invoke(self_inner, prompt):
                captured["prompt"] = prompt
                return {"is_safe": is_safe, "reason": reason}

        return Invoker()

    monkeypatch.setattr(
        chat_model.__class__,
        "with_structured_output",
        fake_with_structured_output,
    )
    return captured


def test_run_command_invokes_subprocess_in_workspace(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    recorded = {}
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def fake_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return SimpleNamespace(stdout="output", stderr="")

    monkeypatch.setattr("ursa.tools.run_command_tool.subprocess.run", fake_run)

    result = run_command.func(
        "echo hi",
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            thread_id="run-thread",
            tool_call_id="run-call",
        ),
    )

    assert result == "STDOUT:\noutput\nSTDERR:\n"
    assert recorded["kwargs"]["cwd"] == tmp_path
    assert recorded["kwargs"]["shell"] is True


def test_run_command_truncates_output(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    long_stdout = "a" * 200
    long_stderr = "b" * 200
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=long_stdout, stderr=long_stderr
        ),
    )

    result = run_command.func(
        "noop",
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            limit=64,
            tool_call_id="truncate",
            thread_id="run-thread",
        ),
    )

    stdout_part, stderr_part = result.split("STDERR:\n", maxsplit=1)
    stdout_body = stdout_part.replace("STDOUT:\n", "", 1).rstrip("\n")
    stderr_body = stderr_part

    assert "... [snipped" in stdout_body
    assert "... [snipped" in stderr_body
    assert len(stdout_body) < len(long_stdout)
    assert len(stderr_body) < len(long_stderr)


def test_run_command_handles_keyboard_interrupt(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    _patch_safety_result(monkeypatch, chat_model, is_safe=True)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run", raise_interrupt
    )

    result = run_command.func(
        "sleep 1",
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="interrupt",
            thread_id="run-thread",
        ),
    )

    assert "KeyboardInterrupt:" in result


def test_run_command_rejects_unicode_input(
    tmp_path: Path, chat_model: BaseChatModel
):
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        thread_id="run-thread",
        tool_call_id="unicode",
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


def test_run_command_blocks_commands_that_fail_safety_check(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    captured = _patch_safety_result(
        monkeypatch, chat_model, is_safe=False, reason="Not safe in test"
    )
    monkeypatch.setattr(
        "ursa.tools.run_command_tool.subprocess.run",
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

    result = run_command.func(
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
