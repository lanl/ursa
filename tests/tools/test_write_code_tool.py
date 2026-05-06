import time
from pathlib import Path

from langchain.chat_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from tests.tools.utils import make_runtime
from ursa.tools.write_code_tool import (
    edit_code,
    write_code,
    write_code_with_repo,
)


def test_write_code_records_store_entry(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    store = InMemoryStore()
    events = []
    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        lambda event_name, payload, config: events.append((
            event_name,
            payload,
            config,
        )),
    )
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        store=store,
        tool_call_id="tc-1",
        thread_id="thread-1",
    )

    write_code.func(code="print(42)", filename="sample.py", runtime=runtime)

    item = store.get(("workspace", "file_edit"), "sample.py")
    assert item is not None
    assert item.value["tool_call_id"] == "tc-1"
    assert item.value["thread_id"] == "thread-1"
    assert item.value["modified"] <= time.time()
    assert events[0] == (
        "ursa_agent_progress",
        {
            "tool": "write_code",
            "tool_call_id": "tc-1",
            "stage": "write",
            "message": "Writing file",
            "phase": "start",
            "filename": "sample.py",
            "path": str(tmp_path / "sample.py"),
        },
        runtime.config,
    )
    event_name, payload, config = events[1]
    assert event_name == "ursa_agent_progress"
    assert config == runtime.config
    assert payload["tool"] == "write_code"
    assert payload["tool_call_id"] == "tc-1"
    assert payload["stage"] == "write"
    assert payload["message"] == "File written"
    assert payload["phase"] == "end"
    assert payload["filename"] == "sample.py"
    assert payload["path"] == str(tmp_path / "sample.py")
    assert isinstance(payload["elapsed_ms"], float)


def test_edit_code_updates_file_and_records(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "app.py"
    target.write_text("print('hello')\nprint('hello')\n", encoding="utf-8")
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        store=store,
        tool_call_id="tc-edit",
        thread_id="thread-7",
    )

    result = edit_code.func(
        old_code="print('hello')",
        new_code="print('bye')",
        filename="app.py",
        runtime=runtime,
    )

    assert "updated successfully" in result
    assert (
        target.read_text(encoding="utf-8") == "print('bye')\nprint('hello')\n"
    )
    item = store.get(("workspace", "file_edit"), "app.py")
    assert item is not None
    assert item.value["tool_call_id"] == "tc-edit"
    assert item.value["thread_id"] == "thread-7"


def test_edit_code_noop_when_old_code_missing(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "script.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        store=store,
        tool_call_id="tc-miss",
    )

    result = edit_code.func(
        old_code="print('world')",
        new_code="print('bye')",
        filename="script.py",
        runtime=runtime,
    )

    assert "No changes made" in result
    assert target.read_text(encoding="utf-8") == "print('hello')\n"
    assert store.get(("workspace", "file_edit"), "script.py") is None


def test_write_code_with_repo_records_store_entry(
    tmp_path: Path, chat_model: BaseChatModel
):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        store=store,
        tool_call_id="tc-2",
        thread_id="thread-2",
    )

    result = write_code_with_repo.func(
        code="print(7)",
        filename="repo/sample.py",
        runtime=runtime,
        repo_path=str(repo),
    )

    assert "written successfully" in result
    item = store.get(("workspace", "file_edit"), "repo/sample.py")
    assert item is not None
    assert item.value["tool_call_id"] == "tc-2"
    assert item.value["thread_id"] == "thread-2"


def test_edit_code_missing_file(tmp_path: Path, chat_model: BaseChatModel):
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        store=store,
        tool_call_id="tc-missing",
    )

    result = edit_code.func(
        old_code="print('hello')",
        new_code="print('bye')",
        filename="missing.py",
        runtime=runtime,
    )

    assert "Failed: missing.py not found" in result
    assert store.get(("workspace", "file_edit"), "missing.py") is None
