import time
from pathlib import Path

from langgraph.store.memory import InMemoryStore

from tests.tools.utils import make_runtime
from ursa.tools.write_code_tool import edit_code, write_code


def test_write_code_strips_fences_and_writes(tmp_path: Path):
    runtime = make_runtime(
        tmp_path, thread_id="thread-1", tool_call_id="write-call"
    )

    fenced = """```python
print(\"hello\")
```"""

    result = write_code.func(code=fenced, filename="hello.py", runtime=runtime)

    target = tmp_path / "hello.py"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == 'print("hello")\n'
    assert "written successfully" in result


def test_write_code_records_store_entry(tmp_path: Path):
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path, store=store, tool_call_id="tc-1", thread_id="thread-1"
    )

    write_code.func(code="print(42)", filename="sample.py", runtime=runtime)

    item = store.get(("workspace", "file_edit"), "sample.py")
    assert item is not None
    assert item.value["tool_call_id"] == "tc-1"
    assert item.value["thread_id"] == "thread-1"
    assert item.value["modified"] <= time.time()


def test_edit_code_updates_file_and_records(tmp_path: Path):
    target = tmp_path / "app.py"
    target.write_text(
        "print('hello')\nprint('hello')\n", encoding="utf-8"
    )
    store = InMemoryStore()
    runtime = make_runtime(
        tmp_path, store=store, tool_call_id="tc-edit", thread_id="thread-7"
    )

    result = edit_code.func(
        old_code="print('hello')",
        new_code="print('bye')",
        filename="app.py",
        runtime=runtime,
    )

    assert "updated successfully" in result
    assert (
        target.read_text(encoding="utf-8")
        == "print('bye')\nprint('hello')\n"
    )
    item = store.get(("workspace", "file_edit"), "app.py")
    assert item is not None
    assert item.value["tool_call_id"] == "tc-edit"
    assert item.value["thread_id"] == "thread-7"


def test_edit_code_noop_when_old_code_missing(tmp_path: Path):
    target = tmp_path / "script.py"
    target.write_text("print('hello')\n", encoding="utf-8")
    store = InMemoryStore()
    runtime = make_runtime(tmp_path, store=store, tool_call_id="tc-miss")

    result = edit_code.func(
        old_code="print('world')",
        new_code="print('bye')",
        filename="script.py",
        runtime=runtime,
    )

    assert "No changes made" in result
    assert target.read_text(encoding="utf-8") == "print('hello')\n"
    assert store.get(("workspace", "file_edit"), "script.py") is None


def test_edit_code_missing_file(tmp_path: Path):
    store = InMemoryStore()
    runtime = make_runtime(tmp_path, store=store, tool_call_id="tc-missing")

    result = edit_code.func(
        old_code="print('hello')",
        new_code="print('bye')",
        filename="missing.py",
        runtime=runtime,
    )

    assert "Failed: missing.py not found" in result
    assert store.get(("workspace", "file_edit"), "missing.py") is None
