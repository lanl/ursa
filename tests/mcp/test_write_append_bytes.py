# pytest -q tests/mcp/test_write_append_bytes.py
import base64
import shutil
import tempfile
from pathlib import Path

import pytest

# Import the tool functions & helpers from your execution_agent module
# Adjust this import if your package name or path differs.
from ursa.agents.execution_agent import (
    _cache_put,  # intentionally black-box-ish but stable enough for tests
    append_bytes,
    write_bytes,
)

# The tools expect InjectedState + tool_call_id style args.
# We'll emulate exactly what ToolNode would pass.


@pytest.fixture
def tmp_ws():
    d = tempfile.mkdtemp(prefix="ea_bytes_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_state(ws):
    return {"workspace": ws, "messages": [], "code_files": []}


def tool_id(n="t1"):
    return f"call_{n}"


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_write_bytes_with_blob_ref(tmp_ws):
    raw = b"%PDF-1.4\nFAKE\n"
    blob = _cache_put(raw)

    out = write_bytes.func(
        data_b64=None,
        filename="data/doc.pdf",
        tool_call_id=tool_id("pdf"),
        state=make_state(tmp_ws),
        allow_text=False,
        overwrite=True,
        blob_ref=blob,
    )
    # write_bytes returns a Command(update=...) so .func unwraps to that Command
    upd = out.update if hasattr(out, "update") else out
    assert "messages" in upd
    assert any("Wrote" in m.content for m in upd["messages"])
    path = Path(tmp_ws) / "data/doc.pdf"
    assert path.exists()
    assert path.read_bytes() == raw  # exact bytes written


def test_write_bytes_idempotent_if_blob_missing_and_file_exists(tmp_ws):
    # First: write successfully
    raw = b"hello, csv\n"
    blob = _cache_put(raw)
    write_bytes.func(
        data_b64=None,
        filename="data/a.csv",
        tool_call_id=tool_id("w1"),
        state=make_state(tmp_ws),
        allow_text=True,
        overwrite=True,
        blob_ref=blob,
    )
    path = Path(tmp_ws) / "data/a.csv"
    assert path.exists()
    assert path.read_bytes() == raw

    # Second: simulate a *duplicate* tool call where the blob was already consumed & deleted.
    out2 = write_bytes.func(
        data_b64=None,
        filename="data/a.csv",
        tool_call_id=tool_id("w2"),
        state=make_state(tmp_ws),
        allow_text=True,
        overwrite=True,
        blob_ref="blob_not_there_anymore",
    )
    upd = out2.update if hasattr(out2, "update") else out2
    # Should *not* crash; should acknowledge existing file as prior success.
    assert any(
        "already exists; assuming prior write succeeded" in m.content
        for m in upd["messages"]
    )


def test_write_bytes_with_base64_fallback(tmp_ws):
    raw = b"col1,col2\n1,2\n"
    out = write_bytes.func(
        data_b64=b64(raw),
        filename="data/small.csv",
        tool_call_id=tool_id("b64"),
        state=make_state(tmp_ws),
        allow_text=True,
        overwrite=True,
        blob_ref=None,
    )
    out.update if hasattr(out, "update") else out
    path = Path(tmp_ws) / "data/small.csv"
    assert path.exists()
    assert path.read_bytes() == raw


def test_write_bytes_refuses_code_when_allow_text_false(tmp_ws):
    raw = b"print('hello')\n"
    blob = _cache_put(raw)
    out = write_bytes.func(
        data_b64=None,
        filename="src/app.py",
        tool_call_id=tool_id("code"),
        state=make_state(tmp_ws),
        allow_text=False,  # important
        overwrite=True,
        blob_ref=blob,
    )
    upd = out.update if hasattr(out, "update") else out
    assert any(
        "Refusing to write probable source/text" in m.content
        for m in upd["messages"]
    )


def test_append_bytes_happy_path(tmp_ws):
    # Start with a first chunk via write_bytes
    first = b"A" * 10
    blob1 = _cache_put(first)
    write_bytes.func(
        data_b64=None,
        filename="data/huge.bin",
        tool_call_id=tool_id("first"),
        state=make_state(tmp_ws),
        allow_text=True,
        overwrite=True,
        blob_ref=blob1,
    )
    path = Path(tmp_ws) / "data/huge.bin"
    assert path.exists()
    assert path.stat().st_size == 10

    # Append a second chunk of 6 bytes with expected_offset=10
    second = b"B" * 6
    blob2 = _cache_put(second)
    append_bytes.func(
        data_b64=None,
        filename="data/huge.bin",
        tool_call_id=tool_id("append"),
        state=make_state(tmp_ws),
        expected_offset=10,
        blob_ref=blob2,
    )
    # Combined file is 16 bytes
    assert path.stat().st_size == 16
    assert path.read_bytes() == first + second


def test_append_bytes_rejects_bad_offset(tmp_ws):
    # Create file with 5 bytes
    first = b"12345"
    blob1 = _cache_put(first)
    write_bytes.func(
        data_b64=None,
        filename="data/x.dat",
        tool_call_id=tool_id("w"),
        state=make_state(tmp_ws),
        allow_text=True,
        overwrite=True,
        blob_ref=blob1,
    )
    # Try appending with wrong expected_offset
    chunk = b"yyy"
    blob2 = _cache_put(chunk)
    out = append_bytes.func(
        data_b64=None,
        filename="data/x.dat",
        tool_call_id=tool_id("bad"),
        state=make_state(tmp_ws),
        expected_offset=7,  # wrong
        blob_ref=blob2,
    )
    upd = out.update if hasattr(out, "update") else out
    assert any("Offset mismatch" in m.content for m in upd["messages"])
    # File should remain unchanged
    assert (Path(tmp_ws) / "data/x.dat").read_bytes() == first
