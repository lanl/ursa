from pathlib import Path

from langchain.chat_models import BaseChatModel

from tests.tools.utils import make_runtime
from ursa.tools.read_file_tool import read_file
from ursa.util import parse


def test_read_file_reads_text_from_workspace(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "example.txt"
    target.write_text("sample text", encoding="utf-8")

    result = read_file.func(
        path=str(target.name),
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-file-call",
        ),
    )

    assert result == {
        "path": str(target.resolve()),
        "type": "file",
        "text": "sample text",
        "next_offset": len("sample text"),
    }


def test_read_file_uses_pdf_reader(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    called = {}

    def fake_pdf_reader(path: Path) -> str:
        called["path"] = path
        return "pdf contents"

    def fail_text_reader(path: Path) -> str:
        raise AssertionError("read_text_file should not be called for PDFs")

    monkeypatch.setattr(
        "ursa.tools.read_file_tool.read_text_from_file", fake_pdf_reader
    )
    monkeypatch.setattr(parse, "read_text_file", fail_text_reader)

    runtime = make_runtime(
        tmp_path,
        llm=chat_model,
        tool_call_id="pdf-call",
    )
    result = read_file.func(path="report.pdf", runtime=runtime)

    assert result["type"] == "file"
    assert result["text"] == "pdf contents"
    assert called["path"] == tmp_path / "report.pdf"
