from pathlib import Path

import pytest
from langchain.chat_models import BaseChatModel

from tests.tools.utils import (
    invoke_with_event_recorder,
    invoke_with_parent_run,
    make_runtime,
)
from ursa.tools.read_file_tool import download_file_tool, read_file
from ursa.util import parse

FIXED_MONOTONIC_TIMESTAMP_NS = 123456789


@pytest.fixture(autouse=True)
def fixed_monotonic_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ursa.util.events.monotonic_ns",
        lambda: FIXED_MONOTONIC_TIMESTAMP_NS,
    )


def test_read_file_reads_text_from_workspace(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
):
    target = tmp_path / "example.txt"
    target.write_text("sample text", encoding="utf-8")
    result, recorder = invoke_with_event_recorder(
        read_file.func,
        filename=str(target.name),
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-file-call",
        ),
    )

    assert result == "sample text"
    assert recorder.events == [
        (
            "ursa_agent_progress",
            {
                "tool": "read_file",
                "tool_call_id": "read-file-call",
                "stage": "read",
                "message": "Reading file",
                "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
                "path": str(target),
            },
        )
    ]


def test_read_file_uses_pdf_reader(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
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

    result = invoke_with_parent_run(
        lambda config: read_file.func(
            filename="report.pdf",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                tool_call_id="pdf-call",
                config=config,
            ),
        )
    )

    assert result == "pdf contents"
    assert called["path"] == tmp_path / "report.pdf"


def test_download_file_rejects_invalid_ascii_output_path_without_cleaning(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
):
    existing = tmp_path / "caf.txt"
    existing.write_text("original", encoding="utf-8")
    runtime = make_runtime(tmp_path, llm=chat_model)
    monkeypatch.setattr(
        "ursa.tools.read_file_tool.requests.get",
        lambda *args, **kwargs: pytest.fail(
            "requests.get should not run for invalid output_path"
        ),
    )

    result = download_file_tool.func(
        url="https://example.com/file.txt",
        output_path="caf\u00e9.txt",
        runtime=runtime,
    )

    assert result.startswith("Invalid output_path:")
    assert "U+00E9" in result
    assert "corrected ASCII string" in result
    assert existing.read_text(encoding="utf-8") == "original"
