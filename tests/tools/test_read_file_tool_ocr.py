import os
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.tools.utils import read_file_content
from ursa.util import parse


def _touch(p: Path, content: bytes = b"%PDF-1.4\n%fake\n") -> None:
    p.write_bytes(content)
    # ensure mtime changes if needed
    os.utime(p, None)


@pytest.fixture(autouse=True)
def ocr_settings(monkeypatch):
    monkeypatch.setenv("READ_FILE_OCR", "1")
    monkeypatch.setenv("READ_FILE_OCR_MIN_PAGES", "3")
    monkeypatch.setenv("READ_FILE_OCR_MIN_CHARS", "3000")


@pytest.fixture
def ocr(monkeypatch):
    operation = Mock()
    monkeypatch.setattr(parse, "_ocr_to_searchable_pdf", operation)
    return operation


def test_no_ocr_when_text_is_sufficient(tmp_path, monkeypatch, ocr):
    pdf = tmp_path / "doc.pdf"
    _touch(pdf)

    # Pretend this PDF already has plenty of text
    monkeypatch.setattr(parse, "read_text_pdf", lambda path: "X" * 5000)
    monkeypatch.setattr(parse, "_pdf_page_count", lambda path: 10)

    out = read_file_content("doc.pdf", tmp_path)

    assert len(out) == 5000
    ocr.assert_not_called()


def test_ocr_runs_and_uses_ocr_pdf(tmp_path, monkeypatch, ocr):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    monkeypatch.setattr(parse, "_pdf_page_count", lambda path: 22)

    # Make read_pdf return tiny text for original, large for *.ocr.pdf
    def fake_read_pdf(path: Path) -> str:
        if ".ocr." in str(path) and str(path).endswith(".pdf"):
            return "OCR_TEXT_" + ("Y" * 4000)
        return "tiny"

    monkeypatch.setattr(parse, "read_text_pdf", fake_read_pdf)

    def fake_ocr(src: str, dst: str, *, mode: str = "skip") -> None:
        Path(dst).write_bytes(b"%PDF-1.4\n%ocr\n")

    ocr.side_effect = fake_ocr

    out = read_file_content("scan.pdf", tmp_path)

    assert out.startswith("OCR_TEXT_")
    assert len(out) > 3000
    assert (tmp_path / "scan.pdf.ocr.skip.pdf").exists()


def test_ocr_cache_skips_second_run(tmp_path, monkeypatch, ocr):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    ocr_pdf = tmp_path / "scan.pdf.ocr.skip.pdf"
    _touch(ocr_pdf, content=b"%PDF-1.4\n%cached\n")

    # Make cached OCR newer than source
    time.sleep(0.01)
    os.utime(ocr_pdf, None)

    monkeypatch.setattr(parse, "_pdf_page_count", lambda path: 22)

    # Original tiny, OCR big
    def fake_read_pdf(path: Path) -> str:
        return "tiny" if ".ocr." not in str(path) else "Z" * 5000

    monkeypatch.setattr(parse, "read_text_pdf", fake_read_pdf)

    out = read_file_content("scan.pdf", tmp_path)

    assert len(out) == 5000
    ocr.assert_not_called()


def test_ocr_failure_returns_original_text(tmp_path, monkeypatch, ocr):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    monkeypatch.setattr(parse, "_pdf_page_count", lambda path: 22)
    monkeypatch.setattr(parse, "read_pdf", lambda path: "tiny")

    ocr.side_effect = RuntimeError("ocr failed")

    out = read_file_content("scan.pdf", tmp_path)

    assert out == "tiny"
