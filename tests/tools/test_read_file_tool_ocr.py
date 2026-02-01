import os
import shutil
import time
from pathlib import Path

import pytest

# import the module (not just the symbol) so monkeypatch works cleanly
import ursa.tools.read_file_tool as rft


def _touch(p: Path, content: bytes = b"%PDF-1.4\n%fake\n") -> None:
    p.write_bytes(content)
    # ensure mtime changes if needed
    os.utime(p, None)


def _call_tool(filename: str, workspace: Path) -> str:
    # If @tool produced a Tool object, it should have .invoke
    # InjectedState usually flows via state; passing state directly works in practice for unit tests.
    return rft.read_file.func(
        filename=filename, state={"workspace": str(workspace)}
    )


def test_no_ocr_when_text_is_sufficient(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    _touch(pdf)

    monkeypatch.setenv("READ_FILE_OCR", "1")
    monkeypatch.setenv("READ_FILE_OCR_MIN_PAGES", "3")
    monkeypatch.setenv("READ_FILE_OCR_MIN_CHARS", "3000")

    # Pretend this PDF already has plenty of text
    monkeypatch.setattr(rft, "read_pdf_text", lambda path: "X" * 5000)
    monkeypatch.setattr(rft, "_pdf_page_count", lambda path: 10)

    called = {"ocr": 0}
    monkeypatch.setattr(
        rft,
        "_ocr_to_searchable_pdf",
        lambda src, dst: called.__setitem__("ocr", called["ocr"] + 1),
    )

    out = _call_tool("doc.pdf", tmp_path)
    print("EXTRACTED_LEN:", len(out))
    print("EXTRACTED_PREVIEW:", out[:300])

    assert len(out) == 5000
    assert called["ocr"] == 0


def test_ocr_runs_and_uses_ocr_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    monkeypatch.setenv("READ_FILE_OCR", "1")
    monkeypatch.setenv("READ_FILE_OCR_MIN_PAGES", "3")
    monkeypatch.setenv("READ_FILE_OCR_MIN_CHARS", "3000")

    monkeypatch.setattr(rft, "_pdf_page_count", lambda path: 22)

    # Make read_pdf_text return tiny text for original, large for *.ocr.pdf
    def fake_read_pdf_text(path: str) -> str:
        if path.endswith(".ocr.pdf"):
            return "OCR_TEXT_" + ("Y" * 4000)
        return "tiny"

    monkeypatch.setattr(rft, "read_pdf_text", fake_read_pdf_text)

    def fake_ocr(src: str, dst: str) -> None:
        Path(dst).write_bytes(b"%PDF-1.4\n%ocr\n")

    monkeypatch.setattr(rft, "_ocr_to_searchable_pdf", fake_ocr)

    out = _call_tool("scan.pdf", tmp_path)
    print("EXTRACTED_LEN:", len(out))
    print("EXTRACTED_PREVIEW:", out[:300])

    assert out.startswith("OCR_TEXT_")
    assert len(out) > 3000
    assert (tmp_path / "scan.pdf.ocr.pdf").exists()


def test_real_ocr_if_available(tmp_path):
    if not shutil.which("ocrmypdf"):
        pytest.skip("ocrmypdf not installed")
    # generate an image-only PDF here, then call the tool and assert output non-trivial


def test_ocr_cache_skips_second_run(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    ocr_pdf = tmp_path / "scan.pdf.ocr.pdf"
    _touch(ocr_pdf, content=b"%PDF-1.4\n%cached\n")

    # Make cached OCR newer than source
    time.sleep(0.01)
    os.utime(ocr_pdf, None)

    monkeypatch.setenv("READ_FILE_OCR", "1")
    monkeypatch.setenv("READ_FILE_OCR_MIN_PAGES", "3")
    monkeypatch.setenv("READ_FILE_OCR_MIN_CHARS", "3000")

    monkeypatch.setattr(rft, "_pdf_page_count", lambda path: 22)

    # Original tiny, OCR big
    def fake_read_pdf_text(path: str) -> str:
        return "tiny" if not path.endswith(".ocr.pdf") else "Z" * 5000

    monkeypatch.setattr(rft, "read_pdf_text", fake_read_pdf_text)

    called = {"ocr": 0}
    monkeypatch.setattr(
        rft,
        "_ocr_to_searchable_pdf",
        lambda src, dst: called.__setitem__("ocr", called["ocr"] + 1),
    )

    out = _call_tool("scan.pdf", tmp_path)
    print("EXTRACTED_LEN:", len(out))
    print("EXTRACTED_PREVIEW:", out[:300])

    assert len(out) == 5000
    assert called["ocr"] == 0


def test_ocr_failure_returns_original_text(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    _touch(pdf)

    monkeypatch.setenv("READ_FILE_OCR", "1")
    monkeypatch.setenv("READ_FILE_OCR_MIN_PAGES", "3")
    monkeypatch.setenv("READ_FILE_OCR_MIN_CHARS", "3000")

    monkeypatch.setattr(rft, "_pdf_page_count", lambda path: 22)
    monkeypatch.setattr(rft, "read_pdf_text", lambda path: "tiny")

    def fail_ocr(src: str, dst: str) -> None:
        raise RuntimeError("ocr failed")

    monkeypatch.setattr(rft, "_ocr_to_searchable_pdf", fail_ocr)

    out = _call_tool("scan.pdf", tmp_path)
    print("EXTRACTED_LEN:", len(out))
    print("EXTRACTED_PREVIEW:", out[:300])

    assert out == "tiny" or out.startswith(
        "[Error]"
    )  # depending on your exact error handling
