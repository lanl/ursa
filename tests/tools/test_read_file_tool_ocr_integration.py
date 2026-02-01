import os
import shutil
from pathlib import Path

import pytest

import ursa.tools.read_file_tool as rft


def _bins_available() -> bool:
    # ocrmypdf usually shells out to these
    needed = ["ocrmypdf", "tesseract", "qpdf", "gs"]
    return all(shutil.which(b) for b in needed)


def _make_scanned_pdf(path: Path, text: str = "Hello OCR 123") -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ModuleNotFoundError:
        pytest.skip(
            "Requires reportlab for generating a scanned PDF in this integration test"
        )

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.text((120, 140), text, fill="black", font=font)

    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(ImageReader(img), 0, 0, width=letter[0], height=letter[1])
    c.showPage()
    c.save()


def _call_tool(filename: str, workspace: Path) -> str:
    # Works with @tool wrappers
    return rft.read_file.func(
        filename=filename, state={"workspace": str(workspace)}
    )


def test_real_ocr_creates_openable_pdf_and_extracts_text(tmp_path):
    if not _bins_available():
        pytest.skip("Requires ocrmypdf+tesseract+qpdf+ghostscript on PATH")

    # Arrange
    os.environ["READ_FILE_OCR"] = "1"
    os.environ["READ_FILE_OCR_MIN_PAGES"] = "1"
    os.environ["READ_FILE_OCR_MIN_CHARS"] = "50"

    pdf = tmp_path / "scanned.pdf"
    _make_scanned_pdf(pdf, text="Hello OCR 123")

    # Act
    out = _call_tool("scanned.pdf", tmp_path)
    print("EXTRACTED_LEN:", len(out))
    print("EXTRACTED_PREVIEW:", out[:300])

    # Assert: text should include our phrase (allow normalization quirks)
    assert "hello" in out.lower()
    assert "ocr" in out.lower()
    assert "123" in out

    # Persist artifacts for humans to inspect
    repo_root = Path(__file__).resolve().parents[2]
    art_dir = repo_root / "artifacts"
    art_dir.mkdir(exist_ok=True)

    ocr_pdf = tmp_path / "scanned.pdf.ocr.pdf"
    assert ocr_pdf.exists()

    (art_dir / "scanned.input.pdf").write_bytes(pdf.read_bytes())
    (art_dir / "scanned.output.ocr.pdf").write_bytes(ocr_pdf.read_bytes())
    (art_dir / "scanned.output.txt").write_text(out, encoding="utf-8")

    # Basic sanity: output PDF should start with real PDF header
    assert (art_dir / "scanned.output.ocr.pdf").read_bytes().startswith(b"%PDF")
