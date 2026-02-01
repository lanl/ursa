import os
import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ursa.util.parse import read_pdf_text, read_text_file


def _pdf_page_count(path: str) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(path).pages)
    except Exception:
        return 0


def _ocr_to_searchable_pdf(src_pdf: str, out_pdf: str) -> None:
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--rotate-pages",
        "--deskew",
        "--clean",
        src_pdf,
        out_pdf,
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@tool
def read_file(filename: str, state: Annotated[dict, InjectedState]) -> str:
    """Read a file from the workspace.

    - If filename ends with .pdf, extract text from the PDF.
    - If extracted text is very small (likely scanned), optionally run OCR to add a text layer.
    - Otherwise read as UTF-8 text.

    Args:
        filename: File name relative to the workspace directory.
        state: Injected graph state containing "workspace".

    Returns:
        Extracted text content.
    """
    workspace_dir = state["workspace"]
    full_filename = os.path.join(workspace_dir, filename)

    print("[READING]:", full_filename)

    try:
        if not full_filename.lower().endswith(".pdf"):
            return read_text_file(full_filename)

        # 1) normal extraction
        text = read_pdf_text(full_filename) or ""

        # 2) decide if OCR fallback is needed
        pages = _pdf_page_count(full_filename)
        ocr_enabled = os.getenv("READ_FILE_OCR", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        min_pages = int(os.getenv("READ_FILE_OCR_MIN_PAGES", "3"))
        min_chars = int(os.getenv("READ_FILE_OCR_MIN_CHARS", "3000"))

        if ocr_enabled and pages >= min_pages and len(text) < min_chars:
            src = Path(full_filename)
            ocr_pdf = str(src.with_suffix(src.suffix + ".ocr.pdf"))

            # cache if already OCR’d and up-to-date
            if not os.path.exists(ocr_pdf) or os.path.getmtime(
                ocr_pdf
            ) < os.path.getmtime(full_filename):
                print(
                    f"[OCR]: low extracted text ({len(text)} chars, {pages} pages) -> {ocr_pdf}"
                )
                _ocr_to_searchable_pdf(full_filename, ocr_pdf)
            else:
                print(f"[OCR]: using cached OCR PDF -> {ocr_pdf}")

            text2 = read_pdf_text(ocr_pdf) or ""
            if len(text2) > len(text):
                text = text2

        return text

    except subprocess.CalledProcessError as e:
        # OCR failed; return whatever we got from normal extraction
        err = (e.stderr or "")[:500]
        print(f"[OCR Error]: {err}")
        return text if text else f"[Error]: OCR failed: {err}"
    except Exception as e:
        print(f"[Error]: {e}")
        return f"[Error]: {e}"
