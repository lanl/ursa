import os
import shutil
import subprocess
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pypdf import PdfReader

from ursa.agents.base import AgentContext
from ursa.util.parse import read_pdf_text, read_text_file


def _pdf_page_count(path: Path) -> int:
    try:
        return len(PdfReader(path).pages)
    except Exception as e:
        print("[Error]: ", e)
        return 0


def ocrmypdf_is_installed() -> bool:
    return shutil.which("ocrmypdf") is not None


def _ocr_to_searchable_pdf(
    src_pdf: str, out_pdf: str, *, mode: str = "skip"
) -> None:
    # mode:
    #  - "skip":  only OCR pages that look like they need it (your current behavior)
    #  - "force": rasterize + OCR everything (fixes vector/outlined “no images” PDFs)
    if not ocrmypdf_is_installed():
        raise ImportError(
            "ocrmypdf was not found in your path. "
            "See installation instructions:"
            "https://github.com/ocrmypdf/OCRmyPDF?tab=readme-ov-file#installation"
        )

    cmd = ["ocrmypdf", "--rotate-pages", "--deskew", "--clean"]

    if mode == "force":
        cmd += ["--force-ocr"]
    else:
        cmd += ["--skip-text"]

    # Optional: dump a sidecar text file for debugging confidence
    if os.getenv("READ_FILE_OCR_SIDECAR", "0").lower() in ("1", "true", "yes"):
        cmd += ["--sidecar", out_pdf + ".txt"]

    cmd += [src_pdf, out_pdf]

    # Don’t swallow stderr/stdout when debugging
    debug = os.getenv("READ_FILE_OCR_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    subprocess.run(
        cmd,
        check=True,
        stdout=None if debug else subprocess.PIPE,
        stderr=None if debug else subprocess.PIPE,
        text=True,
    )


@tool
def read_file(filename: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Read a file from the workspace.

    - If filename ends with .pdf, extract text from the PDF.
    - If extracted text is very small (likely scanned), optionally run OCR to add a text layer.
    - Otherwise read as UTF-8 text.

    Args:
        filename: File name relative to the workspace directory.

    Returns:
        Extracted text content.
    """
    full_filename = runtime.context.workspace.joinpath(filename)

    print("[READING]:", full_filename)

    try:
        if not (full_filename.suffix.lower() == ".pdf"):
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

            mode_env = os.getenv("READ_FILE_OCR_MODE", "auto").lower()
            force_if_still_low = os.getenv(
                "READ_FILE_OCR_FORCE_IF_STILL_LOW", "1"
            ).lower() in ("1", "true", "yes")

            try:
                # First pass (skip-text) unless user forces always-force
                first_mode = "force" if mode_env == "force" else "skip"
                ocr_pdf = str(
                    src.with_suffix(src.suffix + f".ocr.{first_mode}.pdf")
                )

                if not os.path.exists(ocr_pdf) or os.path.getmtime(
                    ocr_pdf
                ) < os.path.getmtime(full_filename):
                    print(
                        f"[OCR]: mode={first_mode} ({len(text)} chars, {pages} pages) -> {ocr_pdf}"
                    )
                    _ocr_to_searchable_pdf(
                        full_filename, ocr_pdf, mode=first_mode
                    )
                else:
                    print(f"[OCR]: using cached OCR PDF -> {ocr_pdf}")

                text2 = read_pdf_text(ocr_pdf) or ""
                if len(text2) > len(text):
                    text = text2

                # Second pass: if still low and we weren’t already forcing, try force-ocr
                if (
                    force_if_still_low
                    and mode_env != "force"
                    and len(text) < min_chars
                ):
                    force_pdf = str(
                        src.with_suffix(src.suffix + ".ocr.force.pdf")
                    )
                    if not os.path.exists(force_pdf) or os.path.getmtime(
                        force_pdf
                    ) < os.path.getmtime(full_filename):
                        print(
                            f"[OCR]: still low after skip-text; retrying with force-ocr -> {force_pdf}"
                        )
                        _ocr_to_searchable_pdf(
                            full_filename, force_pdf, mode="force"
                        )
                    else:
                        print(
                            f"[OCR]: using cached force OCR PDF -> {force_pdf}"
                        )

                    text3 = read_pdf_text(force_pdf) or ""
                    if len(text3) > len(text):
                        text = text3

            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                # Missing ocrmypdf or OCR failed: keep original extraction
                print(f"[OCR Error]: {e}")
            except Exception as e:
                # Any other OCR-related failure: keep original extraction
                print(f"[OCR Error]: {e}")

        return text

    except subprocess.CalledProcessError as e:
        # OCR failed; return whatever we got from normal extraction
        err = (e.stderr or "")[:500]
        print(f"[OCR Error]: {err}")
        return text if text else f"[Error]: OCR failed: {err}"
    except Exception as e:
        print(f"[Error]: {e}")
<<<<<<< HEAD
        file_contents = f"[Error]: {e}"
    return file_contents


@tool
def download_file_tool(url: Annotated[str, "web link for the file"], output_path: Annotated[str, "local path to save the file"]=".") -> str:
    """ Download a file from a URL and save it locally.

    Arg:
        url (str): a string containing the URL of the file to download.
        output_path (str): the local path where the file should be saved.
    Returns:
        Confirmation message with the saved file path.
    """


    try:
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file to disk
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File successfully downloaded to: {output_path}"

    except Exception as e:
        return f"Error downloading file: {str(e)}"

=======
        return f"[Error]: {e}"
>>>>>>> main
