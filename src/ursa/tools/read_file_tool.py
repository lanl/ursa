from pathlib import Path
from typing import Annotated

import requests
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.parse import read_text_from_file


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
    # Move all the reading to a function in the parse util
    text = read_text_from_file(full_filename)
    return text


@tool
def download_file_tool(
    url: Annotated[str, "web link for the file"],
    output_path: Annotated[str, "local path to save the file"] = ".",
) -> str:
    """Download a file from a URL and save it locally.

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

        return f"[Error]: {e}"
