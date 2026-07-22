from pathlib import Path
from typing import Annotated

import requests
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents
from ursa.util.parse import read_text_from_file
from ursa.util.rendering import file_artifact
from ursa.util.types import (
    AsciiValidationError,
    ascii_validation_message,
    validate_ascii,
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
    events = ToolEvents.from_runtime("read_file", runtime)
    with events.range(
        "read",
        "Reading file",
        done="File read",
        error="Failed to read file",
        path=str(full_filename),
    ) as span:
        # Move all the reading to a function in the parse util
        text = read_text_from_file(full_filename)
        span.update(artifact=file_artifact(full_filename, title="File read"))
    return text


@tool
def download_file_tool(
    url: Annotated[str, "web link for the file"],
    output_path: Annotated[
        str, "local path to save the file within the workspace"
    ],
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Download a file from a URL and save it locally.

    Arg:
        url (str): a string containing the URL of the file to download.
        output_path (str): the local path where the file should be saved.
                           Default to '.' for workspace root
    Returns:
        Confirmation message with the saved file path.
    """
    try:
        url = validate_ascii(url)
    except AsciiValidationError as exc:
        return ascii_validation_message("url", exc)
    try:
        output_path = validate_ascii(output_path)
    except AsciiValidationError as exc:
        return ascii_validation_message("output_path", exc)

    try:
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Ensure directory exists
        output_path = runtime.context.workspace / Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ToolEvents.from_runtime("download_file", runtime).emit(
            "Downloaded file",
            stage="Download",
            path=str(output_path),
        )

        # Write file to disk
        with open(output_path, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        return f"File successfully downloaded to: {output_path}"

    except Exception as e:  # noqa: BLE001
        return f"Error downloading file: {e!s}"
