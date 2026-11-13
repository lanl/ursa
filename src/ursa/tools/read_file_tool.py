import os
from pathlib import Path
from typing import Annotated

import requests
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.skills import skill_directories
from ursa.tools.write_code_tool import _validate_file_path
from ursa.util.events import ToolEvents
from ursa.util.rendering import file_artifact
from ursa.util.types import (
    AsciiValidationError,
    ascii_validation_message,
    validate_ascii,
)


def _validate_read_path(
    filename: str, workspace: Path
) -> tuple[Path | None, str | None]:
    """Resolve a read path and enforce workspace/skills containment."""
    workspace = workspace.resolve()
    full_filename = workspace.joinpath(filename).resolve()
    if _allow_unsafe_reads_enabled() or (
        full_filename.is_relative_to(workspace)
        or any(
            full_filename.is_relative_to(root) for root in skill_directories()
        )
    ):
        return full_filename, None
    return None, (
        f"File path '{filename}' resolves outside workspace directory "
        "and any configured skills directory. Files must be read within "
        "the workspace or a configured skills directory."
    )


@tool
def read_file(filename: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Read a file from the workspace or the configured skills directory.

    URSA_SKILLS_PATH permits reads from the configured skills directories.
    URSA_ALLOW_UNSAFE_READS enables reads outside these permitted directories.

    - If filename ends with .pdf, extract text from the PDF.
    - If extracted text is very small (likely scanned), optionally run OCR to add a text layer.
    - Otherwise read as UTF-8 text.

    Args:
        filename: Path relative to the workspace, or an absolute path within
            the workspace or a directory specified by URSA_SKILLS_PATH.
            Paths and symlinks resolving outside these directories are rejected
            unless the user enables URSA_ALLOW_UNSAFE_READS.

    Returns:
        Extracted text content.
    """
    from ursa.util.parse import read_text_from_file

    full_filename, error = _validate_read_path(
        filename, runtime.context.workspace
    )
    events = ToolEvents.from_runtime("read_file", runtime)

    if error is not None:
        events.emit(
            "Failed to read file",
            stage="read",
            phase="error",
            filename=filename,
            error=error,
        )
        return f"Failed to read {filename}: {error}"
    assert full_filename is not None
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


def _allow_unsafe_reads_enabled() -> bool:
    """Return whether outside reads are explicitly enabled via environment.

    Set URSA_ALLOW_UNSAFE_READS to one of: 1, true, yes, on.
    """
    return os.getenv("URSA_ALLOW_UNSAFE_READS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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

    destination, path_error = _validate_file_path(
        output_path, runtime.context.workspace
    )
    if path_error is not None:
        return f"Error downloading file: {path_error}"
    assert destination is not None

    try:
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Ensure directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        ToolEvents.from_runtime("download_file", runtime).emit(
            "Downloaded file",
            stage="Download",
            path=str(destination),
        )

        # Write file to disk
        with open(destination, "wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

        return f"File successfully downloaded to: {destination}"

    except Exception as e:  # noqa: BLE001
        return f"Error downloading file: {e!s}"
