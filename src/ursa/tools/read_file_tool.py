from pathlib import Path

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
    workspace = Path(runtime.context.workspace).resolve()

    # Build the full path
    if Path(filename).is_absolute():
        full_filename = Path(filename).resolve()
    else:
        full_filename = (workspace / filename).resolve()

    # Validate it's within the workspace
    try:
        full_filename.relative_to(workspace)
    except ValueError:
        return (
            f"Error: File path '{filename}' resolves outside workspace directory. "
            f"Files must be read from within the workspace at {workspace}"
        )

    print("[READING]:", full_filename)
    # Move all the reading to a function in the parse util
    return read_text_from_file(full_filename)
