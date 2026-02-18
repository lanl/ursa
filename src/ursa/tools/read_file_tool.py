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
