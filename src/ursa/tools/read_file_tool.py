from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.parse import read_pdf_text, read_text_file


@tool
def read_file(filename: str, runtime: ToolRuntime[AgentContext]) -> str:
    """
    Reads in a file with a given filename into a string. Can read in PDF
    or files that are text/ASCII. Uses a PDF parser if the filename ends
    with .pdf (case-insensitive)

    Args:
        filename: string filename to read in
    """
    full_filename = runtime.context.workspace.joinpath(filename)

    print("[READING]: ", full_filename)
    try:
        if full_filename.suffix.lower() == ".pdf":
            file_contents = read_pdf_text(full_filename)
        else:
            file_contents = read_text_file(full_filename)
    except Exception as e:
        print(f"[Error]: {e}")
        file_contents = f"[Error]: {e}"
    return file_contents
