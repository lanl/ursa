import os
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ..util.parse import read_pdf_text, read_text_file


# Tools for ExecutionAgent
@tool
def read_file(filename: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Reads in a file with a given filename into a string. Can read in PDF
    or files that are text/ASCII. Uses a PDF parser if the filename ends
    with .pdf (case-insensitive)

    Args:
        filename: string filename to read in
    """
    workspace_dir = state["workspace"]
    full_filename = os.path.join(workspace_dir, filename)

    print("[READING]: ", full_filename)
    if full_filename.lower().endswith(".pdf"):
        file_contents = read_pdf_text(full_filename)
    else:
        file_contents = read_text_file(full_filename)
    return file_contents
