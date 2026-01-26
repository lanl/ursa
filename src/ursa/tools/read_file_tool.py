import os
import requests
from pathlib import Path
from typing import Literal
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from ursa.util.parse import read_pdf_text, read_text_file


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
    try:
        if full_filename.lower().endswith(".pdf"):
            file_contents = read_pdf_text(full_filename)
        else:
            file_contents = read_text_file(full_filename)
    except Exception as e:
        print(f"[Error]: {e}")
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

