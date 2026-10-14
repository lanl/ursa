import base64
import logging
import mimetypes
from pathlib import Path

import pymupdf
from langchain.tools import ToolRuntime
from langchain_core.messages.content import (
    ImageContentBlock,
    TextContentBlock,
    create_image_block,
)
from langchain_core.tools import tool

from ursa.agents.base import AgentContext

logger = logging.getLogger(__name__)


@tool
def read_image_tool(
    image_path: str, runtime: ToolRuntime[AgentContext]
) -> list[TextContentBlock | ImageContentBlock]:
    """Read an image from disk to ingest into the workflow"""
    workspace = runtime.context.workspace
    resolved_path = workspace.joinpath(image_path)
    try:
        image_block = image_block_from_file(resolved_path)
    except Exception as e:
        logger.exception(
            "Image read failed",
            exc_info=e,
            extra={"image_path": str(resolved_path)},
        )
        raise

    try:
        display_path = str(resolved_path.relative_to(workspace))
    except ValueError:
        # Not under the workspace (e.g. an absolute path); label it the way
        # the caller referenced it.
        display_path = image_path

    # The text block surfaces the filename to the model. It must stay a
    # separate text block: ImageContentBlock.file_id is reserved for
    # provider-side file-store references (OpenAI file IDs, Google File API
    # URIs), not local paths, and providers may give it priority over the
    # base64 payload.
    text_block: TextContentBlock = {
        "type": "text",
        "text": f"Image file: {display_path}",
    }
    return [text_block, image_block]


def image_block_from_file(
    filename: Path,
    max_size_mb: float = 20,
) -> ImageContentBlock:
    file_size = filename.stat().st_size
    if file_size > (max_size_mb * 1024 * 1024):
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB "
            f"(max: {max_size_mb:0.1f}MB)"
        )

    mime_type, _ = mimetypes.guess_type(filename)
    assert mime_type is not None

    if mime_type == "image/svg+xml":
        with pymupdf.open(filename) as document:
            page = document[0]
            pixmap = page.get_pixmap()
            image_bytes = pixmap.tobytes("png")

        mime_type = "image/png"
    else:
        image_bytes = filename.read_bytes()

    data = base64.b64encode(image_bytes).decode("utf-8")

    return create_image_block(
        base64=data,
        mime_type=mime_type,
    )
