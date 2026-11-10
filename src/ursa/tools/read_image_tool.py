import base64
import logging
import mimetypes
from pathlib import Path

import pymupdf
from langchain.tools import ToolException, ToolRuntime
from langchain_core.messages.content import (
    ImageContentBlock,
    TextContentBlock,
    create_image_block,
)
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.tools.read_file_tool import _validate_read_path

logger = logging.getLogger(__name__)


class ImageReadError(ToolException):
    """A recoverable image path or image-content error from a tool call."""


class ImageSizeError(ImageReadError, ValueError):
    """An image exceeds the tool's input-size limit."""


@tool
def read_image_tool(
    image_path: str, runtime: ToolRuntime[AgentContext]
) -> list[TextContentBlock | ImageContentBlock]:
    """Read an image from the workspace or configured skills directories."""
    workspace = runtime.context.workspace.resolve()
    resolved_path, path_error = _validate_read_path(image_path, workspace)
    if path_error is not None:
        raise ValueError(path_error)
    assert resolved_path is not None
    try:
        image_block = image_block_from_file(resolved_path)
    except (ImageReadError, OSError, pymupdf.FileDataError) as exc:
        logger.exception(
            "Image read failed",
            exc_info=exc,
            extra={"image_path": str(resolved_path)},
        )
        if isinstance(exc, ImageReadError):
            raise
        raise ImageReadError(
            f"Could not read image {image_path!r}: {exc}"
        ) from exc

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
        raise ImageSizeError(
            f"File too large: {file_size / 1024 / 1024:.2f}MB "
            f"(max: {max_size_mb:0.1f}MB)"
        )

    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        raise ImageReadError(f"Unsupported image type: {filename.name}")

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
