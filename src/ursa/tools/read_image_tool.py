import base64
import logging
import mimetypes
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.messages.content import (
    ImageContentBlock,
    create_image_block,
)
from langchain_core.tools import tool

from ursa.agents.base import AgentContext

logger = logging.getLogger(__name__)


@tool
def read_image_tool(
    image_path: str, runtime: ToolRuntime[AgentContext]
) -> list[ImageContentBlock]:
    """Read an image from disk to ingest into the workflow"""
    image_path: Path = runtime.context.workspace.joinpath(image_path)
    try:
        result = image_block_from_file(
            image_path,
            workspace=runtime.context.workspace,
        )
    except Exception as e:
        logger.exception(
            "Image read failed",
            exc_info=e,
            extra={"image_path": str(image_path)},
        )
        raise

    return [result]


def image_block_from_file(
    filename: Path,
    workspace: Path | None = None,
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
    data = base64.b64encode(filename.read_bytes()).decode("utf-8")

    # If workspace is provided, try resolve a local path
    file_id = None
    if workspace:
        try:
            file_id = str(filename.relative_to(workspace))
        except ValueError:
            pass  # relative_to failed

    return create_image_block(
        base64=data,
        mime_type=mime_type,
        file_id=file_id,
    )
