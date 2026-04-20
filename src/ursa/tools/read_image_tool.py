import base64
import mimetypes
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext


@tool
def read_image_tool(
    image_path: str, runtime: ToolRuntime[AgentContext]
) -> dict:
    """Read an image from disk to ingest into the workflow"""
    image_path = runtime.context.workspace.joinpath(image_path)
    return read_image(image_path)


def read_image(
    image_path: str,
    include_base64: bool = True,
    max_size_mb: float = 20.0,
) -> dict[str, str | int | None]:
    """
    Read an image from disk and prepare it for LLM processing.

    This tool reads an image file and returns its content in a format
    suitable for multimodal LLMs. For non-multimodal LLMs, you can
    use the returned metadata and path information.

    Args:
        image_path: Path to the image file (supports jpg, jpeg, png, gif, webp)
        include_base64: Whether to include base64-encoded data (default: True)
        max_size_mb: Maximum file size in MB (default: 20.0)

    Returns:
        Dictionary containing:
            - success: Boolean indicating if read was successful
            - base64_data: Base64-encoded image data (if include_base64=True)
            - mime_type: MIME type of the image
            - file_size: Size in bytes
            - dimensions: Image dimensions (requires PIL)
            - error: Error message if failed
            - path: Original file path

    Example:
        >>> result = read_image("screenshot.png")
        >>> if result["success"]:
        ...     # Use with multimodal LLM
        ...     image_data = result["base64_data"]
    """

    result = {
        "success": False,
        "base64_data": None,
        "mime_type": None,
        "file_size": None,
        "dimensions": None,
        "error": None,
        "path": image_path,
    }
    print(f"[Reading Image]: {image_path}")

    try:
        # Validate path
        img_path = Path(image_path)
        if not img_path.exists():
            result["error"] = f"Image file not found: {image_path}"
            return result

        if not img_path.is_file():
            result["error"] = f"Path is not a file: {image_path}"
            return result

        # Check file size
        file_size = img_path.stat().st_size
        result["file_size"] = file_size

        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            result["error"] = (
                f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {max_size_mb}MB)"
            )
            return result

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if not mime_type or not mime_type.startswith("image/"):
            # Fallback based on extension
            ext = img_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }
            mime_type = mime_map.get(ext)

            if not mime_type:
                result["error"] = f"Unsupported image format: {ext}"
                return result

        result["mime_type"] = mime_type

        # Read and encode image
        if include_base64:
            with open(img_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode("utf-8")
                result["base64_data"] = base64_encoded

        # Try to get dimensions (optional, requires PIL)
        try:
            from PIL import Image

            with Image.open(img_path) as img:
                result["dimensions"] = {
                    "width": img.width,
                    "height": img.height,
                }
        except ImportError:
            pass  # PIL not available, skip dimensions
        except Exception:
            pass  # Error reading dimensions, skip

        result["success"] = True

    except PermissionError:
        result["error"] = f"Permission denied reading file: {image_path}"
    except Exception as e:
        result["error"] = f"Error reading image: {str(e)}"

    return result


def format_image_for_llm(
    image_result: dict, model_type: str = "multimodal"
) -> dict | str:
    """
    Format image data for specific LLM types.

    Args:
        image_result: Result dictionary from read_image()
        model_type: Type of model ("multimodal" or "text-only")

    Returns:
        Formatted data for the LLM:
            - For multimodal: Dict with image_url containing base64 data
            - For text-only: String description of the image metadata
    """
    if not image_result["success"]:
        return f"Error: {image_result['error']}"

    if model_type == "multimodal":
        if not image_result["base64_data"]:
            return "Error: Base64 data not available"

        # Format for OpenAI/Anthropic style multimodal input
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_result['mime_type']};base64,{image_result['base64_data']}"
            },
        }

    else:  # text-only
        # Provide metadata for text-only models
        dims = image_result.get("dimensions", {})
        dim_str = (
            f"{dims.get('width', '?')}x{dims.get('height', '?')}"
            if dims
            else "unknown"
        )

        return (
            f"Image file: {image_result['path']}\n"
            f"Type: {image_result['mime_type']}\n"
            f"Size: {image_result['file_size'] / 1024:.2f} KB\n"
            f"Dimensions: {dim_str}\n"
            f"Note: This is a text-only model. Image content cannot be analyzed."
        )
