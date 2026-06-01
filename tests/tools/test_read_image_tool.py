import base64
import os
import random
import subprocess
from pathlib import Path

import pytest
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image, ImageDraw
from pydantic import BaseModel

from tests.tools.utils import make_runtime
from ursa.cli.config import ChatModelConfig
from ursa.tools.read_image_tool import (
    image_block_from_file,
    read_image_tool,
)


class DigitAnswer(BaseModel):
    number: int


def _write_digit_image(path: Path, number: int) -> None:
    size = 128
    target_extent = int(size * 0.9)
    text = str(number)
    image = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(image)

    font_size = 1
    for candidate in range(1, size * 2):
        left, top, right, bottom = draw.textbbox(
            (0, 0), text, font_size=candidate
        )
        if (right - left) > target_extent or (bottom - top) > target_extent:
            break
        font_size = candidate

    left, top, right, bottom = draw.textbbox((0, 0), text, font_size=font_size)
    x = (size - (right - left)) // 2 - left
    y = (size - (bottom - top)) // 2 - top
    draw.text((x, y), text, fill="black", font_size=font_size)
    image.save(path)


def _require_openai_model(model: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip(f"Requires OPENAI_API_KEY for {model}")


def _require_ollama_model(model: str) -> None:
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Ollama is not available for {model}: {exc}")

    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        pytest.skip(f"Ollama model {model!r} is not available: {message}")


@pytest.fixture(
    params=[
        pytest.param(
            {
                "model": "openai:gpt-5.2",
                "max_completion_tokens": 128,
                "use_responses_api": True,
            },
            id="openai-gpt-5.2",
        ),
        # Add other ChatModelConfig values for local testing here
    ]
)
def vision_chat_model(request, monkeypatch) -> BaseChatModel:
    kwargs = dict(request.param)
    model = kwargs["model"]
    if model.startswith("openai:"):
        _require_openai_model(model.split(":", 1)[1])
    elif model.startswith("ollama:"):
        _require_ollama_model(model.split(":", 1)[1])

    return ChatModelConfig(**kwargs).init_chat_model()


def test_image_block_from_file_returns_base64_image_block(tmp_path: Path):
    target = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(target)

    result = image_block_from_file(target)

    assert result["type"] == "image"
    assert result["mime_type"] == "image/png"
    assert base64.b64decode(result["base64"]).startswith(b"\x89PNG")


def test_image_block_from_file_rejects_large_images(tmp_path: Path):
    target = tmp_path / "large.png"
    max_image_mb = 2
    max_image_bytes = max_image_mb * 1024 * 1024
    with target.open("wb") as image_file:
        image_file.seek(max_image_bytes)
        image_file.write(b"\0")

    with pytest.raises(ValueError, match="File too large"):
        image_block_from_file(target, max_size_mb=max_image_mb)


def test_read_image_tool_reads_from_runtime_workspace(
    tmp_path: Path, chat_model: BaseChatModel
):
    target = tmp_path / "logo.png"
    Image.new("RGB", (2, 1), color=(0, 128, 255)).save(target)

    result = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-image-call",
        ),
    )

    assert len(result) == 1
    assert result[0]["type"] == "image"
    assert result[0]["mime_type"] == "image/png"
    assert result[0]["file_id"] == target.name
    assert base64.b64decode(result[0]["base64"]).startswith(b"\x89PNG")


def test_read_image_tool_output_can_be_read_by_attached_llm(
    tmp_path: Path, vision_chat_model: BaseChatModel
):
    number = random.SystemRandom().randint(10, 99)
    target = tmp_path / "number.png"
    _write_digit_image(target, number)

    tool_llm = vision_chat_model.bind_tools([read_image_tool])
    structured_llm = tool_llm.with_structured_output(
        DigitAnswer,
        method="function_calling",
    )
    tool_call_id = "call_read_number_image"
    image_blocks = read_image_tool.func(
        image_path=target.name,
        runtime=make_runtime(
            tmp_path,
            llm=tool_llm,
            tool_call_id=tool_call_id,
        ),
    )

    response = structured_llm.invoke([
        HumanMessage(
            content=(
                "Use the attached read_image_tool result to identify the "
                "two-digit number shown in number.png."
            )
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_image_tool",
                    "args": {"image_path": target.name},
                    "id": tool_call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=image_blocks,
            name="read_image_tool",
            tool_call_id=tool_call_id,
        ),
    ])

    assert response.number == number
