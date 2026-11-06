from pathlib import Path

import httpx
import pytest

from ursa.cli.config import UrsaConfig
from ursa.runtime import AgentRuntime
from ursa.util.events import DEFAULT_EVENT_NAME


def make_runtime(workspace: Path) -> AgentRuntime:
    """Construct the same concrete runtime used by the CLI."""
    return AgentRuntime(
        UrsaConfig(
            workspace=workspace,
            inference_providers={
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": {"env": "OPENAI_API_KEY"},
                }
            },
            llm_model={
                "model": "openai:test-model",
                "inference_provider": "openai",
            },
            emb_model={
                "model": "openai:test-embedding",
                "inference_provider": "openai",
            },
        )
    )


def make_ollama_runtime(workspace: Path, model: str = "gemma4") -> AgentRuntime:
    """Construct a real runtime backed by an available local Ollama model."""
    try:
        with httpx.Client(timeout=2) as client:
            response = client.get("http://127.0.0.1:11434/api/tags")
            response.raise_for_status()
            available = {
                item["name"].removesuffix(":latest")
                for item in response.json()["models"]
            }
        if model.removesuffix(":latest") not in available:
            raise LookupError(f"model is not in {sorted(available)}")
    except Exception as exc:
        pytest.skip(f"Local Ollama model {model!r} is unavailable: {exc}")
    return AgentRuntime(
        UrsaConfig(
            workspace=workspace,
            llm_model={
                "model": f"ollama:{model}",
                "inference_provider": None,
            },
        )
    )


async def emit_event(handler, payload=None, **details):
    await handler.on_custom_event(
        DEFAULT_EVENT_NAME,
        details if payload is None else payload,
    )


async def emit_tool_start(handler, tool, run_id, /, **inputs):
    """Send a structured tool start through the real Textual callback handler."""
    await handler.on_tool_start(
        {"name": tool}, "", run_id=run_id, inputs=inputs
    )


async def wait_for(pilot, condition, *, timeout=3.0, interval=0.05):
    """Poll a condition instead of trusting one fixed pause."""
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(interval)
        await pilot.pause()
    return bool(condition())
