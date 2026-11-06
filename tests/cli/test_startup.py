"""Exercise delayed loading in fresh processes and response-before-cleanup."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.cli._startup import SCENARIOS, measure
from ursa.cli.config import EmbModelConfig, UrsaConfig
from ursa.cli.one_shot import run_once
from ursa.cli.runtime import HITL


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_cli_defers_unused_imports(scenario):
    _, modules = measure(scenario)
    forbidden = {"chromadb", "mcp", "fastmcp", "langchain_openai", "pymupdf"}
    if scenario == "exec":
        forbidden.add("textual")
        assert "langchain_core.language_models.fake_chat_models" in modules
    else:
        forbidden.update({
            "ursa.agents.base",
            "langchain_core.language_models.fake_chat_models",
            "numpy",
        })
    for prefix in forbidden:
        assert not any(
            name == prefix or name.startswith(prefix + ".") for name in modules
        ), prefix


@pytest.mark.parametrize("rag_tools", [[], ["saved-rag"]])
async def test_embeddings_wait_for_rag_tools(tmp_path, monkeypatch, rag_tools):
    runtime = HITL(
        UrsaConfig(
            workspace=tmp_path,
            rag_tools=rag_tools,
            emb_model=EmbModelConfig(model="unused", model_provider="openai"),
        )
    )
    runtime.model = object()
    calls = []
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", lambda *a: None
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.enforce_model_group_policy", lambda *a: None
    )
    monkeypatch.setattr(
        EmbModelConfig, "init_embedding", lambda self: calls.append(1)
    )
    runtime._agent_dependencies()
    assert bool(calls) == bool(rag_tools)
    await runtime.aclose()


def test_exec_displays_response_before_cleanup(tmp_path):
    output = StringIO()

    async def close():
        assert output.getvalue().strip() == "Hello!"

    runtime = SimpleNamespace(
        workspace=tmp_path,
        agents={"chat": object()},
        run_agent=AsyncMock(return_value="Hello!"),
        aclose=AsyncMock(side_effect=close),
    )
    assert run_once(runtime, "say hello", stdout=output) == "Hello!"
    runtime.aclose.assert_awaited_once()
