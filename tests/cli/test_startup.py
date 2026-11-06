"""Import regressions must run outside pytest's already-imported runtime."""

import asyncio
import threading

import pytest
from pydantic import ValidationError

from benchmark._startup import MCP_SCENARIOS, measure
from ursa.cli.config import UrsaConfig
from ursa.cli.runtime import HITL, AgentHITL


@pytest.mark.parametrize(
    "scenario", ["prompt", "named-prompt", "provider-prompt", *MCP_SCENARIOS]
)
def test_first_prompt_does_not_load_agent_implementations(scenario):
    _, modules = measure(scenario, timeout=30, headless=True)
    for prefix in (
        "ursa.agents.base",
        "ursa.util.mcp",
        "fastmcp",
        "mcp",
        "langchain_mcp_adapters",
        "langchain_openai",
        "langchain_ollama",
        "langgraph.checkpoint.sqlite",
        "chromadb",
        "numpy",
    ):
        assert not any(
            name == prefix or name.startswith(prefix + ".") for name in modules
        ), prefix


async def test_agent_registry_resolves_classes_only_on_metadata_access(
    monkeypatch, tmp_path
):
    loaded = []

    class Agent:
        """An agent loaded on demand."""

    class Catalog:
        def __getattr__(self, name):
            loaded.append(name)
            return Agent

    monkeypatch.setattr("ursa.cli.runtime.agents", Catalog())
    runtime = HITL(UrsaConfig(workspace=tmp_path))
    try:
        assert "chat" in runtime.agents
        assert loaded == []
        assert runtime.agents["chat"].description == Agent.__doc__
        assert runtime.agents["chat"].agent_class is Agent
        assert loaded == ["ChatAgent"]
    finally:
        await runtime.aclose()


def test_runtime_agent_accepts_concrete_class():
    class Agent:
        """Custom agent metadata."""

    agent = AgentHITL(Agent)
    assert agent.agent_class is Agent
    assert agent.description == Agent.__doc__


@pytest.mark.parametrize("cancel_waiters", [False, True])
async def test_first_use_shares_dependencies_and_shutdown_waits(
    monkeypatch, tmp_path, cancel_waiters
):
    runtime = HITL(UrsaConfig(workspace=tmp_path))
    started = threading.Event()
    release = threading.Event()
    calls = []
    model = object()

    def load_dependencies():
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(5)
        return model, None, None

    class Agent:
        def __init__(self, **kwargs):
            assert kwargs["llm"] is model

        async def aclose(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runtime, "_agent_dependencies", load_dependencies)
    runtime.agents = {name: AgentHITL(Agent) for name in ("one", "two")}
    waiters = [
        asyncio.create_task(runtime.get_agent(name)) for name in runtime.agents
    ]
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if cancel_waiters:
            for waiter in waiters:
                waiter.cancel()
            for waiter in waiters:
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            closing = asyncio.create_task(runtime.aclose())
            await asyncio.sleep(0.01)
            assert not closing.done()
            release.set()
            await asyncio.wait_for(closing, 2)
            assert all(
                wrapper._agent is None for wrapper in runtime.agents.values()
            )
        else:
            release.set()
            wrappers = await asyncio.wait_for(asyncio.gather(*waiters), 2)
            assert all(wrapper._agent is not None for wrapper in wrappers)
        assert len(calls) == 1
        assert calls[0] != threading.get_ident()
    finally:
        release.set()
        await asyncio.gather(*waiters, return_exceptions=True)
        await runtime.aclose()


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "unknown", "url": "https://example.com/mcp"},
        {"transport": "stdio"},
        {"transport": "sse", "url": 123},
    ],
)
def test_delayed_mcp_import_preserves_configuration_validation(server):
    with pytest.raises(ValidationError):
        UrsaConfig(mcp_servers={"invalid": server})


def test_delayed_mcp_import_preserves_json_schema():
    schema = UrsaConfig.model_json_schema()
    parameters = schema["properties"]["mcp_servers"]["additionalProperties"]
    assert {
        item["$ref"].rsplit("/", 1)[-1] for item in parameters["anyOf"]
    } == {
        "StdioServerParameters",
        "SseServerParameters",
        "StreamableHttpParameters",
    }
    assert schema["$defs"]["StdioServerParameters"]["required"] == ["command"]
