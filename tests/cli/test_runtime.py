# ruff: noqa: TID251

import asyncio
import io
import logging
import threading
import time
from pathlib import Path
from random import random
from sys import executable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastmcp.client import Client
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import StdioServerParameters
from pydantic import ValidationError
from rich.console import Console as RealConsole

from tests.conftest import extraskip
from ursa import security
from ursa.agents.base import AgentWithTools
from ursa.cli.callbacks import HITLLogEventHandler
from ursa.cli.config import ChatModelConfig, EmbModelConfig, UrsaConfig
from ursa.cli.runtime import EPHEMERAL_AGENT_PREFIX, HITL, AgentHITL
from ursa.cli.tui.agent_info import load_agent_tools
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.util.rendering import event_artifact

LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _isolate_ursa_cache(monkeypatch, tmp_path_factory):
    # Every session now owns a (temporary) agent directory under the cache, so
    # the real runtime must never be exercised against the user's ~/.cache.
    monkeypatch.setattr(
        security, "URSA_CACHE_DIR", tmp_path_factory.mktemp("ursa_cache")
    )


@pytest.fixture(autouse=True)
def stub_duckduckgo(monkeypatch):
    class DummyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, *args, **kwargs):
            yield {
                "href": "https://example.com",
                "title": "Example Result",
                "body": "Example summary",
            }

    monkeypatch.setattr(
        "ursa.agents.acquisition_agents.DDGS",
        lambda: DummyDDGS(),
        raising=False,
    )
    monkeypatch.setattr(
        "ursa.agents.hypothesizer_agent.DDGS",
        lambda: DummyDDGS(),
        raising=False,
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider",
        lambda _config, _model_type: None,
    )


@pytest.fixture(scope="function")
def ursa_config(tmpdir, chat_model, embedding_model):
    config = UrsaConfig(
        workspace=Path(tmpdir),
        llm_model=chat_model._testing_only_kwargs,
        emb_model=embedding_model._testing_only_kwargs,
    )
    print("ursa config:", config)  # Displayed on test failure
    return config


async def test_default_config_smoke(ursa_config):
    hitl = HITL(ursa_config)
    assert hitl is not None
    assert set(hitl.agents.keys()) >= {"chat", "plan", "execute"}
    out = await hitl.run_agent("chat", "Hello! What is your name?")
    print("chat out:", out)
    assert len(out) > 0


DOCS_ROOT = Path(__file__).resolve().parents[2]
DOC_EXAMPLE_CONFIG = DOCS_ROOT / "configs" / "example.yaml"


async def test_agents_use_configured_workspace(ursa_config, tmp_path):
    workspace = tmp_path / "custom-workspace"
    ursa_config.workspace = workspace

    hitl = HITL(ursa_config)
    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.workspace == workspace


@pytest.mark.asyncio
async def test_unnamed_cli_session_uses_temporary_agent_removed_on_close(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    workspace = tmp_path / "ephemeral-workspace"
    hitl = HITL(UrsaConfig(workspace=workspace))

    # The user-facing name stays unset, so the UI never shows the generated one.
    assert hitl.agent_name is None
    name = hitl.session_agent_name
    assert name.startswith(EPHEMERAL_AGENT_PREFIX)
    den = security.group_agents_dir(hitl.group) / name

    agent = await hitl.get_agent("chat")

    # History is checkpointed in the agent den, never in the workspace.
    assert agent._agent is not None
    assert isinstance(agent._agent.checkpointer, AsyncSqliteSaver)
    assert agent._agent.den == den
    assert (den / "db" / "checkpointer.db").exists()
    assert not (workspace / "db" / "checkpointer.db").exists()

    await hitl.aclose()
    assert not den.exists()


@pytest.mark.asyncio
async def test_temporary_agent_names_are_unique_per_session(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    first = HITL(UrsaConfig(workspace=tmp_path / "a"))
    second = HITL(UrsaConfig(workspace=tmp_path / "b"))
    try:
        assert first.session_agent_name != second.session_agent_name
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_named_session_agent_is_kept_on_close(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path, agent_name="keep-me"))
    assert hitl.session_agent_name == "keep-me"
    den = security.group_agents_dir(hitl.group) / "keep-me"

    await hitl.get_agent("chat")
    await hitl.aclose()

    assert den.exists()  # only generated, prefixed agents are ever removed


@pytest.mark.parametrize("cancel_waiters", [False, True])
async def test_concurrent_get_agent_constructs_and_finalizes_once(
    tmp_path, monkeypatch, cancel_waiters
):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path, agent_name="persistent"))
    constructions = 0
    checkpointers = 0
    started = threading.Event()
    release = threading.Event()
    checkpointer = object()

    class SlowPersistentAgent:
        def __init__(self, **_kwargs):
            nonlocal constructions
            constructions += 1
            self.den = tmp_path
            self.checkpointer = None
            started.set()
            assert release.wait(timeout=5)

    async def fake_get_checkpointer(_path):
        nonlocal checkpointers
        checkpointers += 1
        return checkpointer

    wrapper = AgentHITL(agent_class=SlowPersistentAgent)
    hitl.agents["chat"] = wrapper
    monkeypatch.setattr(hitl, "_get_checkpointer", fake_get_checkpointer)

    waiters = [asyncio.create_task(hitl.get_agent("chat")) for _ in range(2)]
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if cancel_waiters:
            for waiter in waiters:
                waiter.cancel()
            for waiter in waiters:
                with pytest.raises(asyncio.CancelledError):
                    await waiter
    finally:
        release.set()
    if not cancel_waiters:
        results = await asyncio.gather(*waiters)
        assert all(result is wrapper for result in results)
    await wrapper.wait_until_initialized()
    assert await hitl.get_agent("chat") is wrapper
    assert wrapper._agent.checkpointer is checkpointer
    assert constructions == 1
    assert checkpointers == 1


async def test_distinct_agents_initialize_concurrently(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    first_started = threading.Event()
    second_started = threading.Event()
    release = threading.Event()

    def agent_class(started):
        class SlowAgent:
            def __init__(self, **_kwargs):
                started.set()
                release.wait(timeout=5)

        return SlowAgent

    hitl.agents = {
        "first": AgentHITL(agent_class=agent_class(first_started)),
        "second": AgentHITL(agent_class=agent_class(second_started)),
    }
    first = asyncio.create_task(hitl.get_agent("first"))
    second = asyncio.create_task(hitl.get_agent("second"))
    try:
        assert await asyncio.to_thread(first_started.wait, 2)
        assert await asyncio.to_thread(second_started.wait, 2)
    finally:
        release.set()
    await asyncio.gather(first, second)


async def test_named_finalizer_failure_cleans_and_allows_retry(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path, agent_name="persistent"))
    instances = []

    class FailingConnection:
        def close(self):
            raise RuntimeError("sync close failed")

    class PersistentAgent:
        def __init__(self, **_kwargs):
            self.den = tmp_path
            self.checkpointer = (
                SqliteSaver(FailingConnection()) if not instances else None
            )
            self.async_closed = False
            self.closed = False
            instances.append(self)

        async def aclose(self):
            self.async_closed = True

        def close(self):
            self.closed = True

    class FailedAsyncConnection:
        def __init__(self):
            self.closed = False
            self.joined = False

        async def close(self):
            self.closed = True

        def join(self):
            self.joined = True

    failed_connection = FailedAsyncConnection()
    failed_checkpointer = SimpleNamespace(conn=failed_connection)
    successful_checkpointer = object()
    finalizer_calls = 0

    async def fake_get_checkpointer(_path):
        nonlocal finalizer_calls
        finalizer_calls += 1
        return (
            failed_checkpointer
            if finalizer_calls == 1
            else successful_checkpointer
        )

    wrapper = AgentHITL(agent_class=PersistentAgent)
    hitl.agents["chat"] = wrapper
    monkeypatch.setattr(hitl, "_get_checkpointer", fake_get_checkpointer)

    with pytest.raises(RuntimeError, match="sync close failed"):
        await hitl.get_agent("chat")

    assert wrapper._agent is None
    assert wrapper._initialization_task is None
    assert instances[0].async_closed
    assert instances[0].closed
    assert failed_connection.closed
    assert failed_connection.joined
    assert hitl._runtime_checkpointers == []

    loaded = await hitl.get_agent("chat")
    assert loaded._agent is instances[1]
    assert loaded._agent.checkpointer is successful_checkpointer
    assert hitl._runtime_checkpointers == [successful_checkpointer]


@pytest.mark.parametrize(
    "instantiate_kwargs",
    [
        {},
        {"agent_name": "persistent-agent"},
        {"checkpointer": object()},
        {"agent_name": "persistent-agent", "checkpointer": object()},
    ],
)
async def test_agent_instantiation_always_runs_off_event_loop(
    instantiate_kwargs,
):
    event_loop_thread = threading.get_ident()
    constructor_threads = []

    class RecordingAgent:
        def __init__(self, **_kwargs):
            constructor_threads.append(threading.get_ident())

    wrapper = AgentHITL(agent_class=RecordingAgent)
    await wrapper.instantiate(**instantiate_kwargs)

    assert constructor_threads
    assert constructor_threads[0] != event_loop_thread


async def test_complete_agent_loading_pipeline_does_not_block_event_loop(
    monkeypatch,
):
    mcp_started = threading.Event()
    mcp_release = threading.Event()

    async def slow_mcp_discovery(_client):
        mcp_started.set()
        mcp_release.wait(timeout=5)
        return [], {"remote_tool": "laboratory"}

    monkeypatch.setattr(
        "ursa.util.mcp.load_mcp_tools_with_sources", slow_mcp_discovery
    )

    class SlowMcpAgent(AgentWithTools):
        def __init__(self, **_kwargs):
            self._tools = {}

        def add_tool(self, tools):
            self._tools.update({tool.name: tool for tool in tools})

    ticks = 0
    loading = True

    async def ticker():
        nonlocal ticks
        while loading:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    wrapper = AgentHITL(agent_class=SlowMcpAgent)
    instantiate_task = asyncio.create_task(
        wrapper.instantiate(mcp_client=object(), agent_name="persistent")
    )
    try:
        assert await asyncio.to_thread(mcp_started.wait, 2)
        ticks_at_mcp_start = ticks
        await asyncio.sleep(0.05)
        assert ticks > ticks_at_mcp_start
        assert not instantiate_task.done()

        mcp_release.set()
        await instantiate_task
        assert wrapper.tool_sources == {"remote_tool": "laboratory"}
    finally:
        mcp_release.set()
        loading = False
        await ticker_task


@pytest.mark.parametrize("agent_name", [None, "persistent"])
async def test_full_get_agent_path_does_not_block_event_loop(
    tmp_path, monkeypatch, agent_name
):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path, agent_name=agent_name))
    event_loop_thread = threading.get_ident()
    close_threads = []

    class SlowConnection:
        def close(self):
            close_threads.append(threading.get_ident())
            time.sleep(0.05)

    class SlowAgent:
        def __init__(self, **_kwargs):
            time.sleep(0.05)
            self.den = tmp_path
            self.checkpointer = (
                SqliteSaver(SlowConnection()) if agent_name else None
            )

    async def fake_get_checkpointer(_path):
        return object()

    hitl.agents["chat"] = AgentHITL(agent_class=SlowAgent)
    monkeypatch.setattr(hitl, "_get_checkpointer", fake_get_checkpointer)
    ticks = 0
    loading = True

    async def ticker():
        nonlocal ticks
        while loading:
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    await hitl.get_agent("chat")
    loading = False
    await ticker_task

    assert ticks > 1
    if agent_name:
        assert close_threads[0] != event_loop_thread


async def test_agent_loading_without_mcp_skips_tool_discovery():
    class NoMcpAgent(AgentWithTools):
        def __init__(self, **_kwargs):
            pass

        async def add_mcp_tools(self, _client):
            pytest.fail("MCP discovery must not run without an MCP client")

    wrapper = AgentHITL(agent_class=NoMcpAgent)
    await wrapper.instantiate()

    assert wrapper.tool_sources == {}


async def test_concurrent_initialization_failure_cleans_once_and_can_retry():
    instances = []

    class FlakyMcpAgent(AgentWithTools):
        def __init__(self, **_kwargs):
            self.attempt = len(instances) + 1
            self.async_close_count = 0
            self.close_count = 0
            instances.append(self)

        async def add_mcp_tools(self, _client):
            if self.attempt == 1:
                await asyncio.sleep(0.05)
                raise RuntimeError("temporary failure")
            return {}

        async def aclose(self):
            self.async_close_count += 1

        def close(self):
            self.close_count += 1

    wrapper = AgentHITL(agent_class=FlakyMcpAgent)
    failures = await asyncio.gather(
        wrapper.instantiate(mcp_client=object()),
        wrapper.instantiate(mcp_client=object()),
        return_exceptions=True,
    )

    assert all(isinstance(error, RuntimeError) for error in failures)
    assert len(instances) == 1
    assert instances[0].async_close_count == 1
    assert instances[0].close_count == 1
    assert wrapper._agent is None
    assert wrapper._initialization_task is None

    await wrapper.instantiate(mcp_client=object())
    assert len(instances) == 2
    assert wrapper._agent is instances[1]


@pytest.mark.asyncio
async def test_named_cli_agent_resources_close_once(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(
        UrsaConfig(
            workspace=tmp_path / "workspace",
            agent_name="persistent-agent",
        )
    )
    persistent_den = tmp_path / "persistent-agent-den"

    class DummyPersistentAgent:
        def __init__(self, **_kwargs):
            self.den = persistent_den
            self.checkpointer = None
            self.async_close_count = 0
            self.close_count = 0

        async def aclose(self):
            self.async_close_count += 1

        def close(self):
            self.close_count += 1

    hitl.agents["chat"] = AgentHITL(agent_class=DummyPersistentAgent)
    wrapper = await hitl.get_agent("chat")
    assert wrapper._agent is not None
    agent = wrapper._agent
    checkpointer = agent.checkpointer
    assert isinstance(checkpointer, AsyncSqliteSaver)
    assert (persistent_den / "db" / "checkpointer.db").is_file()
    assert checkpointer.conn.is_alive()

    await hitl.close()
    await hitl.aclose()

    assert agent.async_close_count == 1
    assert agent.close_count == 1
    assert wrapper._agent is None
    assert checkpointer.conn._connection is None
    assert not checkpointer.conn.is_alive()


async def test_closed_runtime_rejects_agent_loading_without_waiting(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    await asyncio.gather(hitl.aclose(), hitl.aclose())

    with pytest.raises(RuntimeError, match="runtime is closed"):
        await asyncio.wait_for(hitl.get_agent("chat"), timeout=0.5)
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await hitl.reconfigure_model("openai:gpt-5.4", "openai")
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await hitl.reconfigure_models(
            ChatModelConfig(
                model="openai:gpt-5.4", inference_provider="openai"
            ),
            None,
        )


async def test_reconfigure_models_resets_agents_and_uses_selected_providers(
    tmp_path, monkeypatch
):
    initial_model, _ = _stub_hitl_dependencies(monkeypatch)
    replacement_model = MagicMock(name="replacement-llm")
    replacement_embedding = MagicMock(name="replacement-embedding")
    hitl = HITL(UrsaConfig(workspace=tmp_path))

    class DummyAgent:
        checkpointer = None

        def __init__(self, **_kwargs):
            self.async_closed = False
            self.closed = False

        async def aclose(self):
            self.async_closed = True

        def close(self):
            self.closed = True

    wrapper = AgentHITL(
        agent_class=DummyAgent,
        config={"rag_tool_embedding": None},
    )
    hitl.agents["chat"] = wrapper
    await hitl.get_agent("chat")
    old_agent = wrapper._agent
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: replacement_model
    )
    monkeypatch.setattr(
        "langchain.embeddings.init_embeddings",
        lambda **_: replacement_embedding,
    )

    await hitl.reconfigure_models(
        ChatModelConfig(model="openai:gpt-5.4", inference_provider="openai"),
        EmbModelConfig(
            model="openai:text-embedding-3-large",
            inference_provider="openai",
        ),
    )

    assert hitl.model is replacement_model
    assert hitl.model is not initial_model
    assert hitl.config.llm_model.model == "gpt-5.4"
    assert hitl.config.llm_model.model_provider == "openai"
    assert hitl.embedding is replacement_embedding
    assert hitl.config.emb_model.model == "text-embedding-3-large"
    assert hitl.config.emb_model.model_provider == "openai"
    assert wrapper.config["rag_tool_embedding"] is replacement_embedding
    assert wrapper._agent is None
    assert old_agent.async_closed
    assert old_agent.closed


async def test_reconfigure_waits_for_loading_then_discards_old_model_agent(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    replacement_model = MagicMock(name="replacement-llm")
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    started = threading.Event()
    release = threading.Event()
    instances = []

    class SlowAgent:
        checkpointer = None

        def __init__(self, llm, **_kwargs):
            self.llm = llm
            self.async_closed = False
            self.closed = False
            instances.append(self)
            started.set()
            release.wait(timeout=5)

        async def aclose(self):
            self.async_closed = True

        def close(self):
            self.closed = True

    wrapper = AgentHITL(agent_class=SlowAgent)
    hitl.agents["chat"] = wrapper
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: replacement_model
    )

    load_task = asyncio.create_task(hitl.get_agent("chat"))
    assert await asyncio.to_thread(started.wait, 2)
    reconfigure_task = asyncio.create_task(
        hitl.reconfigure_models(
            ChatModelConfig(
                model="openai:gpt-5.4", inference_provider="openai"
            ),
            None,
        )
    )
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(load_task, reconfigure_task)
    # Reconfiguration acquired the lifecycle lock after loading, closed the
    # old-model instance, and reset the wrapper for the next lazy load.
    assert wrapper._agent is None
    assert hitl.model is replacement_model
    assert instances[0].async_closed
    assert instances[0].closed


@pytest.mark.parametrize("operation", ["close", "reconfigure"])
async def test_transition_waits_for_active_agent_run(
    tmp_path, monkeypatch, operation
):
    _stub_hitl_dependencies(monkeypatch)
    replacement_model = MagicMock(name="replacement-llm")
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    run_started = asyncio.Event()
    run_release = asyncio.Event()
    instances = []
    close_threads = []
    event_loop_thread = threading.get_ident()

    class RunningAgent:
        checkpointer = None

        def __init__(self, **_kwargs):
            self.async_closed = False
            self.closed = False
            instances.append(self)

        async def aclose(self):
            self.async_closed = True

        def close(self):
            close_threads.append(threading.get_ident())
            time.sleep(0.05)
            self.closed = True

    class RunningWrapper(AgentHITL):
        async def __call__(self, *_args, **_kwargs):
            run_started.set()
            await run_release.wait()
            return "complete"

    hitl.agents["chat"] = RunningWrapper(agent_class=RunningAgent)
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: replacement_model
    )
    run_task = asyncio.create_task(hitl.run_agent("chat", "work"))
    await run_started.wait()
    transition = asyncio.create_task(
        hitl.aclose()
        if operation == "close"
        else hitl.reconfigure_models(
            ChatModelConfig(
                model="openai:gpt-5.4", inference_provider="openai"
            ),
            None,
        )
    )
    await asyncio.sleep(0.05)

    assert not transition.done()
    assert not instances[0].closed
    run_release.set()
    result, _ = await asyncio.gather(run_task, transition)

    assert result == "complete"
    assert instances[0].async_closed
    assert instances[0].closed
    assert len(close_threads) == 1
    assert close_threads[0] != event_loop_thread
    assert hitl._closed == (operation == "close")


@pytest.mark.parametrize("operation", ["close", "reconfigure"])
async def test_cancelled_waiter_does_not_interrupt_transition(
    tmp_path, monkeypatch, operation
):
    initial_model, _ = _stub_hitl_dependencies(monkeypatch)
    replacement_model = MagicMock(name="replacement-llm")
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    assert hitl.model is initial_model
    close_started = threading.Event()
    close_release = threading.Event()

    class SlowCloseAgent:
        checkpointer = None

        async def aclose(self):
            pass

        def close(self):
            close_started.set()
            close_release.wait(timeout=5)

    wrapper = AgentHITL(agent_class=SlowCloseAgent)
    wrapper._agent = SlowCloseAgent()
    hitl.agents["chat"] = wrapper
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: replacement_model
    )
    waiter = asyncio.create_task(
        hitl.aclose()
        if operation == "close"
        else hitl.reconfigure_models(
            ChatModelConfig(
                model="openai:gpt-5.4", inference_provider="openai"
            ),
            None,
        )
    )
    assert await asyncio.to_thread(close_started.wait, 2)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    close_release.set()
    if operation == "close":
        assert hitl._close_task is not None
        await hitl._close_task
        assert hitl._closed
    else:
        async with asyncio.timeout(2):
            while hitl.model is not replacement_model:
                await asyncio.sleep(0.01)

    assert wrapper._agent is None
    assert hitl._loads_allowed.is_set()


@pytest.mark.parametrize("operation", ["close", "reconfigure"])
@pytest.mark.parametrize("cancel_snapshot", [False, True])
async def test_transition_waits_for_tool_schema_snapshot(
    tmp_path, monkeypatch, operation, cancel_snapshot
):
    _stub_hitl_dependencies(monkeypatch)
    replacement_model = MagicMock(name="replacement-llm")
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    schema_started = threading.Event()
    schema_release = threading.Event()

    class SlowSchema:
        @classmethod
        def model_json_schema(cls):
            schema_started.set()
            schema_release.wait(timeout=5)
            return {"properties": {}}

    class Tool:
        name = "slow_tool"
        description = "Slow schema tool"
        args_schema = SlowSchema
        return_direct = False

    class InitializedAgent:
        checkpointer = None

        def __init__(self):
            self.tools = {"slow_tool": Tool()}
            self.closed = False

        async def aclose(self):
            pass

        def close(self):
            self.closed = True

    initialized = InitializedAgent()
    wrapper = AgentHITL(agent_class=InitializedAgent)
    wrapper._agent = initialized
    hitl.agents["chat"] = wrapper
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: replacement_model
    )

    snapshot_task = asyncio.create_task(load_agent_tools(hitl, "chat"))
    assert await asyncio.to_thread(schema_started.wait, 2)
    if cancel_snapshot:
        snapshot_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(snapshot_task, 0.5)
    transition = asyncio.create_task(
        hitl.aclose()
        if operation == "close"
        else hitl.reconfigure_models(
            ChatModelConfig(
                model="openai:gpt-5.4", inference_provider="openai"
            ),
            None,
        )
    )
    await asyncio.sleep(0.05)

    assert not transition.done()
    assert not initialized.closed
    schema_release.set()
    await transition

    if not cancel_snapshot:
        assert [tool.name for tool in await snapshot_task] == ["slow_tool"]
    assert initialized.closed
    assert hitl._closed == (operation == "close")


def _stub_hitl_dependencies(monkeypatch):
    fake_llm = MagicMock(name="llm")
    fake_embedding = MagicMock(name="embedding")
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_: fake_llm
    )
    monkeypatch.setattr(
        "langchain.embeddings.init_embeddings", lambda **_: fake_embedding
    )
    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", lambda servers: None)
    return fake_llm, fake_embedding


async def test_hitl_first_use_reports_provider_validation_failure(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider",
        MagicMock(side_effect=ValueError("API key is missing")),
    )

    hitl = HITL(UrsaConfig(workspace=tmp_path))
    with pytest.raises(
        ValueError,
        match="API key is missing",
    ):
        await hitl.get_agent("chat")
    await hitl.aclose()


async def test_reconfigure_models_reports_validation_failure_without_change(
    tmp_path, monkeypatch
):
    initial_model, _ = _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    assert hitl.model is initial_model
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider",
        MagicMock(side_effect=ValueError("model is unavailable")),
    )

    with pytest.raises(
        ValueError,
        match="model is unavailable",
    ):
        await hitl.reconfigure_models(
            ChatModelConfig(
                model="missing-model",
                model_provider="openai",
                inference_provider="openai",
            ),
            None,
        )

    assert hitl.model is initial_model
    assert hitl.config.llm_model.model == "gpt-5.4"


@pytest.mark.parametrize(
    "agent_name",
    [
        "chat",
        "arxiv",
        "execute",
        "hypothesize",
        "plan",
        "web",
        "dsi",
    ],
)
async def test_agents_apply_agent_config_overrides(
    agent_name, tmp_path, monkeypatch
):
    if agent_name == "dsi":
        extraskip("dsi")
    _stub_hitl_dependencies(monkeypatch)

    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )

    overrides = {}
    overrides[agent_name] = {
        "workspace": tmp_path / f"{agent_name}-workspace",
        "enable_metrics": random() > 0.5,
    }

    config.agent_config = overrides

    hitl = HITL(config)

    agent = await hitl.get_agent(agent_name)
    override = overrides[agent_name]
    assert agent._agent is not None
    assert agent._agent.workspace == override["workspace"]
    assert agent._agent.telemetry.enable == override["enable_metrics"]


async def test_global_summarization_fallback_and_agent_override(
    tmp_path, monkeypatch
):
    _stub_hitl_dependencies(monkeypatch)
    hitl = HITL(
        UrsaConfig(
            workspace=tmp_path,
            tokens_before_summarize=12000,
            agent_config={"chat": {"tokens_before_summarize": 8000}},
        )
    )
    for name, wrapper in hitl.agents.items():
        assert wrapper.config["tokens_before_summarize"] == (
            8000 if name == "chat" else 12000
        )
    chat = await hitl.get_agent("chat")
    execute = await hitl.get_agent("execute")
    assert chat._agent.tokens_before_summarize == 8000
    assert execute._agent.tokens_before_summarize == 12000


@pytest.mark.asyncio
async def test_thread_id_propagates_from_config(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        thread_id="custom-thread",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )

    hitl = HITL(config)
    assert hitl.thread_id == "custom-thread"

    agent = await hitl.get_agent("chat")
    assert agent._agent is not None
    assert agent._agent.thread_id == "custom-thread"


@pytest.mark.asyncio
async def test_hitl_run_agent_forwards_callbacks(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )
    hitl = HITL(config)
    captured = {}
    previous_agent = object()
    current_agent = object()

    class DummyAgent:
        _agent = current_agent

        async def __call__(
            self,
            prompt: str,
            last_agent_result: str | None = None,
            last_agent=None,
            callbacks=None,
        ) -> str:
            captured["prompt"] = prompt
            captured["last_agent_result"] = last_agent_result
            captured["last_agent"] = last_agent
            captured["callbacks"] = callbacks
            return "agent result"

    async def fake_get_agent(name: str):
        assert name == "chat"
        return DummyAgent()

    hitl.last_agent_result = "previous result"
    hitl.last_agent = previous_agent
    monkeypatch.setattr(hitl, "_get_agent", fake_get_agent)

    callbacks = ["callback-1"]
    result = await hitl.run_agent("chat", "hello", callbacks=callbacks)

    assert result == "agent result"
    assert captured == {
        "prompt": "hello",
        "last_agent_result": "previous result",
        "last_agent": previous_agent,
        "callbacks": callbacks,
    }
    assert hitl.last_agent_result == "agent result"
    assert hitl.last_agent is current_agent


@pytest.mark.asyncio
async def test_agent_hitl_passes_extra_callbacks_only():
    captured = {}
    custom_callback = object()

    class DummyAgent:
        telemetry = type("Telemetry", (), {"callbacks": ["telemetry"]})()

        def format_query(self, prompt: str, state=None):
            captured["prompt"] = prompt
            captured["state"] = state
            return {"messages": [prompt]}

        async def ainvoke(self, query, config=None):
            captured["query"] = query
            captured["config"] = config
            return {"messages": ["done"]}

        def format_result(self, result):
            return "done"

    wrapper = AgentHITL(agent_class=object)
    wrapper._agent = DummyAgent()

    result = await wrapper("hello", callbacks=[custom_callback])

    assert result == "done"
    assert captured["prompt"] == "hello"
    assert captured["state"] is None
    assert captured["query"] == {"messages": ["hello"]}
    assert captured["config"] == {"callbacks": [custom_callback]}


def test_hitl_log_event_handler_renders_events(tmp_path):
    output = io.StringIO()
    console = RealConsole(
        file=output,
        force_terminal=False,
        force_interactive=False,
        color_system=None,
        width=80,
    )
    handler = HITLLogEventHandler(console=console, workspace=tmp_path)

    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "PlanningAgent",
                "stage": "reflect_result",
                "message": "Plan needs another pass",
                "approved": False,
                "reason": "Need one more concrete step.",
            },
            run_id="agent-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "phase": "end",
                "message": "File updated",
                "artifact": event_artifact(
                    "--- app.py\n+++ app.py\n-old\n+new",
                    "text/x-diff",
                    metadata={
                        "title": "Edit diff",
                        "path": "repo/app.py",
                    },
                ),
            },
            run_id="edit-tool-artifact-run",
        )
    )
    asyncio.run(
        handler.on_tool_start(
            {"name": "run_command"},
            '{"query":"uname -s"}',
            run_id="tool-run",
            inputs={"query": "uname -s"},
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "execute",
                "phase": "end",
                "message": "Command finished",
                "artifacts": [
                    event_artifact(
                        "Darwin",
                        "text/plain",
                        metadata={"title": "stdout"},
                    ),
                    event_artifact(
                        "warning",
                        "text/plain",
                        metadata={"title": "stderr"},
                    ),
                ],
            },
            run_id="tool-run",
        )
    )
    asyncio.run(
        handler.on_tool_end(
            "STDOUT:\nDarwin\nSTDERR:\n",
            run_id="tool-run",
        )
    )
    asyncio.run(
        handler.on_tool_start(
            {"name": "write_code_with_repo"},
            '{"filename":"repo/app.py"}',
            run_id="write-tool-run",
            inputs={"filename": "repo/app.py"},
        )
    )
    asyncio.run(
        handler.on_tool_end(
            "File repo/app.py written successfully.",
            run_id="write-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "phase": "start",
                "message": "Editing file",
                "path": str(tmp_path / "repo" / "app.py"),
            },
            run_id="edit-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "message": "No changes made",
                "filename": "repo/app.py",
                "reason": "'old_code' not found in file.",
            },
            run_id="edit-tool-noop-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_web_search",
                "stage": "search",
                "message": "Searching Web",
                "query": "ursa events",
            },
            run_id="search-tool-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_web_search",
                "stage": "search_result",
                "message": "Web search complete",
                "query": "ursa events",
                "result_chars": 42,
            },
            run_id="search-tool-result-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "choose_potential",
                "phase": "end",
                "message": "Potential chosen",
                "chosen_index": 2,
                "potential_id": "pot-2",
                "rationale": "Best fit for the requested elements.",
            },
            run_id="lammps-choice-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "author_input",
                "phase": "end",
                "message": "LAMMPS input authored",
                "preview": "units metal\nrun 100",
                "language": "bash",
                "path": str(tmp_path / "in.lammps"),
            },
            run_id="lammps-author-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "fix_input",
                "phase": "end",
                "message": "LAMMPS input rewritten",
                "old_code": "run 100",
                "new_code": "run 200",
                "path": str(tmp_path / "in.lammps"),
            },
            run_id="lammps-fix-run",
        )
    )
    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "LammpsAgent",
                "stage": "run",
                "phase": "error",
                "message": "LAMMPS run failed",
                "returncode": 1,
                "error_output": "ERROR: Invalid pair style",
            },
            run_id="lammps-failed-run",
        )
    )

    rendered = output.getvalue()

    assert "Edit diff" in rendered
    assert "-old" in rendered
    assert "+new" in rendered

    repo_app_string = str(
        Path("repo") / "app.py"
    )  # Rendering OS specific path string

    assert "Plan" in rendered
    assert "Plan needs another pass" in rendered
    assert "Need one more concrete step." in rendered
    assert "Running command: uname -s" in rendered
    assert "Command finished: uname -s" in rendered
    assert "stdout" in rendered
    assert "stderr" in rendered
    assert "warning" in rendered
    assert any(
        "stdout" in line and "stderr" in line for line in rendered.splitlines()
    )
    assert "Darwin" in rendered
    assert f"Writing file: {repo_app_string}" in rendered
    assert f"File written: {repo_app_string}" in rendered
    assert f"Editing file: {repo_app_string}" in rendered
    assert f"No changes made: {repo_app_string}" in rendered
    assert "'old_code' not found in file." in rendered
    assert "Searching Web: ursa events" in rendered
    assert "Web search complete: ursa events" in rendered
    assert "42 chars" in rendered
    assert "LAMMPS" in rendered
    assert "Chosen Potential" in rendered
    assert "pot-2" in rendered
    assert "Best fit for the requested elements." in rendered
    assert "LAMMPS input authored" in rendered
    assert "units metal" in rendered
    assert "LAMMPS input diff" in rendered
    assert "run 100" in rendered
    assert "run 200" in rendered
    assert "Run error/output" in rendered
    assert "ERROR: Invalid pair style" in rendered


def test_hitl_log_event_handler_renders_named_agent_tool_artifacts(tmp_path):
    output = io.StringIO()
    console = RealConsole(
        file=output,
        force_terminal=False,
        force_interactive=False,
        color_system=None,
        width=80,
    )
    handler = HITLLogEventHandler(console=console, workspace=tmp_path)

    async def emit_events() -> None:
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "dummy_bot_3000",
                "tool": "write_code",
                "stage": "write",
                "phase": "end",
                "message": "File written",
                "filename": "first_10_integers.py",
                "artifact": event_artifact(
                    "for i in range(1, 11):\n    print(i)\n",
                    "text/x-python",
                    metadata={"title": "File written"},
                ),
            },
            run_id="named-write-tool-run",
        )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "dummy_bot_3000",
                "tool": "run_command",
                "stage": "execute",
                "phase": "end",
                "message": "Command finished",
                "query": "python first_10_integers.py",
                "artifacts": [
                    event_artifact(
                        "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
                        "text/plain",
                        metadata={"title": "stdout"},
                    )
                ],
            },
            run_id="named-command-tool-run",
        )

    asyncio.run(emit_events())

    rendered = output.getvalue()
    assert "File written" in rendered
    assert "for i in range(1, 11):" in rendered
    assert "print(i)" in rendered
    assert "stdout" in rendered
    assert "1" in rendered
    assert "10" in rendered


def test_agent_config_unknown_agent_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )
    config.agent_config = {
        "ghost": {"workspace": tmp_path / "ghost-workspace"},
    }

    with pytest.raises(AssertionError, match="Unknown agent ghost"):
        HITL(config)


def test_agent_config_none_value_errors(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )
    with pytest.raises(ValidationError):
        config.agent_config = {"chat": None}


@pytest.mark.asyncio
async def test_agent_config_unknown_option_raises(tmp_path, monkeypatch):
    _stub_hitl_dependencies(monkeypatch)
    config = UrsaConfig(
        workspace=tmp_path / "global-workspace",
        emb_model=EmbModelConfig(model="fake-embedding", model_provider="fake"),
    )
    config.agent_config = {"chat": {"nonexistent_option": True}}

    hitl = HITL(config)

    with pytest.raises(TypeError, match="nonexistent_option"):
        await hitl.get_agent("chat")


async def test_chat(ursa_config):
    hitl = HITL(ursa_config)
    out = await hitl.run_agent(
        "chat",
        "What is your name?",
    )
    print(out)
    assert out is not None


DUMMY_MCP_SERVER_PATH = Path(__file__).parent.parent.joinpath(
    "tools", "dummy_mcp_server.py"
)


async def test_mcp_tools(ursa_config: UrsaConfig):
    ursa_config.mcp_servers["demo"] = StdioServerParameters(
        command=executable,
        args=[str(DUMMY_MCP_SERVER_PATH.resolve())],
    )
    hitl = HITL(ursa_config)
    agent = await hitl.get_agent("execute")
    assert agent._agent is not None
    assert isinstance(agent._agent, AgentWithTools)
    assert "add" in agent._agent.tools
    assert agent.tool_sources["add"] == "demo"


@pytest.fixture
async def mcp_server(ursa_config):
    hitl = HITL(ursa_config)
    server = hitl.as_mcp_server()
    async with Client(transport=server) as client:
        yield client


async def test_mcp_smoke(mcp_server: Client):
    tools = await mcp_server.list_tools()
    assert len(tools) > 0
    await mcp_server.list_resources()
    await mcp_server.list_prompts()


@pytest.mark.parametrize("agent,query", [("chat", "Who are you?")])
async def test_mcp_agents(mcp_server: Client, agent: str, query: str):
    response = await mcp_server.call_tool(agent, {"prompt": query})
    assert isinstance(response.structured_content["result"], str)


def _recording_models(monkeypatch):
    """Every chat model the runtime builds, in order (RecordingChatModel)."""
    from tests.agents.utils import RecordingChatModel

    built = []

    def init_chat_model(self):
        built.append(RecordingChatModel())
        return built[-1]

    monkeypatch.setattr(ChatModelConfig, "init_chat_model", init_chat_model)
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.enforce_model_group_policy", lambda *a, **k: None
    )
    return built


def _saw(model, text):
    return any(text in str(m.content) for m in model.calls[-1])


@pytest.mark.asyncio
async def test_unnamed_session_keeps_history_across_model_switch(
    tmp_path, monkeypatch
):
    # Regression: switching models reset the unnamed session's only copy of
    # the conversation, so every model after a switch started from scratch.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await hitl.run_agent("chat", "My secret word is PINEAPPLE.")
        await hitl.reconfigure_model("second-model", None)
        await hitl.run_agent("chat", "What is my secret word?")
        assert built[-1] is not built[0]  # really a new model
        assert _saw(built[-1], "PINEAPPLE")
    finally:
        await hitl.aclose()


@pytest.mark.asyncio
async def test_unnamed_session_shares_history_across_agent_types(
    tmp_path, monkeypatch
):
    # Same behaviour as a named session: one thread for every agent type.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await hitl.run_agent("chat", "My secret word is PINEAPPLE.")
        await hitl.run_agent("prompt", "Unrelated request.")
        assert _saw(built[-1], "PINEAPPLE")
    finally:
        await hitl.aclose()


class _SharedState(dict):
    __annotations__ = {"messages": list}


class _PlainState(dict):
    __annotations__ = {"context": str}


def _fake_agent_class(state_type, checkpointer=True):
    """Minimal agent: records the query it is given and returns a fixed answer."""

    class FakeAgent:
        def __init__(self):
            self.state_type = state_type
            self.checkpointer = object() if checkpointer else None
            self.queries = []

        def format_query(self, prompt, state=None):
            query = {"prompt": prompt, "state": state}
            self.queries.append(query)
            return query

        async def ainvoke(self, query, config=None):
            return {"messages": ["new"], "extra": "kept"}

        def format_result(self, result):
            return "answer"

    return FakeAgent


SharedA = _fake_agent_class(_SharedState)
SharedB = _fake_agent_class(_SharedState)
Plain = _fake_agent_class(_PlainState)
PlainOther = _fake_agent_class(_PlainState)
Unpersisted = _fake_agent_class(_SharedState, checkpointer=False)


@pytest.mark.parametrize(
    ("previous", "current", "handoff"),
    [
        # Both share the checkpointed history: the output is already there.
        (SharedA, SharedB, False),
        # Either side outside the shared history: the handoff is the only copy.
        (Plain, SharedA, True),  # e.g. /web -> /chat
        (SharedA, Plain, True),  # e.g. /chat -> /web
        (Plain, PlainOther, True),  # e.g. /web -> /lammps
        # Messages schema but no checkpointer: history is not shared.
        (Unpersisted, SharedA, True),
        # Same class (e.g. rebuilt after a model switch) never hands off to itself.
        (Plain, Plain, False),
        (SharedA, SharedA, False),
    ],
    ids=[
        "shared-to-shared",
        "plain-to-shared",
        "shared-to-plain",
        "plain-to-plain",
        "unpersisted-to-shared",
        "same-plain-class",
        "same-shared-class",
    ],
)
async def test_handoff_only_when_output_is_not_already_shared(
    previous, current, handoff
):
    wrapper = AgentHITL(agent_class=current)
    wrapper._agent = current()

    await wrapper(
        "next request", last_agent_result="PREVIOUS", last_agent=previous()
    )

    prompt = wrapper._agent.queries[-1]["prompt"]
    assert ("The last agent output was: PREVIOUS" in prompt) is handoff
    assert "next request" in prompt


@pytest.mark.parametrize("checkpointer", [True, False])
async def test_cached_messages_are_not_resent_to_checkpointed_agents(
    checkpointer,
):
    wrapper = AgentHITL(
        agent_class=_fake_agent_class(_SharedState, checkpointer=checkpointer)
    )
    wrapper._agent = wrapper.agent_class()
    cached = {"messages": ["old-1", "old-2"], "extra": "kept"}
    wrapper.state = cached

    await wrapper("hi")

    sent = wrapper._agent.queries[-1]["state"]
    assert sent["extra"] == "kept"  # non-message state is still passed
    # The checkpoint holds the history; only unpersisted agents need the copy.
    assert sent["messages"] == ([] if checkpointer else ["old-1", "old-2"])
    assert cached["messages"] == ["old-1", "old-2"]  # cache not mutated


@pytest.mark.asyncio
async def test_summarized_messages_are_not_resurrected_by_another_agent(
    tmp_path, monkeypatch
):
    # Regression: chat's cached copy re-sent messages that the prompt agent had
    # since summarized away, putting them back next to their own summary.
    _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await hitl.run_agent("chat", "Turn one mentions PINEAPPLE.")
        prompt = await hitl.get_agent("prompt")
        prompt._agent.tokens_before_summarize = 1
        prompt._agent.messages_to_keep = 1
        await hitl.run_agent("prompt", "Second turn from another agent.")
        await hitl.run_agent("chat", "Third turn, back on chat.")

        graph = await hitl.agents["chat"]._agent._aget_async_compiled_graph()
        snapshot = await graph.aget_state({
            "configurable": {"thread_id": hitl.thread_id}
        })
        messages = snapshot.values["messages"]
        raw = [
            m
            for m in messages
            if "PINEAPPLE" in str(m.content)
            and "[Summary" not in str(m.content)
        ]
        assert any("[Summary" in str(m.content) for m in messages)
        assert raw == []
    finally:
        await hitl.aclose()


@pytest.mark.asyncio
async def test_shared_history_agents_get_no_duplicate_handoff(
    tmp_path, monkeypatch
):
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await hitl.run_agent("chat", "My secret word is PINEAPPLE.")
        await hitl.run_agent("prompt", "Unrelated request.")
        sent = built[-1].calls[-1]
        assert not any(
            "The last agent output was" in str(m.content) for m in sent
        )
        assert _saw(built[-1], "PINEAPPLE")  # it still has the history
    finally:
        await hitl.aclose()
