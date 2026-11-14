import asyncio
import json
import logging
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

# LangChain core bits
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.sqlite.aio import AsyncSqliteStore

# Your project imports
from tests.event_fixtures import (
    FIXED_MONOTONIC_TIMESTAMP_NS,
)
from tests.event_fixtures import (
    fixed_monotonic_timestamp as fixed_monotonic_timestamp,
)
from ursa.agents.base import (
    _INVOKE_DEPTH,
    AgentContext,
    AgentWithTools,
    BaseAgent,
)
from ursa.agents.chat_agent import BasicChatAgent
from ursa.util import Checkpointer
from ursa.util.checkpoint_retention import CheckpointPruneResult
from ursa.util.events import DEFAULT_EVENT_LOGGING_HANDLER


# --- Tiny offline model that triggers LLM callbacks and returns usage ---
class TinyCountingModel(BaseChatModel):
    """
    Offline fake chat model:
      - Pretends to be "openai:o3" (so it matches pricing.json keys)
      - Emits small token usage numbers (4 in, 5 out)
      - Never calls any external APIs
    """

    model: str = "openai:o3"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        # Simulate a small response with usage metadata
        usage = {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9}
        # Also mirror provider-style fields in response_metadata for robustness
        response_metadata = {
            "model": self.model,
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        ai = AIMessage(
            content="ok",
            response_metadata=response_metadata,
            usage_metadata=usage,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])


class SpecState(TypedDict, total=False):
    messages: Annotated[list, add_messages]


# --- Minimal agent under test (subclasses BaseAgent and makes one LLM call) ---
class Agent(BaseAgent):
    def __init__(self, llm, enable_metrics=False, **kwargs):
        super().__init__(llm, enable_metrics=enable_metrics, **kwargs)

    def _run_impl(self, state: SpecState):
        # Make one LLM call with callbacks + metadata wired in via build_config()
        cfg = self.build_config(tags=["TestAgent"])
        _ = self.llm.invoke(state["messages"], config=cfg)
        # Return something that looks like your agents' shape
        return {"messages": [AIMessage(content="done")]}

    def _build_graph(self):
        self.add_node(self._run_impl, "run_impl")
        self.graph.set_entry_point("run_impl")
        self.graph.set_finish_point("run_impl")


class ChildNodeState(TypedDict, total=False):
    value: str


class ParentNodeState(TypedDict, total=False):
    source: str
    result: str


class ChildNodeAgent(BaseAgent[ChildNodeState]):
    state_type = ChildNodeState

    @staticmethod
    def _uppercase(state: ChildNodeState) -> ChildNodeState:
        return {"value": state["value"].upper()}

    def _build_graph(self):
        self.add_node(self._uppercase, "uppercase")
        self.graph.set_entry_point("uppercase")
        self.graph.set_finish_point("uppercase")


class ParentNodeAgent(BaseAgent[ParentNodeState]):
    state_type = ParentNodeState

    def __init__(self, *args, child: ChildNodeAgent, **kwargs):
        self.child = child
        super().__init__(*args, **kwargs)

    def _build_graph(self):
        self.add_agent_node(
            "child",
            self.child,
            input_fn=lambda state: {"value": state["source"]},
            output_fn=lambda state: {"result": state["value"]},
        )
        self.graph.set_entry_point("child")
        self.graph.set_finish_point("child")


def test_add_agent_node_requires_adapters_and_exposes_one_subgraph(
    tmp_path,
):
    model = TinyCountingModel()
    child = ChildNodeAgent(model, workspace=tmp_path, enable_metrics=False)
    parent = ParentNodeAgent(
        model,
        child=child,
        workspace=tmp_path,
        enable_metrics=False,
    )

    result = parent.invoke({"source": "adapt me"})

    assert result == {"source": "adapt me", "result": "ADAPT ME"}
    assert parent.agent_nodes == {"child": child}
    with pytest.raises(TypeError):
        parent.agent_nodes["other"] = child
    assert [name for name, _graph in parent.compiled_graph.get_subgraphs()] == [
        "child"
    ]
    assert len(parent.compiled_graph.get_graph().nodes) == 3
    parent.close()


def test_add_agent_node_rejects_missing_adapters(tmp_path):
    class InvalidParentNodeAgent(ParentNodeAgent):
        def _build_graph(self):
            self.add_agent_node("child", self.child)

    with pytest.raises(TypeError, match="input_fn.*output_fn"):
        InvalidParentNodeAgent(
            TinyCountingModel(),
            child=ChildNodeAgent(
                TinyCountingModel(),
                workspace=tmp_path,
                enable_metrics=False,
            ),
            workspace=tmp_path,
            enable_metrics=False,
        )


def test_child_tool_change_invalidates_parent_compiled_graph(tmp_path):
    class ToolChildNodeAgent(AgentWithTools, ChildNodeAgent):
        pass

    child = ToolChildNodeAgent(
        TinyCountingModel(), workspace=tmp_path, enable_metrics=False
    )
    parent = ParentNodeAgent(
        TinyCountingModel(),
        child=child,
        workspace=tmp_path,
        enable_metrics=False,
    )
    first_graph = parent.compiled_graph

    @tool
    def added_later() -> str:
        """Return a value."""
        return "value"

    child.add_tool(added_later)

    assert "compiled_graph" not in parent.__dict__
    assert parent.compiled_graph is not first_graph
    parent.close()


def test_composite_detects_async_only_tools_on_a_child(tmp_path):
    child = ChildNodeAgent(
        TinyCountingModel(), workspace=tmp_path, enable_metrics=False
    )

    async def async_tool() -> str:
        return "ok"

    child.tools = [SimpleNamespace(func=None, coroutine=async_tool)]
    parent = ParentNodeAgent(
        TinyCountingModel(),
        child=child,
        workspace=tmp_path,
        enable_metrics=False,
    )

    assert parent._has_async_only_tools()
    parent.close()


@pytest.mark.asyncio
async def test_composite_uses_child_lifecycle_overrides_once(tmp_path):
    class LifecycleChildNodeAgent(ChildNodeAgent):
        def __init__(self, *args, **kwargs):
            self.close_calls = 0
            self.aclose_calls = 0
            super().__init__(*args, **kwargs)

        def close(self):
            self.close_calls += 1
            super().close()

        async def aclose(self):
            self.aclose_calls += 1
            await super().aclose()

    child = LifecycleChildNodeAgent(
        TinyCountingModel(), workspace=tmp_path, enable_metrics=False
    )
    parent = ParentNodeAgent(
        TinyCountingModel(),
        child=child,
        workspace=tmp_path,
        enable_metrics=False,
    )

    parent.close()
    await parent.aclose()

    assert child.close_calls == 1
    assert child.aclose_calls == 1


class EventAgent(Agent):
    def _run_impl(self, state: SpecState, config=None):
        self.events(config).emit("Drafting plan", stage="generate")
        return {"messages": [AIMessage(content="done")]}


class ToolAgent(AgentWithTools, Agent):
    pass


def _make_agent(agent_class=Agent, **kwargs):
    return agent_class(llm=TinyCountingModel(), **kwargs)


@pytest.fixture
def pricing_file(tmp_path: Path) -> Path:
    """
    Create a tiny pricing.json that matches the fake model ("openai:o3").
    Prices are per 1K tokens.
      input:  $0.002 / 1K
      output: $0.008 / 1K
    """
    data = {
        "_note": "Test pricing file for unit test",
        "openai:o3": {
            "input_per_1k": 0.002,
            "output_per_1k": 0.008,
            "cached_input_multiplier": 0.25,
        },
    }
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _read_single_metrics_file(metrics_dir: Path) -> dict:
    files = sorted(metrics_dir.glob("*.json"))
    assert files, f"No metrics JSON found in {metrics_dir}"
    # pick the newest
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def test_base_agent_metrics_and_pricing(
    tmpdir: Path, monkeypatch, pricing_file: Path
):
    """
    End-to-end: BaseAgent.invoke() triggers telemetry + pricing using a fake chat model.
    Asserts:
      - llm_events present with 1 call
      - usage_rollup matches fake counts
      - costs computed from pricing.json
      - totals sane
    """
    # Point loader at our local pricing.json
    monkeypatch.setenv("URSA_PRICING_JSON", str(pricing_file))

    # Instantiate agent with metrics enabled and autosave on
    agent = _make_agent(
        enable_metrics=True, autosave_metrics=True, workspace=tmpdir
    )

    # Run once (no network); prints telemetry and writes JSON
    out = agent.invoke("hello tiny model")
    assert isinstance(out, dict)
    assert "messages" in out

    # Read the latest metrics JSON
    assert Path(agent.telemetry.output_dir).is_dir()
    payload = _read_single_metrics_file(agent.telemetry.output_dir)

    # Basic structure present
    assert "llm_events" in payload
    assert isinstance(payload["llm_events"], list)
    assert len(payload["llm_events"]) >= 1

    # Pick the last event
    ev = payload["llm_events"][-1]
    assert ev["ok"] is True
    assert ev["metadata"]["model"] == "openai:o3"

    # Usage rollup should reflect our fake numbers
    roll = ev["metrics"]["usage_rollup"]
    assert roll["input_tokens"] == 4
    assert roll["output_tokens"] == 5
    assert roll["total_tokens"] == 9
    assert roll["prompt_tokens"] == 4
    assert roll["completion_tokens"] == 5
    assert roll["reasoning_tokens"] == 0
    assert roll["cached_tokens"] == 0

    # Costs block computed and totals correct:
    # input: 4 * 0.002 / 1000 = 0.000008
    # output: 5 * 0.008 / 1000 = 0.000040
    # total = 0.000048
    assert "costs" in payload
    total_usd = payload["costs"]["total_usd"]
    by_model = payload["costs"]["by_model_usd"]
    assert pytest.approx(total_usd, rel=1e-9, abs=1e-9) == 0.000048
    assert pytest.approx(by_model["openai:o3"], rel=1e-9, abs=1e-9) == 0.000048

    # Per-event cost details should be annotated as computed
    assert ev["metrics"]["cost_source"] == "computed"
    cd = ev["metrics"]["cost_details"]["components_usd"]
    assert pytest.approx(cd["input_cost"], rel=1e-9, abs=1e-9) == 0.000008
    assert pytest.approx(cd["output_cost"], rel=1e-9, abs=1e-9) == 0.000040
    assert pytest.approx(cd["total_cost"], rel=1e-9, abs=1e-9) == 0.000048


def test_metrics_toggle_off(tmpdir: Path, monkeypatch, pricing_file: Path):
    """
    When metrics=False, callbacks are disabled and render() prints nothing.
    No JSON should be saved to metrics_dir.
    """
    # Still set pricing, but it shouldn't be used
    monkeypatch.setenv("URSA_PRICING_JSON", str(pricing_file))

    agent = _make_agent(
        enable_metrics=False, autosave_metrics=True, workspace=tmpdir
    )

    _ = agent.invoke("hello")
    # No files should be created
    files = list(Path(agent.telemetry.output_dir).glob("*.json"))
    assert files == []


def test_base_agent_provisions_sqlite_store(tmpdir: Path):
    agent = _make_agent(workspace=tmpdir)

    store = agent.storage
    store.put(("tests",), "key", {"value": "ok"})

    item = store.get(("tests",), "key")
    assert item is not None
    assert item.value["value"] == "ok"


def test_build_config_includes_default_event_logging_handler(tmpdir: Path):
    agent = _make_agent(workspace=tmpdir, enable_metrics=False)

    config = agent.build_config()

    assert config["callbacks"] == [DEFAULT_EVENT_LOGGING_HANDLER]


def test_build_config_dedupes_default_callbacks(tmpdir: Path):
    agent = _make_agent(workspace=tmpdir, enable_metrics=True)
    custom_handler = AsyncCallbackHandler()

    config = agent.build_config(
        callbacks=[
            agent.telemetry.callbacks[0],
            DEFAULT_EVENT_LOGGING_HANDLER,
            custom_handler,
        ]
    )

    callbacks = config["callbacks"]
    assert callbacks[:4] == [
        *agent.telemetry.callbacks,
        DEFAULT_EVENT_LOGGING_HANDLER,
    ]
    assert callbacks[-1] is custom_handler
    assert (
        sum(callback is DEFAULT_EVENT_LOGGING_HANDLER for callback in callbacks)
        == 1
    )
    assert (
        sum(callback is agent.telemetry.callbacks[0] for callback in callbacks)
        == 1
    )


class AsyncOnlyFlagAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async def async_tool() -> str:
            return "ok"

        self.tools = [SimpleNamespace(func=None, coroutine=async_tool)]


def test_agent_with_tools_seeds_safe_codes_in_sync_store(tmp_path: Path):
    agent = _make_agent(
        ToolAgent,
        workspace=tmp_path,
        safe_codes=["python", "julia"],
        enable_metrics=False,
    )

    with closing(agent):
        safe_codes = [
            item.key
            for item in agent.storage.search(
                ("workspace", "safe_codes"), limit=1000
            )
        ]

    assert set(safe_codes) == {"python", "julia"}


async def test_agent_with_tools_seeds_safe_codes_in_async_store(tmp_path: Path):
    agent = _make_agent(
        ToolAgent,
        workspace=tmp_path,
        safe_codes=["python", "julia"],
        enable_metrics=False,
    )

    try:
        store = await agent._aget_async_storage()
        safe_codes = [
            item.key
            for item in await store.asearch(
                ("workspace", "safe_codes"), limit=1000
            )
        ]
    finally:
        await agent.aclose()
        agent.close()

    assert set(safe_codes) == {"python", "julia"}


async def test_shared_async_store_serializes_first_use_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = [
        _make_agent(workspace=tmp_path, enable_metrics=False) for _ in range(4)
    ]
    original_setup = AsyncSqliteStore.setup
    active_setups = 0
    max_active_setups = 0

    async def observed_setup(store: AsyncSqliteStore) -> None:
        nonlocal active_setups, max_active_setups
        active_setups += 1
        max_active_setups = max(max_active_setups, active_setups)
        try:
            await asyncio.sleep(0.01)
            await original_setup(store)
        finally:
            active_setups -= 1

    monkeypatch.setattr(AsyncSqliteStore, "setup", observed_setup)
    try:
        stores = await asyncio.gather(
            *(agent._aget_async_storage() for agent in agents)
        )
        assert max_active_setups == 1
        assert all(store is not None for store in stores)
    finally:
        await asyncio.gather(*(agent.aclose() for agent in agents))
        for agent in agents:
            agent.close()


@pytest.fixture
async def persistent_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "groups")
    agents = []

    def create(agent_class=Agent, **kwargs):
        agent = _make_agent(agent_class, enable_metrics=False, **kwargs)
        agents.append(agent)
        return agent

    yield create
    for agent in reversed(agents):
        await agent.aclose()
        agent.close()


async def test_persistent_agent_ainvoke_uses_async_sqlite_resources(
    persistent_agent,
):
    agent = persistent_agent(agent_name="persistent_async_agent")

    result = await agent.ainvoke("hello persistent async")

    assert result["messages"][-1].text == "done"
    assert (agent.den / "db" / "checkpointer.db").is_file()
    assert (agent.den / "graph_store.sqlite").is_file()


async def test_aclose_releases_and_joins_async_sqlite_resources(
    persistent_agent,
):
    """``aclose`` releases async SQLite resources and joins worker threads.

    ``aiosqlite.Connection`` is a non-daemon ``threading.Thread``. ``close()``
    finalizes queued queries and signals the worker to stop, but does not join
    the thread. ``aclose`` explicitly joins after closing (mirroring the HITL
    runtime's own teardown) so shutdown is deterministic and leaves no live
    aiosqlite worker threads, and it resets the cached async resources so a
    subsequent invoke rebuilds them and ``aclose`` is safely idempotent.
    """
    import threading

    import aiosqlite

    agent = persistent_agent(agent_name="aclose_join_agent")

    baseline = {
        t for t in threading.enumerate() if isinstance(t, aiosqlite.Connection)
    }

    # Provision the async SQLite-backed resources directly so we can assert on
    # them before/after aclose (ainvoke would otherwise close them itself).
    storage = await agent._aget_async_storage()
    checkpointer = await agent._aget_async_checkpointer()
    storage_conn = storage.conn
    checkpointer_conn = checkpointer.conn

    live_during = {
        t for t in threading.enumerate() if isinstance(t, aiosqlite.Connection)
    } - baseline
    assert live_during, (
        "expected async resources to open aiosqlite connection threads"
    )

    await agent.aclose()

    # Cached async resources are released so the next invoke rebuilds them.
    assert agent._async_storage is None
    assert agent._async_checkpointer is None
    # Threads were joined (not merely closed), so none remain alive.
    assert not storage_conn.is_alive()
    assert not checkpointer_conn.is_alive()

    # aclose is idempotent: a second call with nothing to close is a no-op.
    await agent.aclose()

    agent.close()


async def test_named_agent_restores_state_in_a_new_instance(persistent_agent):
    first = persistent_agent(
        agent_class=BasicChatAgent,
        agent_name="resumed-agent",
        thread_id="resumed-thread",
    )
    try:
        await first.ainvoke(first.format_query("first turn"))
    finally:
        await first.aclose()
        first.close()

    resumed = persistent_agent(
        agent_class=BasicChatAgent,
        agent_name="resumed-agent",
        thread_id="resumed-thread",
    )
    result = await resumed.ainvoke(resumed.format_query("second turn"))

    human_messages = [
        message.content
        for message in result["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert human_messages == ["first turn", "second turn"]


async def test_persistent_agent_supports_sync_and_async_invocation(
    persistent_agent,
):
    agent = persistent_agent(agent_name="persistent_sync_async_agent")

    sync_result = agent.invoke("hello sync")
    async_result = await agent.ainvoke("hello async")

    assert sync_result["messages"][-1].text == "done"
    assert async_result["messages"][-1].text == "done"


def test_persistent_async_only_agent_sync_invoke_uses_async_sqlite_resources(
    persistent_agent,
):
    agent = persistent_agent(
        agent_class=AsyncOnlyFlagAgent, agent_name="persistent_async_only_agent"
    )

    result = agent.invoke("hello async-only")

    assert result["messages"][-1].text == "done"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("custom_thread", [None, "custom-thread"])
async def test_persistent_agent_prunes_after_terminal_success(
    asynchronous, custom_thread, persistent_agent, monkeypatch
):
    prune = AsyncMock() if asynchronous else Mock()
    prune.return_value = CheckpointPruneResult(pruned=False)
    method = (
        "aprune_sqlite_checkpoints"
        if asynchronous
        else "prune_sqlite_checkpoints"
    )
    monkeypatch.setattr(f"ursa.agents.base.{method}", prune)
    agent = persistent_agent(agent_name="persistent_retention_agent")
    config = (
        {"configurable": {"thread_id": custom_thread}} if custom_thread else {}
    )
    if asynchronous:
        await agent.ainvoke("terminal run", config=config)
    else:
        agent.invoke("terminal run", config=config)
    prune.assert_called_once()
    assert prune.call_args.args[1] == (custom_thread or agent.thread_id)
    if asynchronous:
        prune.assert_awaited_once()


def test_user_supplied_checkpointer_is_not_automatically_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def record_prune(_saver, thread_id):
        calls.append(thread_id)
        return CheckpointPruneResult(pruned=False)

    monkeypatch.setattr(
        "ursa.agents.base.prune_sqlite_checkpoints", record_prune
    )
    checkpointer = Checkpointer.from_workspace(tmp_path / "provided")
    agent = _make_agent(checkpointer=checkpointer, enable_metrics=False)

    with closing(agent):
        agent.invoke("terminal provided-checkpointer run")

    assert calls == []


async def test_ainvoke_merges_callback_config(tmpdir: Path):
    agent = _make_agent(EventAgent, workspace=tmpdir, enable_metrics=False)
    handler = AsyncCallbackHandler()
    handler.on_custom_event = AsyncMock(return_value=None)

    result = await agent.ainvoke(
        "hello",
        config={"callbacks": [handler]},
    )

    assert result["messages"][-1].text == "done"
    handler.on_custom_event.assert_awaited_once()
    assert "run_id" in handler.on_custom_event.await_args.kwargs
    assert handler.on_custom_event.await_args.args == (
        "ursa_agent_progress",
        {
            "agent": "EventAgent",
            "stage": "generate",
            "message": "Drafting plan",
            "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
        },
    )


async def test_ainvoke_logs_progress_event_by_default(
    tmpdir: Path,
    caplog: pytest.LogCaptureFixture,
):
    agent = _make_agent(EventAgent, workspace=tmpdir, enable_metrics=False)

    with caplog.at_level(logging.INFO, logger="ursa.util.events"):
        result = await agent.ainvoke("hello")

    assert result["messages"][-1].text == "done"
    assert caplog.messages == [
        'event="ursa_agent_progress" agent="EventAgent" '
        'stage="generate" message="Drafting plan" '
        'data={"monotonic_timestamp_ns": 123456789}'
    ]


async def test_chat_interface(tmpdir: Path):
    agent = _make_agent(
        enable_metrics=False, autosave_metrics=True, workspace=tmpdir
    )

    # Convert a string to a query
    state = agent.format_query("Hello")
    assert "messages" in state

    state = await agent.ainvoke(state)
    result = agent.format_result(state)
    assert isinstance(result, str)

    # Repeat to check state option
    state = agent.format_query("Who are you?", state=state)
    assert "messages" in state
    state = await agent.ainvoke(state)
    result = agent.format_result(state)
    assert isinstance(result, str)


def test_runtime_injection_preserved_for_nodes(tmp_path: Path):
    captured: dict[str, Runtime[AgentContext]] = {}

    class RuntimeAwareAgent(BaseAgent[SpecState]):
        def __init__(self, **kwargs):
            super().__init__(llm=TinyCountingModel(), **kwargs)

        def _runtime_node(
            self, state: SpecState, runtime: Runtime[AgentContext]
        ) -> SpecState:
            captured["runtime"] = runtime
            assert isinstance(runtime.context, AgentContext)
            assert runtime.store is not None
            return state

        def _build_graph(self) -> None:
            self.add_node(self._runtime_node, "runtime_node")
            self.graph.set_entry_point("runtime_node")
            self.graph.set_finish_point("runtime_node")

    agent = RuntimeAwareAgent(workspace=tmp_path)
    agent.invoke("ensure runtime exists")

    runtime = captured.get("runtime")
    assert runtime is not None
    assert isinstance(runtime, Runtime)
    assert runtime.context.workspace == Path(tmp_path)


def test_default_thread_id_is_unique_per_agent(tmp_path):
    """Two agents built without a thread_id must not share a checkpoint thread.

    The constructor docstring promises a generated identifier. A constant one
    puts every library-built agent on the same thread of whatever checkpoint
    store it is given, so two agents sharing a store share a conversation
    (issue #332).
    """
    first = _make_agent(BasicChatAgent, workspace=tmp_path / "w1")
    second = _make_agent(BasicChatAgent, workspace=tmp_path / "w2")

    assert first.thread_id != second.thread_id


def test_explicit_thread_id_is_respected(tmp_path):
    """Passing a thread_id still pins it, which is how the CLI shares a thread."""
    agent = _make_agent(
        BasicChatAgent, workspace=tmp_path / "w", thread_id="ursa"
    )

    assert agent.thread_id == "ursa"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
async def test_invocation_scope_preserves_inputs_controls_and_error_cleanup(
    tmp_path, asynchronous, failure
):
    agent = _make_agent(workspace=tmp_path)
    agent.telemetry = Mock()
    target = (
        AsyncMock(return_value="done")
        if asynchronous
        else Mock(return_value="done")
    )
    engine = agent._ainvoke_engine if asynchronous else agent._invoke_engine
    result = engine(
        target, question="hello", recursion_limit=3, config={"tags": ["test"]}
    )
    assert (await result if asynchronous else result) == "done"
    target.assert_called_once_with(
        {"question": "hello"}, tags=["test"], recursion_limit=3
    )
    with pytest.raises(
        TypeError, match="Unexpected keyword argument 'question'"
    ):
        result = engine(target, {"question": "hello"}, question="ambiguous")
        if asynchronous:
            await result
    assert target.call_count == 1
    target.side_effect = failure("invocation failed")
    with pytest.raises(failure, match="invocation failed"):
        result = engine(target, "hello")
        if asynchronous:
            await result
    with pytest.raises(TypeError):
        result = engine(target, config=[42])
        if asynchronous:
            await result
    assert agent.telemetry.begin_run.call_count == 3
    assert agent.telemetry.render.call_count == 3
    assert _INVOKE_DEPTH.get() == 0
    assert agent._async_invocation_owner is None
    agent.close()


@pytest.mark.parametrize("fails", [False, True])
def test_stream_finalizes_telemetry_on_failure_or_early_close(tmp_path, fails):
    agent = _make_agent(workspace=tmp_path)
    agent.telemetry = Mock()

    def stream(inputs, **config):
        assert inputs == {"query": "hello"}
        assert config == {"tags": ["stream"], "stream_mode": "values"}
        assert _INVOKE_DEPTH.get() == 1
        yield "first"
        raise ValueError("stream failed")

    agent._stream = stream
    result = agent.stream(
        {"query": "hello"},
        {"tags": ["stream"]},
        stream_mode="values",
        raw_debug=True,
    )
    assert next(result) == "first"
    if fails:
        with pytest.raises(ValueError, match="stream failed"):
            next(result)
    else:
        result.close()
    agent.telemetry.begin_run.assert_called_once()
    agent.telemetry.render.assert_called_once_with(
        raw=True,
        save_json=None,
        save_otel=None,
        filepath=None,
        save_raw_snapshot=None,
        save_raw_records=None,
    )
    assert _INVOKE_DEPTH.get() == 0
    agent.close()
