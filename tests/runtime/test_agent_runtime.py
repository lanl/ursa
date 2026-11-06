from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from tests.runtime.fixtures import runtime_request, set_hierarchy
from ursa.cli.config import UrsaConfig
from ursa.runtime import (
    AgentContext,
    AgentInvocationError,
    AgentManager,
    AgentProcessExitedError,
    AgentRuntime,
    AgentSpec,
)

pytestmark = pytest.mark.process_integration


async def _wait_for_background_agent_tasks(runtime: AgentRuntime) -> None:
    """Wait until every one-way message accepted so far has completed."""
    while runtime._background_agent_tasks:
        await asyncio.gather(
            *tuple(runtime._background_agent_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_process_agent_receives_message_delivery_envelope(process_runner):
    process_agent = process_runner("root/reviewer", result="ok")
    agent = process_agent.agent
    result = await process_agent.ainvoke(
        "Tell one short joke.",
        config={
            "metadata": {
                "ursa_message_sender": "root",
                "ursa_message_recipient": "root/reviewer",
            }
        },
    )
    second_result = await process_agent.ainvoke(
        "Use a penguin this time.",
        config={
            "metadata": {
                "ursa_message_sender": "root/editor",
                "ursa_message_recipient": "root/reviewer",
            }
        },
    )

    assert result == "ok"
    assert second_result == "ok"
    messages = agent.seen[-1]["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Agent instructions"
    assert isinstance(messages[1], SystemMessage)
    assert "Sender: root" in messages[1].content
    assert "Recipient: root/reviewer" in messages[1].content
    assert "exact message content" in messages[1].content
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "Tell one short joke."
    assert isinstance(messages[3], SystemMessage)
    assert "Sender: root/editor" in messages[3].content
    assert "Recipient: root/reviewer" in messages[3].content
    assert isinstance(messages[4], HumanMessage)
    assert messages[4].content == "Use a penguin this time."


@pytest.mark.asyncio
async def test_process_runtime_bridge_forwards_list_wait_options() -> None:
    """Child-side wait calls preserve list targets and all-mode for the host."""
    from ursa.runtime.agent_runtime import _ProcessRuntimeBridge

    calls: list[tuple[str, dict[str, object]]] = []

    class Context:
        async def request_runtime(self, action: str, **payload: object) -> str:
            calls.append((action, payload))
            return "done"

    bridge = _ProcessRuntimeBridge(cast(AgentContext, Context()), "root/child")
    assert (
        await bridge.wait_for_agent(["sibling", "reviewer"], 30, wait_all=True)
        == "done"
    )
    assert calls == [
        (
            "wait_for",
            {
                "target": ["sibling", "reviewer"],
                "timeout_seconds": 30,
                "wait_all": True,
            },
        )
    ]


class _BlockingManager:
    """Minimal manager double that models an interruptible live invocation."""

    def __init__(self) -> None:
        self.invocation_started = asyncio.Event()
        self.kill_started = asyncio.Event()
        self._stopped = asyncio.Event()

    async def start_agent(self, _agent_id: str) -> None:
        pass

    async def invoke(self, *_args, **_kwargs) -> str:
        self.invocation_started.set()
        await self._stopped.wait()
        raise AgentProcessExitedError("agent was stopped")

    async def kill_agent(self, _agent_id: str) -> None:
        self.kill_started.set()
        self._stopped.set()


@pytest.mark.asyncio
async def test_kill_interrupts_an_active_runtime_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _BlockingManager()
    runtime = AgentRuntime(
        UrsaConfig(workspace=tmp_path, llm_model={"model": "fake:chat"}),
        manager=cast(AgentManager, manager),
    )

    async def registered(_name: str) -> None:
        return None

    monkeypatch.setattr(runtime, "_ensure_registered", registered)
    invocation = asyncio.create_task(runtime.run_agent("chat", "wait"))
    await asyncio.wait_for(manager.invocation_started.wait(), 1)

    await asyncio.wait_for(runtime.kill_agent("chat"), 1)
    assert manager.kill_started.is_set()
    with pytest.raises(AgentProcessExitedError, match="stopped"):
        await invocation


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_agent_runtime_runs_cli_agent_outside_host_process(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory(fork=True)
    assert await runtime.run_agent("chat", "hello") == "ok"
    snapshot = await manager.snapshot(runtime.agents["chat"].runtime_id)
    assert snapshot.pid is not None
    assert snapshot.pid != os.getpid()
    assert runtime.agents["chat"]._agent is None


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_runtime_serializes_filtered_tool_schemas() -> None:
    manager = AgentManager(
        start_method="fork",
        heartbeat_interval=0.05,
        heartbeat_timeout=1,
    )
    try:
        await manager.register(
            AgentSpec(
                agent_id="tool-browser",
                factory="tests.runtime.fixtures:tool_describing_factory",
            )
        )
        tools = await manager.call("tool-browser", "describe_tools")
        by_name = {tool["name"]: tool for tool in tools}

        assert "run_command" in by_name
        assert set(by_name["run_command"]["schema"]["properties"]) == {"query"}
    finally:
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_subagents_are_launched_as_managed_processes(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory(
        fork=True,
        emb_model={"model": "openai:root-embeddings"},
        subagents={
            "max_agents": None,
            "max_depth": None,
            "llm_model": {"model": "fake:subagent"},
            "embedding_model": {"model": "openai:subagent-embeddings"},
        },
    )
    root = await runtime.get_agent("chat")
    assert await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "reviewer",
        "review this",
    ) == (
        "root/reviewer",
        "Message queued for root/reviewer.",
    )
    reviewer_id = runtime._hierarchies[root.runtime_id]["root/reviewer"]
    reviewer_config = manager._entry(reviewer_id).spec.config["ursa_config"]
    assert reviewer_config.llm_model.model == "fake:subagent"
    assert reviewer_config.emb_model.model == "subagent-embeddings"
    assert await manager.call(
        reviewer_id,
        "spawn_subagent",
        "critic",
        "critique this",
    ) == (
        "root/reviewer/critic",
        "Message queued for root/reviewer/critic.",
    )

    snapshots = [
        await manager.snapshot(runtime_id)
        for runtime_id in runtime._hierarchies[root.runtime_id].values()
    ]
    pids = {snapshot.pid for snapshot in snapshots}
    assert None not in pids
    assert len(pids) == 3
    assert os.getpid() not in pids
    assert (
        await manager.call(
            root.runtime_id,
            "send_subagent_message",
            "root/reviewer",
            "review this",
        )
        == "Message queued for root/reviewer."
    )
    runtime._active_root_id = root.runtime_id
    assert (
        await runtime.run_active_agent("root/reviewer", "user follow-up")
        == "ok"
    )
    assert await manager.call(root.runtime_id, "list_subagents") == [
        "root",
        "root/reviewer",
        "root/reviewer/critic",
    ]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_hierarchy_shares_checkpoint_file_with_distinct_threads(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable hierarchy members share one SQLite store, never one thread."""
    group_root = tmp_path / "checkpoint-group"
    group_root.mkdir()
    monkeypatch.setattr(
        "ursa.agents.base.group_root_dir", lambda _group: group_root
    )
    monkeypatch.setattr(
        "ursa.agents.base.group_agents_dir",
        lambda _group: group_root / "agents",
    )
    agent_name = "durable-hierarchy"
    root_thread = "hierarchy-session"
    runtime, manager = runtime_factory(
        fork=True, agent_name=agent_name, thread_id=root_thread
    )
    checkpoint_path = (
        group_root / "agents" / agent_name / "db" / "checkpointer.db"
    )
    assert await runtime.run_agent("chat", "Root work") == "ok"
    root = await runtime.get_agent("chat")
    assert await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "reviewer",
        "Review this.",
    ) == (
        "root/reviewer",
        "Message queued for root/reviewer.",
    )
    reviewer_id = runtime._hierarchies[root.runtime_id]["root/reviewer"]
    assert await manager.call(
        reviewer_id,
        "spawn_subagent",
        "critic",
        "Critique this.",
    ) == (
        "root/reviewer/critic",
        "Message queued for root/reviewer/critic.",
    )
    critic_id = runtime._hierarchies[root.runtime_id]["root/reviewer/critic"]

    configs = [
        manager._entry(runtime_id).spec.config["ursa_config"]
        for runtime_id in (root.runtime_id, reviewer_id, critic_id)
    ]
    assert [config.agent_name for config in configs] == [agent_name] * 3
    assert [config.thread_id for config in configs] == [
        root_thread,
        f"{root_thread}:root/reviewer",
        f"{root_thread}:root/reviewer/critic",
    ]
    await _wait_for_background_agent_tasks(runtime)
    assert checkpoint_path.is_file()

    with sqlite3.connect(checkpoint_path) as connection:
        threads = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            )
        }
    assert threads == {
        root_thread,
        f"{root_thread}:root/reviewer",
        f"{root_thread}:root/reviewer/critic",
    }


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_subagent_model_usage_callbacks_keep_recipient_metadata(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token callbacks from a child remain attributable to that child."""
    recipients: list[str | None] = []
    messages: list[tuple[str, tuple[str, ...], str]] = []

    class Callback:
        async def on_llm_end(self, _response, *, metadata=None, **_kwargs):
            recipients.append(
                metadata.get("ursa_message_recipient")
                if isinstance(metadata, dict)
                else None
            )

        async def on_hierarchy_message(self, sender, targets, message) -> None:
            messages.append((sender, tuple(targets), message))

    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    result = await runtime_request(
        runtime,
        root.runtime_id,
        "spawn",
        callbacks=(Callback(),),
        name="reviewer",
        message="Review this.",
    )

    assert result["path"] == "root/reviewer"
    assert result["response"] == "Message queued for root/reviewer."
    await _wait_for_background_agent_tasks(runtime)
    assert recipients
    assert set(recipients) == {"root/reviewer"}
    assert messages == [
        ("root", ("root/reviewer",), "Review this."),
    ]

    recipients.clear()
    messages.clear()
    runtime._active_root_id = root.runtime_id
    assert (
        await runtime.run_active_agent(
            "root/reviewer",
            "Follow up.",
            callbacks=(Callback(),),
        )
        == "ok"
    )
    assert recipients
    assert set(recipients) == {"root/reviewer"}
    assert messages == [
        ("user", ("root/reviewer",), "Follow up."),
        ("root/reviewer", ("user",), "ok"),
    ]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_host_selects_requested_subagent_type(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory(
        fork=True, agent_config={"plan": {"max_reflection_steps": 7}}
    )
    root = await runtime.get_agent("chat")
    child_path, response = await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "planner",
        "Create a plan.",
        "plan",
    )
    assert child_path == "root/planner"
    assert response

    child_id = runtime._hierarchies[root.runtime_id]["root/planner"]
    child_spec = manager._entry(child_id).spec.config
    assert child_spec["agent_name"] == "plan"
    assert runtime._runtime_agent_names[child_id] == "plan"
    assert child_spec["ursa_config"].agent_config == {
        "plan": {"max_reflection_steps": 7}
    }

    with pytest.raises(
        AgentInvocationError, match="Unknown subagent agent_type 'missing'"
    ):
        await manager.call(
            root.runtime_id,
            "spawn_subagent",
            "unknown",
            "This must not start.",
            "missing",
        )
    assert "root/unknown" not in runtime._hierarchies[root.runtime_id]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_subagent_limit_is_a_recoverable_runtime_tool_error(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory(fork=True, subagents={"max_agents": 1})
    root = await runtime.get_agent("chat")
    await manager.call(root.runtime_id, "spawn_subagent", "first", "first task")
    with pytest.raises(AgentInvocationError, match="max_agents=1"):
        await manager.call(
            root.runtime_id,
            "spawn_subagent",
            "overflow",
            "overflow task",
        )

    snapshot = await manager.snapshot(root.runtime_id)
    assert snapshot.status.value == "ready"
    assert await manager.call(root.runtime_id, "list_subagents") == [
        "root",
        "root/first",
    ]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_unknown_subagent_path_does_not_fail_managed_agent(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "reviewer",
        "Review this.",
    )
    reviewer_id = runtime._hierarchies[root.runtime_id]["root/reviewer"]

    with pytest.raises(AgentInvocationError, match="Unknown agent path"):
        await manager.call(
            reviewer_id,
            "send_subagent_message",
            "../foo",
            "Please review this.",
        )

    with pytest.raises(AgentInvocationError, match="Unknown agent path"):
        await manager.call(
            reviewer_id,
            "wait_for_subagent",
            "../foo",
            0,
        )

    assert (await manager.snapshot(reviewer_id)).status.value == "ready"
    assert (
        await manager.call(
            root.runtime_id,
            "send_subagent_message",
            "root/reviewer",
            "Still available?",
        )
        == "Message queued for root/reviewer."
    )


@pytest.mark.asyncio
async def test_host_spawn_returns_before_initial_task_finishes(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory()
    root_id, worker_id = "root-id", "worker-id"
    runtime._runtime_roots[root_id] = root_id
    runtime._runtime_paths[root_id] = "root"
    runtime._runtime_agent_names[root_id] = "chat"
    runtime._hierarchies[root_id] = {"root": root_id}
    started = asyncio.Event()
    release = asyncio.Event()

    async def spawn(_parent_id, _name, _agent_type=None):
        runtime._runtime_roots[worker_id] = root_id
        runtime._runtime_paths[worker_id] = "root/worker"
        runtime._runtime_agent_names[worker_id] = "chat"
        runtime._hierarchies[root_id]["root/worker"] = worker_id
        return "root/worker"

    async def invoke(*_args, **_kwargs):
        started.set()
        await release.wait()
        return "finished"

    monkeypatch.setattr(runtime, "_spawn_runtime_subagent", spawn)
    monkeypatch.setattr(manager, "invoke", invoke)
    try:
        result = await runtime_request(
            runtime, root_id, "spawn", name="worker", message="Do blocked work."
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        assert result == {
            "path": "root/worker",
            "response": "Message queued for root/worker.",
        }
        assert runtime._working_counts == {"root/worker": 1}
        assert not release.is_set()

        release.set()
        await _wait_for_background_agent_tasks(runtime)
        assert runtime._working_counts == {}
    finally:
        release.set()


@pytest.mark.asyncio
async def test_host_preflights_batched_agent_messages_before_any_delivery(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory()
    root_id, analyst_id, reviewer_id = "root-id", "analyst-id", "reviewer-id"
    set_hierarchy(
        runtime,
        {
            root_id: "root",
            analyst_id: "root/analyst",
            reviewer_id: "root/reviewer",
        },
    )
    runtime._active_root_id = root_id
    calls: list[str] = []
    messages: list[tuple[str, tuple[str, ...], str]] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def invoke(agent_id, *_args, **_kwargs):
        calls.append(agent_id)
        if len(calls) == 2:
            started.set()
        await release.wait()
        return f"reply from {agent_id}"

    monkeypatch.setattr(manager, "invoke", invoke)

    class Callback:
        async def on_hierarchy_message(self, sender, targets, message) -> None:
            messages.append((sender, tuple(targets), message))

    callbacks = (Callback(),)
    try:
        with pytest.raises(KeyError, match="Unknown agent path"):
            await runtime_request(
                runtime,
                root_id,
                "invoke_many",
                callbacks=callbacks,
                targets=["root/analyst", "root/missing", "root/reviewer"],
                message="Compare the proposal.",
            )
        assert calls == []
        assert messages == []

        results = await runtime_request(
            runtime,
            root_id,
            "invoke_many",
            callbacks=callbacks,
            targets=["root/reviewer", "root/analyst"],
            message="Compare the proposal.",
        )
        assert results == [
            {"result": "Message queued for root/reviewer."},
            {"result": "Message queued for root/analyst."},
        ]
        assert runtime._working_counts == {
            "root/reviewer": 1,
            "root/analyst": 1,
        }
        await asyncio.wait_for(started.wait(), timeout=1)
        assert messages == [
            (
                "root",
                ("root/reviewer", "root/analyst"),
                "Compare the proposal.",
            )
        ]
        release.set()
        await _wait_for_background_agent_tasks(runtime)
        assert runtime._working_counts == {}
    finally:
        release.set()


@pytest.mark.asyncio
async def test_host_expands_message_target_groups_atomically(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selectors expand only for sends, retain order, and never self-send."""
    runtime, manager = runtime_factory()
    root_id = "root-id"
    analyst_id = "analyst-id"
    reviewer_id = "reviewer-id"
    specialist_id = "specialist-id"
    set_hierarchy(
        runtime,
        {
            root_id: "root",
            analyst_id: "root/analyst",
            reviewer_id: "root/reviewer",
            specialist_id: "root/analyst/specialist",
        },
    )
    calls: list[str] = []
    messages: list[tuple[str, tuple[str, ...], str]] = []

    async def invoke(agent_id, *_args, **_kwargs):
        calls.append(agent_id)
        return f"reply from {agent_id}"

    monkeypatch.setattr(manager, "invoke", invoke)

    class Callback:
        async def on_hierarchy_message(self, sender, targets, message) -> None:
            messages.append((sender, tuple(targets), message))

    callbacks = (Callback(),)
    assert runtime._expand_message_target(analyst_id, "@everyone") == [
        "root",
        "root/reviewer",
        "root/analyst/specialist",
    ]
    for selector in ("@parent", "@siblings", "@children"):
        with pytest.raises(ValueError, match="Unknown message target group"):
            await runtime_request(
                runtime,
                analyst_id,
                "invoke_many",
                callbacks=callbacks,
                targets=["root/reviewer", selector],
                message="Do not partially deliver.",
            )
    assert calls == []
    assert messages == []
    results = await runtime_request(
        runtime,
        analyst_id,
        "invoke_many",
        callbacks=callbacks,
        targets=[
            "@peers",
            "root/reviewer",
            "@reports",
            "@supervisor",
            "root/analyst",
        ],
        message="Compare findings.",
    )
    assert results == [
        {"result": "Message queued for root/reviewer."},
        {"result": "Message queued for root/analyst/specialist."},
        {"result": "Message queued for root."},
    ]
    assert messages == [
        (
            "root/analyst",
            ("root/reviewer", "root/analyst/specialist", "root"),
            "Compare findings.",
        )
    ]
    await _wait_for_background_agent_tasks(runtime)
    assert calls == [reviewer_id, specialist_id, root_id]

    # Any concrete recipient that is stopping rejects the entire selector
    # expansion before a callback or a new delivery can occur.
    runtime._stopping_subtrees.add((root_id, "root/reviewer"))
    with pytest.raises(ValueError, match="is stopping"):
        await runtime_request(
            runtime,
            analyst_id,
            "invoke",
            callbacks=callbacks,
            target="@everyone",
            message="Status update.",
        )
    assert messages == [
        (
            "root/analyst",
            ("root/reviewer", "root/analyst/specialist", "root"),
            "Compare findings.",
        )
    ]
    assert calls == [reviewer_id, specialist_id, root_id]

    for selector in ("@supervisor", "@peers", "@reports", "@unknown"):
        with pytest.raises(ValueError):
            await runtime_request(
                runtime,
                root_id,
                "invoke",
                callbacks=callbacks,
                target=selector,
                message="No recipients.",
            )
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_host_stops_only_a_descendant_subtree(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory()
    root_id, reviewer_id, critic_id, analyst_id = (
        "root-id",
        "reviewer-id",
        "critic-id",
        "analyst-id",
    )
    set_hierarchy(
        runtime,
        {
            root_id: "root",
            reviewer_id: "root/reviewer",
            critic_id: "root/reviewer/critic",
            analyst_id: "root/analyst",
        },
    )
    runtime._runtime_agent_names.update({
        root_id: "chat",
        reviewer_id: "chat",
        critic_id: "chat",
        analyst_id: "chat",
    })
    runtime._active_root_id = root_id
    runtime._dynamic_runtime_ids.update({reviewer_id, critic_id, analyst_id})
    removed: list[tuple[str, dict[str, object]]] = []
    teardown_started = asyncio.Event()
    allow_teardown = asyncio.Event()

    async def unregister(runtime_id: str, **kwargs) -> None:
        removed.append((runtime_id, kwargs))
        teardown_started.set()
        await allow_teardown.wait()

    monkeypatch.setattr(manager, "unregister", unregister)
    try:
        with pytest.raises(
            ValueError, match="only stop one of its descendants"
        ):
            await runtime_request(
                runtime, reviewer_id, "stop", target="root/analyst"
            )
        assert (
            await runtime_request(runtime, root_id, "stop", target="reviewer")
            == "Stopped agent root/reviewer and its subagents."
        )
        await asyncio.wait_for(teardown_started.wait(), timeout=1)
        assert {
            status.path: status.state for status in runtime.hierarchy_status()
        }["root/reviewer"] == "Stopping"
        assert runtime._is_stopping_path(root_id, "root/reviewer")
        assert runtime._is_stopping_path(root_id, "root/reviewer/critic")
        with pytest.raises(ValueError, match="is stopping"):
            await runtime_request(
                runtime,
                root_id,
                "invoke",
                target="root/reviewer/critic",
                message="No more work.",
            )
        with pytest.raises(ValueError, match="is stopping"):
            await runtime_request(
                runtime,
                reviewer_id,
                "spawn",
                name="late-child",
                message="Do not start.",
            )
        assert not any(runtime_id == analyst_id for runtime_id, _ in removed)
        allow_teardown.set()
        for _ in range(100):
            if "root/reviewer" not in runtime._hierarchies[root_id]:
                break
            await asyncio.sleep(0.01)
        assert list(runtime._hierarchies[root_id]) == ["root", "root/analyst"]
        assert [runtime_id for runtime_id, _ in removed] == [
            critic_id,
            reviewer_id,
        ]
        assert all(kwargs == {"force": True} for _runtime_id, kwargs in removed)
    finally:
        allow_teardown.set()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_run_active_agent_message_wakes_a_managed_wait(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user message reaches a waiting process without stalling routing."""
    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    child_path, _ = await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "worker",
        "long task",
    )
    runtime._active_root_id = root.runtime_id
    runtime._mark_working(child_path, 1)
    wait = asyncio.create_task(
        manager.call(
            root.runtime_id,
            "wait_for_subagent",
            child_path,
            1,
        )
    )
    for _ in range(100):
        if runtime._wait_dependencies.get("root"):
            break
        await asyncio.sleep(0.01)
    assert runtime._wait_dependencies.get("root")

    # User messages take the same in-band path as peer messages. The
    # waiting invocation receives the exact delivery, so no second model
    # turn or duplicate response is created.
    assert await runtime.run_active_agent("root", "wake now") is None
    assert await wait == (
        "Received a message while waiting.\n"
        "Sender: user\nRecipient: root\n"
        "Exact message content:\nwake now"
    )
    runtime._mark_working(child_path, -1)
    assert (await manager.snapshot(root.runtime_id)).status.value == "ready"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_agent_message_is_returned_to_managed_wait(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model receives a reply that reaches the host just before its wait."""
    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    child_path, _ = await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "worker",
        "long task",
    )
    child_id = runtime._hierarchies[root.runtime_id][child_path]
    await _wait_for_background_agent_tasks(runtime)
    runtime._active_root_id = root.runtime_id
    runtime._mark_working(child_path, 1)
    runtime._mark_working("root", 1)
    assert (
        await runtime_request(
            runtime, child_id, "invoke", target="root", message="VOTE: player4"
        )
        == "Message queued for root."
    )
    assert len(runtime._buffered_agent_messages["root"]) == 1
    assert not runtime._background_agent_tasks
    wait = asyncio.create_task(
        manager.call(
            root.runtime_id,
            "wait_for_subagent",
            child_path,
            1,
        )
    )
    assert (
        await wait == "Received a message while waiting.\n"
        "Sender: root/worker\n"
        "Recipient: root\n"
        "Exact message content:\n"
        "VOTE: player4"
    )
    assert runtime._buffered_agent_messages == {}
    assert not runtime._background_agent_tasks
    runtime._mark_working("root", -1)
    runtime._mark_working(child_path, -1)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_agent_message_emits_one_activity_lifecycle(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One routed message produces exactly one progress lifecycle."""
    runtime, manager = runtime_factory(fork=True)
    lifecycle: list[tuple[str, str, bool | None]] = []

    class Callback:
        async def on_agent_message_start(self, path: str, **_kwargs) -> None:
            lifecycle.append(("start", path, None))

        async def on_agent_message_end(
            self, path: str, *, succeeded: bool, **_kwargs
        ) -> None:
            lifecycle.append(("end", path, succeeded))

    root = await runtime.get_agent("chat")
    child_path, _ = await manager.call(
        root.runtime_id,
        "spawn_subagent",
        "worker",
        "initial task",
    )
    await _wait_for_background_agent_tasks(runtime)
    assert (
        await runtime_request(
            runtime,
            root.runtime_id,
            "invoke",
            callbacks=(Callback(),),
            target=child_path,
            message="follow-up",
        )
        == f"Message queued for {child_path}."
    )
    await _wait_for_background_agent_tasks(runtime)
    assert lifecycle == [
        ("start", child_path, None),
        ("end", child_path, True),
    ]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_runtimes_share_manager_without_agent_id_or_ownership_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ursa.runtime.agent_runtime.validate_model_provider",
        lambda *_args: None,
    )
    manager = AgentManager(
        start_method="fork",
        heartbeat_interval=0.05,
        heartbeat_timeout=1,
    )
    first = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path / "first",
            llm_model={"model": "fake:chat"},
        ),
        manager=manager,
    )
    second = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path / "second",
            llm_model={"model": "fake:chat"},
        ),
        manager=manager,
    )
    try:
        assert (
            first.agents["chat"].runtime_id != second.agents["chat"].runtime_id
        )
        assert await first.run_agent("chat", "first") == "ok"
        assert await second.run_agent("chat", "second") == "ok"

        await first.aclose()

        assert await second.run_agent("chat", "still running") == "ok"
    finally:
        await first.aclose()
        await second.aclose()
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_runtime_resolves_selected_keyring_in_host_and_skips_unused(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolution_pid = tmp_path / "secret-resolution.pid"

    def get_password(service: str, username: str) -> str:
        assert (service, username) == ("ursa", "selected-account")
        resolution_pid.write_text(str(os.getpid()))
        return "selected-secret"

    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(get_password=get_password),
    )
    runtime, manager = runtime_factory(
        fork=True,
        inference_providers={
            "unused": {
                "model_provider": "openai",
                "api_key": {"keyring": "aiportal"},
            }
        },
        llm_model={
            "model": "fake:chat",
            "api_key": {"keyring": "selected-account"},
        },
    )
    assert await runtime.run_agent("chat", "hello") == "ok"
    assert int(resolution_pid.read_text()) == os.getpid()
    api_key = runtime.config.llm_model.api_key
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "selected-secret"


@pytest.mark.asyncio
async def test_runner_config_contains_only_host_materialized_secrets(
    runtime_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_TOKEN", "model-secret")
    monkeypatch.setenv("MCP_TOKEN", "mcp-secret")
    runtime, manager = runtime_factory(
        llm_model={"model": "fake:chat", "api_key": {"env": "MODEL_TOKEN"}},
        subagents={
            "llm_model": {
                "model": "fake:child",
                "api_key": {"env": "UNUSED_CHILD_TOKEN"},
            }
        },
        mcp_servers={
            "tools": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
                "headers": {
                    "Authorization": {
                        "env": "MCP_TOKEN",
                        "template": "Bearer %s",
                    }
                },
            }
        },
    )
    assert runtime.config.llm_model.api_key == SecretStr("model-secret")
    assert runtime.config.mcp_servers["tools"].headers == {
        "Authorization": "Bearer mcp-secret"
    }

    await runtime._ensure_registered("chat")
    runtime_id = runtime.agents["chat"].runtime_id
    runner_config = manager._entry(runtime_id).spec.config["ursa_config"]

    assert runner_config.inference_providers == {}
    assert runner_config.subagents.llm_model is None
    assert runner_config.llm_model.api_key == SecretStr("model-secret")
    assert runner_config.mcp_servers["tools"].headers == {
        "Authorization": "Bearer mcp-secret"
    }
