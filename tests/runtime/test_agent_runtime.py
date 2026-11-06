from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from tests.runtime.fixtures import runtime_request, set_hierarchy, spawn_managed
from ursa.runtime import (
    AgentContext,
    AgentInvocationError,
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
    deliveries = [
        ("root", "Tell one short joke."),
        ("root/editor", "Use a penguin this time."),
    ]
    for sender, content in deliveries:
        assert (
            await process_agent.ainvoke(
                content,
                config={
                    "metadata": {
                        "ursa_message_sender": sender,
                        "ursa_message_recipient": "root/reviewer",
                    }
                },
            )
            == "ok"
        )

    messages = agent.seen[-1]["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Agent instructions"
    for index, (sender, content) in enumerate(deliveries):
        envelope, message = messages[1 + index * 2 : 3 + index * 2]
        assert isinstance(envelope, SystemMessage)
        assert f"Sender: {sender}" in envelope.content
        assert "Recipient: root/reviewer" in envelope.content
        assert "exact message content" in envelope.content
        assert isinstance(message, HumanMessage)
        assert message.content == content


@pytest.mark.asyncio
async def test_process_runtime_bridge_forwards_list_wait_options() -> None:
    """Child-side wait calls preserve list targets and all-mode for the host."""
    from ursa.runtime.agent_runtime import _ProcessRuntimeBridge

    context = SimpleNamespace(request_runtime=AsyncMock(return_value="done"))
    bridge = _ProcessRuntimeBridge(cast(AgentContext, context), "root/child")
    assert (
        await bridge.wait_for_agent(["sibling", "reviewer"], 30, wait_all=True)
        == "done"
    )
    context.request_runtime.assert_awaited_once_with(
        "wait_for",
        target=["sibling", "reviewer"],
        timeout_seconds=30,
        wait_all=True,
    )


async def test_kill_interrupts_an_active_runtime_invocation(
    runtime_factory, monkeypatch
):
    runtime, manager = runtime_factory()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def invoke(*_args, **_kwargs):
        started.set()
        await stopped.wait()
        raise AgentProcessExitedError("agent was stopped")

    monkeypatch.setattr(runtime, "_ensure_registered", AsyncMock())
    monkeypatch.setattr(manager, "start_agent", AsyncMock())
    monkeypatch.setattr(manager, "invoke", invoke)
    monkeypatch.setattr(
        manager, "kill_agent", AsyncMock(side_effect=lambda _id: stopped.set())
    )
    invocation = asyncio.create_task(runtime.run_agent("chat", "wait"))
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(runtime.kill_agent("chat"), 1)
    manager.kill_agent.assert_awaited_once_with(
        runtime.agents["chat"].runtime_id
    )
    with pytest.raises(AgentProcessExitedError, match="stopped"):
        await invocation


@pytest.mark.asyncio
async def test_agent_runtime_runs_cli_agent_outside_host_process(
    runtime_factory,
) -> None:
    runtime, manager = runtime_factory(fork=True)
    assert await runtime.run_agent("chat", "hello") == "ok"
    snapshot = await manager.snapshot(runtime.agents["chat"].runtime_id)
    assert snapshot.pid is not None
    assert snapshot.pid != os.getpid()
    assert runtime.agents["chat"]._agent is None


@pytest.mark.asyncio
async def test_runtime_serializes_filtered_tool_schemas(
    runtime_factory,
) -> None:
    _, manager = runtime_factory(fork=True)
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


@pytest.mark.asyncio
async def test_subagents_are_launched_as_managed_processes(
    runtime_factory,
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
    reviewer_id = await spawn_managed(runtime, "root/reviewer", "review this")
    reviewer_config = manager._entry(reviewer_id).spec.config["ursa_config"]
    assert reviewer_config.llm_model.model == "fake:subagent"
    assert reviewer_config.emb_model.model == "subagent-embeddings"
    await spawn_managed(runtime, "root/reviewer/critic", "critique this")

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
    reviewer_id = await spawn_managed(runtime, "root/reviewer", "Review this.")
    critic_id = await spawn_managed(
        runtime, "root/reviewer/critic", "Critique this."
    )

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


@pytest.mark.asyncio
async def test_subagent_model_usage_callbacks_keep_recipient_metadata(
    runtime_factory,
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
    recipients.clear()
    assert (
        await manager.send_message(
            root.runtime_id,
            {
                "prompt": "Message the root directly.",
                "config": {
                    "callbacks": None,
                    "metadata": {"ursa_message_recipient": "root"},
                },
            },
            callbacks=(Callback(),),
        )
        == "ok"
    )
    assert recipients and set(recipients) == {"root"}


@pytest.mark.asyncio
async def test_host_selects_requested_subagent_type(runtime_factory) -> None:
    runtime, manager = runtime_factory(
        fork=True, agent_config={"plan": {"max_reflection_steps": 7}}
    )
    root = await runtime.get_agent("chat")
    child_id = await spawn_managed(
        runtime, "root/planner", "Create a plan.", "plan"
    )
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


@pytest.mark.asyncio
async def test_subagent_limit_is_a_recoverable_runtime_tool_error(
    runtime_factory,
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


@pytest.mark.asyncio
async def test_unknown_subagent_path_does_not_fail_managed_agent(
    runtime_factory,
) -> None:
    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    reviewer_id = await spawn_managed(runtime, "root/reviewer", "Review this.")

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
    runtime_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, manager = runtime_factory()
    root_id, worker_id = "root-id", "worker-id"
    set_hierarchy(runtime, {root_id: "root"})
    runtime._runtime_agent_names[root_id] = "chat"
    started = asyncio.Event()
    release = asyncio.Event()

    async def spawn(_parent_id, _name, _agent_type=None):
        set_hierarchy(runtime, {root_id: "root", worker_id: "root/worker"})
        runtime._runtime_agent_names[worker_id] = "chat"
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
    runtime_factory, monkeypatch: pytest.MonkeyPatch
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
async def test_host_stops_only_a_descendant_subtree(
    runtime_factory, monkeypatch: pytest.MonkeyPatch
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
    runtime._runtime_agent_names.update(
        dict.fromkeys((root_id, reviewer_id, critic_id, analyst_id), "chat")
    )
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


@pytest.mark.parametrize(
    ("sender", "buffered", "message"),
    [("user", False, "wake now"), ("root/worker", True, "VOTE: player4")],
)
async def test_message_reaches_managed_wait_without_duplicate_invocation(
    runtime_factory, sender, buffered, message
):
    """Preserve delivery both before a wait and while its tool call is active."""
    runtime, manager = runtime_factory(fork=True)
    root = await runtime.get_agent("chat")
    child_path = "root/worker"
    child_id = await spawn_managed(runtime, child_path, "long task")
    await _wait_for_background_agent_tasks(runtime)
    runtime._active_root_id = root.runtime_id
    runtime._mark_working(child_path, 1)
    if buffered:
        runtime._mark_working("root", 1)
        assert (
            await runtime_request(
                runtime, child_id, "invoke", target="root", message=message
            )
            == "Message queued for root."
        )
        assert len(runtime._buffered_agent_messages["root"]) == 1
        assert not runtime._background_agent_tasks

    wait = asyncio.create_task(
        manager.call(root.runtime_id, "wait_for_subagent", child_path, 1)
    )
    if not buffered:
        async with asyncio.timeout(1):
            while not runtime._wait_dependencies.get("root"):
                await asyncio.sleep(0.01)
        assert runtime._wait_dependencies.get("root")
        # Delivery is consumed by the existing turn, with no separate reply.
        assert await runtime.run_active_agent("root", message) is None
    assert await wait == (
        "Received a message while waiting.\n"
        f"Sender: {sender}\nRecipient: root\n"
        f"Exact message content:\n{message}"
    )
    assert runtime._buffered_agent_messages == {}
    assert not runtime._background_agent_tasks
    if buffered:
        runtime._mark_working("root", -1)
    runtime._mark_working(child_path, -1)
    assert (await manager.snapshot(root.runtime_id)).status.value == "ready"


@pytest.mark.asyncio
async def test_agent_message_emits_one_activity_lifecycle(
    runtime_factory,
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
    child_path = "root/worker"
    await spawn_managed(runtime, child_path, "initial task")
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


@pytest.mark.asyncio
async def test_runtimes_share_manager_without_agent_id_or_ownership_collision(
    runtime_factory, tmp_path: Path
) -> None:
    first, manager = runtime_factory(fork=True, workspace=tmp_path / "first")
    second, _ = runtime_factory(
        fork=True, manager=manager, workspace=tmp_path / "second"
    )
    assert first.agents["chat"].runtime_id != second.agents["chat"].runtime_id
    assert await first.run_agent("chat", "first") == "ok"
    assert await second.run_agent("chat", "second") == "ok"

    await first.aclose()

    assert await second.run_agent("chat", "still running") == "ok"


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
    runtime_factory, monkeypatch: pytest.MonkeyPatch
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
