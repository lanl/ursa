from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import SecretStr

from ursa.cli.config import UrsaConfig
from ursa.runtime import (
    AgentInvocationError,
    AgentManager,
    AgentProcessExitedError,
    AgentRuntime,
    AgentSpec,
)

pytestmark = pytest.mark.process_integration


class _BlockingManager:
    """Minimal manager double that models an interruptible live invocation."""

    def __init__(self) -> None:
        self.invocation_started = asyncio.Event()
        self.kill_started = asyncio.Event()
        self._stopped = asyncio.Event()

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
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
        ),
        manager=manager,
    )
    try:
        assert await runtime.run_agent("chat", "hello") == "ok"
        snapshot = await manager.snapshot(runtime.agents["chat"].runtime_id)
        assert snapshot.pid is not None
        assert snapshot.pid != os.getpid()
        assert runtime.agents["chat"]._agent is None
    finally:
        await runtime.aclose()


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
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            emb_model={"model": "openai:root-embeddings"},
            subagents={
                "max_agents": None,
                "max_depth": None,
                "llm_model": {"model": "fake:subagent"},
                "embedding_model": {"model": "openai:subagent-embeddings"},
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        assert await manager.call(
            root.runtime_id,
            "spawn_subagent",
            "reviewer",
            "review this",
        ) == ("root/reviewer", "ok")
        reviewer_id = runtime._hierarchies[root.runtime_id]["root/reviewer"]
        reviewer_config = manager._entry(reviewer_id).spec.config["ursa_config"]
        assert reviewer_config.llm_model.model == "fake:subagent"
        assert reviewer_config.emb_model.model == "subagent-embeddings"
        assert await manager.call(
            reviewer_id,
            "spawn_subagent",
            "critic",
            "critique this",
        ) == ("root/reviewer/critic", "ok")

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
            == "ok"
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
    finally:
        await runtime.aclose()
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_host_selects_requested_subagent_type(
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
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            agent_config={"plan": {"max_reflection_steps": 7}},
        ),
        manager=manager,
    )
    try:
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
    finally:
        await runtime.aclose()
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_subagent_limit_is_a_recoverable_runtime_tool_error(
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
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            subagents={"max_agents": 1},
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        await manager.call(
            root.runtime_id, "spawn_subagent", "first", "first task"
        )
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
    finally:
        await runtime.aclose()
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_unknown_subagent_path_does_not_fail_managed_agent(
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
    runtime = AgentRuntime(
        UrsaConfig(workspace=tmp_path, llm_model={"model": "fake:chat"}),
        manager=manager,
    )
    try:
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

        assert (await manager.snapshot(reviewer_id)).status.value == "ready"
        assert (
            await manager.call(
                root.runtime_id,
                "send_subagent_message",
                "root/reviewer",
                "Still available?",
            )
            == "ok"
        )
    finally:
        await runtime.aclose()
        await manager.aclose()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="test fixtures are inherited through fork",
)
@pytest.mark.asyncio
async def test_runtime_interrupts_active_work_and_drains_queued_messages():
    manager = AgentManager(
        start_method="fork",
        heartbeat_interval=0.05,
        heartbeat_timeout=1,
    )
    await manager.register(
        AgentSpec(
            agent_id="interruptible",
            factory=("tests.runtime.fixtures:interruptible_ursa_factory"),
        )
    )
    try:
        active = asyncio.create_task(manager.invoke("interruptible", "block"))
        started = await manager.receive_message(timeout=2)
        assert started.payload == "started"

        first = asyncio.create_task(
            manager.send_message(
                "interruptible",
                {"action": "request", "prompt": "first"},
            )
        )
        second = asyncio.create_task(
            manager.send_message(
                "interruptible",
                {"action": "request", "prompt": "second"},
            )
        )
        await asyncio.sleep(0.1)

        assert await manager.send_message(
            "interruptible", {"action": "interrupt_queued"}
        )
        with pytest.raises(AgentInvocationError, match="interrupted"):
            await active
        assert await first == "first:True"
        assert await second == "second:False"

        processed = [
            (await manager.receive_message(timeout=2)).payload,
            (await manager.receive_message(timeout=2)).payload,
        ]
        assert processed == [
            {"processed": "first", "interrupted": True},
            {"processed": "second", "interrupted": False},
        ]
    finally:
        await manager.aclose()


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ursa.runtime.agent_runtime.validate_model_provider",
        lambda *_args: None,
    )
    resolution_pid = tmp_path / "secret-resolution.pid"

    def get_password(service: str, username: str) -> str:
        assert (service, username) == ("ursa", "selected-account")
        resolution_pid.write_text(str(os.getpid()))
        return "selected-secret"

    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(get_password=get_password),
    )
    manager = AgentManager(
        start_method="fork",
        heartbeat_interval=0.05,
        heartbeat_timeout=1,
    )
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
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
        ),
        manager=manager,
    )
    try:
        assert await runtime.run_agent("chat", "hello") == "ok"
        assert int(resolution_pid.read_text()) == os.getpid()
        api_key = runtime.config.llm_model.api_key
        assert isinstance(api_key, SecretStr)
        assert api_key.get_secret_value() == "selected-secret"
    finally:
        await runtime.aclose()
        await manager.aclose()


@pytest.mark.asyncio
async def test_runner_config_contains_only_host_materialized_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_TOKEN", "model-secret")
    monkeypatch.setenv("MCP_TOKEN", "mcp-secret")
    manager = AgentManager()
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={
                "model": "fake:chat",
                "api_key": {"env": "MODEL_TOKEN"},
            },
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
        ),
        manager=manager,
    )
    try:
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
    finally:
        await runtime.aclose()
        await manager.aclose()
