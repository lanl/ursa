from __future__ import annotations

import asyncio
import multiprocessing
import os
from pathlib import Path
from typing import cast

import pytest

from ursa.cli.config import UrsaConfig
from ursa.runtime import (
    AgentManager,
    AgentProcessExitedError,
    AgentRuntime,
    AgentSpec,
)
from ursa.util.secrets import SecretReference

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
async def test_runtime_resolves_selected_keyring_in_child_and_skips_unused(
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

    monkeypatch.setattr("keyring.get_password", get_password)
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
        assert int(resolution_pid.read_text()) != os.getpid()
        api_key = runtime.config.llm_model.api_key
        assert isinstance(api_key, SecretReference)
        assert api_key.keyring == "selected-account"
    finally:
        await runtime.aclose()
        await manager.aclose()
