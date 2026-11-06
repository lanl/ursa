from __future__ import annotations

import asyncio

import pytest

from tests.runtime.fixtures import runtime_request
from ursa.messaging import SubagentLimitError


@pytest.fixture
async def runtime(runtime_factory):
    runtime, _ = runtime_factory(subagents={"max_agents": 2})
    await runtime._ensure_registered("chat")
    runtime._active_root_id = runtime.agents["chat"].runtime_id
    return runtime


@pytest.mark.asyncio
async def test_starts_overlap_without_blocking_messages_or_exceeding_limits(
    runtime, monkeypatch
):
    root = runtime.agents["chat"].runtime_id
    both_started = asyncio.Event()
    release = {name: asyncio.Event() for name in ("first", "second")}
    starting = set()
    delivered = []

    async def start(agent_id):
        name = agent_id.rsplit("/", 1)[-1]
        starting.add(name)
        if len(starting) == 2:
            both_started.set()
        await release[name].wait()

    async def invoke(agent_id, message, **_kwargs):
        delivered.append((agent_id, message))
        return "ok"

    monkeypatch.setattr(runtime._manager, "start_agent", start)
    monkeypatch.setattr(runtime._manager, "invoke", invoke)
    spawns = [
        asyncio.create_task(
            runtime_request(runtime, root, "spawn", name=name, message=name)
        )
        for name in release
    ]
    try:
        # A serialized implementation cannot reach this barrier.
        await asyncio.wait_for(both_started.wait(), 1)
        assert {
            item.path: item.state for item in runtime.hierarchy_status()
        } == {
            "root": "Ready",
            "root/first": "Starting",
            "root/second": "Starting",
        }
        with pytest.raises(ValueError, match="already exists"):
            await runtime._spawn_runtime_subagent(root, "first")
        with pytest.raises(SubagentLimitError, match="limit reached"):
            await runtime._spawn_runtime_subagent(root, "third")
        assert "did not finish" in await runtime._wait_for_runtime_agent(
            root, "first", 0
        )

        release["first"].set()
        result = await asyncio.wait_for(spawns[0], 1)
        assert result["response"] == "Message queued for root/first."
        assert not spawns[1].done()
        await asyncio.gather(*runtime._background_agent_tasks)
        assert delivered == [(f"{root}:sub:root/first", "first")]

        release["second"].set()
        await asyncio.wait_for(spawns[1], 1)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(*spawns, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_start_releases_name_and_capacity(runtime, monkeypatch):
    root = runtime.agents["chat"].runtime_id
    fail = True

    async def start(_agent_id):
        if fail:
            raise RuntimeError("startup failed")

    monkeypatch.setattr(runtime._manager, "start_agent", start)
    with pytest.raises(RuntimeError, match="startup failed"):
        await runtime._spawn_runtime_subagent(root, "worker")
    assert runtime._hierarchies[root] == {"root": root}
    assert not runtime._dynamic_runtime_ids
    assert not runtime._subagent_start_tasks
    assert set(runtime._manager._agents) == {root}
    fail = False
    assert (
        await runtime._spawn_runtime_subagent(root, "worker") == "root/worker"
    )


@pytest.mark.parametrize("operation", ["stop", "close"])
async def test_shutdown_cancels_pending_startup(
    runtime, monkeypatch, operation
):
    root = runtime.agents["chat"].runtime_id
    started = asyncio.Event()

    async def start(_agent_id):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runtime._manager, "start_agent", start)
    spawn = asyncio.create_task(runtime._spawn_runtime_subagent(root, "worker"))
    await asyncio.wait_for(started.wait(), 1)
    if operation == "stop":
        result = await asyncio.wait_for(
            runtime._stop_runtime_subagent(root, "worker"), 1
        )
        assert result == "Stopped agent root/worker and its subagents."
        with pytest.raises(
            (ValueError, KeyError), match="is stopping|Unknown agent path"
        ):
            runtime._validate_message_target(root, "worker")
    else:
        await asyncio.wait_for(runtime.aclose(), 1)
    with pytest.raises(ValueError, match="stopped during startup"):
        await asyncio.wait_for(spawn, 1)
    await asyncio.gather(*runtime._background_teardown_tasks)
    assert not runtime._subagent_start_tasks
    assert not runtime._dynamic_runtime_ids
    if operation == "stop":
        assert runtime._hierarchies[root] == {"root": root}
        assert set(runtime._manager._agents) == {root}
    else:
        assert not runtime._manager._agents


@pytest.mark.asyncio
async def test_cancel_before_startup_task_runs_releases_reservation(
    runtime, monkeypatch
):
    root = runtime.agents["chat"].runtime_id
    create_task = asyncio.create_task

    def cancelled_start(coro, *, name=None, **kwargs):
        task = create_task(coro, name=name, **kwargs)
        if name and name.startswith("ursa-start-agent-"):
            task.cancel()
        return task

    monkeypatch.setattr(asyncio, "create_task", cancelled_start)
    with pytest.raises(ValueError, match="stopped during startup"):
        await runtime._spawn_runtime_subagent(root, "worker")
    assert runtime._hierarchies[root] == {"root": root}
    assert not runtime._subagent_start_tasks
    assert set(runtime._manager._agents) == {root}
