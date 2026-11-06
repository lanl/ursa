from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading
from pathlib import Path

import pytest

from ursa.runtime import (
    AgentInvocationError,
    AgentManager,
    AgentMessage,
    AgentProcessExitedError,
    AgentSpec,
    AgentStatus,
    SyncAgentManager,
)

FACTORY = "tests.runtime.fixtures:stateful_factory"

pytestmark = pytest.mark.process_integration


@pytest.fixture
async def manager():
    async with AgentManager(
        heartbeat_interval=0.05, heartbeat_timeout=1
    ) as instance:
        yield instance


@pytest.mark.asyncio
async def test_agent_process_isolated_config_and_heartbeat(
    manager,
) -> None:
    await manager.register(
        AgentSpec("first", FACTORY, {"label": "one"}), start=True
    )
    await manager.register(
        AgentSpec("second", FACTORY, {"label": "two"}), start=True
    )
    configured_first = await manager.invoke("first", {})
    second = await manager.invoke("second", {})
    before = await manager.snapshot("first")
    async with asyncio.timeout(1):
        while True:
            snapshot = await manager.snapshot("first")
            if snapshot.last_heartbeat != before.last_heartbeat:
                break
            await asyncio.sleep(0.01)

    assert configured_first["label"] == "one"
    assert second["label"] == "two"
    assert second["pid"] != os.getpid()
    assert snapshot.status is AgentStatus.READY
    assert snapshot.heartbeat_age is not None
    assert snapshot.heartbeat_age < 1


@pytest.mark.asyncio
async def test_agent_process_routes_all_output_through_logging(
    manager, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    await manager.register(
        AgentSpec(
            "output",
            "tests.runtime.fixtures:output_factory",
        ),
        start=True,
    )
    assert await manager.invoke("output", {}) == "ok"

    messages = [record.getMessage() for record in caplog.records]
    assert any("agent factory marker" in message for message in messages)
    assert any("agent stdout marker" in message for message in messages)
    assert any("agent stderr marker" in message for message in messages)
    assert any("agent log marker" in message for message in messages)

    captured = capfd.readouterr()
    terminal = captured.out + captured.err
    assert "agent factory marker" not in terminal
    assert "agent stdout marker" not in terminal
    assert "agent stderr marker" not in terminal
    assert "agent log marker" not in terminal


@pytest.mark.asyncio
async def test_restart_keeps_only_committed_invocation_state(
    manager,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "agent-state.pickle"
    await manager.register(
        AgentSpec("agent", FACTORY, state_path=state_path), start=True
    )
    assert (await manager.invoke("agent", {}))["calls"] == 1

    interrupted = asyncio.create_task(
        manager.invoke("agent", {"announce": "host", "sleep": 30})
    )
    assert await manager.receive_message(timeout=1) == AgentMessage(
        sender="agent", recipient="host", payload="started"
    )
    await manager.kill_agent("agent", graceful_timeout=0.05)
    with pytest.raises(AgentProcessExitedError):
        await interrupted

    await manager.start_agent("agent")
    assert (await manager.invoke("agent", {}))["calls"] == 2
    assert state_path.is_file()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_discard_completed_state(
    manager,
) -> None:
    await manager.register(AgentSpec("agent", FACTORY), start=True)
    waiter = asyncio.create_task(
        manager.invoke("agent", {"announce": "host", "sleep": 0.15})
    )
    await manager.receive_message(timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert (await manager.invoke("agent", {}))["calls"] == 2


@pytest.mark.asyncio
async def test_shutdown_observes_exception_for_cancelled_invocation(
    manager,
) -> None:
    """Hard TUI shutdown must not leak an unhandled Future exception."""
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    unhandled: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        await manager.register(AgentSpec("agent", FACTORY), start=True)
        waiter = asyncio.create_task(
            manager.invoke("agent", {"announce": "host", "sleep": 30})
        )
        await manager.receive_message(timeout=1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        await manager.aclose()
        del waiter
        gc.collect()
        await asyncio.sleep(0)

        assert not any(
            context.get("message") == "Future exception was never retrieved"
            for context in unhandled
        )
    finally:
        loop.set_exception_handler(original_handler)


@pytest.mark.asyncio
async def test_host_and_agent_messages_are_routed(
    manager,
) -> None:
    for agent_id in ("sender", "receiver"):
        await manager.register(AgentSpec(agent_id, FACTORY), start=True)

    await manager.send_message("receiver", "from host")
    await manager.invoke(
        "sender", {"send": "receiver", "payload": "from agent"}
    )
    await manager.invoke("sender", {"send": "host", "payload": "for host"})

    await manager.restart_agent("receiver")
    received = await manager.call("receiver", "received")
    host_message = await manager.receive_message(timeout=1)
    assert received == [
        {"sender": "host", "payload": "from host"},
        {"sender": "sender", "payload": "from agent"},
    ]
    assert host_message == AgentMessage(
        sender="sender", recipient="host", payload="for host"
    )


@pytest.mark.asyncio
async def test_send_message_returns_after_recipient_state_is_committed(
    manager,
) -> None:
    await manager.register(AgentSpec("agent", FACTORY), start=True)

    await manager.send_message("agent", {"status": "delivered"})

    snapshot = await manager.snapshot("agent")
    assert snapshot.state["received"] == [
        {
            "sender": "host",
            "payload": {"status": "delivered"},
        }
    ]


@pytest.mark.asyncio
async def test_send_message_propagates_recipient_handler_failure(
    manager,
) -> None:
    await manager.register(
        AgentSpec("agent", FACTORY, {"raise_on_message": True}),
        start=True,
    )

    with pytest.raises(AgentInvocationError, match="message handler failed"):
        await manager.send_message("agent", "will fail")


@pytest.mark.asyncio
async def test_callbacks_are_dispatched_in_host_process(
    manager,
) -> None:
    calls: list[tuple[str, dict, str, dict, int]] = []

    class Callback:
        async def on_custom_event(
            self, name, data, *, run_id, metadata=None, **_kwargs
        ):
            calls.append((name, data, run_id, metadata, os.getpid()))

    await manager.register(AgentSpec("agent", FACTORY))
    await manager.invoke(
        "agent",
        {},
        config={
            "metadata": {
                "ursa_message_recipient": "root/reviewer",
                "request_kind": "delegated",
            }
        },
        callbacks=[Callback()],
    )
    assert calls == [
        (
            "test",
            {"calls": 1},
            "run",
            {
                "ursa_message_recipient": "root/reviewer",
                "request_kind": "delegated",
            },
            os.getpid(),
        )
    ]


@pytest.mark.asyncio
async def test_start_waits_for_concurrent_kill_before_new_generation(
    manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await manager.register(AgentSpec("agent", FACTORY), start=True)
    original = await manager.snapshot("agent")
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    send = manager._send

    async def controlled_send(entry, message, *, connection=None):
        if message.get("type") == "shutdown":
            shutdown_started.set()
            await release_shutdown.wait()
        await send(entry, message, connection=connection)

    monkeypatch.setattr(manager, "_send", controlled_send)
    killing = asyncio.create_task(manager.kill_agent("agent"))
    await asyncio.wait_for(shutdown_started.wait(), 1)
    starting = asyncio.create_task(manager.start_agent("agent"))
    release_shutdown.set()
    await asyncio.wait_for(asyncio.gather(killing, starting), 5)

    restarted = await manager.snapshot("agent")
    assert restarted.status is AgentStatus.READY
    assert restarted.generation == original.generation + 1
    assert restarted.pid is not None
    assert restarted.pid != original.pid


@pytest.mark.asyncio
async def test_cancelled_close_waiter_does_not_cancel_process_cleanup(
    manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await manager.register(AgentSpec("agent", FACTORY), start=True)
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    send = manager._send

    async def controlled_send(entry, message, *, connection=None):
        if message.get("type") == "shutdown":
            shutdown_started.set()
            await release_shutdown.wait()
        await send(entry, message, connection=connection)

    monkeypatch.setattr(manager, "_send", controlled_send)
    waiter = asyncio.create_task(manager.aclose())
    await asyncio.wait_for(shutdown_started.wait(), 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_shutdown.set()
    await asyncio.wait_for(manager.aclose(), 5)
    assert (await manager.snapshot("agent")).status is AgentStatus.STOPPED


@pytest.mark.asyncio
async def test_kill_waits_for_received_state_commit(
    manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit_started = threading.Event()
    release_commit = threading.Event()
    write_state = manager._write_state

    def controlled_write(path: Path, state: object) -> None:
        commit_started.set()
        assert release_commit.wait(5)
        write_state(path, state)

    monkeypatch.setattr(manager, "_write_state", controlled_write)
    await manager.register(
        AgentSpec(
            "agent",
            FACTORY,
            state_path=tmp_path / "state.pickle",
        ),
        start=True,
    )
    invoking = asyncio.create_task(manager.invoke("agent", {}))
    await asyncio.wait_for(asyncio.to_thread(commit_started.wait), 1)
    killing = asyncio.create_task(manager.kill_agent("agent"))
    await asyncio.sleep(0)
    assert not killing.done()

    release_commit.set()
    assert (await asyncio.wait_for(invoking, 5))["calls"] == 1
    await asyncio.wait_for(killing, 5)
    assert (await manager.snapshot("agent")).state == {
        "calls": 1,
        "received": [],
    }
    await manager.aclose()


def test_synchronous_manager_facade() -> None:
    with SyncAgentManager(
        heartbeat_interval=0.05, heartbeat_timeout=1
    ) as manager:
        manager.register(AgentSpec("agent", FACTORY), start=True)
        assert manager.invoke("agent", {})["calls"] == 1
        manager.send_message("agent", "from host")
        assert manager.call("agent", "received") == [
            {
                "sender": "host",
                "payload": "from host",
            }
        ]
        manager.broadcast("from broadcast")
        assert manager.call("agent", "received")[-1] == {
            "sender": "host",
            "payload": "from broadcast",
        }
        assert manager.snapshot("agent").status is AgentStatus.READY
        manager.kill_agent("agent")
        assert manager.snapshot("agent").status is AgentStatus.STOPPED
        manager.start_agent("agent")
        manager.unregister("agent")
