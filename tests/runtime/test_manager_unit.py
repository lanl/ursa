from __future__ import annotations

import asyncio
import multiprocessing
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ursa.runtime import (
    AgentManager,
    AgentMessage,
    AgentSpec,
    AgentStatus,
)
from ursa.runtime.manager import (
    AgentProcess,
    _freeze_message,
    _ManagedAgent,
    _PendingRequest,
)


class _Connection:
    def send(self, _message: Any) -> None:
        pass

    def close(self) -> None:
        pass


class _UnitAgent:
    def __init__(self) -> None:
        self.state = {"calls": 0}
        self.messages: list[AgentMessage] = []

    async def ainvoke(self, inputs: Any, config=None) -> Any:
        del config
        self.state["calls"] += 1
        return {"input": inputs, "calls": self.state["calls"]}

    async def on_message(self, message: AgentMessage) -> None:
        self.messages.append(message)


def _process() -> AgentProcess:
    process = AgentProcess(
        AgentSpec("unit", lambda _config, _context: _UnitAgent()),
        None,
        _Connection(),
        SimpleNamespace(value=0.0),
        1,
    )
    process.agent = _UnitAgent()
    return process


def test_freeze_message_copies_nested_mutable_state() -> None:
    state = {"calls": [1]}
    frozen = _freeze_message({"state": state})

    state["calls"].append(2)

    assert frozen == {"state": {"calls": [1]}}


@pytest.mark.parametrize(
    ("agent_id", "factory", "message"),
    [
        (
            " ",
            "tests.runtime.fixtures:stateful_factory",
            "agent_id cannot be blank",
        ),
        (
            "agent",
            "missing_separator",
            "string agent factories must use 'package.module:qualname'",
        ),
    ],
)
def test_agent_spec_rejects_invalid_identity(
    agent_id: str, factory: str, message: str
) -> None:
    with pytest.raises(ValueError) as error:
        AgentSpec(agent_id, factory)
    assert str(error.value) == message


def test_agent_spec_accepts_import_factory_reference() -> None:
    spec = AgentSpec("agent", "package.module:factory")

    assert spec.agent_id == "agent"
    assert spec.factory == "package.module:factory"


@pytest.mark.asyncio
async def test_spawn_manager_primes_resource_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def ensure_running() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        "multiprocessing.resource_tracker.ensure_running", ensure_running
    )
    manager = AgentManager(start_method="spawn")

    await manager.aclose()

    assert calls == 1


@pytest.mark.asyncio
async def test_process_start_failure_rolls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AgentManager(start_method="fork")
    await manager.register(AgentSpec("agent", lambda *_args: None))

    def fail_start(_process: Any) -> None:
        raise ValueError("bad value(s) in fds_to_keep")

    monkeypatch.setattr("multiprocessing.process.BaseProcess.start", fail_start)

    with pytest.raises(ValueError, match="fds_to_keep"):
        await manager.start_agent("agent")

    snapshot = await manager.snapshot("agent")
    assert snapshot.status is AgentStatus.FAILED
    assert snapshot.pid is None
    await manager.aclose()


@pytest.mark.asyncio
async def test_agent_process_dispatches_invoke_and_snapshots_exact_state() -> (
    None
):
    process = _process()

    await process._dispatch({
        "type": "invoke",
        "request_id": "request-1",
        "inputs": "hello",
    })

    assert await process.outgoing.get() == {
        "type": "response",
        "request_id": "request-1",
        "ok": True,
        "result": {"input": "hello", "calls": 1},
        "has_state": True,
        "state": {"calls": 1},
    }


@pytest.mark.asyncio
async def test_agent_process_rejects_private_calls() -> None:
    process = _process()

    await process._dispatch({
        "type": "call",
        "request_id": "request-2",
        "method": "_private",
    })

    response = await process.outgoing.get()
    assert response["type"] == "response"
    assert response["request_id"] == "request-2"
    assert response["ok"] is False
    assert response["error_type"] == "ValueError"
    assert response["message"] == "only public agent methods may be called"


@pytest.mark.asyncio
async def test_agent_process_delivers_exact_message_before_snapshot() -> None:
    process = _process()
    message = AgentMessage("host", "unit", {"value": 7})

    await process._dispatch({
        "type": "message_handler",
        "request_id": "request-3",
        "message": message,
    })

    assert process.agent.messages == [message]
    assert await process.outgoing.get() == {
        "type": "response",
        "request_id": "request-3",
        "ok": True,
        "result": None,
        "has_state": True,
        "state": {"calls": 0},
    }


@pytest.mark.asyncio
async def test_stopping_process_releases_its_process_handle() -> None:
    class Process:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    manager = AgentManager()
    entry = _ManagedAgent(AgentSpec("unit", lambda *_args: None), None)
    process = Process()
    entry.process = cast(multiprocessing.Process, process)
    entry.generation = 1

    await manager._finish_stopped(
        entry, 1, _Connection(), cast(multiprocessing.Process, process)
    )

    assert process.closed
    assert entry.process is None


@pytest.mark.asyncio
async def test_stopping_old_process_preserves_newer_generation() -> None:
    class Process:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    manager = AgentManager()
    entry = _ManagedAgent(AgentSpec("unit", lambda *_args: None), None)
    stopped = Process()
    newer = Process()
    entry.process = cast(multiprocessing.Process, newer)
    entry.generation = 2

    await manager._finish_stopped(
        entry, 1, _Connection(), cast(multiprocessing.Process, stopped)
    )

    assert stopped.closed
    assert not newer.closed
    assert entry.process is newer


@pytest.mark.asyncio
async def test_manager_response_commits_state_before_resolving() -> None:
    from ursa.runtime import AgentManager

    manager = AgentManager()
    entry = _ManagedAgent(AgentSpec("unit", lambda *_args: None), None)
    future = asyncio.get_running_loop().create_future()
    entry.pending["request"] = _PendingRequest(future)

    await manager._handle_response(
        entry,
        {
            "type": "response",
            "request_id": "request",
            "ok": True,
            "result": "done",
            "has_state": True,
            "state": {"calls": 1},
        },
    )

    assert future.result() == "done"
    assert entry.state == {"calls": 1}
    assert entry.pending == {}
