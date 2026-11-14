from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

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
        Mock(spec=["send", "close"]),
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


async def test_spawn_manager_primes_resource_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_running = Mock()
    monkeypatch.setattr(
        "multiprocessing.resource_tracker.ensure_running", ensure_running
    )
    manager = AgentManager(start_method="spawn")

    await manager.aclose()

    ensure_running.assert_called_once_with()


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


@pytest.mark.parametrize("kind", ["invoke", "message_handler"])
async def test_process_dispatch_returns_exact_result_and_committed_state(kind):
    process = _process()
    message = AgentMessage("host", "unit", {"value": 7})
    invoking = kind == "invoke"
    payload = {"inputs": "hello"} if invoking else {"message": message}
    calls = 1 if invoking else 0
    await process._dispatch({"type": kind, "request_id": "request", **payload})
    assert process.agent.messages == ([] if invoking else [message])
    assert await process.outgoing.get() == {
        "type": "response",
        "request_id": "request",
        "ok": True,
        "result": {"input": "hello", "calls": calls} if invoking else None,
        "has_state": True,
        "state": {"calls": calls},
    }


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


@pytest.mark.parametrize("kind", ["invoke", "call", "message_handler"])
async def test_unserializable_result_fails_only_its_request(kind, monkeypatch):
    process = _process()
    method = "on_message" if kind == "message_handler" else "ainvoke"
    monkeypatch.setattr(
        process.agent, method, AsyncMock(side_effect=[lambda: None, "ok"])
    )
    request = {
        "type": kind,
        "method": method,
        "inputs": "hello",
        "args": ["hello"],
        "message": AgentMessage("host", "unit", "hello"),
    }
    await process._dispatch(request | {"request_id": "bad"})
    failed = await process.outgoing.get()
    assert not failed["ok"] and failed["request_id"] == "bad"
    assert "pickle" in failed["message"]
    assert process.context.active_request_id is None

    await process._dispatch(request | {"request_id": "next"})
    succeeded = await process.outgoing.get()
    assert succeeded["ok"] and succeeded["request_id"] == "next"
    assert succeeded["result"] == "ok"
    assert succeeded["state"] == {"calls": 0}


@pytest.mark.parametrize("new_generation", [False, True])
async def test_stopping_process_closes_only_its_own_generation(new_generation):
    manager = AgentManager()
    entry = _ManagedAgent(AgentSpec("unit", lambda *_args: None), None)
    stopped = Mock(spec=["close"])
    current = Mock(spec=["close"]) if new_generation else stopped
    entry.process = current
    entry.generation = 2 if new_generation else 1

    await manager._finish_stopped(
        entry, 1, Mock(spec=["send", "close"]), stopped
    )

    stopped.close.assert_called_once_with()
    assert entry.process is (current if new_generation else None)
    if new_generation:
        current.close.assert_not_called()


async def test_manager_response_commits_state_before_resolving() -> None:
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
