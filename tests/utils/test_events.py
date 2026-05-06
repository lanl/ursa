from pathlib import Path

import pytest
from langchain.tools import ToolRuntime

from ursa.agents.base import AgentContext
from ursa.util.events import AgentEvents, ToolEvents


def test_emit_dispatches_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_dispatch(event_name: str, payload: dict, config: dict) -> None:
        calls.append((event_name, payload, config))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    config = {"metadata": {"thread_id": "thread-1"}}
    events = AgentEvents(agent="planner", config=config)
    payload = events.emit(
        "Drafting plan",
        stage="generate",
        step_count=2,
    )

    assert payload == {
        "agent": "planner",
        "stage": "generate",
        "message": "Drafting plan",
        "step_count": 2,
    }
    assert calls == [("ursa_agent_progress", payload, config)]


def test_emit_is_noop_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_dispatch(*args, **kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    events = AgentEvents(agent="planner", config=None)
    assert events.emit("Drafting plan", stage="generate") is None
    assert called is False


def test_emit_tolerates_missing_parent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dispatch(*args, **kwargs) -> None:
        raise RuntimeError(
            "Unable to dispatch an adhoc event without a parent run id."
        )

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    payload = AgentEvents(
        agent="planner",
        config={"metadata": {"thread_id": "thread-2"}},
    ).emit("Drafting plan", stage="generate")

    assert payload == {
        "agent": "planner",
        "stage": "generate",
        "message": "Drafting plan",
    }


def test_tool_events_emit_tool_payload_from_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_dispatch(event_name: str, payload: dict, config: dict) -> None:
        calls.append((event_name, payload, config))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    runtime = ToolRuntime(
        state={},
        context=AgentContext(llm=None, workspace=Path("workspace")),
        config={"metadata": {"thread_id": "thread-9"}},
        stream_writer=lambda _: None,
        tool_call_id="tool-call-9",
        store=None,
    )

    events = ToolEvents.from_runtime("write_code", runtime)
    payload = events.emit(
        "Writing file",
        stage="write",
        path="workspace/sample.py",
    )

    assert payload == {
        "tool": "write_code",
        "tool_call_id": "tool-call-9",
        "stage": "write",
        "message": "Writing file",
        "path": "workspace/sample.py",
    }
    assert calls == [("ursa_agent_progress", payload, runtime.config)]


def test_range_emits_start_and_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_dispatch(event_name: str, payload: dict, config: dict) -> None:
        assert event_name == "ursa_agent_progress"
        assert config == {"callbacks": []}
        calls.append(payload)

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    events = AgentEvents(agent="planner", config={"callbacks": []})
    with events.range(
        "generate",
        "Drafting plan",
        done="Plan ready",
        iteration=1,
    ) as span:
        span.update(step_count=3)

    assert calls[0] == {
        "agent": "planner",
        "stage": "generate",
        "message": "Drafting plan",
        "phase": "start",
        "iteration": 1,
    }
    assert calls[1]["agent"] == "planner"
    assert calls[1]["stage"] == "generate"
    assert calls[1]["message"] == "Plan ready"
    assert calls[1]["phase"] == "end"
    assert calls[1]["iteration"] == 1
    assert calls[1]["step_count"] == 3
    assert isinstance(calls[1]["elapsed_ms"], float)


def test_range_emits_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_dispatch(event_name: str, payload: dict, config: dict) -> None:
        assert event_name == "ursa_agent_progress"
        assert config == {"callbacks": []}
        calls.append(payload)

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    events = AgentEvents(agent="planner", config={"callbacks": []})
    with (
        pytest.raises(RuntimeError, match="boom"),
        events.range(
            "generate",
            "Drafting plan",
            error="Plan failed",
            iteration=2,
        ) as span,
    ):
        span.update(step_count=1)
        raise RuntimeError("boom")

    assert calls[0]["phase"] == "start"
    assert calls[1]["phase"] == "error"
    assert calls[1]["message"] == "Plan failed"
    assert calls[1]["error_type"] == "RuntimeError"
    assert calls[1]["error"] == "boom"
    assert calls[1]["iteration"] == 2
    assert calls[1]["step_count"] == 1


@pytest.mark.asyncio
async def test_aemit_dispatches_and_async_range_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    async def fake_dispatch(
        event_name: str,
        payload: dict,
        config: dict,
    ) -> None:
        assert event_name == "ursa_agent_progress"
        assert config == {"callbacks": []}
        calls.append(payload)

    monkeypatch.setattr(
        "ursa.util.events.adispatch_custom_event",
        fake_dispatch,
    )

    events = AgentEvents(agent="acquisition", config={"callbacks": []})
    payload = await events.aemit(
        "Acquisition query ready",
        stage="search_query",
        query="heat transfer",
    )

    assert payload == {
        "agent": "acquisition",
        "stage": "search_query",
        "message": "Acquisition query ready",
        "query": "heat transfer",
    }

    async with events.range(
        "search_query",
        "Generating acquisition query",
        done="Acquisition query ready",
    ) as span:
        span.update(query="cooling load")

    assert calls[0] == payload
    assert calls[1]["phase"] == "start"
    assert calls[2]["phase"] == "end"
    assert calls[2]["query"] == "cooling load"
