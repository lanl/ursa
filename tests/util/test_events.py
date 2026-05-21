import logging
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from langchain.tools import ToolRuntime

from ursa.agents.base import AgentContext
from ursa.util.events import (
    LOGGER,
    AgentEvents,
    EventConsoleFormatter,
    EventLoggingHandler,
    ToolEvents,
    configure_event_logging,
)

FIXED_MONOTONIC_TIMESTAMP_NS = 123456789


@pytest.fixture(autouse=True)
def fixed_monotonic_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ursa.util.events.monotonic_ns",
        lambda: FIXED_MONOTONIC_TIMESTAMP_NS,
    )


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
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
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


def test_emit_propagates_dispatch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_dispatch(*args, **kwargs) -> None:
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

    with pytest.raises(RuntimeError, match="dispatch failed"):
        AgentEvents(
            agent="planner",
            config={"metadata": {"thread_id": "thread-2"}},
        ).emit("Drafting plan", stage="generate")


def test_event_logging_handler_logs_structured_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = EventLoggingHandler()

    with caplog.at_level(logging.INFO, logger="ursa.util.events"):
        handler.on_custom_event(
            "ursa_agent_progress",
            {
                "agent": "planner",
                "stage": "generate",
                "message": "Drafting plan",
                "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
                "step_count": 2,
            },
            run_id=uuid4(),
        )

    assert caplog.messages == [
        'event="ursa_agent_progress" agent="planner" '
        'stage="generate" message="Drafting plan" '
        'data={"monotonic_timestamp_ns": 123456789, "step_count": 2}'
    ]


def test_configure_event_logging_sets_console_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_handlers = list(LOGGER.handlers)
    original_level = LOGGER.level
    original_propagate = LOGGER.propagate
    monkeypatch.setattr(LOGGER, "handlers", [])

    try:
        configure_event_logging()

        assert len(LOGGER.handlers) == 1
        assert isinstance(LOGGER.handlers[0].formatter, EventConsoleFormatter)
        assert LOGGER.level == logging.INFO
        assert LOGGER.propagate is False
    finally:
        LOGGER.handlers = original_handlers
        LOGGER.setLevel(original_level)
        LOGGER.propagate = original_propagate


def test_event_console_formatter_renders_readable_event() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(EventConsoleFormatter())
    logger = logging.getLogger("test.ursa.events.console")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        logger.info(
            "raw event message",
            extra={
                "ursa_event_payload": {
                    "tool": "write_code",
                    "stage": "write",
                    "phase": "end",
                    "message": "File written",
                    "path": "workspace/sample.py",
                    "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
                }
            },
        )

        assert stream.getvalue().strip() == (
            "[ursa] write_code write/end: File written "
            "(path=workspace/sample.py)"
        )
    finally:
        logger.handlers = []
        logger.propagate = True


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
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
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
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
        "phase": "start",
        "iteration": 1,
    }
    assert calls[1]["agent"] == "planner"
    assert calls[1]["stage"] == "generate"
    assert calls[1]["message"] == "Plan ready"
    assert calls[1]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
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
    assert calls[0]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert calls[1]["phase"] == "error"
    assert calls[1]["message"] == "Plan failed"
    assert calls[1]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
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
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
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
    assert calls[1]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert calls[2]["phase"] == "end"
    assert calls[2]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert calls[2]["query"] == "cooling load"
