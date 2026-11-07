import logging
from contextlib import AsyncExitStack, nullcontext
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langchain.tools import ToolRuntime

from ursa.agents.base import AgentContext
from ursa.util.events import (
    AgentEvents,
    EnvironmentEvents,
    EventLoggingHandler,
    ToolEvents,
    configure_event_logging,
)
from ursa.util.rendering import (
    EventConsoleFormatter,
    event_artifact,
    file_artifact,
)

FIXED_MONOTONIC_TIMESTAMP_NS = 123456789


@pytest.fixture(autouse=True)
def fixed_monotonic_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ursa.util.events.monotonic_ns",
        lambda: FIXED_MONOTONIC_TIMESTAMP_NS,
    )


@pytest.fixture
def dispatch(monkeypatch):
    mock = Mock()
    monkeypatch.setattr("ursa.util.events.dispatch_custom_event", mock)
    return mock


@pytest.fixture
def log_record():
    return logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )


@pytest.fixture
def tool_runtime():
    def create(agent_name, metadata, call_id):
        return ToolRuntime(
            state={},
            context=AgentContext(
                llm=None,
                workspace=Path("workspace"),
                den=Path("workspace"),
                agent_name=agent_name,
                group="default",
            ),
            config={"metadata": metadata},
            stream_writer=lambda _: None,
            tool_call_id=call_id,
            store=None,
        )

    return create


def test_emit_dispatches_structured_payload(dispatch) -> None:
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
    dispatch.assert_called_once_with(
        "ursa_agent_progress", payload, config=config
    )


def test_emit_is_noop_without_config(dispatch) -> None:
    events = AgentEvents(agent="planner", config=None)
    assert events.emit("Drafting plan", stage="generate") is None
    dispatch.assert_not_called()


def test_emit_propagates_dispatch_errors(dispatch) -> None:
    dispatch.side_effect = RuntimeError("dispatch failed")
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


@pytest.mark.parametrize("rich", [True, False])
def test_configure_event_logging_sets_console_handler(monkeypatch, rich):
    root_logger = logging.getLogger()
    ursa_logger = logging.getLogger("ursa")
    original_levels = root_logger.level, ursa_logger.level
    monkeypatch.setattr(root_logger, "handlers", [])
    try:
        if rich:
            configure_event_logging()
        else:
            configure_event_logging(rich=False)

        assert len(root_logger.handlers) == 1
        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, EventConsoleFormatter)
        assert formatter.render_artifacts is rich
        assert root_logger.level == logging.WARNING
        assert ursa_logger.level == logging.INFO
    finally:
        # setLevel also clears logging's enabled-level cache.
        root_logger.setLevel(original_levels[0])
        ursa_logger.setLevel(original_levels[1])


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


@pytest.mark.parametrize("render_artifacts", [True, False])
def test_event_console_formatter_renders_mime_artifact(
    log_record, render_artifacts
):
    formatter = (
        EventConsoleFormatter()
        if render_artifacts
        else EventConsoleFormatter(render_artifacts=False)
    )
    record = log_record
    record.ursa_event_payload = {
        "tool": "edit_code",
        "stage": "edit",
        "phase": "end",
        "message": "File updated",
        "artifact": event_artifact(
            "--- app.py\n+++ app.py\n-old\n+new",
            "text/x-diff",
            metadata={"title": "Edit diff", "path": "app.py"},
        ),
    }

    rendered = formatter.format(record)

    if render_artifacts:
        assert "[ursa] edit_code edit/end: File updated" in rendered
        assert "Edit diff" in rendered
        assert "-old" in rendered
        assert "+new" in rendered
    else:
        assert rendered == "[ursa] edit_code edit/end: File updated"


def test_event_console_formatter_renders_multiple_artifacts_side_by_side(
    log_record,
) -> None:
    formatter = EventConsoleFormatter(force_terminal=False)
    record = log_record
    record.ursa_event_payload = {
        "tool": "run_command",
        "stage": "execute",
        "phase": "end",
        "message": "Command finished",
        "artifacts": [
            event_artifact(
                "command output " * 20,
                "text/plain",
                metadata={"title": "stdout"},
            ),
            event_artifact(
                "command warning " * 20,
                "text/plain",
                metadata={"title": "stderr"},
            ),
        ],
    }

    rendered = formatter.format(record)

    assert any(
        "stdout" in line and "stderr" in line for line in rendered.splitlines()
    )
    assert any(
        "command output" in line and "command warning" in line
        for line in rendered.splitlines()
    )


def test_event_console_formatter_renders_structured_json_content(
    log_record,
) -> None:
    formatter = EventConsoleFormatter()
    record = log_record
    record.ursa_event_payload = {
        "tool": "inspect",
        "stage": "result",
        "message": "Inspection complete",
        "artifact": event_artifact(
            {"answer": 42},
            "application/json",
            metadata={"title": "Result"},
        ),
    }

    rendered = formatter.format(record)

    assert "Result" in rendered
    assert '"answer": 42' in rendered


def test_event_console_formatter_renders_referenced_file_contents(
    tmp_path: Path, log_record
) -> None:
    target = tmp_path / "example.py"
    target.write_text("print('highlight me')\n", encoding="utf-8")
    formatter = EventConsoleFormatter()
    record = log_record
    record.ursa_event_payload = {
        "tool": "read_file",
        "stage": "read",
        "phase": "end",
        "message": "File read",
        "artifact": file_artifact(target, title="File read"),
    }

    rendered = formatter.format(record)

    assert "File read" in rendered
    assert "print('highlight me')" in rendered


def test_environment_events_include_environment_metadata(dispatch) -> None:
    config = {"metadata": {"thread_id": "thread-env"}}
    events = EnvironmentEvents(
        environment="team",
        config=config,
        environment_type="agent_team",
        environment_id="team-id",
        path=["team"],
    )
    payload = events.emit("Team started", stage="team", phase="start")

    assert payload == {
        "environment": "team",
        "stage": "team",
        "message": "Team started",
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
        "environment_type": "agent_team",
        "environment_id": "team-id",
        "path": ["team"],
        "phase": "start",
    }
    dispatch.assert_called_once_with(
        "ursa_agent_progress", payload, config=config
    )


def test_tool_events_emit_tool_payload_from_runtime(
    dispatch, tool_runtime
) -> None:
    runtime = tool_runtime(None, {"thread_id": "thread-9"}, "tool-call-9")

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
    dispatch.assert_called_once_with(
        "ursa_agent_progress", payload, config=runtime.config
    )


def test_tool_events_include_owner_from_named_agent_context(
    dispatch, tool_runtime
) -> None:
    runtime = tool_runtime(
        "dummy_bot_3000", {"thread_id": "thread-10"}, "tool-call-10"
    )

    events = ToolEvents.from_runtime("write_code", runtime)
    payload = events.emit(
        "File written",
        stage="write",
        phase="end",
    )

    assert payload["tool"] == "write_code"
    assert payload["agent"] == "dummy_bot_3000"
    assert payload["tool_call_id"] == "tool-call-10"
    dispatch.assert_called_once_with(
        "ursa_agent_progress", payload, config=runtime.config
    )


def test_tool_events_include_owner_payload_from_runtime_metadata(
    dispatch, tool_runtime
) -> None:
    runtime = tool_runtime(
        "fallback_agent",
        {
            "environment_id": "team",
            "environment_member": "analyst",
            "environment_member_id": "team.analyst",
            "environment_member_role": "analysis",
            "environment_member_path": ["team", "analyst"],
        },
        "tool-call-10",
    )

    events = ToolEvents.from_runtime("run_web_search", runtime)
    payload = events.emit("Searching Web", stage="search", query="fusion")

    assert payload["tool"] == "run_web_search"
    assert payload["tool_call_id"] == "tool-call-10"
    assert payload["agent"] == "analyst"
    assert payload["agent_id"] == "team.analyst"
    assert payload["environment_id"] == "team"
    assert payload["environment_member"] == "analyst"
    assert payload["environment_member_id"] == "team.analyst"
    assert payload["environment_member_role"] == "analysis"
    assert payload["environment_member_path"] == ["team", "analyst"]
    dispatch.assert_called_once_with(
        "ursa_agent_progress", payload, config=runtime.config
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_range_emits_start_and_terminal_payload(
    monkeypatch, asynchronous, failure
):
    dispatch = AsyncMock() if asynchronous else Mock()
    method = (
        "adispatch_custom_event" if asynchronous else "dispatch_custom_event"
    )
    monkeypatch.setattr(f"ursa.util.events.{method}", dispatch)
    config = {"callbacks": []}
    events = AgentEvents(agent="planner", config=config)
    iteration = 2 if failure else 1
    steps = 1 if failure else 3
    span = events.range(
        "generate",
        "Drafting plan",
        done="Plan ready",
        error="Plan failed",
        iteration=iteration,
    )
    with (
        pytest.raises(RuntimeError, match="boom") if failure else nullcontext()
    ):
        async with AsyncExitStack() as stack:
            if asynchronous:
                await stack.enter_async_context(span)
            else:
                stack.enter_context(span)
            span.update(step_count=steps)
            if failure:
                raise RuntimeError("boom")

    assert len(dispatch.call_args_list) == 2
    if asynchronous:
        assert dispatch.await_count == 2
    for call in dispatch.call_args_list:
        assert call.args[0] == "ursa_agent_progress"
        assert call.kwargs == {"config": config}
    start, end = (call.args[1] for call in dispatch.call_args_list)
    assert start == {
        "agent": "planner",
        "stage": "generate",
        "message": "Drafting plan",
        "monotonic_timestamp_ns": FIXED_MONOTONIC_TIMESTAMP_NS,
        "phase": "start",
        "iteration": iteration,
    }
    assert end["agent"] == "planner"
    assert end["stage"] == "generate"
    assert end["message"] == ("Plan failed" if failure else "Plan ready")
    assert end["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert end["phase"] == ("error" if failure else "end")
    assert end["iteration"] == iteration
    assert end["step_count"] == steps
    assert isinstance(end["elapsed_ms"], float)
    if failure:
        assert end["error_type"] == "RuntimeError"
        assert end["error"] == "boom"


async def test_aemit_dispatches_and_async_range_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "ursa.util.events.adispatch_custom_event",
        dispatch,
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

    assert dispatch.await_count == dispatch.call_count == 3
    for call in dispatch.await_args_list:
        assert call.args[0] == "ursa_agent_progress"
        assert call.kwargs == {"config": {"callbacks": []}}
    calls = [call.args[1] for call in dispatch.await_args_list]
    assert calls[0] == payload
    assert calls[1]["phase"] == "start"
    assert calls[1]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert calls[2]["phase"] == "end"
    assert calls[2]["monotonic_timestamp_ns"] == FIXED_MONOTONIC_TIMESTAMP_NS
    assert calls[2]["query"] == "cooling load"
