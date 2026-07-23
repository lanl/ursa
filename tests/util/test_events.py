import logging
from io import StringIO
from pathlib import Path
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
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    ursa_logger = logging.getLogger("ursa")
    original_ursa_level = ursa_logger.level
    monkeypatch.setattr(root_logger, "handlers", [])

    try:
        configure_event_logging()

        assert len(root_logger.handlers) == 1
        assert isinstance(
            root_logger.handlers[0].formatter, EventConsoleFormatter
        )
        assert root_logger.level == logging.WARNING
        assert ursa_logger.level == logging.INFO
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        ursa_logger.setLevel(original_ursa_level)


def test_configure_event_logging_can_disable_rich_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    ursa_logger = logging.getLogger("ursa")
    original_ursa_level = ursa_logger.level
    monkeypatch.setattr(root_logger, "handlers", [])

    try:
        configure_event_logging(rich=False)

        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, EventConsoleFormatter)
        assert formatter.render_artifacts is False
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        ursa_logger.setLevel(original_ursa_level)


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


def test_event_console_formatter_renders_mime_artifact() -> None:
    formatter = EventConsoleFormatter()
    record = logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )
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

    assert "[ursa] edit_code edit/end: File updated" in rendered
    assert "Edit diff" in rendered
    assert "-old" in rendered
    assert "+new" in rendered


def test_event_console_formatter_can_omit_mime_artifact() -> None:
    formatter = EventConsoleFormatter(render_artifacts=False)
    record = logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )
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

    assert rendered == "[ursa] edit_code edit/end: File updated"


def test_event_console_formatter_renders_multiple_artifacts_side_by_side() -> (
    None
):
    formatter = EventConsoleFormatter(force_terminal=False)
    record = logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )
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


def test_event_console_formatter_renders_structured_json_content() -> None:
    formatter = EventConsoleFormatter()
    record = logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )
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
    tmp_path: Path,
) -> None:
    target = tmp_path / "example.py"
    target.write_text("print('highlight me')\n", encoding="utf-8")
    formatter = EventConsoleFormatter()
    record = logging.LogRecord(
        name="ursa.util.events",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="raw event message",
        args=(),
        exc_info=None,
    )
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


def test_environment_events_include_environment_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_dispatch(event_name: str, payload: dict, config: dict) -> None:
        calls.append((event_name, payload, config))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        fake_dispatch,
    )

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
    assert calls == [("ursa_agent_progress", payload, config)]


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
        context=AgentContext(
            llm=None,
            workspace=Path("workspace"),
            den=Path("workspace"),
            agent_name=None,
            group="default",
        ),
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


def test_tool_events_include_owner_payload_from_runtime_metadata(
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
        context=AgentContext(
            llm=None,
            workspace=Path("workspace"),
            den=Path("workspace"),
            agent_name="fallback_agent",
            group="default",
        ),
        config={
            "metadata": {
                "environment_id": "team",
                "environment_member": "analyst",
                "environment_member_id": "team.analyst",
                "environment_member_role": "analysis",
                "environment_member_path": ["team", "analyst"],
            }
        },
        stream_writer=lambda _: None,
        tool_call_id="tool-call-10",
        store=None,
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
