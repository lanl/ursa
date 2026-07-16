"""Structured progress events for URSA agents and tools.

URSA runs inside LangGraph/LangChain, so progress updates can be surfaced as
custom events instead of being printed to stdout. This module provides a small,
opinionated wrapper around LangChain's custom event APIs so both agents and
tools can do two common things without repeating the same plumbing:

1. Emit a one-off event with ``emit`` / ``aemit``.
2. Mark a scoped range with ``with events.range(...)`` or
   ``async with events.range(...)``.

Every emitted payload includes:

- a source identifier such as ``agent="planner"`` or ``tool="write_code"``
- ``stage``: a machine-readable lifecycle marker
- ``message``: a concise human-readable status string

BaseAgent also installs a default callback handler from this module that writes
these structured events to Python logging at ``INFO`` level. That keeps the
event stream visible in logs even when no interactive renderer is attached.

Any extra keyword arguments are merged into the payload unchanged. Tools reuse
the same custom event channel as agents so downstream consumers only need one
subscription.

Examples
--------
One-off agent progress update:

>>> events = AgentEvents(agent="planner", config=config)
>>> events.emit("Drafting plan", stage="generate")

Scoped agent range with automatic start/end/error events:

>>> with events.range("generate", "Drafting plan", done="Plan ready") as span:
...     plan = build_plan()
...     span.update(step_count=len(plan.steps))

Tool usage from a ``ToolRuntime``:

>>> events = ToolEvents.from_runtime("write_code", runtime)
>>> with events.range("write", "Writing file", done="File written"):
...     path.write_text(code)
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from time import monotonic_ns, perf_counter
from typing import Any, Mapping, Self
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
    dispatch_custom_event,
)
from langchain_core.runnables import RunnableConfig

DEFAULT_EVENT_NAME = "ursa_agent_progress"
LOGGER = logging.getLogger(__name__)


class EventConsoleFormatter(logging.Formatter):
    """Render URSA progress events as compact console messages."""

    DETAIL_KEYS = (
        "path",
        "filename",
        "query",
        "output_path",
        "returncode",
        "stdout_chars",
        "stderr_chars",
        "result_chars",
        "error",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "ursa_event_payload", None)
        if not isinstance(payload, dict):
            return super().format(record)

        source = (
            payload.get("agent")
            or payload.get("tool")
            or payload.get("name")
            or "ursa"
        )
        stage = payload.get("stage")
        phase = payload.get("phase")
        message = payload.get("message") or record.getMessage()

        label = str(source)
        if stage:
            label += f" {stage}"
        if phase:
            label += f"/{phase}"

        details = [
            f"{key}={payload[key]}"
            for key in self.DETAIL_KEYS
            if payload.get(key) not in (None, "")
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        return f"[ursa] {label}: {message}{suffix}"


def configure_event_logging(
    *,
    level: int = logging.INFO,
    formatter: logging.Formatter | None = None,
) -> None:
    """Enable console logging for URSA progress events.

    Examples call this helper so the default event logging callback installed
    by ``BaseAgent`` is visible without requiring users to configure Python
    logging manually.
    """
    formatter = formatter or EventConsoleFormatter()
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


class EventLoggingHandler(BaseCallbackHandler):
    """Write structured URSA progress events to the Python logger.

    The handler is intentionally narrow: it only logs URSA's structured custom
    progress events and ignores all other callback activity. The emitted log
    line keeps a compact summary for humans while preserving any remaining
    payload fields as JSON for debugging and ingestion.
    """

    def __init__(
        self,
        *,
        event_name: str = DEFAULT_EVENT_NAME,
        logger: logging.Logger | None = None,
    ) -> None:
        self.event_name = event_name
        self.logger = logger or LOGGER

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log matching progress events at ``INFO`` level."""
        if name != self.event_name or not isinstance(data, dict):
            return
        if not self.logger.isEnabledFor(logging.INFO):
            return
        self.logger.info(
            self._format_message(name, data),
            extra={
                "ursa_event_name": name,
                "ursa_event_payload": data,
            },
        )

    def _format_message(self, name: str, data: dict[str, Any]) -> str:
        parts = [f"event={json.dumps(name)}"]
        for key in ("agent", "tool", "name", "stage", "phase", "message"):
            value = data.get(key)
            if value in (None, ""):
                continue
            parts.append(f"{key}={json.dumps(str(value), ensure_ascii=False)}")

        extras = {
            key: value
            for key, value in data.items()
            if key not in {"agent", "tool", "name", "stage", "phase", "message"}
        }
        if extras:
            parts.append(
                "data="
                + json.dumps(
                    extras,
                    default=str,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return " ".join(parts)


DEFAULT_EVENT_LOGGING_HANDLER = EventLoggingHandler()


@dataclass(slots=True)
class ProgressEvents:
    """Emit standardized URSA progress events for a single source.

    Parameters
    ----------
    name:
        Stable short identifier included in every payload.
    config:
        Runnable config propagated by LangGraph/LangChain. When ``None``, all
        methods become safe no-ops.
    event_name:
        Underlying LangChain custom event name. The default preserves the
        contract already used by the deployed branch work.
    name_key:
        Payload key used for the source identifier, for example ``"agent"`` or
        ``"tool"``.
    default_payload:
        Extra fields merged into every emitted payload.
    """

    name: str
    config: RunnableConfig | None = None
    event_name: str = DEFAULT_EVENT_NAME
    name_key: str = "name"
    default_payload: dict[str, Any] = field(default_factory=dict)

    def emit(
        self,
        message: str,
        *,
        stage: str,
        **payload: Any,
    ) -> dict[str, Any] | None:
        """Emit a single synchronous progress event."""
        if self.config is None:
            return None
        body = self._payload(message=message, stage=stage, **payload)
        dispatch_custom_event(self.event_name, body, config=self.config)
        return body

    async def aemit(
        self,
        message: str,
        *,
        stage: str,
        **payload: Any,
    ) -> dict[str, Any] | None:
        """Emit a single asynchronous progress event."""
        if self.config is None:
            return None
        body = self._payload(message=message, stage=stage, **payload)
        await adispatch_custom_event(
            self.event_name,
            body,
            config=self.config,
        )
        return body

    def range(
        self,
        stage: str,
        start: str,
        *,
        done: str | None = None,
        error: str | None = None,
        **payload: Any,
    ) -> EventRange:
        """Create a scoped range that emits start/end/error events."""
        return EventRange(
            events=self,
            stage=stage,
            start_message=start,
            done_message=done,
            error_message=error,
            payload=dict(payload),
        )

    def _payload(
        self,
        *,
        message: str,
        stage: str,
        **payload: Any,
    ) -> dict[str, Any]:
        return {
            self.name_key: self.name,
            "stage": stage,
            "message": message,
            "monotonic_timestamp_ns": monotonic_ns(),
            **self.default_payload,
            **payload,
        }


class AgentEvents(ProgressEvents):
    """Emit standardized URSA progress events for a single agent."""

    def __init__(
        self,
        agent: str,
        config: RunnableConfig | None = None,
        event_name: str = DEFAULT_EVENT_NAME,
    ) -> None:
        super().__init__(
            name=agent,
            config=config,
            event_name=event_name,
            name_key="agent",
        )


class EnvironmentEvents(ProgressEvents):
    """Emit standardized URSA progress events for an environment.

    Environment events use the same custom LangChain event channel as agent and
    tool events so a single callback recorder can capture complete nested runs.
    Environments may also be invoked directly rather than from inside a
    LangChain runnable. In that case LangChain refuses ad-hoc custom events
    because there is no parent run ID, so this helper falls back to directly
    notifying callbacks from the runnable config.
    """

    def __init__(
        self,
        environment: str,
        config: RunnableConfig | None = None,
        event_name: str = DEFAULT_EVENT_NAME,
        *,
        environment_type: str | None = None,
        environment_id: str | None = None,
        path: list[str] | None = None,
    ) -> None:
        default_payload: dict[str, Any] = {}
        if environment_type is not None:
            default_payload["environment_type"] = environment_type
        if environment_id is not None:
            default_payload["environment_id"] = environment_id
        if path is not None:
            default_payload["path"] = path
        super().__init__(
            name=environment,
            config=config,
            event_name=event_name,
            name_key="environment",
            default_payload=default_payload,
        )

    def emit(
        self,
        message: str,
        *,
        stage: str,
        **payload: Any,
    ) -> dict[str, Any] | None:
        if self.config is None:
            return None
        body = self._payload(message=message, stage=stage, **payload)
        try:
            dispatch_custom_event(self.event_name, body, config=self.config)
        except RuntimeError as exc:
            if not self._is_missing_parent_run_error(exc):
                raise
            self._direct_emit(body)
        return body

    async def aemit(
        self,
        message: str,
        *,
        stage: str,
        **payload: Any,
    ) -> dict[str, Any] | None:
        if self.config is None:
            return None
        body = self._payload(message=message, stage=stage, **payload)
        try:
            await adispatch_custom_event(
                self.event_name,
                body,
                config=self.config,
            )
        except RuntimeError as exc:
            if not self._is_missing_parent_run_error(exc):
                raise
            await self._adirect_emit(body)
        return body

    @staticmethod
    def _is_missing_parent_run_error(exc: RuntimeError) -> bool:
        return (
            "Unable to dispatch an adhoc event without a parent run id"
            in str(exc)
        )

    def _callbacks(self) -> list[Any]:
        if self.config is None:
            return []
        callbacks = self.config.get("callbacks", [])
        if callbacks is None:
            return []
        if isinstance(callbacks, list):
            return callbacks
        return [callbacks]

    def _direct_emit(self, body: dict[str, Any]) -> None:
        run_id = uuid4()
        for callback in self._callbacks():
            handler = getattr(callback, "on_custom_event", None)
            if handler is None:
                continue
            handler(self.event_name, body, run_id=run_id)

    async def _adirect_emit(self, body: dict[str, Any]) -> None:
        run_id = uuid4()
        for callback in self._callbacks():
            handler = getattr(callback, "on_custom_event", None)
            if handler is None:
                continue
            result = handler(self.event_name, body, run_id=run_id)
            if inspect.isawaitable(result):
                await result


class ToolEvents(ProgressEvents):
    """Emit standardized URSA progress events for a single tool."""

    def __init__(
        self,
        tool: str,
        config: RunnableConfig | None = None,
        event_name: str = DEFAULT_EVENT_NAME,
        *,
        tool_call_id: str | None = None,
        owner_payload: Mapping[str, Any] | None = None,
    ) -> None:
        default_payload: dict[str, Any] = {}
        if owner_payload:
            default_payload.update(owner_payload)
        if tool_call_id is not None:
            default_payload["tool_call_id"] = tool_call_id
        super().__init__(
            name=tool,
            config=config,
            event_name=event_name,
            name_key="tool",
            default_payload=default_payload,
        )

    @staticmethod
    def _owner_payload_from_runtime(
        runtime: ToolRuntime[Any],
    ) -> dict[str, Any]:
        """Return agent/member identity fields recorded on a tool runtime."""
        config = runtime.config if isinstance(runtime.config, Mapping) else {}
        metadata = config.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        context = getattr(runtime, "context", None)

        environment_id = metadata.get("environment_id")
        member = metadata.get("environment_member")
        member_id = metadata.get("environment_member_id")
        member_path = metadata.get("environment_member_path")
        member_role = metadata.get("environment_member_role")

        agent = metadata.get("agent") or member
        agent_id = metadata.get("agent_id") or member_id
        if agent is None and context is not None:
            agent = getattr(context, "agent_name", None)
        if agent_id is None and environment_id and agent:
            agent_id = f"{environment_id}.{agent}"

        payload: dict[str, Any] = {}
        if agent is not None:
            payload["agent"] = str(agent)
        if agent_id is not None:
            payload["agent_id"] = str(agent_id)
        if member is not None:
            payload["environment_member"] = str(member)
        if member_id is not None:
            payload["environment_member_id"] = str(member_id)
        if member_role is not None:
            payload["environment_member_role"] = str(member_role)
        if environment_id is not None:
            payload["environment_id"] = str(environment_id)
        if member_path is not None:
            payload["environment_member_path"] = member_path
        return payload

    @classmethod
    def from_runtime(
        cls,
        tool: str,
        runtime: ToolRuntime[Any] | None,
        *,
        event_name: str = DEFAULT_EVENT_NAME,
    ) -> ToolEvents:
        """Create a tool event helper from a LangGraph ``ToolRuntime``."""
        if runtime is None:
            return cls(tool=tool, event_name=event_name)
        return cls(
            tool=tool,
            config=runtime.config,
            event_name=event_name,
            tool_call_id=runtime.tool_call_id,
            owner_payload=cls._owner_payload_from_runtime(runtime),
        )


@dataclass(slots=True)
class EventRange:
    """Context manager that marks the lifecycle of a unit of work.

    The range emits a ``phase="start"`` event on entry, followed by either
    ``phase="end"`` or ``phase="error"`` when the block exits.

    Notes
    -----
    - Updating ``payload`` inside the block only affects the terminal event.
    - ``done_message`` and ``error_message`` may be reassigned inside the block
      when the final outcome is only known after work completes.
    """

    events: ProgressEvents
    stage: str
    start_message: str
    done_message: str | None = None
    error_message: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    include_elapsed_ms: bool = True
    _started_at: float | None = field(default=None, init=False, repr=False)

    def update(self, **payload: Any) -> EventRange:
        """Merge additional payload fields into the terminal event."""
        self.payload.update(payload)
        return self

    def __enter__(self) -> Self:
        self._started_at = perf_counter()
        self.events.emit(
            self.start_message,
            stage=self.stage,
            phase="start",
            **self.payload,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._emit_terminal(exc)
        return False

    async def __aenter__(self) -> Self:
        self._started_at = perf_counter()
        await self.events.aemit(
            self.start_message,
            stage=self.stage,
            phase="start",
            **self.payload,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self._aemit_terminal(exc)
        return False

    def _terminal_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        if self.include_elapsed_ms and self._started_at is not None:
            payload["elapsed_ms"] = (perf_counter() - self._started_at) * 1000
        return payload

    def _emit_terminal(self, exc: BaseException | None) -> None:
        payload = self._terminal_payload()
        if exc is None:
            self.events.emit(
                self.done_message or self.start_message,
                stage=self.stage,
                phase="end",
                **payload,
            )
            return
        self.events.emit(
            self.error_message or self.start_message,
            stage=self.stage,
            phase="error",
            error_type=type(exc).__name__,
            error=str(exc),
            **payload,
        )

    async def _aemit_terminal(self, exc: BaseException | None) -> None:
        payload = self._terminal_payload()
        if exc is None:
            await self.events.aemit(
                self.done_message or self.start_message,
                stage=self.stage,
                phase="end",
                **payload,
            )
            return
        await self.events.aemit(
            self.error_message or self.start_message,
            stage=self.stage,
            phase="error",
            error_type=type(exc).__name__,
            error=str(exc),
            **payload,
        )


__all__ = [
    "DEFAULT_EVENT_LOGGING_HANDLER",
    "DEFAULT_EVENT_NAME",
    "AgentEvents",
    "EventConsoleFormatter",
    "EnvironmentEvents",
    "EventLoggingHandler",
    "EventRange",
    "ProgressEvents",
    "ToolEvents",
    "configure_event_logging",
]
