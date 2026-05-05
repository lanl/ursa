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

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Self

from langchain.tools import ToolRuntime
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
    dispatch_custom_event,
)
from langchain_core.runnables import RunnableConfig

DEFAULT_EVENT_NAME = "ursa_agent_progress"
_MISSING_PARENT_RUN_ERROR = (
    "Unable to dispatch an adhoc event without a parent run id."
)


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
        try:
            dispatch_custom_event(self.event_name, body, config=self.config)
        except RuntimeError as exc:
            if _MISSING_PARENT_RUN_ERROR not in str(exc):
                raise
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
        try:
            await adispatch_custom_event(
                self.event_name,
                body,
                config=self.config,
            )
        except RuntimeError as exc:
            if _MISSING_PARENT_RUN_ERROR not in str(exc):
                raise
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


class ToolEvents(ProgressEvents):
    """Emit standardized URSA progress events for a single tool."""

    def __init__(
        self,
        tool: str,
        config: RunnableConfig | None = None,
        event_name: str = DEFAULT_EVENT_NAME,
        *,
        tool_call_id: str | None = None,
    ) -> None:
        default_payload: dict[str, Any] = {}
        if tool_call_id is not None:
            default_payload["tool_call_id"] = tool_call_id
        super().__init__(
            name=tool,
            config=config,
            event_name=event_name,
            name_key="tool",
            default_payload=default_payload,
        )

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
            payload["elapsed_ms"] = round(
                (perf_counter() - self._started_at) * 1000,
                3,
            )
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
    "DEFAULT_EVENT_NAME",
    "AgentEvents",
    "EventRange",
    "ProgressEvents",
    "ToolEvents",
]
