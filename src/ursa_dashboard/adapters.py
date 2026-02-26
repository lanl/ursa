from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .events import Event


class EventSink(Protocol):
    def emit(self, event: Event) -> None:
        pass


@dataclass(frozen=True)
class RunContext:
    run_id: str
    agent_id: str
    workspace_dir: Path


class AgentAdapter(Protocol):
    """Uniform adapter interface used by the dashboard runner."""

    def invoke(self, *, ctx: RunContext, inputs: Any, sink: EventSink) -> str:
        """Run the agent to completion; emit standardized events; return final string."""


class BaseAgentInProcessAdapter:
    """Adapter for URSA BaseAgent-derived classes executed in-process.

    IMPORTANT: Runs in the dashboard are executed inside a *separate worker subprocess*.
    The dashboard runner already captures that subprocess's stdout/stderr and persists
    them to the run event log.

    Therefore, this adapter intentionally does **not** redirect/capture stdout/stderr
    in-process (doing so would hide logs from the runner).

    Exceptions are allowed to propagate so the worker can mark the run as failed.
    """

    def __init__(
        self,
        agent_factory: Callable[[Path, Any], Any],
        *,
        supports_streaming: bool,
    ):
        self._agent_factory = agent_factory
        self._supports_streaming = supports_streaming
        self._setup_hook: Callable[[Any, RunContext, Any], Any] | None = None

    def set_setup_hook(
        self, hook: Callable[[Any, RunContext, Any], Any] | None
    ) -> None:
        """Set an optional hook invoked after agent construction and before invoke().

        The hook may be sync or async.
        """

        self._setup_hook = hook

    def invoke(self, *, ctx: RunContext, inputs: Any, sink: EventSink) -> str:  # noqa: ARG002
        agent = self._agent_factory(ctx.workspace_dir, inputs)
        if self._setup_hook is not None:
            r = self._setup_hook(agent, ctx, inputs)
            if asyncio.iscoroutine(r):
                asyncio.run(r)
        result = agent.invoke(inputs)

        # Prefer BaseAgent.format_result if available; else just string-ify.
        if hasattr(agent, "format_result"):
            try:
                return str(agent.format_result(result))
            except Exception:
                return str(result)
        return str(result)


class DirectInvokeAdapter:
    """Adapter that invokes an agent in-process *without* redirecting stdout/stderr.

    This is useful for demo/smoke-test agents where we want the worker subprocess
    stdout/stderr to be streamed directly by the dashboard runner.

    The adapter ignores the EventSink.
    """

    def __init__(self, agent_factory: Callable[[Path, Any], Any]):
        self._agent_factory = agent_factory
        self._setup_hook: Callable[[Any, RunContext, Any], Any] | None = None

    def set_setup_hook(
        self, hook: Callable[[Any, RunContext, Any], Any] | None
    ) -> None:
        """Set an optional hook invoked after agent construction and before invoke().

        The hook may be sync or async.
        """

        self._setup_hook = hook

    def invoke(self, *, ctx: RunContext, inputs: Any, sink: EventSink) -> str:  # noqa: ARG002
        agent = self._agent_factory(ctx.workspace_dir, inputs)
        if self._setup_hook is not None:
            r = self._setup_hook(agent, ctx, inputs)
            if asyncio.iscoroutine(r):
                asyncio.run(r)
        result = agent.invoke(inputs)
        if hasattr(agent, "format_result"):
            try:
                return str(agent.format_result(result))
            except Exception:
                return str(result)
        return str(result)
