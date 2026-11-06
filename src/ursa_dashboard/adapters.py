# ruff: noqa: TID251

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console

from ursa.cli.callbacks import AgentLogEventHandler

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

    async def ainvoke(
        self, *, ctx: RunContext, inputs: Any, sink: EventSink
    ) -> str:
        """Async run variant used by the dashboard worker."""


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

    def invoke(self, *, ctx: RunContext, inputs: Any, sink: EventSink) -> str:
        """Synchronous compatibility wrapper."""
        return asyncio.run(self.ainvoke(ctx=ctx, inputs=inputs, sink=sink))

    async def ainvoke(
        self, *, ctx: RunContext, inputs: Any, sink: EventSink
    ) -> str:
        del sink
        agent = self._agent_factory(ctx.workspace_dir, inputs)
        if not callable(getattr(agent, "ainvoke", None)):
            # Some dashboard targets are synchronous workflows.  Run setup and
            # invocation in a worker thread so they do not execute inside this
            # event loop; that preserves their existing sync invoke semantics.
            return await asyncio.to_thread(
                _invoke_sync_target_with_setup,
                agent,
                self._setup_hook,
                ctx,
                inputs,
                ctx.workspace_dir,
            )

        try:
            if self._setup_hook is not None:
                await _maybe_await(self._setup_hook(agent, ctx, inputs))
            result = await _ainvoke_with_cli_handler(
                agent,
                inputs,
                workspace_dir=ctx.workspace_dir,
            )
            return _format_agent_result(agent, result)
        finally:
            await _close_agent(agent)


class BaseAgentInProcessAdapter(DirectInvokeAdapter):
    """Adapter for URSA BaseAgent-derived classes executed in-process.

    IMPORTANT: Runs in the dashboard are executed inside a *separate worker subprocess*.
    The dashboard runner already captures that subprocess's stdout/stderr and persists
    them to the run event log.

    Therefore, this adapter intentionally does **not** redirect/capture stdout/stderr
    in-process (doing so would hide logs from the runner). Instead it attaches the
    shared CLI callback handler so structured progress is rendered directly into the
    worker's stdout stream.

    Exceptions are allowed to propagate so the worker can mark the run as failed.
    """

    def __init__(
        self,
        agent_factory: Callable[[Path, Any], Any],
        *,
        supports_streaming: bool,
    ):
        super().__init__(agent_factory)
        self._supports_streaming = supports_streaming


def _dashboard_console() -> Console:
    """Create the plain-text console used for dashboard worker log streaming."""
    return Console(
        file=sys.stdout,
        force_terminal=False,
        force_interactive=False,
        color_system="standard",
        width=120,
    )


def _callback_config(invoke: Callable, workspace_dir: Path) -> dict[str, Any]:
    """Attach CLI progress only to targets that accept runnable configuration."""
    try:
        params = inspect.signature(invoke).parameters.values()
    except (AttributeError, TypeError, ValueError):
        return {}
    if not any(
        param.name == "config" or param.kind is inspect.Parameter.VAR_KEYWORD
        for param in params
    ):
        return {}
    return {
        "config": {
            "callbacks": [
                AgentLogEventHandler(
                    console=_dashboard_console(), workspace=workspace_dir
                )
            ]
        }
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _format_agent_result(agent: Any, result: Any) -> str:
    if hasattr(agent, "format_result"):
        try:
            return str(agent.format_result(result))
        except (
            AttributeError,
            IndexError,
            KeyError,
            NotImplementedError,
            TypeError,
            ValueError,
        ):
            return str(result)
    return str(result)


async def _close_agent(agent: Any) -> None:
    """Close resources through the agent's native async/sync lifecycle."""
    async_close = getattr(agent, "aclose", None)
    if callable(async_close):
        await _maybe_await(async_close())
    close = getattr(agent, "close", None)
    if callable(close):
        await asyncio.to_thread(close)


def _invoke_sync_target_with_setup(
    agent: Any,
    setup_hook: Callable[[Any, RunContext, Any], Any] | None,
    ctx: RunContext,
    inputs: Any,
    workspace_dir: Path,
) -> str:
    if setup_hook is not None:
        result = setup_hook(agent, ctx, inputs)
        if inspect.isawaitable(result):
            asyncio.run(result)
    result = _invoke_with_cli_handler(
        agent,
        inputs,
        workspace_dir=workspace_dir,
    )
    return _format_agent_result(agent, result)


def _invoke_with_cli_handler(
    agent: Any,
    inputs: Any,
    *,
    workspace_dir: Path,
) -> Any:
    """Invoke a dashboard target, attaching the shared CLI callback when supported."""
    return agent.invoke(inputs, **_callback_config(agent.invoke, workspace_dir))


async def _ainvoke_with_cli_handler(
    agent: Any,
    inputs: Any,
    *,
    workspace_dir: Path,
) -> Any:
    """Async invoke a dashboard target with the shared CLI callback when supported."""
    return await _maybe_await(
        agent.ainvoke(inputs, **_callback_config(agent.ainvoke, workspace_dir))
    )
