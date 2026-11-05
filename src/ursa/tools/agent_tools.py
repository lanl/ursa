"""Instance-bound tools for managing an agent hierarchy."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.tools import BaseTool, ToolException
from langchain_core.tools import tool

from ursa.agents.base import SubagentLimitError

if TYPE_CHECKING:
    from ursa.agents.base import BaseAgent

logger = logging.getLogger(__name__)


def _routing_tool_error(error: BaseException) -> ToolException | None:
    """Translate invalid hierarchy addresses into recoverable tool failures."""
    message = str(error)
    if isinstance(error, (KeyError, TypeError)) or any(
        phrase in message.casefold()
        for phrase in (
            "unknown agent path",
            "agent path cannot be empty",
            "root agent has no parent",
            "path escapes hierarchy root",
            "cannot synchronously message",
        )
    ):
        return ToolException(message)
    return None


async def _notify_callbacks(
    config: Mapping[str, Any] | None,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Call optional hierarchy lifecycle hooks on configured callbacks."""
    configured = config.get("callbacks") if config is not None else None
    if isinstance(configured, Sequence) and not isinstance(
        configured, (str, bytes)
    ):
        callbacks = configured
    else:
        callbacks = getattr(configured, "handlers", ())
    for callback in callbacks:
        method = getattr(callback, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Agent hierarchy callback %s failed", method_name)


def build_agent_tools(agent: BaseAgent) -> list[BaseTool]:
    """Build hierarchy tools bound to ``agent`` as their caller."""

    async def request_local_agent(recipient: BaseAgent, message: str) -> Any:
        current = agent.current_message
        config = dict(current.config) if current is not None else None
        started = False
        message_id: str | None = None

        async def on_start(_message: Any) -> None:
            nonlocal message_id, started
            started = True
            message_id = str(_message.message_id)
            await _notify_callbacks(
                config,
                "on_agent_message_start",
                recipient.agent_path,
                message_id=message_id,
            )

        succeeded = False
        try:
            result = await agent.ask_agent(
                recipient,
                message,
                config=config,
                on_start=on_start,
            )
            succeeded = True
            return result
        finally:
            if started:
                await _notify_callbacks(
                    config,
                    "on_agent_message_end",
                    recipient.agent_path,
                    succeeded=succeeded,
                    message_id=message_id,
                )

    @tool
    async def spawn_agent(
        name: str,
        message: str,
        agent_type: str | None = None,
    ) -> str:
        """Spawn a child, send its initial task immediately, and return its reply.

        The name is a single path component. It is joined to the launching
        agent's path, so a child named ``reviewer`` launched by ``root`` is
        addressed as ``root/reviewer``. The call waits for the child's reply.
        In a managed runtime, ``agent_type`` optionally selects the configured
        URSA agent implementation; omit it to inherit the caller's type.
        """
        try:
            runtime = getattr(agent, "_process_runtime", None)
            if runtime is not None:
                path, result = await runtime.spawn_subagent(
                    name, message, agent_type
                )
            else:
                if agent_type is not None:
                    raise ToolException(
                        "agent_type selection requires a managed AgentRuntime"
                    )
                child = agent.spawn_subagent(name)
                path = child.agent_path
                result = await request_local_agent(child, message)
        except SubagentLimitError as exc:
            raise ToolException(str(exc)) from exc
        except RuntimeError as exc:
            # Runtime policy failures cross the process boundary as a generic
            # runtime error. They remain recoverable tool failures so the
            # invoking model can revise its delegation plan.
            if not any(
                term in str(exc).casefold()
                for term in ("subagent", "agent_type")
            ):
                raise
            raise ToolException(str(exc)) from exc
        return f"Spawned agent {path}\n\n{_format_agent_result(child if runtime is None else None, result)}"

    @tool
    async def send_message(target: str, message: str) -> str:
        """Send a message to another agent and wait for its response.

        Target may be a relative path, a sibling name, or a root-qualified path
        such as ``root/reviewer``. Independent tool calls to different agents
        may run concurrently.
        """
        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            try:
                return _stringify_result(
                    await runtime.send_message(target, message)
                )
            except RuntimeError as exc:
                if tool_error := _routing_tool_error(exc):
                    raise tool_error from exc
                raise

        try:
            recipient = agent.resolve_agent(target)
        except (KeyError, TypeError) as exc:
            raise ToolException(str(exc)) from exc
        if recipient is agent:
            raise ToolException("An agent cannot synchronously message itself")

        # Waiting on an ancestor that is currently waiting for this agent would
        # deadlock. Child and sibling requests are safe and cover delegation.
        ancestor = agent.parent_agent
        while ancestor is not None:
            if recipient is ancestor:
                raise ToolException(
                    "Cannot synchronously message an ancestor; send work to "
                    "a child or sibling agent instead"
                )
            ancestor = ancestor.parent_agent

        result = await request_local_agent(recipient, message)
        return _format_agent_result(recipient, result)

    @tool
    async def list_agents() -> str:
        """List all agents in the hierarchy and mark the calling agent."""
        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            paths = await runtime.list_agents()
            return "\n".join(
                f"{path}{' (self)' if path == agent.agent_path else ''}"
                for path in paths
            )
        entries = []
        for member in agent.root_agent.walk_agents():
            suffix = " (self)" if member is agent else ""
            entries.append(f"{member.agent_path}{suffix}")
        return "\n".join(entries)

    return [spawn_agent, send_message, list_agents]


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return repr(result)


def _format_agent_result(agent: BaseAgent | None, result: Any) -> str:
    if agent is not None and isinstance(result, Mapping):
        try:
            return agent.format_result(result)
        except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            NotImplementedError,
        ):
            pass
    return _stringify_result(result)
