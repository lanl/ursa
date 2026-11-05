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

    @tool
    def spawn_agent(name: str) -> str:
        """Spawn a child agent and return its full hierarchy path.

        The name is a single path component. It is joined to the launching
        agent's path, so a child named ``reviewer`` launched by ``root`` is
        addressed as ``root/reviewer``.
        """
        try:
            child = agent.spawn_subagent(name)
        except SubagentLimitError as exc:
            raise ToolException(str(exc)) from exc
        return f"Spawned agent {child.agent_path}"

    @tool
    async def send_message(target: str, message: str) -> str:
        """Send a message to another agent and wait for its response.

        Target may be a relative path, a sibling name, or a root-qualified path
        such as ``root/reviewer``. Independent tool calls to different agents
        may run concurrently.
        """
        recipient = agent.resolve_agent(target)
        if recipient is agent:
            raise ValueError("An agent cannot synchronously message itself")

        # Waiting on an ancestor that is currently waiting for this agent would
        # deadlock. Child and sibling requests are safe and cover delegation.
        ancestor = agent.parent_agent
        while ancestor is not None:
            if recipient is ancestor:
                raise ValueError(
                    "Cannot synchronously message an ancestor; send work to "
                    "a child or sibling agent instead"
                )
            ancestor = ancestor.parent_agent

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
        finally:
            if started:
                await _notify_callbacks(
                    config,
                    "on_agent_message_end",
                    recipient.agent_path,
                    succeeded=succeeded,
                    message_id=message_id,
                )
        if isinstance(result, Mapping):
            try:
                return recipient.format_result(result)
            except (
                KeyError,
                TypeError,
                ValueError,
                AttributeError,
                NotImplementedError,
            ):
                pass
        return _stringify_result(result)

    @tool
    def list_agents() -> str:
        """List all agents in the hierarchy and mark the calling agent."""
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
