"""Instance-bound tools for managing an agent hierarchy."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.tools import BaseTool, ToolException
from langchain_core.tools import tool

from ursa.messaging.coordination import (
    SubagentLimitError,
    wait_for_completion,
    wait_targets,
    wait_timeout,
)

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
            "unknown message target group",
            "message target group",
            "agent path cannot be empty",
            "root agent has no parent",
            "path escapes hierarchy root",
            "cannot synchronously message",
            "cannot wait for itself",
            "cannot wait for an ancestor",
            "target must be a non-empty agent path or list of paths",
            "timeout_seconds must be",
            "can only stop one of its descendants",
            "is stopping",
            "already stopping",
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

    async def post_local_agent(recipient: BaseAgent, message: str) -> str:
        current = agent.current_message
        config = dict(current.config) if current is not None else None

        await _notify_callbacks(
            config,
            "on_hierarchy_message",
            agent.agent_path,
            (recipient.agent_path,),
            message,
        )

        async def on_start(_message: Any) -> None:
            await _notify_callbacks(
                config,
                "on_agent_message_start",
                recipient.agent_path,
                message_id=str(_message.message_id),
            )

        async def on_end(_message: Any, succeeded: bool) -> None:
            await _notify_callbacks(
                config,
                "on_agent_message_end",
                recipient.agent_path,
                succeeded=succeeded,
                message_id=str(_message.message_id),
            )

        await recipient.post_message(
            message,
            sender=agent.agent_path,
            config=config,
            on_start=on_start,
            on_end=on_end,
        )
        return f"Message queued for {recipient.agent_path}."

    def resolve_local_recipient(target: str) -> BaseAgent:
        """Validate one local target before scheduling any work for it."""
        try:
            recipient = agent.resolve_agent(target)
        except (KeyError, TypeError) as exc:
            raise ToolException(str(exc)) from exc
        try:
            recipient._ensure_hierarchy_accepting_messages()
        except RuntimeError as exc:
            raise ToolException(str(exc)) from exc
        return recipient

    @tool
    async def spawn_agent(
        name: str,
        message: str,
        agent_type: str | None = None,
    ) -> str:
        """Spawn a child and queue its initial task without waiting for a reply.

        The name is a single path component. It is joined to the launching
        agent's path, so a child named ``reviewer`` launched by ``root`` is
        addressed as ``root/reviewer``. Use ``wait_for`` when synchronization
        is needed. In a managed runtime, ``agent_type`` optionally selects the
        configured URSA agent implementation; omit it to inherit the caller's
        type.

        When creating independent teammates, emit all their spawn_agent calls
        together in one response, then wait. Include the child's identity,
        useful peer paths, and explicit send_message reply instructions in
        its initial task when coordination is needed.
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
                result = await post_local_agent(child, message)
        except SubagentLimitError as exc:
            raise ToolException(str(exc)) from exc
        except RuntimeError as exc:
            # Runtime policy failures cross the process boundary as a generic
            # runtime error. They remain recoverable tool failures so the
            # invoking model can revise its delegation plan.
            if _routing_tool_error(exc) is None and not any(
                term in str(exc).casefold()
                for term in ("subagent", "agent_type", "did not become ready")
            ):
                raise
            raise ToolException(str(exc)) from exc
        return f"Spawned agent {path}\n\n{_stringify_result(result)}"

    @tool
    async def send_message(target: str | list[str], message: str) -> str:
        """Queue a one-way message to one agent or a list of agents.

        Each target may be a relative path, a sibling name, or a root-qualified
        path such as ``root/reviewer``. Message groups are ``@everyone`` for
        every other hierarchy member, ``@supervisor`` for the agent that
        directly launched you, ``@peers`` for agents sharing your supervisor,
        and ``@reports`` for agents you directly launched.
        Empty groups are errors. URSA returns after queueing the message; it
        does not wait for the receiving agent's response. For a target list,
        every address is validated before anything is delivered, and delivery
        acknowledgements retain input order.

        Coordinate directly with peers as well as supervisors. Use a target list
        for the same message to a group (exclude yourself), or emit multiple
        send_message calls together for independent, different messages.
        Ask recipients to call send_message with an explicit reply target;
        their ordinary final responses are not routed back automatically.
        """
        if not isinstance(target, str) and (
            not isinstance(target, list) or not target
        ):
            raise ToolException(
                "target must be a non-empty path or list of paths"
            )
        routes = [target] if isinstance(target, str) else target
        if not all(isinstance(item, str) and item.strip() for item in routes):
            raise ToolException("each target must be a non-empty agent path")

        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            try:
                if isinstance(target, str):
                    return _stringify_result(
                        await runtime.send_message(target, message)
                    )
                results = await runtime.send_messages(target, message)
            except RuntimeError as exc:
                if tool_error := _routing_tool_error(exc):
                    raise tool_error from exc
                raise
            # The host expands selectors transactionally and returns one
            # acknowledgement per concrete recipient. Do not zip that result
            # with the caller's raw selector list: one selector can name many
            # agents, and acknowledgements already include the concrete path.
            acknowledgements = []
            for result in results:
                if isinstance(result, Mapping):
                    if "error" in result:
                        acknowledgements.append(f"Error: {result['error']}")
                        continue
                    result = result.get("result")
                acknowledgements.append(_stringify_result(result))
            return "\n\n".join(acknowledgements)
        # Validate the complete broadcast before scheduling any local delivery.
        try:
            recipients = agent.resolve_message_targets(target)
            for recipient in recipients:
                recipient._ensure_hierarchy_accepting_messages()
            results = await asyncio.gather(
                *(
                    post_local_agent(recipient, message)
                    for recipient in recipients
                )
            )
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ToolException(str(exc)) from exc
        if isinstance(target, str) and len(results) == 1:
            return results[0]
        return "\n\n".join(
            f"{recipient.agent_path}: {result}"
            for recipient, result in zip(recipients, results, strict=True)
        )

    @tool
    async def wait_for(
        target: str | list[str],
        timeout_seconds: float = 120,
        wait_all: bool = False,
    ) -> str:
        """Wait for one or more agents, for at most 120 seconds.

        ``target`` accepts one path or a non-empty list using the same relative
        and root-qualified paths as ``send_message``. A list returns when any
        target is ready by default; set ``wait_all=True`` to wait until every
        target is ready. Prefer one list-target wait to several separate waits:
        it represents one synchronization point and avoids redundant tool
        calls. All list addresses are validated before waiting begins.

        A target is considered ready when that wait would close a dependency
        cycle, preventing reciprocal waits from deadlocking without treating
        unrelated waiting agents as finished. If this agent receives a new
        message while waiting, the wait ends immediately so the model can
        handle it. Completion alone is not a reply; returned messages name
        their sender. A managed wait may return several queued messages at
        once, each with its own sender, recipient, and exact content.
        """
        try:
            timeout = wait_timeout(timeout_seconds)
            targets = wait_targets(target)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            try:
                return _stringify_result(
                    await runtime.wait_for_agent(
                        target, timeout, wait_all=wait_all
                    )
                )
            except RuntimeError as exc:
                if tool_error := _routing_tool_error(exc):
                    raise tool_error from exc
                raise

        # Resolve every entry before registering the wait. That keeps a batch
        # invalid if any route is invalid, without partially changing state.
        recipients = [resolve_local_recipient(item) for item in targets]
        if any(recipient is agent for recipient in recipients):
            raise ToolException("An agent cannot wait for itself")
        target_paths = [recipient.agent_path for recipient in recipients]

        waiter = agent._begin_message_wait(recipients)
        try:
            return await wait_for_completion(
                waiter,
                target_paths,
                lambda: _completed_local_wait_targets(recipients, agent),
                timeout,
                wait_all,
            )
        finally:
            agent._end_message_wait(waiter)

    @tool
    async def stop_agent(target: str) -> str:
        """Stop a descendant agent and every agent below it.

        ``target`` accepts the same paths as ``send_message``, but only a
        child, grandchild, or deeper descendant of the calling agent may be
        stopped. The caller itself, ancestors, and siblings are not allowed.
        """
        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            try:
                return _stringify_result(await runtime.stop_subagent(target))
            except RuntimeError as exc:
                if tool_error := _routing_tool_error(exc):
                    raise tool_error from exc
                raise

        try:
            recipient = agent.resolve_agent(target)
        except (KeyError, TypeError) as exc:
            raise ToolException(str(exc)) from exc
        caller_path = agent.agent_path
        parent = recipient.parent_agent
        if parent is None or not recipient.agent_path.startswith(
            f"{caller_path}/"
        ):
            raise ToolException("An agent can only stop one of its descendants")
        stopped_path = recipient.agent_path
        parent.stop_subagent(recipient.hierarchy_name)
        return f"Stopped agent {stopped_path} and its subagents."

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

    return [spawn_agent, send_message, wait_for, stop_agent, list_agents]


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return repr(result)


def _completed_local_wait_targets(
    recipients: Sequence[BaseAgent], waiting_agent: BaseAgent
) -> list[tuple[str, str | None]]:
    """Return idle targets or ones that would close a wait dependency cycle."""
    completed = []
    for recipient in recipients:
        if not recipient.is_working:
            completed.append((recipient.agent_path, None))
        elif relation := recipient._wait_dependency_relation(waiting_agent):
            completed.append((recipient.agent_path, relation))
    return completed
