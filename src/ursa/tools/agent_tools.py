"""Instance-bound tools for managing an agent hierarchy."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.tools import BaseTool, ToolException
from langchain_core.tools import tool

from ursa.agents.base import SubagentLimitError, agent_message_wait_result

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

    async def send_one_message(target: str, message: str) -> str:
        """Queue one message and return its delivery acknowledgement."""
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

        recipient = resolve_local_recipient(target)
        return await post_local_agent(recipient, message)

    @tool
    async def send_message(target: str | list[str], message: str) -> str:
        """Queue a one-way message to one agent or a list of agents.

        Each target may be a relative path, a sibling name, or a root-qualified
        path such as ``root/reviewer``. URSA returns after queueing the message;
        it does not wait for the receiving agent's response. For a target list,
        every address is validated before anything is delivered, and delivery
        acknowledgements retain input order.

        Coordinate directly with peers as well as parents. Use a target list
        for the same message to a group (exclude yourself), or emit multiple
        send_message calls together for independent, different messages.
        Ask recipients to call send_message with an explicit reply target;
        their ordinary final responses are not routed back automatically.
        """
        if isinstance(target, str):
            return await send_one_message(target, message)
        if not isinstance(target, list) or not target:
            raise ToolException(
                "target must be a non-empty path or list of paths"
            )
        if not all(isinstance(item, str) and item.strip() for item in target):
            raise ToolException("each target must be a non-empty agent path")

        runtime = getattr(agent, "_process_runtime", None)
        if runtime is not None:
            try:
                results = await runtime.send_messages(target, message)
            except RuntimeError as exc:
                if tool_error := _routing_tool_error(exc):
                    raise tool_error from exc
                raise
        else:
            # Complete preflight before calling any inbox. This is deliberately
            # separate from post_local_agent so an invalid sibling cannot
            # race with an otherwise valid recipient.
            recipients = [resolve_local_recipient(item) for item in target]
            results = await asyncio.gather(
                *(
                    post_local_agent(recipient, message)
                    for recipient in recipients
                )
            )
        formatted = []
        for item, result in zip(target, results, strict=True):
            if isinstance(result, Mapping):
                if "error" in result:
                    formatted.append(f"{item}: Error: {result['error']}")
                    continue
                result = result.get("result")
            if isinstance(result, Exception):
                formatted.append(
                    f"{item}: Error: {type(result).__name__}: {result}"
                )
            else:
                formatted.append(f"{item}: {result}")
        return "\n\n".join(formatted)

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
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ToolException("timeout_seconds must be a number") from exc
        if not 0 <= timeout <= 120:
            raise ToolException("timeout_seconds must be between 0 and 120")

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

        if isinstance(target, str):
            targets = [target]
        elif (
            isinstance(target, list)
            and target
            and all(isinstance(item, str) and item.strip() for item in target)
        ):
            targets = list(target)
        else:
            raise ToolException(
                "target must be a non-empty agent path or list of paths"
            )

        # Resolve every entry before registering the wait. That keeps a batch
        # invalid if any route is invalid, without partially changing state.
        recipients = [resolve_local_recipient(item) for item in targets]
        if any(recipient is agent for recipient in recipients):
            raise ToolException("An agent cannot wait for itself")
        target_paths = [recipient.agent_path for recipient in recipients]

        waiter = agent._begin_message_wait(recipients)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            completed = _completed_local_wait_targets(recipients, agent)
            if _wait_satisfied(completed, len(recipients), wait_all):
                return _format_local_wait_result(
                    completed, target_paths, wait_all, already_finished=True
                )
            while True:
                if waiter.event.is_set():
                    if waiter.message is not None:
                        return agent_message_wait_result(
                            waiter.message.sender,
                            waiter.message.recipient,
                            str(waiter.message.content),
                        )
                    return "Stopped waiting because a new message arrived."
                completed = _completed_local_wait_targets(recipients, agent)
                if _wait_satisfied(completed, len(recipients), wait_all):
                    return _format_local_wait_result(
                        completed, target_paths, wait_all
                    )
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return _local_wait_timeout_result(
                        target_paths, timeout, wait_all
                    )
                try:
                    await asyncio.wait_for(
                        waiter.event.wait(), timeout=min(remaining, 0.1)
                    )
                except TimeoutError:
                    pass
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
        if not recipient.agent_path.startswith(f"{caller_path}/"):
            raise ToolException("An agent can only stop one of its descendants")
        child_name = recipient.agent_path[len(caller_path) + 1 :].split("/", 1)[
            0
        ]
        stopped_path = recipient.agent_path
        agent.stop_subagent(child_name)
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
) -> list[tuple[BaseAgent, str | None]]:
    """Return idle targets or ones that would close a wait dependency cycle."""
    completed = []
    for recipient in recipients:
        if not recipient.is_working:
            completed.append((recipient, None))
        elif relation := recipient._wait_dependency_relation(waiting_agent):
            completed.append((recipient, relation))
    return completed


def _wait_satisfied(
    completed: Sequence[tuple[BaseAgent, str | None]],
    target_count: int,
    wait_all: bool,
) -> bool:
    return len(completed) == target_count if wait_all else bool(completed)


def _format_local_wait_result(
    completed: Sequence[tuple[BaseAgent, str | None]],
    targets: Sequence[str],
    wait_all: bool,
    *,
    already_finished: bool = False,
) -> str:
    """Describe list wait completion while preserving single-target wording."""
    recipient, relation = completed[0]
    if relation is not None:
        cycle_result = (
            f"Agent {recipient.agent_path} is ready for this wait because it "
            f"is waiting on you {relation}."
        )
    else:
        cycle_result = None
    if len(targets) == 1:
        if cycle_result is not None:
            return cycle_result
        verb = "has already finished" if already_finished else "finished"
        return f"Agent {recipient.agent_path} {verb} its work."
    if wait_all:
        if any(item_relation is not None for _, item_relation in completed):
            details = ", ".join(
                f"{item.agent_path} ({'finished' if item_relation is None else f'waiting on you {item_relation}'})"
                for item, item_relation in completed
            )
            return f"All agents are ready for this wait: {details}."
        verb = "have already finished" if already_finished else "finished"
        paths = [item.agent_path for item, _relation in completed]
        return f"All agents {verb} their work: {', '.join(paths)}."
    if cycle_result is not None:
        return cycle_result
    verb = "has already finished" if already_finished else "finished"
    return f"Agent {recipient.agent_path} {verb} its work."


def _local_wait_timeout_result(
    targets: Sequence[str], timeout: float, wait_all: bool
) -> str:
    if len(targets) == 1:
        return f"Agent {targets[0]} did not finish within {timeout:g} seconds."
    if wait_all:
        return (
            f"Not all agents finished within {timeout:g} seconds: "
            f"{', '.join(targets)}."
        )
    return (
        f"No agents finished within {timeout:g} seconds: {', '.join(targets)}."
    )
