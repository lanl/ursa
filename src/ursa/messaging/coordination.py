"""Coordination rules shared by local agents and the process host."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class SubagentLimitError(RuntimeError):
    """A hierarchy policy rejected creation of another subagent."""


def check_capacity(
    count: int, depth: int, max_agents: int | None, max_depth: int | None
) -> None:
    for name, value, limit in (
        ("max_agents", count, max_agents),
        ("max_depth", depth, max_depth),
    ):
        if limit == 0:
            raise SubagentLimitError(
                f"Subagents are disabled by subagents.{name}=0"
            )
        if limit is not None and value > limit:
            kind = "depth limit" if name == "max_depth" else "limit"
            raise SubagentLimitError(
                f"Subagent {kind} reached: subagents.{name}={limit}"
            )


def message_group(
    sender: str, target: str, paths: Collection[str]
) -> list[str]:
    """Expand a send-only selector, excluding its sender."""
    supervisor = sender.rpartition("/")[0]
    selector = target.casefold()
    if selector == "@everyone":
        candidates = paths
    elif selector == "@supervisor":
        candidates = [supervisor] if supervisor else []
    elif selector == "@peers":
        candidates = [
            path for path in paths if path.rpartition("/")[0] == supervisor
        ]
    elif selector == "@reports":
        candidates = [
            path for path in paths if path.rpartition("/")[0] == sender
        ]
    else:
        raise ValueError(f"Unknown message target group {target!r}")
    recipients = [path for path in candidates if path != sender]
    if not recipients:
        raise ValueError(f"Message target group {target!r} has no recipients")
    return recipients


def resolve_message_routes(
    sender: str,
    targets: Sequence[str],
    hierarchy: Collection[str],
    resolve: Callable[[str], tuple[T, str]],
) -> list[tuple[T, str]]:
    """Expand and validate every route, preserving its first input position."""
    resolved: dict[str, tuple[T, str]] = {}
    expands_groups = any(target.strip().startswith("@") for target in targets)
    for target in targets:
        target = target.strip()
        paths = (
            message_group(sender, target, hierarchy)
            if target.startswith("@")
            else [target]
        )
        for path in paths:
            recipient, path = resolve(path)
            if not (expands_groups and path == sender):
                resolved.setdefault(path, (recipient, path))
    return list(resolved.values())


@dataclass(eq=False)
class MessageWait(Generic[T]):
    targets: tuple[T, ...] = ()
    event: asyncio.Event = field(default_factory=asyncio.Event)
    delivery: str | None = None


def wake_waiters(
    waiters: Iterable[MessageWait], delivery: str | None = None
) -> bool:
    available = [waiter for waiter in waiters if not waiter.event.is_set()]
    if not available:
        return False
    available[0].delivery = delivery
    for waiter in available:
        waiter.event.set()
    return delivery is not None


def wait_dependency(
    source: T, target: T, dependencies: Callable[[T], Iterable[T]]
) -> str | None:
    """Find a direct or transitive wait edge without looping through cycles."""
    pending = list(dependencies(source))
    if target in pending:
        return "directly"
    visited = {source}
    while pending:
        current = pending.pop()
        if current == target:
            return "indirectly"
        if current not in visited:
            visited.add(current)
            pending.extend(dependencies(current))
    return None


def wait_timeout(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be a number") from exc
    if not 0 <= timeout <= 120:
        raise ValueError("timeout_seconds must be between 0 and 120")
    return timeout


def wait_targets(target: str | list[str]) -> list[str]:
    targets = [target] if isinstance(target, str) else target
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) and item.strip() for item in targets)
    ):
        raise ValueError(
            "target must be a non-empty agent path or list of paths"
        )
    return targets


def wait_result(
    completed: Sequence[tuple[str, str | None]],
    wait_all: bool,
    already_finished: bool,
) -> str:
    if wait_all and len(completed) > 1:
        if any(relation for _, relation in completed):
            details = ", ".join(
                f"{path} ({'finished' if relation is None else f'waiting on you {relation}'})"
                for path, relation in completed
            )
            return f"All agents are ready for this wait: {details}."
        verb = "have already finished" if already_finished else "finished"
        return f"All agents {verb} their work: {', '.join(path for path, _ in completed)}."
    path, relation = completed[0]
    if relation:
        return f"Agent {path} is ready for this wait because it is waiting on you {relation}."
    verb = "has already finished" if already_finished else "finished"
    return f"Agent {path} {verb} its work."


async def wait_for_completion(
    waiter: MessageWait,
    targets: Sequence[str],
    completed: Callable[[], list[tuple[str, str | None]]],
    timeout: float,
    wait_all: bool,
) -> str:
    """Wait for readiness or delivery, prioritizing already-ready targets."""
    deadline = asyncio.get_running_loop().time() + timeout
    first = True
    while True:
        if not first and waiter.event.is_set():
            return (
                waiter.delivery
                or "Stopped waiting because a new message arrived."
            )
        ready = completed()
        satisfied = len(ready) == len(targets) if wait_all else bool(ready)
        if satisfied:
            return wait_result(ready, wait_all, already_finished=first)
        first = False
        if waiter.event.is_set():
            continue
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            if len(targets) == 1:
                return f"Agent {targets[0]} did not finish within {timeout:g} seconds."
            subject = "Not all agents" if wait_all else "No agents"
            return f"{subject} finished within {timeout:g} seconds: {', '.join(targets)}."
        try:
            await asyncio.wait_for(waiter.event.wait(), min(remaining, 0.1))
        except TimeoutError:
            pass
