"""Immutable inbox envelopes and model-visible delivery context."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

InputLike = str | Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentMessage:
    """A message delivered to an agent's asynchronous inbox."""

    content: InputLike
    sender: str
    recipient: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict, repr=False)
    system_message: str | None = None
    message_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(slots=True)
class _QueuedAgentMessage:
    message: AgentMessage
    response: asyncio.Future[Any] | None = None
    on_start: Callable[[AgentMessage], Awaitable[None] | None] | None = None
    on_end: Callable[[AgentMessage, bool], Awaitable[None] | None] | None = None


def agent_message_system_instruction(sender: str, recipient: str) -> str:
    """Describe a routed message without changing its original content."""
    return (
        "Message delivery metadata (trusted URSA runtime):\n"
        f"Sender: {sender}\n"
        f"Recipient: {recipient}\n"
        "The following human message is the exact message content sent by "
        "the sender."
    )


def agent_message_wait_result(sender: str, recipient: str, content: str) -> str:
    """Expose one routed message directly to a model blocked in ``wait_for``."""
    return (
        "Received a message while waiting.\n"
        f"Sender: {sender}\n"
        f"Recipient: {recipient}\n"
        "Exact message content:\n"
        f"{content}"
    )


def attach_agent_message_instructions(
    inputs: InputLike,
    *,
    sender: str,
    recipient: str,
    system_message: str | None = None,
) -> Mapping[str, Any]:
    """Attach delivery context immediately before the delivered human input."""
    normalized: dict[str, Any]
    if isinstance(inputs, str):
        normalized = {"messages": [HumanMessage(content=inputs)]}
    else:
        normalized = dict(inputs)
    messages = list(normalized.get("messages") or [])
    insert_at = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            insert_at = index
            break
    instructions = []
    if system_message:
        instructions.append(SystemMessage(content=system_message))
    instructions.append(
        SystemMessage(
            content=agent_message_system_instruction(sender, recipient)
        )
    )
    messages[insert_at:insert_at] = instructions
    normalized["messages"] = messages
    return normalized
