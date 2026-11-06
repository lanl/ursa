"""Base agent class providing telemetry, configuration, and execution abstractions.

This module defines the BaseAgent abstract class, which serves as the foundation for all
agent implementations in the Ursa framework. It provides:

- Standardized initialization with LLM configuration
- Telemetry and metrics collection
- Thread and checkpoint management
- Input normalization and validation
- Execution flow control with invoke/stream methods
- Graph integration utilities for LangGraph compatibility
- Runtime enforcement of the agent interface contract

Agents built on this base class benefit from consistent behavior, observability, and
integration capabilities while only needing to implement the core _invoke method.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from contextvars import ContextVar, copy_context
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    final,
)
from uuid import uuid4

from filelock import AsyncFileLock, FileLock
from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain.tools import BaseTool, ToolException
from langchain_core.load import dumps
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import merge_configs
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import (
    CompiledStateGraph,
    StateGraph,
    coerce_to_runnable,
)
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Overwrite

from ursa.agents.coordination import (
    MessageWait,
    check_capacity,
    message_group,
    wait_dependency,
    wake_waiters,
)
from ursa.agents.coordination import (
    SubagentLimitError as SubagentLimitError,
)
from ursa.observability.timing import (
    Telemetry,  # for timing / telemetry / metrics
)
from ursa.security import (
    DEFAULT_GROUP_NAME,
    enforce_model_group_policy,
    group_agents_dir,
    group_root_dir,
    validate_group_name,
)
from ursa.util import Checkpointer
from ursa.util.checkpoint_retention import (
    aprune_sqlite_checkpoints,
    is_terminal_checkpoint,
    prune_sqlite_checkpoints,
)
from ursa.util.events import DEFAULT_EVENT_LOGGING_HANDLER, AgentEvents
from ursa.util.mcp import load_mcp_tools_with_sources
from ursa.util.model_context import (
    DEFAULT_SUMMARIZATION_THRESHOLD,
    summarization_threshold,
)

logger = logging.getLogger(__name__)

InputLike = str | Mapping[str, Any]
TState = TypeVar("TState", bound=Mapping[str, Any])

_INVOKE_DEPTH: ContextVar[int] = ContextVar("ursa_invoke_depth", default=0)

try:
    URSA_VERSION = importlib.metadata.version("ursa-ai")
except importlib.metadata.PackageNotFoundError:
    # Fallback for local development if the package isn't installed via pip/uv yet
    URSA_VERSION = "0.0.0-dev"


@dataclass(frozen=True, kw_only=True)
class AgentContext:
    """Immutable context provided during graph execution"""

    llm: BaseChatModel
    """ Chat model for use during tool calls """

    workspace: Path
    """ Workspace path for the agent """

    agent_name: str
    """ Name for persisting the agent """

    group: str
    """ Name for the group for the agent to operate in. Controls data and endpoint availability"""

    den: Path
    """ Path to the persistent storage for the agent"""

    tool_character_limit: int = 30000
    """ Suggested limit on tool call responses """

    rag_tools: tuple[str, ...] = ()
    """Persisted URSA RAG collections bound to this agent as tools."""

    rag_tool_group: str | None = None
    """Group used to resolve persisted RAG tool collections."""

    rag_tool_return_k: int = 10
    """Number of chunks retrieved by persisted RAG tools."""


def _to_snake(s: str) -> str:
    """Convert a string to snake_case format.

    This function transforms various string formats (CamelCase, PascalCase, etc.) into
    snake_case. It handles special cases like acronyms at the beginning of strings
    (e.g., "RAGAgent" becomes "rag_agent") and replaces hyphens and spaces with
    underscores.

    Args:
        s: The input string to convert to snake_case.

    Returns:
        The snake_case version of the input string.
    """
    s = re.sub(
        r"^([A-Z]{2,})([A-Z][a-z])",
        lambda m: m.group(1)[0] + m.group(1)[1:].lower() + m.group(2),
        str(s),
    )  # RAGAgent -> RagAgent
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)  # CamelCase -> snake_case
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


_SUMMARIZE_INSTRUCTIONS = (
    "Your only task is to provide a detailed, comprehensive summary of the "
    "following conversation.\n\n"
    "Your summary will be the only information retained from the conversation, "
    "so ensure it contains all details that need to be remembered to meet the "
    "goals of the work: decisions made, files created or modified (with paths), "
    "results obtained, open questions, and the current step."
)


def _message_text(m: BaseMessage) -> str:
    text = getattr(m, "text", None)
    if isinstance(text, str) and text:
        return text
    if isinstance(m.content, str):
        return m.content
    parts = []
    for block in m.content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)


def _render_for_summary(msgs: Sequence[BaseMessage]) -> str:
    out: list[str] = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            out.append(f"[tool result {m.tool_call_id}]\n{_message_text(m)}")
            continue
        role = type(m).__name__.removesuffix("Message").lower()
        if text := _message_text(m):
            out.append(f"[{role}]\n{text}")
        for tc in getattr(m, "tool_calls", None) or []:
            out.append(
                f"[tool call {tc.get('id')}] {tc.get('name')}({tc.get('args')})"
            )
    return "\n\n".join(out)


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


_STOP_MESSAGE_LOOP = object()


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


class AsyncMessageMixin:
    """Actor-style async messaging and dynamic agent hierarchy support.

    Each agent processes one inbox message at a time. Separate agents can run
    concurrently, so parents and siblings can exchange messages without sharing
    mutable invocation state.
    """

    _SUBAGENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._parent_agent: BaseAgent | None = None
        self._subagents: dict[str, BaseAgent] = {}
        self._subagent_stop_tasks: set[asyncio.Task[None]] = set()
        self._subagent_hierarchy_lock = RLock()
        self._hierarchy_name: str | None = None
        self._hierarchy_stopping = False
        self._stopping_agent_path: str | None = None
        self._message_inbox: (
            asyncio.Queue[_QueuedAgentMessage | object] | None
        ) = None
        self._message_loop_task: asyncio.Task[None] | None = None
        self._message_event_loop: asyncio.AbstractEventLoop | None = None
        self._message_loop_stopping = False
        self._current_agent_message: AgentMessage | None = None
        self._active_message_task: asyncio.Task[Any] | None = None
        # Tools such as ``wait_for`` can await another member's work without
        # blocking this actor's inbox.  A newly accepted message wakes those
        # tools so the model can handle the message promptly.
        self._message_waiters: set[MessageWait[BaseAgent]] = set()
        self._async_invocation_lock: asyncio.Lock | None = None
        self._async_invocation_loop: asyncio.AbstractEventLoop | None = None
        self._async_invocation_owner: asyncio.Task[Any] | None = None
        super().__init__(*args, **kwargs)

    @property
    def parent_agent(self) -> BaseAgent | None:
        """Return this agent's parent, if it has one."""
        return self._parent_agent

    @property
    def subagents(self) -> Mapping[str, BaseAgent]:
        """Return a read-only view of this agent's immediate children."""
        return MappingProxyType(self._subagents)

    @property
    def hierarchy_name(self) -> str:
        # Hierarchy paths have one stable, user-facing root regardless of the
        # configured top-level agent implementation (for example, ``chat``).
        if self._parent_agent is None:
            return "root"
        if self._hierarchy_name:
            return self._hierarchy_name
        return _to_snake(self.__class__.__name__)

    @property
    def agent_path(self) -> str:
        """Return this agent's slash-delimited path from the root."""
        if runtime_path := getattr(self, "_runtime_agent_path", None):
            return str(runtime_path)
        names: list[str] = []
        current: BaseAgent | None = self  # type: ignore[assignment]
        seen: set[int] = set()
        while current is not None:
            if id(current) in seen:
                raise RuntimeError("Cycle detected in agent hierarchy")
            seen.add(id(current))
            names.append(current.hierarchy_name)
            current = current.parent_agent
        return "/".join(reversed(names))

    @property
    def root_agent(self) -> BaseAgent:
        current: BaseAgent = self  # type: ignore[assignment]
        seen: set[int] = set()
        while current.parent_agent is not None:
            if id(current) in seen:
                raise RuntimeError("Cycle detected in agent hierarchy")
            seen.add(id(current))
            current = current.parent_agent
        return current

    async def post_message(
        self,
        content: InputLike,
        *,
        sender: str = "user",
        metadata: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        system_message: str | None = None,
        on_start: Callable[[AgentMessage], Awaitable[None] | None]
        | None = None,
        on_end: Callable[[AgentMessage, bool], Awaitable[None] | None]
        | None = None,
    ) -> AgentMessage:
        """Queue a one-way message to this agent."""
        message = self._make_agent_message(
            content,
            sender=sender,
            metadata=metadata,
            config=config,
            system_message=system_message,
        )
        await self._enqueue_agent_message(
            message,
            on_start=on_start,
            on_end=on_end,
        )
        return message

    async def request(
        self,
        content: InputLike,
        *,
        sender: str = "user",
        metadata: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        system_message: str | None = None,
        on_start: Callable[[AgentMessage], Awaitable[None] | None]
        | None = None,
    ) -> Any:
        """Queue a message and wait for the agent's result."""
        message = self._make_agent_message(
            content,
            sender=sender,
            metadata=metadata,
            config=config,
            system_message=system_message,
        )
        return await self._request_agent_message(
            message,
            on_start=on_start,
        )

    async def send_message(
        self,
        target: str | BaseAgent,
        content: InputLike,
        *,
        metadata: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> AgentMessage:
        """Send a one-way message to another agent."""
        recipient = self.resolve_agent(target)
        return await recipient.post_message(
            content,
            sender=self.agent_path,
            metadata=metadata,
            config=config,
        )

    async def ask_agent(
        self,
        target: str | BaseAgent,
        content: InputLike,
        *,
        metadata: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        on_start: Callable[[AgentMessage], Awaitable[None] | None]
        | None = None,
    ) -> Any:
        """Send a message to another agent and wait for its result."""
        recipient = self.resolve_agent(target)
        return await recipient.request(
            content,
            sender=self.agent_path,
            metadata=metadata,
            config=config,
            on_start=on_start,
        )

    def start_message_loop(self) -> asyncio.Task[None]:
        """Start this agent's inbox worker in the current event loop."""
        loop = asyncio.get_running_loop()
        if self._message_loop_stopping:
            raise RuntimeError(
                "Agent inbox is stopping; try again after shutdown"
            )
        task = self._message_loop_task
        if task is not None and not task.done():
            if self._message_event_loop is not loop:
                raise RuntimeError(
                    "Agent inbox is active in another event loop"
                )
            return task
        if self._message_event_loop is not loop:
            if (
                self._message_inbox is not None
                and not self._message_inbox.empty()
            ):
                raise RuntimeError(
                    "Pending messages belong to another event loop"
                )
            self._message_inbox = asyncio.Queue()
            self._message_event_loop = loop
        elif self._message_inbox is None:
            self._message_inbox = asyncio.Queue()
        message_context = copy_context()
        message_context.run(_INVOKE_DEPTH.set, 0)
        task = loop.create_task(
            self._run_message_loop(),
            name=f"ursa-message-loop:{self.agent_path}",
            # A child inbox may be started from inside its parent's invocation.
            # It is an independent actor, so it must not inherit that parent's
            # invocation depth. Preserve unrelated context such as Textual's
            # active-app binding, which UI callbacks need while rendering.
            context=message_context,
        )
        self._message_loop_task = task
        return task

    async def wait_for_messages(self) -> None:
        """Wait for all messages accepted by the inbox to finish."""
        if self._message_inbox is not None:
            await self._message_inbox.join()

    async def stop_message_loop(
        self,
        *,
        descendants: bool = True,
        cancel: bool = False,
    ) -> None:
        """Stop this inbox, gracefully unless ``cancel`` is true."""
        if descendants and self._subagents:
            await asyncio.gather(
                *(
                    child.stop_message_loop(descendants=True, cancel=cancel)
                    for child in tuple(self._subagents.values())
                )
            )
        task = self._message_loop_task
        inbox = self._message_inbox
        if task is None or task.done():
            self._cancel_queued_responses()
            return
        if task is asyncio.current_task():
            raise RuntimeError("An agent cannot stop its own message loop")
        if self._message_loop_stopping:
            await task
            return
        # Close admission before adding the sentinel. Otherwise a request can
        # be queued behind the sentinel and wait forever after the worker exits.
        self._message_loop_stopping = True
        try:
            if cancel:
                task.cancel()
            elif inbox is not None:
                await inbox.put(_STOP_MESSAGE_LOOP)
            try:
                await task
            except asyncio.CancelledError:
                if not cancel:
                    raise
        finally:
            if cancel:
                self._cancel_queued_responses()
            self._message_loop_task = None
            self._message_loop_stopping = False

    async def ahandle_message(self, message: AgentMessage) -> Any:
        """Handle an inbox message; subclasses may override this hook."""
        inputs = attach_agent_message_instructions(
            message.content,
            sender=message.sender,
            recipient=message.recipient,
            system_message=message.system_message,
        )
        config = dict(message.config)
        metadata = dict(config.get("metadata") or {})
        metadata.update(message.metadata)
        metadata.update({
            "ursa_message_id": message.message_id,
            "ursa_message_sender": message.sender,
            "ursa_message_recipient": message.recipient,
        })
        config["metadata"] = metadata
        return await self.ainvoke(inputs, config=config)

    def _make_agent_message(
        self,
        content: InputLike,
        *,
        sender: str,
        metadata: Mapping[str, Any] | None,
        config: Mapping[str, Any] | None,
        system_message: str | None,
    ) -> AgentMessage:
        return AgentMessage(
            content=content,
            sender=sender,
            recipient=self.agent_path,
            metadata=metadata or {},
            config=config or {},
            system_message=system_message,
        )

    async def _request_agent_message(
        self,
        message: AgentMessage,
        *,
        on_start: Callable[[AgentMessage], Awaitable[None] | None]
        | None = None,
    ) -> Any:
        response = asyncio.get_running_loop().create_future()
        await self._enqueue_agent_message(
            message,
            response=response,
            on_start=on_start,
        )
        return await response

    async def _enqueue_agent_message(
        self,
        message: AgentMessage,
        *,
        response: asyncio.Future[Any] | None = None,
        on_start: Callable[[AgentMessage], Awaitable[None] | None]
        | None = None,
        on_end: Callable[[AgentMessage, bool], Awaitable[None] | None]
        | None = None,
    ) -> None:
        self._ensure_hierarchy_accepting_messages()
        if response is None and self._wake_message_waiters(message):
            return
        self._wake_message_waiters()
        self.start_message_loop()
        if self._message_inbox is None:  # pragma: no cover
            raise RuntimeError("Agent inbox failed to initialize")
        await self._message_inbox.put(
            _QueuedAgentMessage(
                message=message,
                response=response,
                on_start=on_start,
                on_end=on_end,
            )
        )

    def _begin_message_wait(
        self, targets: Sequence[BaseAgent] = ()
    ) -> MessageWait[BaseAgent]:
        """Register an interruptible wait for this agent's next message."""
        waiter = MessageWait[BaseAgent](targets=tuple(targets))
        self._message_waiters.add(waiter)
        return waiter

    def _end_message_wait(self, waiter: MessageWait[BaseAgent]) -> None:
        """Remove a wait previously created by :meth:`_begin_message_wait`."""
        self._message_waiters.discard(waiter)

    def _wake_message_waiters(
        self, message: AgentMessage | None = None
    ) -> bool:
        delivery = (
            agent_message_wait_result(
                message.sender, message.recipient, str(message.content)
            )
            if message is not None
            else None
        )
        return wake_waiters(self._message_waiters, delivery)

    def _wait_dependency_relation(self, target: BaseAgent) -> str | None:
        return wait_dependency(
            self,
            target,
            lambda agent: (
                dependency
                for waiter in agent._message_waiters
                for dependency in waiter.targets
            ),
        )

    async def _run_message_loop(self) -> None:
        inbox = self._message_inbox
        if inbox is None:  # pragma: no cover
            return
        try:
            while True:
                item = await inbox.get()
                if item is _STOP_MESSAGE_LOOP:
                    inbox.task_done()
                    break
                if not isinstance(item, _QueuedAgentMessage):
                    inbox.task_done()
                    continue
                self._current_agent_message = item.message
                if item.on_start is not None:
                    try:
                        started = item.on_start(item.message)
                        if inspect.isawaitable(started):
                            await started
                    except Exception:
                        logger.exception(
                            "Agent %s message-start callback failed for %s",
                            self.agent_path,
                            item.message.message_id,
                        )
                handler = asyncio.create_task(
                    self.ahandle_message(item.message)
                )
                self._active_message_task = handler
                succeeded = False
                try:
                    result = await handler
                except asyncio.CancelledError:
                    if item.response is not None and not item.response.done():
                        item.response.cancel()
                    if asyncio.current_task().cancelling():
                        raise
                except Exception as exc:
                    if item.response is not None and not item.response.done():
                        item.response.set_exception(exc)
                    else:
                        logger.exception(
                            "Agent %s failed handling message %s",
                            self.agent_path,
                            item.message.message_id,
                        )
                else:
                    succeeded = True
                    if item.response is not None and not item.response.done():
                        item.response.set_result(result)
                finally:
                    if item.on_end is not None:
                        try:
                            ended = item.on_end(item.message, succeeded)
                            if inspect.isawaitable(ended):
                                await ended
                        except Exception:
                            logger.exception(
                                "Agent %s message-end callback failed for %s",
                                self.agent_path,
                                item.message.message_id,
                            )
                    self._active_message_task = None
                    self._current_agent_message = None
                    inbox.task_done()
        finally:
            self._active_message_task = None
            self._current_agent_message = None

    def _cancel_queued_responses(self) -> None:
        inbox = self._message_inbox
        if inbox is None:
            return
        while True:
            try:
                item = inbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _QueuedAgentMessage):
                if item.response is not None and not item.response.done():
                    item.response.cancel()
            inbox.task_done()

    def _subagent_factory_kwargs(
        self, agent_kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        kwargs = dict(agent_kwargs)
        root = self.root_agent
        child_llm = kwargs.setdefault("llm", self._subagent_llm)
        kwargs.setdefault("workspace", self.workspace)
        kwargs.setdefault("group", self.group)
        child_embedding = kwargs.setdefault(
            "rag_tool_embedding", self._subagent_embedding
        )
        kwargs.setdefault("subagent_llm", child_llm)
        kwargs.setdefault("subagent_embedding", child_embedding)
        kwargs.setdefault("subagent_max_agents", root._subagent_max_agents)
        kwargs.setdefault("subagent_max_depth", root._subagent_max_depth)
        return kwargs

    def _validate_subagent_capacity(
        self, agent: BaseAgent | None = None
    ) -> None:
        root = self.root_agent
        additions = list(agent.walk_agents()) if agent is not None else []
        subtree_depth = max(
            (
                member.agent_path.count("/") - agent.agent_path.count("/")
                for member in additions
            ),
            default=0,
        )
        check_capacity(
            sum(1 for _ in root.walk_agents()) - 1 + max(1, len(additions)),
            self.agent_path.count("/") + 1 + subtree_depth,
            root._subagent_max_agents,
            root._subagent_max_depth,
        )

    @classmethod
    def _validate_subagent_name(cls, name: str) -> None:
        if (
            not isinstance(name, str)
            or name in {".", "..", "self", "parent", "root"}
            or cls._SUBAGENT_NAME.fullmatch(name) is None
        ):
            raise ValueError(
                "Subagent names must start with an alphanumeric character and "
                "contain only letters, digits, '.', '-', or '_'"
            )

    def _get_async_invocation_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._async_invocation_loop is not loop:
            if (
                self._async_invocation_lock
                and self._async_invocation_lock.locked()
            ):
                raise RuntimeError(
                    "Agent is already running in another event loop"
                )
            self._async_invocation_lock = asyncio.Lock()
            self._async_invocation_loop = loop
        elif self._async_invocation_lock is None:
            self._async_invocation_lock = asyncio.Lock()
        return self._async_invocation_lock

    @property
    def current_message(self) -> AgentMessage | None:
        """Return the inbox envelope currently being handled."""
        return self._current_agent_message

    @property
    def is_working(self) -> bool:
        """Whether this agent is currently handling an invocation."""
        task = self._async_invocation_owner or self._active_message_task
        return (
            self._current_agent_message is not None
            or self.pending_messages > 0
            or (task is not None and not task.done())
        )

    @property
    def is_waiting(self) -> bool:
        """Whether a hierarchy ``wait_for`` call currently blocks this agent.

        A waiting agent has yielded its model turn. Wait tools use the active
        dependency graph to recognize only reciprocal wait cycles as ready.
        """
        return bool(self._message_waiters)

    @property
    def message_loop_running(self) -> bool:
        task = self._message_loop_task
        return task is not None and not task.done()

    @property
    def pending_messages(self) -> int:
        return self._message_inbox.qsize() if self._message_inbox else 0

    def walk_agents(self) -> Iterator[BaseAgent]:
        """Yield this agent and all descendants in depth-first order."""
        yield self  # type: ignore[misc]
        for child in tuple(self._subagents.values()):
            yield from child.walk_agents()

    def _ensure_hierarchy_accepting_messages(self) -> None:
        if self._hierarchy_stopping:
            path = self._stopping_agent_path or self.agent_path
            raise RuntimeError(f"Agent {path!r} is stopping")

    def attach_subagent(self, name: str, agent: BaseAgent) -> BaseAgent:
        """Attach an existing agent as an immediate child."""
        self._ensure_hierarchy_accepting_messages()
        self._validate_subagent_name(name)
        if not isinstance(agent, BaseAgent):
            raise TypeError("subagent must be a BaseAgent instance")
        with self.root_agent._subagent_hierarchy_lock:
            if name in self._subagents:
                raise ValueError(f"Subagent name already exists: {name!r}")
            if agent is self:
                raise ValueError("An agent cannot be its own subagent")
            if agent.parent_agent is not None:
                raise ValueError(
                    f"Agent {agent.agent_path!r} already has a parent"
                )
            ancestor: BaseAgent | None = self  # type: ignore[assignment]
            while ancestor is not None:
                if ancestor is agent:
                    raise ValueError(
                        "Attaching this agent would create a cycle"
                    )
                ancestor = ancestor.parent_agent
            self._validate_subagent_capacity(agent)
            agent._parent_agent = self  # type: ignore[assignment]
            agent._hierarchy_name = name
            self._subagents[name] = agent
        return agent

    def spawn_subagent(
        self,
        name: str,
        agent: BaseAgent | Callable[..., BaseAgent] | None = None,
        /,
        **agent_kwargs: Any,
    ) -> BaseAgent:
        """Construct and attach a child, inheriting common configuration."""
        self._ensure_hierarchy_accepting_messages()
        if isinstance(agent, BaseAgent):
            if agent_kwargs:
                raise TypeError("agent_kwargs cannot accompany an instance")
            return self.attach_subagent(name, agent)
        self._validate_subagent_capacity()
        factory = agent or self.__class__
        child = factory(**self._subagent_factory_kwargs(agent_kwargs))
        if inspect.isawaitable(child):
            raise TypeError(
                "Async factories require `await aspawn_subagent(...)`"
            )
        if not isinstance(child, BaseAgent):
            raise TypeError("subagent factory must return a BaseAgent")
        return self.attach_subagent(name, child)

    async def aspawn_subagent(
        self,
        name: str,
        agent: BaseAgent
        | Callable[..., BaseAgent | Awaitable[BaseAgent]]
        | None = None,
        /,
        **agent_kwargs: Any,
    ) -> BaseAgent:
        """Async-factory variant of :meth:`spawn_subagent`."""
        self._ensure_hierarchy_accepting_messages()
        if isinstance(agent, BaseAgent):
            if agent_kwargs:
                raise TypeError("agent_kwargs cannot accompany an instance")
            return self.attach_subagent(name, agent)
        self._validate_subagent_capacity()
        factory = agent or self.__class__
        child = factory(**self._subagent_factory_kwargs(agent_kwargs))
        if inspect.isawaitable(child):
            child = await child
        if not isinstance(child, BaseAgent):
            raise TypeError("subagent factory must return a BaseAgent")
        return self.attach_subagent(name, child)

    def detach_subagent(self, name: str) -> BaseAgent:
        """Detach and return an idle immediate child."""
        with self.root_agent._subagent_hierarchy_lock:
            try:
                child = self._subagents[name]
            except KeyError:
                raise KeyError(
                    f"Unknown immediate subagent: {name!r}"
                ) from None
            if child.message_loop_running:
                raise RuntimeError(
                    "Cannot detach a running child; use despawn_subagent"
                )
            del self._subagents[name]
            child._parent_agent = None
            child._hierarchy_name = None
        return child

    async def despawn_subagent(
        self, name: str, *, cancel: bool = False
    ) -> BaseAgent:
        try:
            child = self._subagents[name]
        except KeyError:
            raise KeyError(f"Unknown immediate subagent: {name!r}") from None
        await child.stop_message_loop(descendants=True, cancel=cancel)
        return self.detach_subagent(name)

    def stop_subagent(self, name: str) -> BaseAgent:
        """Detach a child immediately and stop its subtree in the background."""
        with self.root_agent._subagent_hierarchy_lock:
            try:
                child = self._subagents.pop(name)
            except KeyError:
                raise KeyError(
                    f"Unknown immediate subagent: {name!r}"
                ) from None
            child_path = child.agent_path
            for member in child.walk_agents():
                member._hierarchy_stopping = True
                member._stopping_agent_path = member.agent_path
            child._parent_agent = None
            child._hierarchy_name = None
        task = asyncio.create_task(
            child.stop_message_loop(descendants=True, cancel=True),
            name=f"ursa-stop-subagent:{child_path}",
        )
        self._subagent_stop_tasks.add(task)

        def stopped(completed: asyncio.Task[None]) -> None:
            self._subagent_stop_tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                completed.result()
            except Exception:
                logger.exception("Failed stopping subagent %s", name)

        task.add_done_callback(stopped)
        return child

    def resolve_agent(self, target: str | BaseAgent) -> BaseAgent:
        """Resolve an object, relative path, or root-qualified path."""
        if isinstance(target, BaseAgent):
            return target
        if not isinstance(target, str) or not target.strip():
            raise TypeError("target must be a BaseAgent or non-empty path")
        raw = target.strip()
        aliases = {
            ".": self,
            "self": self,
            "root": self.root_agent,
            "parent": self.parent_agent,
        }
        if raw in aliases:
            resolved = aliases[raw]
            if resolved is None:
                raise KeyError("The root agent has no parent")
            return resolved  # type: ignore[return-value]

        root = self.root_agent
        absolute = raw.startswith("/")
        parts = [part for part in raw.strip("/").split("/") if part]
        if parts and parts[0] == root.hierarchy_name:
            absolute = True
            parts.pop(0)
        current: BaseAgent = root if absolute else self  # type: ignore[assignment]
        if (
            not absolute
            and len(parts) == 1
            and parts[0] not in current.subagents
            and current.parent_agent is not None
            and parts[0] in current.parent_agent.subagents
        ):
            current = current.parent_agent
        for part in parts:
            if part in {".", "self"}:
                continue
            if part in {"..", "parent"}:
                if current.parent_agent is None:
                    raise KeyError(f"Path escapes hierarchy root: {raw!r}")
                current = current.parent_agent
                continue
            try:
                current = current.subagents[part]
            except KeyError:
                raise KeyError(
                    f"Unknown agent path {raw!r} from {self.agent_path!r}"
                ) from None
        return current

    def resolve_message_targets(
        self, target: str | Sequence[str]
    ) -> list[BaseAgent]:
        """Resolve message routes, expanding hierarchy group selectors.

        ``@everyone`` addresses every other member of this hierarchy.
        ``@supervisor``, ``@peers``, and ``@reports`` address the immediate
        supervisor, same-supervisor peers, and immediate reports respectively.
        Group selectors are only message routes; ordinary path
        resolution remains unchanged for other hierarchy operations.
        """
        routes = [target] if isinstance(target, str) else list(target)
        if not routes or not all(
            isinstance(item, str) and item.strip() for item in routes
        ):
            raise TypeError(
                "target must be a non-empty agent path or list of paths"
            )

        hierarchy = {
            member.agent_path: member
            for member in self.root_agent.walk_agents()
        }
        recipients: list[BaseAgent] = []
        seen: set[str] = set()
        expands_groups = any(route.strip().startswith("@") for route in routes)
        for route in routes:
            name = route.strip()
            if name.startswith("@"):
                try:
                    expanded = [
                        hierarchy[path]
                        for path in message_group(
                            self.agent_path, name, hierarchy
                        )
                    ]
                except ValueError as exc:
                    raise KeyError(str(exc)) from exc
            else:
                expanded = [self.resolve_agent(name)]
            for recipient in expanded:
                if expands_groups and recipient is self:
                    continue
                path = recipient.agent_path
                if path not in seen:
                    seen.add(path)
                    recipients.append(recipient)
        return recipients


class UnregisteredAgentStateWarning(UserWarning):
    """A BaseAgent subclass declares a typed state it never registers."""


class BaseAgent(AsyncMessageMixin, Generic[TState], ABC):
    """Abstract base class for all agent implementations in the Ursa framework.

    BaseAgent provides a standardized foundation for building LLM-powered agents with
    built-in telemetry, configuration management, and execution flow control. It handles
    common tasks like input normalization, thread management, metrics collection, and
    LangGraph integration.

    Subclasses only need to implement the _invoke method to define their core
    functionality, while inheriting standardized invocation patterns, telemetry, and
    graph integration capabilities. The class enforces a consistent interface through
    runtime checks that prevent subclasses from overriding critical methods like
    invoke().

    The agent supports both direct invocation with inputs and streaming responses, with
    automatic tracking of token usage, execution time, and other metrics. It also
    provides utilities for integrating with LangGraph through node wrapping and
    configuration.

    Subclass Inheritance Guidelines:
        - Must Override: _invoke() - Define your agent's core functionality
        - Can Override: _stream() - Enable streaming support
                        _normalize_inputs() - Customize input handling
                        Various helper methods (_default_node_tags, etc.)
        - Never Override: invoke() - Final method with runtime enforcement
                          stream() - Handles telemetry and delegates to _stream
                          __call__() - Delegates to invoke
                          Other public methods (build_config, write_state, add_node)

    To create a custom agent, inherit from this class and implement the _invoke method:

    ```python
    class MyAgent(BaseAgent):
        def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
            # Process inputs and return results
            ...
    ```
    """

    _TELEMETRY_KW = {
        "raw_debug",
        "save_json",
        "save_otel",
        "metrics_path",
        "save_raw_snapshot",
        "save_raw_records",
        "otel_endpoint",
        "otel_headers",
    }

    _CONTROL_KW = {"config", "recursion_limit", "tags", "metadata", "callbacks"}

    state_type: type[TState] = dict

    def __init__(
        self,
        llm: BaseChatModel,
        workspace: Optional[Path] = None,
        agent_name: Optional[str] = None,
        group: Optional[str] = "default",
        checkpointer: Optional[BaseCheckpointSaver] = None,
        enable_metrics: bool = True,
        metrics_dir: str = "ursa_metrics",  # dir to save metrics, with a default
        autosave_metrics: bool = True,
        otel_metrics: Optional[bool] = None,
        thread_id: Optional[str] = None,
        tokens_before_summarize: int = DEFAULT_SUMMARIZATION_THRESHOLD,
        messages_to_keep: int = 20,
        max_single_tool_message_tokens: int = 100000,
        rag_tools: str | Sequence[str] | None = None,
        rag_tool_group: str | None = None,
        rag_tool_embedding: Embeddings | None = None,
        rag_tool_return_k: int = 10,
        subagent_llm: BaseChatModel | None = None,
        subagent_embedding: Embeddings | None = None,
        subagent_max_agents: int | None = 32,
        subagent_max_depth: int | None = 3,
    ):
        """Initializes the base agent with a language model and optional configurations.

        Args:
            llm: a BaseChatModel instance.
            tokens_before_summarize: Fallback threshold when the model context
                window is unknown. Otherwise summarize at 70% of that window.
            checkpointer: Optional checkpoint saver for persisting agent state.
            enable_metrics: Whether to collect performance and usage metrics.
            metrics_dir: Directory path where metrics will be saved.
            autosave_metrics: Whether to automatically save metrics to disk.
            thread_id: Unique identifier for this agent instance. Generated if not
                       provided.
            subagent_llm: Optional model override inherited by all subagents.
            subagent_embedding: Optional embedding override for subagent RAG.
            subagent_max_agents: Maximum children in the whole hierarchy. Zero
                disables subagents and None removes the count limit.
            subagent_max_depth: Maximum depth below the root. Zero disables
                subagents and None removes the depth limit.
        """
        super().__init__()
        headers = {
            "User-Agent": f"ursa/{URSA_VERSION}",
            "x-ursa-user-agent": f"ursa/{URSA_VERSION}",
        }
        root_llm = llm
        llm = llm.bind(extra_headers=headers)
        if subagent_llm is None or subagent_llm is root_llm:
            subagent_llm = llm
        else:
            subagent_llm = subagent_llm.bind(extra_headers=headers)
        for name, value in (
            ("subagent_max_agents", subagent_max_agents),
            ("subagent_max_depth", subagent_max_depth),
        ):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if otel_metrics is not None:
            import warnings

            warnings.warn(
                "The otel_metrics argument has never had an effect and is "
                "deprecated; OpenTelemetry export is configured per invoke "
                "via save_otel (see issue #259, Tier 2 will supersede it).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.llm: BaseChatModel = llm
        self._subagent_llm = subagent_llm
        self._subagent_embedding = (
            subagent_embedding
            if subagent_embedding is not None
            else rag_tool_embedding
        )
        self._subagent_max_agents = subagent_max_agents
        self._subagent_max_depth = subagent_max_depth
        self.workspace = Path(workspace or ".")
        self.agent_name = agent_name
        self.group = validate_group_name(group)
        self.tokens_before_summarize = tokens_before_summarize
        self.messages_to_keep = messages_to_keep
        self.max_single_tool_message_tokens = max_single_tool_message_tokens
        from ursa.rag.persistence import normalize_rag_tool_names

        self.rag_tools = tuple(normalize_rag_tool_names(rag_tools))
        self.rag_tool_group = rag_tool_group
        self.rag_tool_embedding = rag_tool_embedding
        self.rag_tool_return_k = rag_tool_return_k
        enforce_model_group_policy(self.llm, self.group)
        enforce_model_group_policy(self._subagent_llm, self.group)
        if (
            not group_root_dir(self.group).exists()
            and self.group != DEFAULT_GROUP_NAME
        ):
            raise ValueError(
                (
                    f"Group '{self.group}' does not exist. "
                    f"Please use `ursa create-group {self.group} <group_config_file>` to create"
                )
            )
        set_checkpointer = True if checkpointer else False
        set_name = True if agent_name else False
        self.checkpointer = checkpointer
        self._checkpointer_was_provided = set_checkpointer
        self._async_checkpointer: BaseCheckpointSaver | None = None
        self._async_storage: BaseStore | None = None
        self._async_compiled_graph: CompiledStateGraph | None = None
        self._checkpoint_retention_enabled = False
        persist_agent = (agent_name is not None) or set_checkpointer
        if persist_agent:
            if set_checkpointer:
                if set_name:
                    logger.warning(
                        "Both checkpointer and den persistence are set. "
                        "Using checkpointer; only use one in the future."
                    )
                self.den = self.workspace
            else:
                den_name = Path(self.group) / "agents" / agent_name
                self.den = group_agents_dir(self.group) / agent_name
                if not self.den.exists():
                    logger.info("Agent created: %s", den_name)
                self.checkpointer = Checkpointer.from_workspace(self.den)
                self._checkpoint_retention_enabled = True
        else:
            # Keep current behavior if the user is not persisting.
            self.den = self.workspace
        # A generated default, as the docstring above promises. A constant here
        # means two agents that share a checkpoint store share one thread, so
        # the second one's model request carries the first one's messages.
        # langgraph would otherwise refuse to run a checkpointed graph with no
        # thread selector at all; a constant turns that error into a silent
        # merge. The CLI and the dashboard pin "ursa" themselves, so only
        # library callers reach this. See issue #332.
        self.thread_id = thread_id or uuid4().hex
        self.telemetry = Telemetry(
            enable=enable_metrics,
            output_dir=self.den.joinpath(metrics_dir),
            save_json_default=autosave_metrics,
        )

        self.workspace.mkdir(exist_ok=True, parents=True)
        self.den.mkdir(exist_ok=True, parents=True)

    @property
    def name(self) -> str:
        """Agent name."""
        return self.__class__.__name__

    @property
    def context(self) -> AgentContext:
        """Immutable run-scoped information provided to the Agent's graph"""
        return AgentContext(
            llm=self.llm,
            workspace=self.workspace,
            agent_name=self.agent_name,
            group=self.group,
            den=self.den,
            rag_tools=self.rag_tools,
            rag_tool_group=self.rag_tool_group,
            rag_tool_return_k=self.rag_tool_return_k,
        )

    @staticmethod
    def _dedupe_callbacks(callbacks: list[Any]) -> list[Any]:
        """Preserve callback order while removing duplicate handler instances."""
        deduped: list[Any] = []
        seen: set[int] = set()
        for callback in callbacks:
            marker = id(callback)
            if marker in seen:
                continue
            deduped.append(callback)
            seen.add(marker)
        return deduped

    def events(self, config: RunnableConfig | None = None) -> AgentEvents:
        """Return a helper for emitting structured agent progress events."""
        return AgentEvents(agent=self.name, config=config)

    def add_node(
        self,
        f: Callable[..., Mapping[str, Any]],
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> StateGraph:
        """Add a node to the state graph with token usage tracking.

        This method adds a function as a node to the state graph, wrapping it to track
        token usage during execution. The node is identified by either the provided
        node_name or the function's name.

        Args:
            f: The function to add as a node. Should return a mapping of string keys to
                any values.
            node_name: Optional name for the node. If not provided, the function's name
                will be used.
            agent_name: Optional agent name for tracking. If not provided, the agent's
                name in snake_case will be used.

        Returns:
            The updated StateGraph with the new node added.
        """
        _node_name = node_name or f.__name__
        _agent_name = agent_name or _to_snake(self.name)
        wrapped_node = self._wrap_node(f, _node_name, _agent_name)
        return self.graph.add_node(_node_name, wrapped_node, **kwargs)

    def write_state(self, filename: str, state: dict) -> None:
        """Writes agent state to a JSON file.

        Serializes the provided state dictionary to JSON format and writes it to the
        specified file. The JSON is written with non-ASCII characters preserved.

        Args:
            filename: Path to the file where state will be written.
            state: Dictionary containing the agent state to be serialized.
        """
        json_state = dumps(state, ensure_ascii=False)
        with open(filename, "w") as f:
            f.write(json_state)

    def build_config(self, **overrides) -> dict:
        """Constructs a config dictionary for agent operations with telemetry support.

        This method creates a standardized configuration dictionary that includes thread
        identification, telemetry callbacks, and other metadata needed for agent
        operations. The configuration can be customized through override parameters.

        Args:
            **overrides: Optional configuration overrides that can include keys like
                'recursion_limit', 'configurable', 'metadata', 'tags', etc.

        Returns:
            dict: A complete configuration dictionary with all necessary parameters.
        """
        # Create the base configuration with essential fields.
        base = {
            "configurable": {"thread_id": self.thread_id},
            "metadata": {
                "thread_id": self.thread_id,
                "telemetry_run_id": self.telemetry.context.get("run_id"),
            },
            "tags": [self.name],
            "callbacks": [
                *self.telemetry.callbacks,
                DEFAULT_EVENT_LOGGING_HANDLER,
            ],
        }

        # Try to determine the model name from either direct or nested attributes
        model_name = getattr(self, "llm_model", None) or getattr(
            getattr(self, "llm", None), "model", None
        )

        # Add model name to metadata if available
        if model_name:
            base["metadata"]["model"] = model_name

        # Handle configurable dictionary overrides by merging with base configurable
        if "configurable" in overrides and isinstance(
            overrides["configurable"], dict
        ):
            base["configurable"].update(overrides.pop("configurable"))

        # Handle metadata dictionary overrides by merging with base metadata
        if "metadata" in overrides and isinstance(overrides["metadata"], dict):
            base["metadata"].update(overrides.pop("metadata"))

        # Merge tags from caller-provided overrides, avoid duplicates
        if "tags" in overrides and isinstance(overrides["tags"], list):
            base["tags"] = base["tags"] + [
                t for t in overrides.pop("tags") if t not in base["tags"]
            ]

        if "callbacks" in overrides:
            callbacks = merge_configs(
                {"callbacks": base["callbacks"]},
                {"callbacks": overrides.pop("callbacks")},
            ).get("callbacks")
            if isinstance(callbacks, list):
                callbacks = self._dedupe_callbacks(callbacks)
            base["callbacks"] = callbacks

        # Apply any remaining overrides directly to the base configuration
        base.update(overrides)

        return base

    def _invoke_engine(
        self,
        invoke_method,
        inputs: Optional[InputLike] = None,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_headers: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ):
        depth = _INVOKE_DEPTH.get() + 1
        depth_token = _INVOKE_DEPTH.set(depth)
        invoke_config = dict(config or {})

        try:
            # Start telemetry tracking for the top-level invocation
            if depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # Handle the case where inputs are provided as keyword arguments
            if inputs is None:
                # Separate kwargs into input parameters and control parameters
                kw_inputs: dict[str, Any] = {}
                control_kwargs: dict[str, Any] = {}
                for k, v in kwargs.items():
                    if k in self._TELEMETRY_KW or k in self._CONTROL_KW:
                        control_kwargs[k] = v
                    else:
                        kw_inputs[k] = v
                inputs = kw_inputs

                # Only control kwargs remain for further processing
                kwargs = control_kwargs

            # Handle the case where inputs are provided as a positional argument
            else:
                # Ensure no ambiguous keyword arguments are present
                for k in kwargs.keys():
                    if not (k in self._TELEMETRY_KW or k in self._CONTROL_KW):
                        raise TypeError(
                            f"Unexpected keyword argument '{k}'. "
                            "Pass inputs as a single mapping or omit the positional "
                            "inputs and pass them as keyword arguments."
                        )

            # Allow subclasses to normalize or transform the input format
            normalized = self._normalize_inputs(inputs)

            # Delegate to the subclass implementation with the normalized inputs
            # and any control parameters
            return invoke_method(normalized, **invoke_config, **kwargs)

        finally:
            _INVOKE_DEPTH.reset(depth_token)
            if depth == 1:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    otel_endpoint=otel_endpoint,
                    otel_headers=otel_headers,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    async def _ainvoke_engine(self, *args: Any, **kwargs: Any) -> Any:
        """Serialize async invocations of one agent instance."""
        current = asyncio.current_task()
        if self._async_invocation_owner is current:
            return await self._ainvoke_engine_unlocked(*args, **kwargs)
        lock = self._get_async_invocation_lock()
        async with lock:
            self._async_invocation_owner = current
            try:
                return await self._ainvoke_engine_unlocked(*args, **kwargs)
            finally:
                self._async_invocation_owner = None

    async def _ainvoke_engine_unlocked(
        self,
        invoke_method,
        inputs: Optional[InputLike] = None,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_headers: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ):
        depth = _INVOKE_DEPTH.get() + 1
        depth_token = _INVOKE_DEPTH.set(depth)
        invoke_config = dict(config or {})

        try:
            if depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            if inputs is None:
                kw_inputs: dict[str, Any] = {}
                control_kwargs: dict[str, Any] = {}
                for k, v in kwargs.items():
                    if k in self._TELEMETRY_KW or k in self._CONTROL_KW:
                        control_kwargs[k] = v
                    else:
                        kw_inputs[k] = v
                inputs = kw_inputs
                kwargs = control_kwargs
            else:
                for k in kwargs.keys():
                    if not (k in self._TELEMETRY_KW or k in self._CONTROL_KW):
                        raise TypeError(
                            f"Unexpected keyword argument '{k}'. "
                            "Pass inputs as a single mapping or omit the "
                            "positional inputs and pass them as keyword "
                            "arguments."
                        )

            normalized = self._normalize_inputs(inputs)
            return await invoke_method(normalized, **invoke_config, **kwargs)

        finally:
            _INVOKE_DEPTH.reset(depth_token)
            if depth == 1:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    otel_endpoint=otel_endpoint,
                    otel_headers=otel_headers,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    # NOTE: The `invoke` method uses the PEP 570 `/,*` notation to explicitly state which
    # arguments can and cannot be passed as positional or keyword arguments.
    @final
    def invoke(
        self,
        inputs: Optional[InputLike] = None,
        /,
        *,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Executes the agent with the provided inputs and configuration.

        This is the main entry point for agent execution. It handles input normalization,
        telemetry tracking, and proper execution context management. The method supports
        flexible input formats - either as a positional argument or as keyword arguments.

        Args:
            inputs: Optional positional input to the agent. If provided, all non-control
                keyword arguments will be rejected to avoid ambiguity.
            raw_debug: If True, displays raw telemetry data for debugging purposes.
            save_json: If True, saves telemetry data as JSON.
            save_otel: If True, saves telemetry data to OpenTelemetry endpoint.
            metrics_path: Optional file path where telemetry metrics should be saved.
            save_raw_snapshot: If True, saves a raw snapshot of the telemetry data.
            save_raw_records: If True, saves raw telemetry records.
            config: Optional configuration dictionary to override default settings.
            **kwargs: Additional keyword arguments that can be either:
                - Input parameters (when no positional input is provided)
                - Control parameters recognized by the agent

        Returns:
            The result of the agent's execution.

        Raises:
            TypeError: If both positional inputs and non-control keyword arguments are
                provided simultaneously.
        """
        return self._invoke_engine(
            invoke_method=self._invoke,
            inputs=inputs,
            raw_debug=raw_debug,
            save_json=save_json,
            save_otel=save_otel,
            metrics_path=metrics_path,
            save_raw_snapshot=save_raw_snapshot,
            save_raw_records=save_raw_records,
            config=config,
            **kwargs,
        )

    # NOTE: The `ainvoke` method uses the PEP 570 `/,*` notation to explicitly state which
    # arguments can and cannot be passed as positional or keyword arguments.
    @final
    async def ainvoke(
        self,
        inputs: Optional[InputLike] = None,
        /,
        *,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchrnously executes the agent with the provided inputs and configuration.

        (Async version of `invoke`.)

        This is the main entry point for agent execution. It handles input normalization,
        telemetry tracking, and proper execution context management. The method supports
        flexible input formats - either as a positional argument or as keyword arguments.

        Args:
            inputs: Optional positional input to the agent. If provided, all non-control
                keyword arguments will be rejected to avoid ambiguity.
            raw_debug: If True, displays raw telemetry data for debugging purposes.
            save_json: If True, saves telemetry data as JSON.
            save_otel: If True, saves telemetry data to OpenTelemetry endpoint.
            metrics_path: Optional file path where telemetry metrics should be saved.
            save_raw_snapshot: If True, saves a raw snapshot of the telemetry data.
            save_raw_records: If True, saves raw telemetry records.
            config: Optional configuration dictionary to override default settings.
            **kwargs: Additional keyword arguments that can be either:
                - Input parameters (when no positional input is provided)
                - Control parameters recognized by the agent

        Returns:
            The result of the agent's execution.

        Raises:
            TypeError: If both positional inputs and non-control keyword arguments are
                provided simultaneously.
        """
        return await self._ainvoke_engine(
            invoke_method=self._ainvoke,
            inputs=inputs,
            raw_debug=raw_debug,
            save_json=save_json,
            save_otel=save_otel,
            metrics_path=metrics_path,
            save_raw_snapshot=save_raw_snapshot,
            save_raw_records=save_raw_records,
            config=config,
            **kwargs,
        )

    def format_query(self, prompt: str, state: TState | None = None) -> TState:
        """Format a plain text prompt into the agent's input schema
        possibly incorporating the prior state.

        Agents should override this method for their operation
        """

        if state is not None and "messages" in state:
            state["messages"].append(HumanMessage(content=str(prompt)))
            return state
        return self._normalize_inputs(prompt)

    def format_result(self, result: TState) -> str:
        """Extracts a plain text response from the Agent's output schema

        Agents should override this method for their operation
        """

        if "messages" in result:
            if isinstance(result["messages"], list) and isinstance(
                result["messages"][-1], BaseMessage
            ):
                return result["messages"][-1].text
        raise NotImplementedError()

    def _message_tool_call_ids(self, msg: Any) -> list[str]:
        """Return LangChain tool-call IDs requested by a message, if any."""
        return [
            call["id"]
            for call in getattr(msg, "tool_calls", []) or []
            if "id" in call
        ]

    def _sanitize_history(
        self,
        state: Mapping[str, Any],
        summarized: bool = False,
        config: RunnableConfig | None = None,
        *,
        keep_last_human: bool = True,
    ) -> tuple[Mapping[str, Any], bool]:
        """Drop mid-list system messages, orphan tool results, and unanswered
        trailing prompts left behind by crashed steps. Dangling *tool calls*
        are left for ``_patch_dangling``."""
        if "messages" not in state:
            return state, summarized

        new_state = deepcopy(state)
        events = self.events(config)
        original = list(new_state["messages"])

        msgs = [
            m
            for i, m in enumerate(original)
            if not (isinstance(m, SystemMessage) and i != 0)
        ]

        seen_calls: set[str] = set()
        paired: list[BaseMessage] = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                if m.tool_call_id in seen_calls:
                    paired.append(m)
                continue
            seen_calls.update(self._message_tool_call_ids(m))
            paired.append(m)
        msgs = paired

        # A leading system prompt is not part of the conversation tail.
        head = msgs[:1] if msgs and isinstance(msgs[0], SystemMessage) else []
        body = msgs[len(head) :]
        last_reply = max(
            (
                i
                for i, m in enumerate(body)
                if isinstance(m, (AIMessage, ToolMessage))
            ),
            default=-1,
        )
        tail = body[last_reply + 1 :]
        if tail:
            keep = []
            if keep_last_human:
                keep = [m for m in tail if isinstance(m, HumanMessage)][-1:]
            msgs = head + body[: last_reply + 1] + keep

        if len(msgs) != len(original):
            events.emit(
                "Sanitized message history",
                stage="sanitize_history",
                removed=len(original) - len(msgs),
            )
            new_state["messages"] = msgs
            summarized = True
        return new_state, summarized

    def _ensure_system_first(
        self,
        messages: Sequence[BaseMessage],
        system_prompt: str | None,
        *,
        replace: bool = False,
    ) -> list[BaseMessage]:
        msgs = list(messages)
        if system_prompt is None:
            return msgs
        if msgs and isinstance(msgs[0], SystemMessage):
            if replace and msgs[0].content != system_prompt:
                msgs[0] = SystemMessage(content=system_prompt, id=msgs[0].id)
            return msgs
        return [SystemMessage(content=system_prompt)] + msgs

    def _patch_dangling(
        self,
        state: Mapping[str, Any],
        summarized: bool = False,
        config: RunnableConfig | None = None,
    ) -> tuple[Mapping[str, Any], bool]:
        """Patch missing tool responses in a LangGraph message state.

        Tool-calling chat models require every AI tool call to be followed by a
        corresponding ``ToolMessage``. If a tool result is missing because a run
        timed out, failed, or summarization trimmed it away, later model calls can
        fail with provider-side message validation errors. This utility inserts a
        synthetic response for any dangling tool call ID and truncates extremely
        large tool messages.

        Args:
            state: A graph state containing a ``messages`` list.
            summarized: Existing overwrite flag from prior state mutation.

        Returns:
            ``(new_state, full_overwrite_required)``. When the boolean is true,
            callers using an ``add_messages`` reducer should return an
            ``Overwrite(new_state["messages"])`` update rather than appending.
        """
        if "messages" not in state:
            return state, summarized

        new_state = deepcopy(state)
        events = self.events(config)
        dangling_response = (
            "Response Not Found from tool. "
            "May have timed out or been forgotten due to summarization."
        )

        pending_tool_ids: list[str] = []
        for msg in new_state["messages"]:
            if isinstance(msg, ToolMessage):
                if (
                    self.max_single_tool_message_tokens is not None
                    and count_tokens_approximately([msg])
                    > self.max_single_tool_message_tokens
                ):
                    msg.content = "Message too long - truncated."
                    summarized = True
                try:
                    pending_tool_ids.remove(msg.tool_call_id)
                except ValueError:
                    # Orphan tool messages are not ideal, but do not create the
                    # provider-side validation error that missing tool responses do.
                    pass
                continue

            pending_tool_ids.extend(self._message_tool_call_ids(msg))

        if pending_tool_ids:
            summarized = True
            events.emit(
                "Dangling tool calls patched",
                stage="dangling_tool_calls",
                tool_call_ids=list(pending_tool_ids),
            )
            for tool_id in pending_tool_ids:
                for msg_ind, msg in enumerate(new_state["messages"]):
                    if tool_id in self._message_tool_call_ids(msg):
                        new_state["messages"].insert(
                            msg_ind + 1,
                            ToolMessage(
                                content=dangling_response,
                                tool_call_id=tool_id,
                            ),
                        )
                        break

        return new_state, summarized

    def _summarize_context(
        self,
        state: Mapping[str, Any],
        *,
        system_prompt: str | None = None,
        summary_prompt: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[Mapping[str, Any], bool]:
        """Summarize long ``state['messages']`` histories.

        This preserves the first/system message and the last
        ``self.messages_to_keep`` messages, replacing the middle of the
        transcript with an LLM-generated summary when the approximate token count
        reaches 70% of the model context window, or
        ``self.tokens_before_summarize`` when that window is unknown.

        The method also avoids creating dangling tool-call pairs when possible by
        moving tool responses that correspond to summarized-away calls into the
        summarized block.

        The summary request goes through ``tool_llm`` when the agent has one
        and through ``llm`` otherwise, so plain ``BaseAgent`` subclasses can
        summarize too.
        """
        if "messages" not in state:
            return state, False

        new_state = deepcopy(state)
        events = self.events(config)
        messages = list(new_state.get("messages") or [])
        if len(messages) <= 1:
            return new_state, False

        if isinstance(messages[0], SystemMessage):
            head, body = [messages[0]], messages[1:]
        else:
            head, body = [], messages

        tokens_before = count_tokens_approximately(
            messages, tools=list(getattr(self, "tools", {}).values())
        )
        threshold = summarization_threshold(
            self.llm, self.tokens_before_summarize
        )
        if tokens_before < threshold:
            return new_state, False

        events.emit(
            message=f"Summarizing ~{tokens_before} token history to compress context",
            stage="summarize_context",
        )
        keep_count = max(0, int(self.messages_to_keep or 0))
        if keep_count == 0:
            to_summarize, to_keep = body, []
        elif len(body) > keep_count:
            to_summarize, to_keep = body[:-keep_count], body[-keep_count:]
        else:
            # Nothing can be summarized while still preserving the requested tail.
            return new_state, False

        open_ids: set[str] = set()
        for msg in to_summarize:
            open_ids.update(self._message_tool_call_ids(msg))
            if isinstance(msg, ToolMessage):
                open_ids.discard(msg.tool_call_id)
        while (
            to_keep
            and isinstance(to_keep[0], ToolMessage)
            and to_keep[0].tool_call_id in open_ids
        ):
            moved = to_keep.pop(0)
            to_summarize.append(moved)
            open_ids.discard(moved.tool_call_id)
        if open_ids:
            events.emit(
                "Dangling tool calls found during summarization",
                stage="summarize_context",
                tool_call_ids=list(open_ids),
            )

        # Transcript goes over as plain text in one user turn, so no
        # tool-call structure crosses the wire and slice boundaries can't
        # violate any provider's turn-ordering rules.
        request = [
            SystemMessage(content=summary_prompt or _SUMMARIZE_INSTRUCTIONS),
            HumanMessage(
                content=(
                    "The conversation to summarize is below.\n\n<conversation>\n"
                    + _render_for_summary(to_summarize)
                    + "\n</conversation>\n\nSummarize that conversation per your instruction."
                )
            ),
        ]
        summarizer = getattr(self, "tool_llm", None) or self.llm
        summary = summarizer.invoke(request)

        if system_prompt is not None:
            head = [SystemMessage(content=system_prompt)]

        # The summary lands as human-role context rather than as the raw
        # assistant message: with messages_to_keep=0 (and in the absorbed
        # tool-tail case) the summary is the final message, and a history
        # ending on an assistant turn is a request shape Claude 4.6 and
        # newer reject (issue 296).
        summary_message = HumanMessage(
            content="[Summary of the earlier conversation]\n"
            + _message_text(summary),
            id=summary.id,
        )
        new_state["messages"] = head + [summary_message] + to_keep
        return new_state, True

    def prepare_messages_context(
        self,
        state: Mapping[str, Any],
        *,
        system_prompt: str | None = None,
        summary_prompt: str | None = None,
        patch_dangling: bool = True,
        summarize: bool = True,
        sanitize: bool = True,
    ) -> tuple[Mapping[str, Any], bool]:
        """Apply standard message-history maintenance before an LLM call.

        Returns ``(new_state, full_overwrite_required)``. Graph nodes using the
        ``add_messages`` reducer should use :meth:`messages_update` with the
        returned flag to avoid appending duplicate history after summarization or
        dangling-tool patching.
        """
        new_state = deepcopy(state)
        full_overwrite = False
        if sanitize:
            new_state, full_overwrite = self._sanitize_history(
                new_state, full_overwrite
            )
        if system_prompt is not None and "messages" in new_state:
            fixed = self._ensure_system_first(
                new_state["messages"], system_prompt
            )
            if fixed is not new_state["messages"] and len(fixed) != len(
                new_state["messages"]
            ):
                new_state["messages"] = fixed
                full_overwrite = True
        if summarize:
            new_state, full_overwrite = self._summarize_context(
                new_state,
                system_prompt=system_prompt,
                summary_prompt=summary_prompt,
            )
        if patch_dangling:
            new_state, full_overwrite = self._patch_dangling(
                new_state, full_overwrite
            )
        return new_state, full_overwrite

    def messages_update(
        self,
        state: Mapping[str, Any],
        message_update: Any,
        *,
        full_overwrite: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a LangGraph-safe update for states with a ``messages`` reducer."""
        update = dict(extra or {})
        if full_overwrite:
            messages = list(state.get("messages") or [])
            if message_update is None:
                update["messages"] = Overwrite(messages)
            else:
                if isinstance(message_update, list):
                    messages.extend(message_update)
                else:
                    messages.append(message_update)
                update["messages"] = Overwrite(messages)
        else:
            update["messages"] = message_update
        return update

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        """Normalizes various input formats into a standardized mapping.

        This method converts different input types into a consistent dictionary format
        that can be processed by the agent. String inputs are wrapped as messages, while
        mappings are passed through unchanged.

        Args:
            inputs: The input to normalize. Can be a string (which will be converted to a
                message) or a mapping (which will be returned as-is).

        Returns:
            A mapping containing the normalized inputs, with keys appropriate for agent
            processing.

        Raises:
            TypeError: If the input type is not supported (neither string nor mapping).
        """
        if isinstance(inputs, str):
            # Adjust to your message type
            return {"messages": [HumanMessage(content=inputs)]}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    @cached_property
    def compiled_graph(self) -> CompiledStateGraph:
        """Return the sync compiled StateGraph application for the agent."""
        graph = self.build_graph()
        compiled = graph.compile(
            checkpointer=self.checkpointer,
            store=self.storage,
        ).with_config({"recursion_limit": 50000})
        return self._finalize_graph(compiled)

    async def _aget_async_compiled_graph(self) -> CompiledStateGraph:
        """Return an async-compatible compiled graph for the agent.

        LangGraph SQLite checkpointers are not interchangeable between sync
        and async execution. A graph compiled with ``SqliteSaver`` cannot be
        awaited with ``ainvoke``. Build a separate graph using async SQLite
        persistence resources for async runs.
        """
        if self._async_compiled_graph is None:
            graph = self.build_graph()
            compiled = graph.compile(
                checkpointer=await self._aget_async_checkpointer(),
                store=await self._aget_async_storage(),
            ).with_config({"recursion_limit": 50000})
            self._async_compiled_graph = self._finalize_graph(compiled)
        return self._async_compiled_graph

    async def _aget_async_checkpointer(self) -> BaseCheckpointSaver | None:
        """Return an async-compatible checkpointer for async graph runs."""
        if self.checkpointer is None:
            return None

        if isinstance(self.checkpointer, AsyncSqliteSaver):
            return self.checkpointer

        if self._async_checkpointer is not None:
            return self._async_checkpointer

        if isinstance(self.checkpointer, SqliteSaver):
            if self._checkpointer_was_provided:
                raise RuntimeError(
                    "This agent was configured with a synchronous SqliteSaver "
                    "checkpointer. Async agent execution requires an async "
                    "checkpointer, such as AsyncSqliteSaver, or URSA-managed "
                    "persistence via agent_name."
                )
            self._async_checkpointer = await Checkpointer.async_from_workspace(
                self.den
            )
            return self._async_checkpointer

        if self._checkpointer_has_async_methods(self.checkpointer):
            return self.checkpointer

        raise RuntimeError(
            "The configured checkpointer does not appear to support LangGraph "
            "async checkpoint methods. Use an async checkpointer for "
            "agent.ainvoke(...), or use URSA-managed persistence via "
            "agent_name."
        )

    @staticmethod
    def _checkpointer_has_async_methods(checkpointer: Any) -> bool:
        required = ("aget_tuple", "alist", "aput", "aput_writes")
        return all(
            inspect.iscoroutinefunction(getattr(type(checkpointer), name, None))
            or inspect.isasyncgenfunction(
                getattr(type(checkpointer), name, None)
            )
            for name in required
        )

    @cached_property
    def storage(self) -> BaseStore:
        """Create a sync SQLite-backed LangGraph store."""
        store_path = self.den / "graph_store.sqlite"
        conn = sqlite3.connect(
            store_path, check_same_thread=False, isolation_level=None
        )
        store = SqliteStore(conn)
        # Hierarchy members share this store across processes. LangGraph's
        # migration check and insert are separate statements, so concurrent
        # first-use can otherwise insert the same migration version twice.
        with FileLock(f"{store_path}.setup.lock", mode=0o600):
            store.setup()
        self.hook_storage_setup(store)
        return store

    async def _aget_async_storage(self) -> BaseStore:
        """Create an async SQLite-backed LangGraph store."""
        if self._async_storage is None:
            import aiosqlite

            store_path = self.den / "graph_store.sqlite"
            conn = await aiosqlite.connect(store_path, isolation_level=None)
            store = AsyncSqliteStore(conn)
            async with AsyncFileLock(f"{store_path}.setup.lock", mode=0o600):
                await store.setup()
            await self.ahook_storage_setup(store)
            self._async_storage = store
        return self._async_storage

    def hook_storage_setup(self, store: BaseStore) -> None:
        pass

    async def ahook_storage_setup(self, store: BaseStore) -> None:
        pass

    def close(self) -> None:
        """Close sync SQLite resources owned by this agent when possible."""
        if "storage" in self.__dict__:
            self.storage.conn.close()
        if isinstance(self.checkpointer, SqliteSaver):
            self.checkpointer.conn.close()

    async def aclose(self) -> None:
        """Close async SQLite resources owned by this agent when possible."""
        if self._async_storage is not None:
            storage_conn = self._async_storage.conn
            await self._async_storage.__aexit__(None, None, None)
            await storage_conn.close()
            # ``aiosqlite.Connection`` is a non-daemon ``threading.Thread``.
            # ``close()`` only finalizes queued queries and signals the worker
            # to stop; it does NOT join the thread, which stays alive. Without
            # joining, each async invoke leaks a live non-daemon thread that
            # ultimately blocks interpreter shutdown (the TUI "hangs" on quit).
            await asyncio.to_thread(storage_conn.join)
            self._async_storage = None
            self._async_compiled_graph = None
        if isinstance(self._async_checkpointer, AsyncSqliteSaver):
            checkpointer_conn = self._async_checkpointer.conn
            await checkpointer_conn.close()
            await asyncio.to_thread(checkpointer_conn.join)
            self._async_checkpointer = None
            self._async_compiled_graph = None

    @final
    def build_graph(self) -> StateGraph:
        """Build and return the StateGraph backing this agent."""
        self.graph = StateGraph(
            self.state_type,
            context_schema=AgentContext,
        )
        self._build_graph()
        return self.graph

    @abstractmethod
    def _build_graph(self) -> None:
        """Construct the StateGraph for this agent without compiling.

        Called during `__post_init__()` after the Agent has been fully
        Initialized (`__post_init__` is called after `__init__`) to
        instantiate `self.graph`

        Agents should implement this to define their their behavior.

        Agents should treat `self.graph` as read-only
        """
        ...

    def _finalize_graph(
        self, graph_app: CompiledStateGraph
    ) -> CompiledStateGraph:
        """Hook for subclasses to wrap or modify the compiled graph."""
        return graph_app

    def _tool_is_async_only(self, tool: Any) -> bool:
        """Return True for tools that can only be invoked asynchronously.

        MCP tools are commonly exposed as StructuredTool instances with a
        coroutine implementation but no synchronous function implementation.
        Those raise errors like:

            "StructuredTool does not support sync invocation."

        when called via `.invoke()`.
        """

        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        return func is None and coroutine is not None

    def _has_async_only_tools(self) -> bool:
        tools_obj = getattr(self, "tools", None)
        if not tools_obj:
            return False

        try:
            tool_iter = (
                tools_obj.values() if isinstance(tools_obj, dict) else tools_obj
            )
        except Exception:
            return False

        return any(self._tool_is_async_only(t) for t in tool_iter)

    def _invoke(self, input, **config):
        config = self.build_config(**config)

        # If we have async-only tools (e.g. MCP StructuredTools), we must run the
        # graph via `ainvoke` so ToolNode dispatches tools asynchronously.
        if self._has_async_only_tools():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._ainvoke(input, **config))

            raise RuntimeError(
                "This agent has async-only tools, but `.invoke()` was called "
                "from an async context (a running event loop was detected). "
                "Use `await agent.ainvoke(...)` instead."
            )

        try:
            graph = self.compiled_graph
            result = graph.invoke(input, config=config, context=self.context)
            self._prune_checkpoints_after_terminal_success(graph, config)
            return result
        except Exception as e:
            # Fallback: if a tool raises the canonical sync-invoke error, retry
            # with ainvoke for backwards compatibility.
            if "does not support sync invocation" in str(e):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(self._ainvoke(input, **config))
            raise

    async def _ainvoke(self, input, **config):
        config = self.build_config(**config)
        graph = await self._aget_async_compiled_graph()
        try:
            result = await graph.ainvoke(
                input, config=config, context=self.context
            )
            await self._aprune_checkpoints_after_terminal_success(graph, config)
            return result
        finally:
            await self.aclose()

    def _stream(self, input, **config):
        config = self.build_config(**config)
        graph = self.compiled_graph
        yield from graph.stream(input, config=config, context=self.context)
        self._prune_checkpoints_after_terminal_success(graph, config)

    def _prune_checkpoints_after_terminal_success(
        self, graph: CompiledStateGraph, config: RunnableConfig
    ) -> None:
        """Prune completed history for URSA-managed sync SQLite agents."""
        if not self._checkpoint_retention_enabled:
            return
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, SqliteSaver):
            return
        try:
            snapshot = graph.get_state(config)
            if not is_terminal_checkpoint(snapshot):
                return
            thread_id = str(
                config.get("configurable", {}).get("thread_id", self.thread_id)
            )
            result = prune_sqlite_checkpoints(checkpointer, thread_id)
            if result.pruned:
                logger.info(
                    "Pruned %d checkpoints and %d writes for thread %s "
                    "(%d -> %d bytes)",
                    result.checkpoints_deleted,
                    result.writes_deleted,
                    thread_id,
                    result.size_before,
                    result.size_after,
                )
        except Exception:
            # Retention is maintenance. It must not convert a successful agent
            # result into a failed invocation.
            logger.warning("Checkpoint pruning failed", exc_info=True)

    async def _aprune_checkpoints_after_terminal_success(
        self, graph: CompiledStateGraph, config: RunnableConfig
    ) -> None:
        """Prune completed history for URSA-managed async SQLite agents."""
        if not self._checkpoint_retention_enabled:
            return
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, AsyncSqliteSaver):
            return
        try:
            snapshot = await graph.aget_state(config)
            if not is_terminal_checkpoint(snapshot):
                return
            thread_id = str(
                config.get("configurable", {}).get("thread_id", self.thread_id)
            )
            result = await aprune_sqlite_checkpoints(checkpointer, thread_id)
            if result.pruned:
                logger.info(
                    "Pruned %d checkpoints and %d writes for thread %s "
                    "(%d -> %d bytes)",
                    result.checkpoints_deleted,
                    result.writes_deleted,
                    thread_id,
                    result.size_before,
                    result.size_after,
                )
        except Exception:
            logger.warning("Checkpoint pruning failed", exc_info=True)

    def __call__(self, inputs: InputLike, /, **kwargs: Any) -> Any:
        """Specify calling behavior for class instance."""
        return self.invoke(inputs, **kwargs)

    # Runtime enforcement: forbid subclasses from overriding invoke
    def __init_subclass__(cls, **kwargs):
        """Ensure subclass does not override key method."""
        super().__init_subclass__(**kwargs)
        if "invoke" in cls.__dict__:
            err_msg = (
                f"{cls.__name__} must not override BaseAgent.invoke(); "
                "implement _invoke() only."
            )
            raise TypeError(err_msg)

        cls._warn_if_state_unregistered()

        # Init graph after subclass has been fully constructed
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.__post_init__()

        cls.__init__ = __init__

    @classmethod
    def _warn_if_state_unregistered(cls):
        """Warn when a subclass declares a typed state it never registers.

        The graph compiles from ``state_type`` (default ``dict``), so a
        TypedDict named only in the ``BaseAgent[X]`` generic parameter or
        the legacy ``agent_state`` attribute contributes nothing and its
        reducers silently never apply.
        """
        import warnings
        from typing import TypeVar, get_args

        declared = cls.__dict__.get("agent_state")
        if declared is None:
            for base in cls.__dict__.get("__orig_bases__", ()):
                for arg in get_args(base):
                    if isinstance(arg, TypeVar):
                        continue
                    if (
                        isinstance(arg, type)
                        and arg is not dict
                        and issubclass(arg, dict)
                        and hasattr(arg, "__annotations__")
                    ):
                        declared = arg
                        break
                if declared is not None:
                    break
        if declared is None:
            return
        registered = getattr(cls, "state_type", dict)
        if registered is declared:
            return
        warnings.warn(
            f"{cls.__name__} declares state "
            f"{getattr(declared, '__name__', declared)} but its graph "
            f"compiles with "
            f"{getattr(registered, '__name__', registered)}; set "
            f"`state_type = {getattr(declared, '__name__', declared)}` "
            f"so its reducers apply.",
            UnregisteredAgentStateWarning,
            stacklevel=3,
        )

    def __post_init__(self):
        self.build_graph()

    def stream(
        self,
        inputs: InputLike,
        config: Any | None = None,  # allow positional/keyword like LangGraph
        /,
        *,
        raw_debug: bool = False,
        save_json: bool | None = None,
        save_otel: bool | None = None,
        metrics_path: str | None = None,
        save_raw_snapshot: bool | None = None,
        save_raw_records: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Streams agent responses with telemetry tracking.

        This method serves as the public streaming entry point for agent interactions.
        It wraps the actual streaming implementation with telemetry tracking to capture
        metrics and debugging information.

        Args:
            inputs: The input to process, which will be normalized internally.
            config: Optional configuration for the agent, compatible with LangGraph
                positional/keyword argument style.
            raw_debug: If True, renders raw debug information in telemetry output.
            save_json: If True, saves telemetry data as JSON.
            metrics_path: Optional file path where metrics should be saved.
            save_raw_snapshot: If True, saves raw snapshot data in telemetry.
            save_raw_records: If True, saves raw record data in telemetry.
            **kwargs: Additional keyword arguments passed to the streaming
                implementation.

        Returns:
            An iterator yielding the agent's responses.

        Note:
            This method tracks invocation depth to properly handle nested agent calls
            and ensure telemetry is only rendered once at the top level.
        """
        depth = _INVOKE_DEPTH.get() + 1
        depth_token = _INVOKE_DEPTH.set(depth)

        try:
            # Start telemetry tracking for top-level invocations only
            if depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # Normalize inputs and delegate to the actual streaming implementation
            normalized = self._normalize_inputs(inputs)
            stream_config = dict(config or {})
            yield from self._stream(normalized, **stream_config, **kwargs)

        finally:
            _INVOKE_DEPTH.reset(depth_token)
            if depth == 1:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    def _default_node_tags(
        self, name: str, extra: Sequence[str] | None = None
    ) -> list[str]:
        """Generate default tags for a graph node.

        Args:
            name: The name of the node.
            extra: Optional sequence of additional tags to include.

        Returns:
            list[str]: A list of tags for the node, including the agent name, 'graph',
                the node name, and any extra tags provided.
        """
        # Start with standard tags: agent name, graph indicator, and node name
        tags = [self.name, "graph", name]

        # Add any extra tags if provided
        if extra:
            tags.extend(extra)

        return tags

    def _node_cfg(self, name: str, *extra_tags: str) -> dict:
        """Build a consistent configuration for a node/runnable.

        Creates a configuration dict that can be reapplied after operations like
        .map(), subgraph compile, etc.

        Args:
            name: The name of the node.
            *extra_tags: Additional tags to include in the node configuration.

        Returns:
            dict: A configuration dictionary with run_name, tags, and metadata.
        """
        # Determine the namespace - use first extra tag if available, otherwise
        # convert agent name to snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Combine all tags: agent name, graph indicator, node name, and any extra tags
        tags = [self.name, "graph", name, *extra_tags]

        # Return the complete configuration dictionary
        return dict(
            run_name="node",  # keep "node:" prefixing in the timer
            tags=tags,
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )

    def ns(self, runnable_or_fn, name: str, *extra_tags: str):
        """Return a runnable with node configuration applied.

        Applies the agent's node configuration to a runnable or callable. This method
        should be called again after operations like .map() or subgraph .compile() as
        these operations may drop configuration.

        Args:
            runnable_or_fn: A runnable or callable to configure.
            name: The name to assign to this node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured runnable with the agent's node configuration applied.
        """
        # Convert input to a runnable if it's not already one
        r = coerce_to_runnable(runnable_or_fn, name=name, trace=True)
        # Apply node configuration and return the configured runnable
        return r.with_config(**self._node_cfg(name, *extra_tags))

    def _wrap_node(self, fn_or_runnable, name: str, *extra_tags: str):
        """Wrap a function or runnable as a node in the graph.

        This is a convenience wrapper around the ns() method.

        Args:
            fn_or_runnable: A function or runnable to wrap as a node.
            name: The name to assign to this node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured runnable with the agent's node configuration applied.
        """
        return self.ns(fn_or_runnable, name, *extra_tags)

    def _wrap_cond(self, fn: Any, name: str, *extra_tags: str):
        """Wrap a conditional function as a routing node in the graph.

        Creates a runnable lambda with routing-specific configuration.

        Args:
            fn: The conditional function to wrap.
            name: The name of the routing node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured RunnableLambda with routing-specific metadata.
        """
        # Use the first extra tag as namespace, or fall back to agent name in snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Create and return a configured RunnableLambda for routing
        return RunnableLambda(fn).with_config(
            run_name="node",
            tags=[
                self.name,
                "graph",
                f"route:{name}",
                *extra_tags,
            ],
            metadata={
                "langgraph_node": f"route:{name}",
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )

    def _named(self, runnable: Any, name: str, *extra_tags: str):
        """Apply a specific name and configuration to a runnable.

        Configures a runnable with a specific name and the agent's metadata.

        Args:
            runnable: The runnable to configure.
            name: The name to assign to this runnable.
            *extra_tags: Additional tags to apply to the runnable.

        Returns:
            A configured runnable with the specified name and agent metadata.
        """
        # Use the first extra tag as namespace, or fall back to agent name in snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Apply configuration and return the configured runnable
        return runnable.with_config(
            run_name=name,
            tags=[self.name, "graph", name, *extra_tags],
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )


def _default_tool_error_handler(e: Exception):
    if isinstance(e, ToolInvocationError):
        return e.message
    elif isinstance(e, ToolException):
        return str(e)
    raise e


class AgentWithTools:
    """Mixin that equips an agent with LangGraph tools management."""

    def __init__(
        self,
        *args,
        tools: list[BaseTool] | dict[str, BaseTool] | None = None,
        safe_codes: list[str] | None = None,
        handle_tool_errors=_default_tool_error_handler,
        **kwargs,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.safe_codes = set(safe_codes or [])
        self.handle_tool_errors = handle_tool_errors
        self.tool_node = ToolNode([])
        self._apply_tools(tools, rebuild_graph=False)
        super().__init__(*args, **kwargs)
        self.tool_llm = self.llm.model_copy()

        if getattr(self, "rag_tools", ()):
            from ursa.rag.tools import build_rag_tools

            rag_tools = build_rag_tools(
                names=self.rag_tools,
                group=self.rag_tool_group or self.group or "default",
                llm=self.llm,
                embedding=getattr(self, "rag_tool_embedding", None),
                return_k=self.rag_tool_return_k,
            )
            if rag_tools:
                merged = dict(self._tools)
                merged.update({tool.name: tool for tool in rag_tools})
                self._apply_tools(merged, rebuild_graph=False)

    def __post_init__(self):
        super().__post_init__()

    def hook_storage_setup(self, store: BaseStore) -> None:
        if store is None:
            return
        for safe_code in self.safe_codes:
            store.put(
                ("workspace", "safe_codes"),
                safe_code,
                {},
            )

    async def ahook_storage_setup(self, store: BaseStore) -> None:
        if store is None:
            return
        for safe_code in self.safe_codes:
            await store.aput(
                ("workspace", "safe_codes"),
                safe_code,
                {},
            )

    @property
    def tools(self) -> dict[str, BaseTool]:
        return dict(self._tools)

    @tools.setter
    def tools(self, tools: dict[str, BaseTool] | list[BaseTool] | None):
        self._apply_tools(tools)

    @staticmethod
    def bind_tools_prefer_parallel(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
    ) -> Any:
        """Bind tools while explicitly enabling OpenAI parallel tool calls.

        LangGraph's :class:`~langgraph.prebuilt.ToolNode` already executes all
        calls from one model turn concurrently. OpenAI models also expose a
        request-level ``parallel_tool_calls`` switch; set it explicitly so a
        provider default cannot accidentally disable that behavior. Other
        LangChain providers keep their existing ``bind_tools`` call and
        semantics.
        """
        from langchain_openai.chat_models.base import BaseChatOpenAI

        underlying_model = llm
        # BaseAgent binds URSA's request headers before this mixin sees the
        # model. LangChain represents that as a RunnableBinding, whose
        # ``bound`` member is the actual provider model.
        while (
            bound := getattr(underlying_model, "bound", None)
        ) is not None and bound is not underlying_model:
            underlying_model = bound
        if isinstance(underlying_model, BaseChatOpenAI):
            return llm.bind_tools(tools, parallel_tool_calls=True)
        return llm.bind_tools(tools)

    def add_tool(self, tools: BaseTool | list[BaseTool]) -> None:
        bundle = tools if isinstance(tools, list) else [tools]
        merged = dict(self._tools)
        merged.update({tool.name: tool for tool in bundle})
        self._apply_tools(merged)

    async def add_mcp_tools(
        self,
        client: MultiServerMCPClient,
        tool_name: None | str | list[str] = None,
    ) -> dict[str, str]:
        """Add tools from an MCP client to the agent

        Args:
           client: the MCP client to add tools from
           tool_name: if provided, only add named tools

        Returns:
            MCP server names keyed by the attached tool name. Clients that do
            not expose named connections return an empty mapping.
        """

        def discover_and_apply() -> dict[str, str]:
            # MCP adapters perform synchronous session/schema setup around
            # their async I/O, and applying tools rebuilds the graph. Own one
            # worker boundary for the complete operation so every caller—not
            # only Textual—keeps its event loop responsive.
            tools, tool_sources = asyncio.run(
                load_mcp_tools_with_sources(client)
            )
            selected = tool_name
            if selected is not None:
                selected = (
                    selected if isinstance(selected, list) else [selected]
                )
                tools = [tool for tool in tools if tool.name in selected]
                attached_names = {tool.name for tool in tools}
                tool_sources = {
                    name: server
                    for name, server in tool_sources.items()
                    if name in attached_names
                }
            self.add_tool(tools)
            return tool_sources

        return await asyncio.to_thread(discover_and_apply)

    def remove_tool(self, tool_names: str | list[str]) -> None:
        names = tool_names if isinstance(tool_names, list) else [tool_names]
        trimmed = {
            name: tool
            for name, tool in self._tools.items()
            if name not in names
        }
        self._apply_tools(trimmed)

    def _apply_tools(
        self,
        tools: dict[str, BaseTool] | list[BaseTool] | None,
        *,
        rebuild_graph: bool = True,
    ) -> None:
        if tools is None:
            mapping: dict[str, BaseTool] = {}
        elif isinstance(tools, dict):
            mapping = dict(tools)
        else:
            mapping = {tool.name: tool for tool in tools}

        self._tools = mapping
        self.tool_node = ToolNode(
            list(self._tools.values()),
            handle_tool_errors=self.handle_tool_errors,
        )

        if rebuild_graph and hasattr(self, "build_graph"):
            self.__dict__.pop("compiled_graph", None)
            if hasattr(self, "graph"):
                self.build_graph()
