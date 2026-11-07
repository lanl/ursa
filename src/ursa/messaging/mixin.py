"""Per-agent inbox lifecycle and local agent hierarchies."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from contextlib import asynccontextmanager
from contextvars import Context, copy_context
from threading import RLock
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)

from ursa.util.async_utils import maybe_await

from .coordination import (
    MessageWait,
    check_capacity,
    resolve_message_routes,
    wait_dependency,
    wake_waiters,
)
from .messages import (
    AgentMessage,
    InputLike,
    _QueuedAgentMessage,
    agent_message_wait_result,
    attach_agent_message_instructions,
)

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel
    from langchain.embeddings import Embeddings

    from ursa.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AsyncMessageMixin:
    """Actor-style async messaging and dynamic agent hierarchy support.

    Each agent processes one inbox message at a time. Separate agents can run
    concurrently, so parents and siblings can exchange messages without sharing
    mutable invocation state.
    """

    _SUBAGENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def __init__(
        self,
        *args: Any,
        subagent_llm: BaseChatModel | None = None,
        subagent_embedding: Embeddings | None = None,
        subagent_max_agents: int | None = 32,
        subagent_max_depth: int | None = 3,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("subagent_max_agents", subagent_max_agents),
            ("subagent_max_depth", subagent_max_depth),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        self._subagent_llm = subagent_llm
        self._subagent_embedding = subagent_embedding
        self._subagent_max_agents = subagent_max_agents
        self._subagent_max_depth = subagent_max_depth
        self._parent_agent: BaseAgent | None = None
        self._subagents: dict[str, BaseAgent] = {}
        self._subagent_stop_tasks: set[asyncio.Task[None]] = set()
        self._subagent_hierarchy_lock = RLock()
        self._hierarchy_name: str | None = None
        self._hierarchy_stopping = False
        self._stopping_agent_path: str | None = None
        self._message_inbox: (
            asyncio.Queue[_QueuedAgentMessage | None] | None
        ) = None
        self._message_loop_task: asyncio.Task[None] | None = None
        self._message_event_loop: asyncio.AbstractEventLoop | None = None
        self._message_loop_stopping = False
        self._current_agent_message: AgentMessage | None = None
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
        from ursa.agents.base import _to_snake

        return _to_snake(self.__class__.__name__)

    @property
    def agent_path(self) -> str:
        """Return this agent's slash-delimited path from the root."""
        if runtime_path := getattr(self, "_runtime_agent_path", None):
            return str(runtime_path)
        names = [agent.hierarchy_name for agent in self._ancestors()]
        return "/".join(reversed(names))

    def _ancestors(self) -> Iterator[BaseAgent]:
        """Walk from this agent to its root, rejecting hierarchy cycles."""
        current: BaseAgent | None = self  # type: ignore[assignment]
        seen: set[int] = set()
        while current is not None:
            if id(current) in seen:
                raise RuntimeError("Cycle detected in agent hierarchy")
            seen.add(id(current))
            yield current
            current = current.parent_agent

    @property
    def root_agent(self) -> BaseAgent:
        return next(
            agent for agent in self._ancestors() if agent.parent_agent is None
        )

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
        message = AgentMessage(
            content=content,
            sender=sender,
            recipient=self.agent_path,
            metadata=metadata or {},
            config=config or {},
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
        message = AgentMessage(
            content=content,
            sender=sender,
            recipient=self.agent_path,
            metadata=metadata or {},
            config=config or {},
            system_message=system_message,
        )
        response = asyncio.get_running_loop().create_future()
        await self._enqueue_agent_message(
            message,
            response=response,
            on_start=on_start,
        )
        return await response

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

    def _message_context(self) -> Context:
        """Copy caller context for a new inbox; subclasses may reset run state."""
        return copy_context()

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
        task = loop.create_task(
            self._run_message_loop(),
            name=f"ursa-message-loop:{self.agent_path}",
            # A child inbox may be started from inside its parent's invocation.
            # It is an independent actor, so it must not inherit that parent's
            # invocation depth. Preserve unrelated context such as Textual's
            # active-app binding, which UI callbacks need while rendering.
            context=self._message_context(),
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
                await inbox.put(None)
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
        waiter = MessageWait(targets=tuple(targets))
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

    async def _notify_message_callback(
        self, item: _QueuedAgentMessage, phase: str, *args: Any
    ) -> None:
        callback = getattr(item, f"on_{phase}")
        if callback is not None:
            try:
                await maybe_await(callback(item.message, *args))
            except Exception:
                logger.exception(
                    f"Agent %s message-{phase} callback failed for %s",
                    self.agent_path,
                    item.message.message_id,
                )

    async def _run_message_loop(self) -> None:
        inbox = self._message_inbox
        if inbox is None:  # pragma: no cover
            return
        try:
            while True:
                item = await inbox.get()
                if item is None:
                    inbox.task_done()
                    break
                self._current_agent_message = item.message
                await self._notify_message_callback(item, "start")
                handler = asyncio.create_task(
                    self.ahandle_message(item.message)
                )
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
                    await self._notify_message_callback(item, "end", succeeded)
                    self._current_agent_message = None
                    inbox.task_done()
        finally:
            self._current_agent_message = None

    def _cancel_queued_responses(self) -> None:
        inbox = self._message_inbox
        if inbox is None:
            return
        while not inbox.empty():
            item = inbox.get_nowait()
            if item is not None:
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
        lock = self._async_invocation_lock
        if lock is None or self._async_invocation_loop is not loop:
            if lock is not None and lock.locked():
                raise RuntimeError(
                    "Agent is already running in another event loop"
                )
            lock = asyncio.Lock()
            self._async_invocation_lock = lock
            self._async_invocation_loop = loop
        return lock

    @asynccontextmanager
    async def _async_invocation(self):
        """Serialize calls per agent, allowing reentry by the owning task."""
        current = asyncio.current_task()
        if self._async_invocation_owner is current:
            yield
            return
        async with self._get_async_invocation_lock():
            self._async_invocation_owner = current
            try:
                yield
            finally:
                self._async_invocation_owner = None

    @property
    def current_message(self) -> AgentMessage | None:
        """Return the inbox envelope currently being handled."""
        return self._current_agent_message

    @property
    def is_working(self) -> bool:
        """Whether this agent is currently handling an invocation."""
        task = self._async_invocation_owner
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
        from ursa.agents.base import BaseAgent

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
            if any(ancestor is agent for ancestor in self._ancestors()):
                raise ValueError("Attaching this agent would create a cycle")
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
        from ursa.agents.base import BaseAgent

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
        from ursa.agents.base import BaseAgent

        self._ensure_hierarchy_accepting_messages()
        if isinstance(agent, BaseAgent):
            if agent_kwargs:
                raise TypeError("agent_kwargs cannot accompany an instance")
            return self.attach_subagent(name, agent)
        self._validate_subagent_capacity()
        factory = agent or self.__class__
        child = factory(**self._subagent_factory_kwargs(agent_kwargs))
        child = await maybe_await(child)
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
        from ursa.agents.base import BaseAgent

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

        def resolve(path: str) -> tuple[BaseAgent, str]:
            recipient = self.resolve_agent(path)
            return recipient, recipient.agent_path

        try:
            return [
                recipient
                for recipient, _path in resolve_message_routes(
                    self.agent_path,
                    routes,
                    [
                        member.agent_path
                        for member in self.root_agent.walk_agents()
                    ],
                    resolve,
                )
            ]
        except ValueError as exc:
            raise KeyError(str(exc)) from exc
