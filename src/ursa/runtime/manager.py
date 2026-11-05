"""Asynchronous, process-isolated agent execution.

The manager is deliberately independent of the CLI and dashboard.  It owns all
``multiprocessing.Process`` objects and all durable invocation-boundary state;
the child owns only the live agent and an asyncio event loop.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import multiprocessing
import os
import pickle
import tempfile
import threading
import traceback
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackHandler

LOGGER = logging.getLogger(__name__)


class AsyncAgent(Protocol):
    """The minimum interface an isolated agent must implement."""

    async def ainvoke(
        self, inputs: Any, config: Mapping[str, Any] | None = None
    ) -> Any: ...


AgentFactory: TypeAlias = Callable[[Mapping[str, Any], "AgentContext"], Any]


def _validate_agent_id(agent_id: str) -> None:
    if not agent_id.strip():
        raise ValueError("agent_id cannot be blank")


def _validate_factory_reference(factory: AgentFactory | str) -> None:
    if isinstance(factory, str) and ":" not in factory:
        raise ValueError(
            "string agent factories must use 'package.module:qualname'"
        )


class AgentStatus(StrEnum):
    REGISTERED = "registered"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, init=False)
class AgentSpec:
    """Everything needed to construct one independently configured agent.

    ``factory`` may be a top-level callable or an import string in
    ``"package.module:qualname"`` form.  Import strings are the most portable
    choice because the default multiprocessing start method is ``spawn``.
    """

    agent_id: str
    factory: AgentFactory | str
    config: Mapping[str, Any]
    initial_state: Any = None
    state_path: Path | None = None
    restart_on_failure: bool = False
    environment: Mapping[str, str] | None = None
    working_directory: Path | None = None

    def __init__(
        self,
        agent_id: str,
        factory: AgentFactory | str,
        config: Mapping[str, Any] | None = None,
        initial_state: Any = None,
        state_path: Path | None = None,
        restart_on_failure: bool = False,
        environment: Mapping[str, str] | None = None,
        working_directory: Path | None = None,
    ) -> None:
        _validate_agent_id(agent_id)
        _validate_factory_reference(factory)
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "factory", factory)
        object.__setattr__(self, "config", dict(config or {}))
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "state_path", state_path)
        object.__setattr__(self, "restart_on_failure", restart_on_failure)
        object.__setattr__(
            self,
            "environment",
            dict(environment) if environment is not None else None,
        )
        object.__setattr__(self, "working_directory", working_directory)


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """A message routed through the manager."""

    sender: str
    recipient: str
    payload: Any


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """Read-only manager view of one registered agent."""

    agent_id: str
    status: AgentStatus
    pid: int | None
    generation: int
    last_heartbeat: float | None
    heartbeat_age: float | None
    state: Any


class AgentRuntimeError(RuntimeError):
    """Base exception for process runtime failures."""


class AgentNotFoundError(AgentRuntimeError):
    pass


class AgentProcessExitedError(AgentRuntimeError):
    pass


class AgentInvocationError(AgentRuntimeError):
    """An operation raised inside the isolated process."""

    def __init__(
        self,
        agent_id: str,
        error_type: str,
        message: str,
        remote_traceback: str = "",
    ) -> None:
        super().__init__(f"{agent_id}: {error_type}: {message}")
        self.agent_id = agent_id
        self.error_type = error_type
        self.remote_message = message
        self.remote_traceback = remote_traceback


def _prime_spawn_resource_tracker() -> None:
    """Start CPython's tracker before UI libraries redirect ``sys.stderr``.

    Textual's console proxy reports ``fileno() == -1``.  If the spawn
    resource tracker is first started through that proxy, CPython passes the
    invalid descriptor to ``fork_exec`` and process creation fails.  Starting
    the tracker while the manager is constructed keeps process launch lazy
    while preserving the host's original standard streams.
    """
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()


def _load_factory(reference: AgentFactory | str) -> AgentFactory:
    if not isinstance(reference, str):
        return reference
    module_name, qualname = reference.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"Agent factory {reference!r} is not callable")
    return value


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _call_without_blocking(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    value = await asyncio.to_thread(function, *args, **kwargs)
    return await _maybe_await(value)


def _picklable(value: Any) -> Any:
    """Keep callback transport alive when a provider returns an odd object."""
    try:
        pickle.dumps(value)
    except Exception:
        if isinstance(value, Mapping):
            return {str(key): _picklable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_picklable(item) for item in value]
        if isinstance(value, BaseException):
            return RuntimeError(str(value))
        return repr(value)
    return value


def _freeze_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze mutable state at an agent-operation boundary."""
    return pickle.loads(pickle.dumps(dict(message)))


class AgentContext:
    """Async services made available to an agent process."""

    def __init__(
        self,
        agent_id: str,
        outgoing: asyncio.Queue[dict[str, Any]],
    ) -> None:
        self.agent_id = agent_id
        self._outgoing = outgoing
        self._responses: dict[str, asyncio.Future[Any]] = {}
        self._messages: asyncio.Queue[AgentMessage] = asyncio.Queue()

    async def _request(self, kind: str, **payload: Any) -> Any:
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._responses[request_id] = future
        await self._outgoing.put({
            "type": kind,
            "request_id": request_id,
            **payload,
        })
        try:
            return await future
        finally:
            self._responses.pop(request_id, None)

    async def send_message(self, recipient: str, payload: Any) -> None:
        """Send a message to another agent, or to ``"host"``."""
        await self._request(
            "route_message", recipient=recipient, payload=payload
        )

    async def receive_message(
        self, *, timeout: float | None = None
    ) -> AgentMessage:
        """Wait for a message delivered by the manager."""
        if timeout is None:
            return await self._messages.get()
        async with asyncio.timeout(timeout):
            return await self._messages.get()

    async def emit_event(
        self,
        request_id: str,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        await self._outgoing.put({
            "type": "callback",
            "request_id": request_id,
            "method": method,
            "args": _picklable(args),
            "kwargs": _picklable(kwargs),
        })

    def _resolve(self, message: Mapping[str, Any]) -> None:
        future = self._responses.get(str(message.get("request_id") or ""))
        if future is None or future.done():
            return
        if message.get("ok", False):
            future.set_result(message.get("result"))
        else:
            future.set_exception(
                AgentRuntimeError(str(message.get("error") or "request failed"))
            )

    async def _deliver(self, message: AgentMessage) -> None:
        await self._messages.put(message)


class _RemoteCallbackHandler(AsyncCallbackHandler):
    """Forward the callback families used by the CLI over the control pipe."""

    def __init__(self, context: AgentContext, request_id: str) -> None:
        self._context = context
        self._request_id = request_id

    async def _send(self, method: str, *args: Any, **kwargs: Any) -> None:
        await self._context.emit_event(self._request_id, method, args, kwargs)

    async def on_custom_event(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_custom_event", *args, **kwargs)

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_llm_start", *args, **kwargs)

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_chat_model_start", *args, **kwargs)

    async def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_llm_new_token", *args, **kwargs)

    async def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_llm_end", *args, **kwargs)

    async def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_llm_error", *args, **kwargs)

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_tool_start", *args, **kwargs)

    async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_tool_end", *args, **kwargs)

    async def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        await self._send("on_tool_error", *args, **kwargs)


class AgentProcess:
    """Child-side, fully asynchronous command dispatcher."""

    def __init__(
        self,
        spec: AgentSpec,
        state: Any,
        connection: Any,
        heartbeat: Any,
        heartbeat_interval: float,
    ) -> None:
        self.spec = spec
        self.state = state
        self.connection = connection
        self.heartbeat = heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.context = AgentContext(spec.agent_id, self.outgoing)
        self.agent: Any = None
        self._commands: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        writer = asyncio.create_task(self._writer())
        heartbeat = asyncio.create_task(self._heartbeat())
        reader = asyncio.create_task(self._reader())
        worker = asyncio.create_task(self._command_worker())
        try:
            await self._construct_agent()
            await self.outgoing.put({"type": "ready", "pid": os.getpid()})
            done, _pending = await asyncio.wait(
                {reader, writer}, return_when=asyncio.FIRST_COMPLETED
            )
            if writer in done:
                await writer
            await reader
        except BaseException as exc:
            if not isinstance(exc, (asyncio.CancelledError, EOFError)):
                await self.outgoing.put({
                    "type": "fatal",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
        finally:
            self._stopping.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await self._close_agent()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not writer.done():
                await self.outgoing.put({"type": "stopped"})
                await self.outgoing.join()
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _construct_agent(self) -> None:
        factory = _load_factory(self.spec.factory)
        self.agent = await _call_without_blocking(
            factory, dict(self.spec.config), self.context
        )
        if self.state is None:
            return
        restore = getattr(self.agent, "restore_state", None)
        if callable(restore):
            await _call_without_blocking(restore, self.state)
        elif hasattr(self.agent, "state"):
            self.agent.state = self.state

    async def _writer(self) -> None:
        while True:
            message = await self.outgoing.get()
            try:
                await asyncio.to_thread(self.connection.send, message)
            finally:
                self.outgoing.task_done()

    async def _reader(self) -> None:
        while not self._stopping.is_set():
            available = await asyncio.to_thread(self.connection.poll, 0.2)
            if not available:
                continue
            message = await asyncio.to_thread(self.connection.recv)
            kind = message.get("type")
            if kind == "shutdown":
                await self._commands.put({"type": "_shutdown"})
                continue
            if kind == "context_response":
                self.context._resolve(message)
                continue
            if kind == "message":
                routed = self._routed_message(message)
                await self.context._deliver(routed)
                await self._commands.put({
                    "type": "message_handler",
                    "request_id": str(message.get("request_id") or ""),
                    "message": routed,
                })
                continue
            await self._commands.put(message)

    async def _command_worker(self) -> None:
        """Serialize every operation that can observe or mutate the agent."""
        while True:
            message = await self._commands.get()
            try:
                if message.get("type") == "_shutdown":
                    self._stopping.set()
                    return
                await self._dispatch(message)
            finally:
                self._commands.task_done()

    async def _dispatch(self, message: Mapping[str, Any]) -> None:
        request_id = str(message.get("request_id") or "")
        try:
            kind = message.get("type")
            if kind == "invoke":
                result, has_state, state = await self._invoke(
                    request_id,
                    message.get("inputs"),
                    message.get("config"),
                    bool(message.get("callbacks")),
                )
                response = {
                    "type": "response",
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                    "has_state": has_state,
                    "state": state,
                }
                # Fail this invocation explicitly instead of silently killing
                # the pipe writer when a custom result/state cannot cross a
                # multiprocessing connection.
                await self.outgoing.put(_freeze_message(response))
            elif kind == "call":
                result = await self._call(
                    str(message.get("method") or ""),
                    tuple(message.get("args") or ()),
                    dict(message.get("kwargs") or {}),
                )
                await self._respond(request_id, result, snapshot=True)
            elif kind == "message_handler":
                await self._handle_message(message["message"])
                await self._respond(request_id, None, snapshot=True)
            else:
                raise ValueError(f"Unknown process command {kind!r}")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self.outgoing.put({
                "type": "response",
                "request_id": request_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })

    async def _respond(
        self, request_id: str, result: Any, *, snapshot: bool = False
    ) -> None:
        has_state, state = (
            await self._snapshot_state() if snapshot else (False, None)
        )
        response = {
            "type": "response",
            "request_id": request_id,
            "ok": True,
            "result": result,
            "has_state": has_state,
            "state": state,
        }
        await self.outgoing.put(_freeze_message(response))

    async def _invoke(
        self,
        request_id: str,
        inputs: Any,
        config: Mapping[str, Any] | None,
        callbacks: bool,
    ) -> tuple[Any, bool, Any]:
        invoke = getattr(self.agent, "ainvoke", None)
        if not callable(invoke):
            raise TypeError(
                f"{type(self.agent).__name__} must implement async ainvoke()"
            )
        invoke_config = dict(config or {})
        if callbacks:
            configured: list[Any] = list(invoke_config.get("callbacks") or ())
            configured.append(_RemoteCallbackHandler(self.context, request_id))
            invoke_config["callbacks"] = configured
        result = await invoke(inputs, config=invoke_config or None)
        has_state, state = await self._snapshot_state()
        return result, has_state, state

    async def _snapshot_state(self) -> tuple[bool, Any]:
        for method_name in ("snapshot_state", "export_state"):
            method = getattr(self.agent, method_name, None)
            if callable(method):
                return True, await _call_without_blocking(method)
        if hasattr(self.agent, "state"):
            return True, self.agent.state
        return False, None

    async def _call(
        self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if not method_name or method_name.startswith("_"):
            raise ValueError("only public agent methods may be called")
        method = getattr(self.agent, method_name)
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        value = await asyncio.to_thread(method, *args, **kwargs)
        return await _maybe_await(value)

    def _routed_message(self, message: Mapping[str, Any]) -> AgentMessage:
        return AgentMessage(
            sender=str(message.get("sender") or "host"),
            recipient=self.spec.agent_id,
            payload=message.get("payload"),
        )

    async def _handle_message(self, routed: AgentMessage) -> None:
        handler = getattr(self.agent, "on_message", None)
        if callable(handler):
            if inspect.iscoroutinefunction(handler):
                await handler(routed)
            else:
                await asyncio.to_thread(handler, routed)

    async def _heartbeat(self) -> None:
        while not self._stopping.is_set():
            self.heartbeat.value = monotonic()
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), self.heartbeat_interval
                )
            except TimeoutError:
                pass

    async def _close_agent(self) -> None:
        if self.agent is None:
            return
        async_close = getattr(self.agent, "aclose", None)
        if callable(async_close):
            try:
                await _maybe_await(async_close())
            except Exception:
                LOGGER.exception("Failed to close isolated agent")
        close = getattr(self.agent, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception:
                LOGGER.exception("Failed to close isolated agent")


def _agent_process_main(
    spec: AgentSpec,
    state: Any,
    connection: Any,
    heartbeat: Any,
    heartbeat_interval: float,
) -> None:
    try:
        if spec.environment is not None:
            os.environ.clear()
            os.environ.update(spec.environment)
        if spec.working_directory is not None:
            os.chdir(spec.working_directory)
        asyncio.run(
            AgentProcess(
                spec, state, connection, heartbeat, heartbeat_interval
            ).run()
        )
    finally:
        connection.close()


@dataclass(slots=True)
class _PendingRequest:
    future: asyncio.Future[Any]
    callbacks: Sequence[Any] = ()
    callback_tail: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _ManagedAgent:
    spec: AgentSpec
    state: Any
    status: AgentStatus = AgentStatus.REGISTERED
    process: multiprocessing.Process | None = None
    connection: Any = None
    generation: int = 0
    last_heartbeat: float | None = None
    reader_task: asyncio.Task[None] | None = None
    ready: asyncio.Future[None] | None = None
    pending: dict[str, _PendingRequest] = field(default_factory=dict)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    commit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    response_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    heartbeat: Any = None


class AgentManager:
    """Own and supervise isolated agent processes from the host process."""

    def __init__(
        self,
        *,
        start_method: str = "spawn",
        heartbeat_interval: float = 1.0,
        heartbeat_timeout: float = 10.0,
        startup_timeout: float = 30.0,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if heartbeat_timeout <= heartbeat_interval:
            raise ValueError(
                "heartbeat_timeout must be greater than heartbeat_interval"
            )
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        self._mp: Any = multiprocessing.get_context(start_method)
        if self._mp.get_start_method() == "spawn":
            _prime_spawn_resource_tracker()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.startup_timeout = startup_timeout
        self._agents: dict[str, _ManagedAgent] = {}
        self._lock = asyncio.Lock()
        self._host_messages: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._monitor_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def register(self, spec: AgentSpec, *, start: bool = False) -> None:
        """Register an independently configured agent."""
        state = spec.initial_state
        if spec.state_path is not None and spec.state_path.is_file():
            state = await asyncio.to_thread(self._read_state, spec.state_path)
        async with self._lock:
            if self._closed:
                raise AgentRuntimeError("AgentManager is closed")
            if spec.agent_id in self._agents:
                raise ValueError(
                    f"Agent {spec.agent_id!r} is already registered"
                )
            self._agents[spec.agent_id] = _ManagedAgent(spec=spec, state=state)
        if start:
            await self.start_agent(spec.agent_id)

    async def start(self) -> None:
        """Start every registered agent and the heartbeat monitor."""
        self._ensure_monitor()
        await asyncio.gather(
            *(self.start_agent(agent_id) for agent_id in tuple(self._agents))
        )

    async def start_agent(self, agent_id: str) -> None:
        """Start one agent and wait until its factory and state restore finish."""
        self._ensure_monitor()
        entry = self._entry(agent_id)
        async with entry.lifecycle_lock:
            await self._start_locked(entry)

    async def _start_locked(self, entry: _ManagedAgent) -> None:
        if self._closed:
            raise AgentRuntimeError("AgentManager is closed")
        if entry.status is AgentStatus.FAILED and entry.process is not None:
            await self._kill_locked(entry)
        if entry.status in {AgentStatus.STARTING, AgentStatus.READY}:
            ready = entry.ready
        else:
            ready = self._spawn(entry)
        assert ready is not None
        try:
            async with asyncio.timeout(self.startup_timeout):
                await asyncio.shield(ready)
        except TimeoutError:
            await self._kill_locked(entry)
            raise AgentProcessExitedError(
                f"Agent {entry.spec.agent_id!r} did not become ready within "
                f"{self.startup_timeout:g} seconds"
            ) from None

    def _spawn(self, entry: _ManagedAgent) -> asyncio.Future[None]:
        parent, child = self._mp.Pipe(duplex=True)
        process: multiprocessing.Process | None = None
        try:
            heartbeat = self._mp.RawValue("d", monotonic())
            ready = asyncio.get_running_loop().create_future()
            process = self._mp.Process(
                target=_agent_process_main,
                args=(
                    entry.spec,
                    entry.state,
                    child,
                    heartbeat,
                    self.heartbeat_interval,
                ),
                name=f"ursa-agent-{entry.spec.agent_id}",
                daemon=True,
            )
            process.start()
        except BaseException:
            parent.close()
            if process is not None:
                try:
                    process.close()
                except ValueError:
                    LOGGER.exception(
                        "Failed to close partially started process"
                    )
            entry.status = AgentStatus.FAILED
            entry.process = None
            entry.connection = None
            entry.ready = None
            entry.heartbeat = None
            raise
        finally:
            child.close()

        entry.generation += 1
        generation = entry.generation
        entry.status = AgentStatus.STARTING
        entry.process = process
        entry.connection = parent
        entry.ready = ready
        entry.last_heartbeat = None
        entry.heartbeat = heartbeat
        entry.reader_task = asyncio.create_task(
            self._reader_loop(entry.spec.agent_id, generation, parent, process)
        )
        return ready

    async def invoke(
        self,
        agent_id: str,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        callbacks: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Invoke an agent and commit its state before returning the result."""
        await self.start_agent(agent_id)
        result = self._request(
            agent_id,
            "invoke",
            inputs=inputs,
            config=dict(config or {}),
            callbacks=bool(callbacks),
            host_callbacks=tuple(callbacks or ()),
        )
        if timeout is None:
            return await asyncio.shield(result)
        async with asyncio.timeout(timeout):
            return await asyncio.shield(result)

    async def call(
        self,
        agent_id: str,
        method: str,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call another public method on an isolated agent."""
        await self.start_agent(agent_id)
        result = self._request(
            agent_id,
            "call",
            method=method,
            args=args,
            kwargs=kwargs,
        )
        if timeout is None:
            return await asyncio.shield(result)
        async with asyncio.timeout(timeout):
            return await asyncio.shield(result)

    def _request(
        self,
        agent_id: str,
        kind: str,
        *,
        host_callbacks: Sequence[Any] = (),
        **payload: Any,
    ) -> asyncio.Future[Any]:
        entry = self._entry(agent_id)
        if entry.status is not AgentStatus.READY:
            raise AgentProcessExitedError(f"Agent {agent_id!r} is not ready")
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        entry.pending[request_id] = _PendingRequest(future, host_callbacks)
        connection = entry.connection
        task = asyncio.create_task(
            self._send(
                entry,
                {"type": kind, "request_id": request_id, **payload},
                connection=connection,
            )
        )
        self._track_task(entry, task)
        task.add_done_callback(
            lambda task: self._send_done(entry, request_id, task)
        )
        return future

    @staticmethod
    def _send_done(
        entry: _ManagedAgent,
        request_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        pending = entry.pending.pop(request_id, None)
        if pending is not None and not pending.future.done():
            pending.future.set_exception(error)

    async def send_message(
        self,
        agent_id: str,
        payload: Any,
        *,
        sender: str = "host",
    ) -> None:
        """Deliver a host or agent message to one process."""
        await self.start_agent(agent_id)
        await self._request(agent_id, "message", sender=sender, payload=payload)

    async def broadcast(self, payload: Any, *, sender: str = "host") -> None:
        await asyncio.gather(
            *(
                self.send_message(agent_id, payload, sender=sender)
                for agent_id in tuple(self._agents)
            )
        )

    async def receive_message(
        self, *, timeout: float | None = None
    ) -> AgentMessage:
        """Receive a child message addressed to ``"host"``."""
        if timeout is None:
            return await self._host_messages.get()
        async with asyncio.timeout(timeout):
            return await self._host_messages.get()

    async def kill_agent(
        self,
        agent_id: str,
        *,
        graceful_timeout: float = 2.0,
    ) -> None:
        """Stop a process while retaining its last committed state."""
        entry = self._entry(agent_id)
        async with entry.lifecycle_lock:
            await self._kill_locked(entry, graceful_timeout=graceful_timeout)

    async def _kill_locked(
        self,
        entry: _ManagedAgent,
        *,
        graceful_timeout: float = 2.0,
    ) -> None:
        process = entry.process
        if process is None:
            entry.status = AgentStatus.STOPPED
            return
        connection = entry.connection
        generation = entry.generation
        dead = not process.is_alive()
        entry.status = AgentStatus.STOPPING
        if dead:
            await asyncio.to_thread(process.join)
            await self._finish_stopped(entry, generation, connection, process)
            return
        try:
            await self._send(entry, {"type": "shutdown"}, connection=connection)
        except (BrokenPipeError, EOFError, OSError):
            pass
        await asyncio.to_thread(process.join, graceful_timeout)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, graceful_timeout)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join)
        await self._finish_stopped(entry, generation, connection, process)

    async def restart_agent(self, agent_id: str) -> None:
        """Restart from the last state committed after a successful invoke."""
        self._ensure_monitor()
        entry = self._entry(agent_id)
        async with entry.lifecycle_lock:
            await self._kill_locked(entry)
            await self._start_locked(entry)

    async def unregister(self, agent_id: str) -> None:
        entry = self._entry(agent_id)
        async with entry.lifecycle_lock:
            await self._kill_locked(entry)
            async with self._lock:
                if self._agents.get(agent_id) is entry:
                    self._agents.pop(agent_id)

    async def snapshot(self, agent_id: str) -> AgentSnapshot:
        entry = self._entry(agent_id)
        last_heartbeat = (
            float(entry.heartbeat.value)
            if entry.heartbeat is not None
            else entry.last_heartbeat
        )
        age = (
            monotonic() - last_heartbeat if last_heartbeat is not None else None
        )
        return AgentSnapshot(
            agent_id=agent_id,
            status=entry.status,
            pid=entry.process.pid if entry.process is not None else None,
            generation=entry.generation,
            last_heartbeat=last_heartbeat,
            heartbeat_age=age,
            state=entry.state,
        )

    async def aclose(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._aclose_once())
            self._close_task = task
        await asyncio.shield(task)

    async def _aclose_once(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            agent_ids = tuple(self._agents)
        monitor = self._monitor_task
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        await asyncio.gather(
            *(self.kill_agent(agent_id) for agent_id in agent_ids),
        )

    async def __aenter__(self) -> AgentManager:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    def _entry(self, agent_id: str) -> _ManagedAgent:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise AgentNotFoundError(f"Unknown agent {agent_id!r}") from None

    async def _send(
        self,
        entry: _ManagedAgent,
        message: Mapping[str, Any],
        *,
        connection: Any = None,
    ) -> None:
        connection = connection if connection is not None else entry.connection
        if connection is None:
            raise AgentProcessExitedError(
                f"Agent {entry.spec.agent_id!r} has no process connection"
            )
        async with entry.send_lock:
            await asyncio.to_thread(connection.send, dict(message))

    async def _reader_loop(
        self,
        agent_id: str,
        generation: int,
        connection: Any,
        process: multiprocessing.Process,
    ) -> None:
        entry = self._entry(agent_id)
        try:
            while generation == entry.generation:
                message = await asyncio.to_thread(connection.recv)
                await self._handle_message(
                    entry, generation, connection, message
                )
        except (EOFError, BrokenPipeError, OSError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            if generation == entry.generation:
                await self._process_exited(entry, generation, process)

    async def _handle_message(
        self,
        entry: _ManagedAgent,
        generation: int,
        connection: Any,
        message: Mapping[str, Any],
    ) -> None:
        kind = message.get("type")
        if kind == "ready":
            entry.status = AgentStatus.READY
            if entry.ready is not None and not entry.ready.done():
                entry.ready.set_result(None)
            return
        if kind == "fatal":
            error = AgentInvocationError(
                entry.spec.agent_id,
                str(message.get("error_type") or "AgentProcessError"),
                str(message.get("message") or "agent process failed"),
                str(message.get("traceback") or ""),
            )
            entry.status = AgentStatus.FAILED
            if entry.ready is not None and not entry.ready.done():
                entry.ready.set_exception(error)
            return
        if kind == "response":
            task = asyncio.create_task(self._handle_response(entry, message))
            self._track_task(entry, task)
            entry.response_tasks.add(task)
            task.add_done_callback(entry.response_tasks.discard)
            return
        if kind == "state_update":
            task = asyncio.create_task(
                self._commit_state(entry, message.get("state"))
            )
            self._track_task(entry, task)
            entry.response_tasks.add(task)
            task.add_done_callback(entry.response_tasks.discard)
            return
        if kind == "callback":
            self._queue_callback(entry, message)
            return
        if kind == "route_message":
            self._track_task(
                entry,
                asyncio.create_task(
                    self._route_message(entry, connection, message)
                ),
            )

    async def _handle_response(
        self, entry: _ManagedAgent, message: Mapping[str, Any]
    ) -> None:
        request_id = str(message.get("request_id") or "")
        pending = entry.pending.get(request_id)
        if pending is None or pending.future.done():
            return
        if not message.get("ok", False):
            pending.future.set_exception(
                AgentInvocationError(
                    entry.spec.agent_id,
                    str(message.get("error_type") or "AgentError"),
                    str(message.get("message") or "agent operation failed"),
                    str(message.get("traceback") or ""),
                )
            )
            entry.pending.pop(request_id, None)
            return
        try:
            if message.get("has_state", False):
                await self._commit_state(entry, message.get("state"))
            if pending.callback_tail is not None:
                await asyncio.shield(pending.callback_tail)
        except Exception as exc:
            pending.future.set_exception(
                AgentRuntimeError(
                    f"Could not complete operation for "
                    f"{entry.spec.agent_id!r}: {exc}"
                )
            )
        else:
            pending.future.set_result(message.get("result"))
        finally:
            entry.pending.pop(request_id, None)

    async def _commit_state(self, entry: _ManagedAgent, state: Any) -> None:
        async with entry.commit_lock:
            if entry.spec.state_path is not None:
                await asyncio.to_thread(
                    self._write_state, entry.spec.state_path, state
                )
            entry.state = state

    def _queue_callback(
        self, entry: _ManagedAgent, message: Mapping[str, Any]
    ) -> None:
        pending = entry.pending.get(str(message.get("request_id") or ""))
        if pending is None:
            return
        previous = pending.callback_tail

        async def dispatch() -> None:
            if previous is not None:
                await previous
            await self._dispatch_callback(entry, pending, message)

        task = asyncio.create_task(dispatch())
        pending.callback_tail = task
        self._track_task(entry, task)

    async def _dispatch_callback(
        self,
        entry: _ManagedAgent,
        pending: _PendingRequest,
        message: Mapping[str, Any],
    ) -> None:
        method_name = str(message.get("method") or "")
        args = tuple(message.get("args") or ())
        kwargs = dict(message.get("kwargs") or {})
        for callback in pending.callbacks:
            method = getattr(callback, method_name, None)
            if not callable(method):
                continue
            try:
                await _call_without_blocking(method, *args, **kwargs)
            except Exception:
                LOGGER.exception(
                    "Host callback %s failed for agent %s",
                    method_name,
                    entry.spec.agent_id,
                )

    @staticmethod
    def _track_task(entry: _ManagedAgent, task: asyncio.Task[Any]) -> None:
        entry.background_tasks.add(task)
        task.add_done_callback(entry.background_tasks.discard)

    async def _route_message(
        self,
        entry: _ManagedAgent,
        connection: Any,
        message: Mapping[str, Any],
    ) -> None:
        request_id = str(message.get("request_id") or "")
        recipient = str(message.get("recipient") or "")
        try:
            if recipient == "host":
                await self._host_messages.put(
                    AgentMessage(
                        sender=entry.spec.agent_id,
                        recipient="host",
                        payload=message.get("payload"),
                    )
                )
            else:
                await self.send_message(
                    recipient,
                    message.get("payload"),
                    sender=entry.spec.agent_id,
                )
            response = {
                "type": "context_response",
                "request_id": request_id,
                "ok": True,
                "result": None,
            }
        except Exception as exc:
            response = {
                "type": "context_response",
                "request_id": request_id,
                "ok": False,
                "error": str(exc),
            }
        await self._send(entry, response, connection=connection)

    async def _process_exited(
        self,
        entry: _ManagedAgent,
        generation: int,
        process: multiprocessing.Process,
    ) -> None:
        if generation != entry.generation or process is not entry.process:
            return
        if entry.status in {AgentStatus.STOPPED, AgentStatus.STOPPING}:
            return
        response_tasks = tuple(entry.response_tasks)
        if response_tasks:
            await asyncio.gather(*response_tasks, return_exceptions=True)
        entry.status = AgentStatus.FAILED
        error = AgentProcessExitedError(
            f"Agent process {entry.spec.agent_id!r} exited"
        )
        if entry.ready is not None and not entry.ready.done():
            entry.ready.set_exception(error)
        self._fail_pending(entry, error)
        if entry.spec.restart_on_failure and not self._closed:
            task = asyncio.create_task(self.restart_agent(entry.spec.agent_id))
            self._track_task(entry, task)

    async def _finish_stopped(
        self,
        entry: _ManagedAgent,
        generation: int,
        connection: Any,
        process: multiprocessing.Process,
    ) -> None:
        reader = entry.reader_task
        if reader is not None and reader is not asyncio.current_task():
            try:
                async with asyncio.timeout(1):
                    await asyncio.shield(reader)
            except TimeoutError:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
        current = asyncio.current_task()
        response_tasks = tuple(
            task for task in entry.response_tasks if task is not current
        )
        if response_tasks:
            await asyncio.gather(*response_tasks, return_exceptions=True)
        tasks = tuple(
            task
            for task in entry.background_tasks
            if task is not current and task not in entry.response_tasks
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if connection is not None:
            connection.close()
        try:
            process.close()
        except ValueError:
            # ``multiprocessing.Process.close`` rejects a still-running
            # process.  The process has already been joined by the caller,
            # but preserve shutdown progress if a custom Process
            # implementation reports a contradictory state.
            LOGGER.exception(
                "Failed to close joined process for agent %s",
                entry.spec.agent_id,
            )
        if generation != entry.generation or process is not entry.process:
            return
        entry.reader_task = None
        entry.connection = None
        entry.process = None
        entry.heartbeat = None
        entry.status = AgentStatus.STOPPED
        self._fail_pending(
            entry,
            AgentProcessExitedError(
                f"Agent process {entry.spec.agent_id!r} was stopped"
            ),
        )

    @staticmethod
    def _fail_pending(entry: _ManagedAgent, error: BaseException) -> None:
        for pending in entry.pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        entry.pending.clear()

    def _ensure_monitor(self) -> None:
        if self._closed:
            raise AgentRuntimeError("AgentManager is closed")
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor())

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            now = monotonic()
            for entry in tuple(self._agents.values()):
                heartbeat = entry.heartbeat
                last_heartbeat = (
                    float(heartbeat.value) if heartbeat is not None else None
                )
                entry.last_heartbeat = last_heartbeat
                if (
                    entry.status is AgentStatus.READY
                    and last_heartbeat is not None
                    and now - last_heartbeat > self.heartbeat_timeout
                ):
                    LOGGER.error(
                        "Agent %s missed its heartbeat deadline",
                        entry.spec.agent_id,
                    )
                    task = asyncio.create_task(
                        self._recover_unresponsive(entry)
                    )
                    self._track_task(entry, task)

    async def _recover_unresponsive(self, entry: _ManagedAgent) -> None:
        async with entry.lifecycle_lock:
            if entry.status is not AgentStatus.READY:
                return
            await self._kill_locked(entry)
            if entry.spec.restart_on_failure and not self._closed:
                await self._start_locked(entry)

    @staticmethod
    def _read_state(path: Path) -> Any:
        with path.open("rb") as stream:
            return pickle.load(stream)

    @staticmethod
    def _write_state(path: Path, state: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                pickle.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


class SyncAgentManager:
    """Optional blocking facade backed by one persistent asyncio loop."""

    def __init__(self, **manager_kwargs: Any) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._manager: AgentManager | None = None

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._manager = AgentManager(**manager_kwargs)
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=run, name="ursa-agent-manager", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    @property
    def manager(self) -> AgentManager:
        assert self._manager is not None
        return self._manager

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def register(self, spec: AgentSpec, *, start: bool = False) -> None:
        self._run(self.manager.register(spec, start=start))

    def invoke(
        self,
        agent_id: str,
        inputs: Any,
        *,
        config: Mapping[str, Any] | None = None,
        callbacks: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return self._run(
            self.manager.invoke(
                agent_id,
                inputs,
                config=config,
                callbacks=callbacks,
                timeout=timeout,
            )
        )

    def start(self) -> None:
        self._run(self.manager.start())

    def start_agent(self, agent_id: str) -> None:
        self._run(self.manager.start_agent(agent_id))

    def call(
        self,
        agent_id: str,
        method: str,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._run(
            self.manager.call(
                agent_id, method, *args, timeout=timeout, **kwargs
            )
        )

    def send_message(
        self, agent_id: str, payload: Any, *, sender: str = "host"
    ) -> None:
        self._run(self.manager.send_message(agent_id, payload, sender=sender))

    def receive_message(self, *, timeout: float | None = None) -> AgentMessage:
        return self._run(self.manager.receive_message(timeout=timeout))

    def broadcast(self, payload: Any, *, sender: str = "host") -> None:
        self._run(self.manager.broadcast(payload, sender=sender))

    def kill_agent(
        self, agent_id: str, *, graceful_timeout: float = 2.0
    ) -> None:
        self._run(
            self.manager.kill_agent(agent_id, graceful_timeout=graceful_timeout)
        )

    def restart_agent(self, agent_id: str) -> None:
        self._run(self.manager.restart_agent(agent_id))

    def unregister(self, agent_id: str) -> None:
        self._run(self.manager.unregister(agent_id))

    def snapshot(self, agent_id: str) -> AgentSnapshot:
        return self._run(self.manager.snapshot(agent_id))

    def close(self) -> None:
        self._run(self.manager.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def __enter__(self) -> SyncAgentManager:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
