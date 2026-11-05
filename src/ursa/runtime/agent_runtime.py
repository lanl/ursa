"""Process-isolated URSA agent facade for interactive hosts."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP

from ursa import agents
from ursa.agents.base import (
    DEFAULT_INTERRUPT_MESSAGE,
    AgentWithTools,
    BaseAgent,
    SubagentLimitError,
)
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    UrsaConfig,
    resolve_ursa_config,
)
from ursa.security import enforce_model_group_policy
from ursa.util.has_optional_dep_group import has_optional_dep_group
from ursa.util.inference_providers import validate_model_provider
from ursa.util.mcp import start_mcp_client
from ursa.util.secrets import SecretReference

from .manager import AgentContext, AgentManager, AgentSpec

LOGGER = logging.getLogger(__name__)

_AGENT_TYPES = (
    ("chat", "ChatAgent", None),
    ("arxiv", "ArxivAgent", None),
    ("dsi", "DSIAgent", "dsi"),
    ("execute", "ExecutionAgent", None),
    ("deep_review", "DeepReviewAgent", None),
    ("hypothesize", "HypothesizerAgent", None),
    ("plan", "PlanningAgent", None),
    ("prompt", "PromptingAgent", None),
    ("web", "WebSearchAgent", None),
    ("lammps", "LammpsAgent", "lammps"),
)


def _resolve_host_secrets(config: UrsaConfig) -> UrsaConfig:
    """Resolve runtime secrets before configuration crosses a process boundary."""
    resolved = resolve_ursa_config(config)
    for owner, field_name in (
        (resolved, "llm_model"),
        (resolved, "emb_model"),
    ):
        model = getattr(owner, field_name)
        if model is None or not isinstance(model.api_key, SecretReference):
            continue
        setattr(
            owner,
            field_name,
            model.model_copy(update={"api_key": model.api_key.resolve()}),
        )

    for server_name, server in tuple(resolved.mcp_servers.items()):
        headers = getattr(server, "headers", None)
        if not headers:
            continue
        materialized = {}
        for name, value in headers.items():
            if not isinstance(value, SecretReference):
                materialized[name] = value
                continue
            secret = value.resolve()
            if secret is None:
                raise ValueError(
                    f"Secret for MCP server {server_name!r} is not set"
                )
            materialized[name] = secret.get_secret_value()
        resolved.mcp_servers[server_name] = server.model_copy(
            update={"headers": materialized}
        )
    return resolved


def _runner_config(config: UrsaConfig) -> UrsaConfig:
    """Remove host-only secret sources from a materialized runner config."""
    subagents = config.subagents.model_copy(
        update={"llm_model": None, "embedding_model": None}
    )
    return config.model_copy(
        update={"inference_providers": {}, "subagents": subagents}
    )


@dataclass(slots=True)
class _RuntimePrompt:
    prompt: str
    config: dict[str, Any]
    response: asyncio.Future[str]
    request_id: str | None = None
    interrupted: bool = False
    on_start: Any = None


class _ProcessRuntimeBridge:
    """Child-side facade for hierarchy operations owned by the host."""

    def __init__(self, context: AgentContext, agent_path: str) -> None:
        self.context = context
        self.agent_path = agent_path

    async def spawn_subagent(
        self,
        name: str,
        message: str,
        agent_type: str | None = None,
    ) -> tuple[str, Any]:
        payload: dict[str, Any] = {"name": name, "message": message}
        if agent_type is not None:
            payload["agent_type"] = agent_type
        result = await self.context.request_runtime("spawn", **payload)
        if not isinstance(result, Mapping):
            raise TypeError("The host returned an invalid subagent result")
        return str(result["path"]), result.get("response")

    async def send_message(self, target: str, message: str) -> Any:
        return await self.context.request_runtime(
            "invoke", target=target, message=message
        )

    async def list_agents(self) -> list[str]:
        result = await self.context.request_runtime("list")
        return [str(path) for path in result]


class _UrsaProcessAgent:
    """One fully constructed URSA agent living exclusively in its child."""

    def __init__(
        self,
        agent: Any,
        context: AgentContext,
        agent_path: str,
        tool_sources: Mapping[str, str] | None = None,
    ) -> None:
        self.agent = agent
        self.context = context
        self.agent_path = agent_path
        self.tool_sources = dict(tool_sources or {})
        self.state: Any = None
        self._inbox: asyncio.Queue[_RuntimePrompt | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._active_task: asyncio.Task[Any] | None = None

    async def ainvoke(
        self, inputs: Any, config: Mapping[str, Any] | None = None
    ) -> str:
        return await self._submit(str(inputs), config=config)

    async def on_message(self, message: Any) -> Any:
        """Accept queued user messages without blocking an active invoke."""
        payload = message.payload
        if not isinstance(payload, Mapping):
            return await self._submit(str(payload))
        action = str(payload.get("action") or "request")
        if action == "interrupt_queued":
            return self._interrupt_for_queued_messages()
        if action != "request":
            raise ValueError(f"Unknown runtime message action {action!r}")
        config = dict(payload.get("config") or {})
        callbacks = list(config.get("callbacks") or ())
        callbacks.append(self.context.callback_handler())
        config["callbacks"] = callbacks

        async def on_start() -> None:
            await self.context.emit_callback(
                "on_agent_message_start", self.agent_path
            )

        succeeded = False
        try:
            result = await self._submit(
                str(payload.get("prompt") or ""),
                config=config,
                interrupted=bool(payload.get("interrupt", False)),
                on_start=on_start,
            )
            succeeded = True
            return result
        finally:
            await self.context.emit_callback(
                "on_agent_message_end",
                self.agent_path,
                succeeded=succeeded,
            )

    async def _submit(
        self,
        prompt: str,
        *,
        config: Mapping[str, Any] | None = None,
        interrupted: bool = False,
        on_start: Any = None,
    ) -> str:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run_inbox(),
                name=f"ursa-runtime-inbox:{self.agent_path}",
            )
        response = asyncio.get_running_loop().create_future()
        item = _RuntimePrompt(
            prompt=prompt,
            config=dict(config or {}),
            response=response,
            request_id=self.context.active_request_id,
            interrupted=interrupted,
            on_start=on_start,
        )
        if interrupted:
            pending: list[_RuntimePrompt | None] = []
            while True:
                try:
                    pending.append(self._inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
                self._inbox.task_done()
            await self._inbox.put(item)
            for queued in pending:
                await self._inbox.put(queued)
            self._cancel_active()
        else:
            await self._inbox.put(item)
        return await response

    def _interrupt_for_queued_messages(self) -> bool:
        pending: list[_RuntimePrompt | None] = []
        while True:
            try:
                pending.append(self._inbox.get_nowait())
            except asyncio.QueueEmpty:
                break
            self._inbox.task_done()
        for item in pending:
            if isinstance(item, _RuntimePrompt):
                item.interrupted = True
                break
        for item in pending:
            self._inbox.put_nowait(item)
        return self._cancel_active()

    def _cancel_active(self) -> bool:
        task = self._active_task
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _run_inbox(self) -> None:
        while True:
            item = await self._inbox.get()
            if item is None:
                self._inbox.task_done()
                return
            request_token = self.context._enter_request(item.request_id or "")
            try:
                if item.on_start is not None:
                    started = item.on_start()
                    if inspect.isawaitable(started):
                        await started
                query = self.agent.format_query(item.prompt, state=self.state)
                if item.interrupted:
                    query = self.agent._prepend_system_message(
                        query, DEFAULT_INTERRUPT_MESSAGE
                    )
                task = asyncio.create_task(
                    self.agent.ainvoke(query, config=item.config or None)
                )
                self._active_task = task
                try:
                    self.state = await task
                except asyncio.CancelledError:
                    if not item.response.done():
                        item.response.set_exception(
                            RuntimeError(
                                "Agent invocation interrupted by a newer message"
                            )
                        )
                except Exception as exc:
                    if not item.response.done():
                        item.response.set_exception(exc)
                else:
                    if not item.response.done():
                        item.response.set_result(
                            str(self.agent.format_result(self.state))
                        )
                finally:
                    self._active_task = None
            finally:
                self.context._leave_request(request_token)
                self._inbox.task_done()

    async def describe_tools(self) -> list[dict[str, Any]]:
        tools = getattr(self.agent, "tools", None)
        if isinstance(tools, Mapping):
            items = tools.items()
        elif isinstance(tools, (list, tuple)):
            items = (
                (str(getattr(tool, "name", index)), tool)
                for index, tool in enumerate(tools)
            )
        else:
            return []
        details = []
        for fallback_name, tool in items:
            schema_json = await _tool_schema(tool)
            details.append({
                "name": str(getattr(tool, "name", None) or fallback_name),
                "class_name": tool.__class__.__name__,
                "description": str(
                    getattr(tool, "description", None)
                    or "No description available."
                ).strip(),
                "schema": schema_json,
                "return_direct": getattr(tool, "return_direct", None),
                "mcp_server": str(
                    self.tool_sources.get(
                        str(getattr(tool, "name", None) or fallback_name), ""
                    )
                ),
            })
        return details

    async def spawn_subagent(
        self,
        name: str,
        message: str,
        agent_type: str | None = None,
    ) -> tuple[str, Any]:
        """Spawn through the bridge used by the model-facing tool."""
        return await self.agent._process_runtime.spawn_subagent(
            name, message, agent_type
        )

    async def send_subagent_message(self, target: str, message: str) -> Any:
        """Send through the bridge used by the model-facing tool."""
        return await self.agent._process_runtime.send_message(target, message)

    async def list_subagents(self) -> list[str]:
        return await self.agent._process_runtime.list_agents()

    async def aclose(self) -> None:
        self._cancel_active()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        while True:
            try:
                item = self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _RuntimePrompt) and not item.response.done():
                item.response.cancel()
            self._inbox.task_done()
        close = getattr(self.agent, "aclose", None)
        if callable(close):
            await close()
        sync_close = getattr(self.agent, "close", None)
        if callable(sync_close):
            await asyncio.to_thread(sync_close)


async def _tool_schema(tool: Any) -> dict[str, Any] | None:
    """Return the filtered, display-safe input schema for one tool."""
    for attribute in ("tool_call_schema", "args_schema"):
        try:
            schema = getattr(tool, attribute, None)
        except Exception:
            continue
        if isinstance(schema, Mapping):
            return dict(schema)
        for method_name in ("model_json_schema", "schema"):
            method = getattr(schema, method_name, None)
            if not callable(method):
                continue
            try:
                schema_json = await asyncio.to_thread(method)
            except Exception:
                continue
            if isinstance(schema_json, Mapping):
                return dict(schema_json)
    return None


async def _ursa_agent_factory(
    config: Mapping[str, Any], context: AgentContext
) -> _UrsaProcessAgent:
    """Build exactly one requested URSA agent in the isolated process."""
    ursa_config = config["ursa_config"]
    if not isinstance(ursa_config, UrsaConfig):
        ursa_config = UrsaConfig.model_validate(ursa_config)
    # The host resolves every secret before serializing this configuration.
    # Runners must never reach back into the host's environment or keyring.
    resolved = ursa_config
    agent_name = str(config["agent_name"])
    agent_path = str(config.get("agent_path") or "root")
    try:
        _name, class_name, dependency = next(
            definition
            for definition in _AGENT_TYPES
            if definition[0] == agent_name
        )
    except StopIteration:
        raise ValueError(f"Unknown agent {agent_name!r}") from None
    if dependency is not None and not has_optional_dep_group(dependency):
        raise ImportError(
            f"Agent {agent_name!r} requires optional dependency "
            f"group {dependency!r}"
        )
    agent_class = getattr(agents, class_name)

    if isinstance(resolved.llm_model.api_key, SecretReference):
        raise RuntimeError("The host did not resolve the chat-model secret")
    await asyncio.to_thread(validate_model_provider, resolved.llm_model, "chat")
    model = await asyncio.to_thread(resolved.llm_model.init_chat_model)
    enforce_model_group_policy(model, resolved.group)
    embedding = None
    if resolved.emb_model is not None:
        if isinstance(resolved.emb_model.api_key, SecretReference):
            raise RuntimeError(
                "The host did not resolve the embedding-model secret"
            )
        await asyncio.to_thread(
            validate_model_provider, resolved.emb_model, "embedding"
        )
        embedding = await asyncio.to_thread(resolved.emb_model.init_embedding)
        enforce_model_group_policy(embedding, resolved.group)

    agent_kwargs: dict[str, Any] = dict(
        resolved.agent_config.get(agent_name) or {}
    )
    agent_kwargs.setdefault(
        "subagent_max_agents", resolved.subagents.max_agents
    )
    agent_kwargs.setdefault("subagent_max_depth", resolved.subagents.max_depth)
    if agent_name in {"chat", "execute", "deep_review", "dsi"}:
        agent_kwargs = {
            "rag_tools": resolved.rag_tools,
            "rag_tool_embedding": embedding,
            **agent_kwargs,
        }
    try:
        agent = await asyncio.to_thread(
            agent_class,
            llm=model,
            workspace=resolved.workspace,
            agent_name=resolved.agent_name,
            group=resolved.group,
            thread_id=resolved.thread_id or "ursa",
            **agent_kwargs,
        )
        agent._runtime_agent_path = agent_path
        agent._process_runtime = _ProcessRuntimeBridge(context, agent_path)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {agent_class.__name__} with config "
            f"{agent_kwargs}: {exc}"
        ) from exc

    tool_sources: Mapping[str, str] = {}
    try:
        if resolved.mcp_servers and isinstance(agent, AgentWithTools):
            client = await asyncio.to_thread(
                start_mcp_client, resolved.mcp_servers
            )
            tool_sources = await agent.add_mcp_tools(client)
    except BaseException:
        await _close_built_agent(agent)
        raise
    return _UrsaProcessAgent(
        agent,
        context,
        agent_path,
        tool_sources,
    )


async def _close_built_agent(agent: Any) -> None:
    try:
        close = getattr(agent, "aclose", None)
        if callable(close):
            await close()
        sync_close = getattr(agent, "close", None)
        if callable(sync_close):
            await asyncio.to_thread(sync_close)
    except Exception:
        LOGGER.exception("Failed to close partially constructed URSA agent")


@dataclass(slots=True)
class RuntimeAgent:
    """Display metadata and remote operations for one managed agent."""

    name: str
    agent_class: type[Any]
    config: dict[str, Any]
    manager: AgentManager = field(repr=False)
    runtime_id: str = field(repr=False)
    tool_sources: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def description(self) -> str | None:
        return self.agent_class.__doc__

    @property
    def _agent(self) -> None:
        """Compatibility marker: the live object never exists in the host."""
        return None

    async def remote_tools(self) -> list[dict[str, Any]]:
        return await self.manager.call(self.runtime_id, "describe_tools")


@dataclass(frozen=True, slots=True)
class RuntimeHierarchyStatus:
    """Host-side live state for one member of the active hierarchy."""

    path: str
    parent: str | None
    state: str
    pending_messages: int = 0


class _MessageStartCallback:
    def __init__(self, callback: Any) -> None:
        self.callback = callback

    async def on_agent_message_start(self, *_args: Any, **_kwargs: Any) -> None:
        result = self.callback(None)
        if inspect.isawaitable(result):
            await result


class AgentRuntime:
    """Host facade backed by one isolated process per configured agent."""

    def __init__(
        self,
        config: UrsaConfig,
        *,
        manager: AgentManager | None = None,
    ) -> None:
        self.config = _resolve_host_secrets(config)
        self.thread_id = self.config.thread_id or "ursa"
        self.workspace = self.config.workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.agent_name = self.config.agent_name
        self.group = self.config.group
        self.inference_provider = self.config.llm_model.inference_provider
        self.embedding_inference_provider = (
            self.config.emb_model.inference_provider
            if self.config.emb_model is not None
            else None
        )
        # Model objects intentionally do not exist in the host process.
        self.model = None
        self.embedding = None
        self._namespace = f"ursa:{uuid4().hex}"
        self._owns_manager = manager is None
        self._manager = manager or AgentManager()
        self._registered: set[str] = set()
        self._dynamic_runtime_ids: set[str] = set()
        self._runtime_paths: dict[str, str] = {}
        self._runtime_roots: dict[str, str] = {}
        self._runtime_agent_names: dict[str, str] = {}
        self._hierarchies: dict[str, dict[str, str]] = {}
        self._working_counts: dict[str, int] = {}
        self._pending_counts: dict[str, int] = {}
        self._active_root_id: str | None = None
        self._active_agent_ready = asyncio.Event()
        self._register_lock = asyncio.Lock()
        self._hierarchy_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

        self.agents: dict[str, RuntimeAgent] = {}
        for name, class_name, dependency in _AGENT_TYPES:
            if dependency is not None and not has_optional_dep_group(
                dependency
            ):
                continue
            agent_config = dict(self.config.agent_config.get(name) or {})
            self.agents[name] = RuntimeAgent(
                name=name,
                agent_class=getattr(agents, class_name),
                config=agent_config,
                manager=self._manager,
                runtime_id=f"{self._namespace}:{name}",
            )
        unknown = set(self.config.agent_config) - set(self.agents)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown agent configuration: {names}")

    async def _ensure_registered(self, name: str) -> None:
        if self._closed:
            raise RuntimeError("Agent runtime is closed")
        if name not in self.agents:
            raise ValueError(f"Unknown agent {name}")
        if name in self._registered:
            return
        async with self._register_lock:
            if name in self._registered:
                return
            await self._manager.register(
                AgentSpec(
                    agent_id=self.agents[name].runtime_id,
                    factory="ursa.runtime.agent_runtime:_ursa_agent_factory",
                    config={
                        "agent_name": name,
                        "agent_path": "root",
                        # multiprocessing serializes this independently for
                        # every child. Avoid deep-copying TemporaryDirectory's
                        # cleanup finalizer in the host.
                        "ursa_config": _runner_config(
                            self.config.model_copy(
                                update={
                                    "agent_config": {
                                        name: dict(
                                            self.config.agent_config.get(name)
                                            or {}
                                        )
                                    }
                                },
                            )
                        ),
                    },
                )
            )
            runtime_id = self.agents[name].runtime_id
            self._manager.set_runtime_request_handler(
                runtime_id, self._handle_runtime_request
            )
            self._runtime_paths[runtime_id] = "root"
            self._runtime_roots[runtime_id] = runtime_id
            self._runtime_agent_names[runtime_id] = name
            self._hierarchies.setdefault(runtime_id, {"root": runtime_id})
            self._registered.add(name)

    async def get_agent(self, name: str) -> RuntimeAgent:
        async with self._operation_lock:
            await self._ensure_registered(name)
            await self._manager.start_agent(self.agents[name].runtime_id)
            return self.agents[name]

    @asynccontextmanager
    async def use_agent(self, name: str) -> AsyncGenerator[RuntimeAgent]:
        async with self._operation_lock:
            await self._ensure_registered(name)
            await self._manager.start_agent(self.agents[name].runtime_id)
            yield self.agents[name]

    async def run_agent(
        self,
        name: str,
        prompt: str,
        callbacks: Sequence[Any] | None = None,
    ) -> str:
        """Run one agent without blocking lifecycle control operations."""
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
            self._active_root_id = runtime_id
            self._active_agent_ready.set()
        self._mark_working("root", 1)
        try:
            result = await self._manager.invoke(
                runtime_id,
                prompt,
                config={"metadata": {"ursa_message_recipient": "root"}},
                callbacks=callbacks,
            )
        finally:
            self._mark_working("root", -1)
        if not isinstance(result, str):
            raise TypeError(f"Agent {name!r} returned a non-string result")
        return result

    @property
    def active_agent_path(self) -> str | None:
        return "root" if self._active_root_id is not None else None

    async def wait_for_active_agent(self) -> str:
        await self._active_agent_ready.wait()
        if self._active_root_id is None:
            raise RuntimeError("No active agent hierarchy")
        return "root"

    def subagent_paths(self) -> list[str]:
        root_id = self._active_root_id
        if root_id is None:
            return []
        return list(self._hierarchies.get(root_id, {}))

    def hierarchy_status(self) -> list[RuntimeHierarchyStatus]:
        root_id = self._active_root_id
        if root_id is None:
            return []
        statuses = []
        for path in self._hierarchies.get(root_id, {}):
            pending = self._pending_counts.get(path, 0)
            if self._working_counts.get(path, 0):
                state = "Working"
            elif pending:
                state = "Queued"
            else:
                state = "Ready"
            statuses.append(
                RuntimeHierarchyStatus(
                    path=path,
                    parent=path.rsplit("/", 1)[0] if "/" in path else None,
                    state=state,
                    pending_messages=pending,
                )
            )
        return statuses

    async def run_active_agent(
        self,
        path: str,
        prompt: str,
        *,
        callbacks: Sequence[Any] | None = None,
        interrupt: bool = False,
        on_start: Any = None,
    ) -> str:
        """Queue a user message for any process in the active hierarchy."""
        runtime_id, resolved_path = self._resolve_active_path(path)
        configured = list(callbacks or ())
        if on_start is not None:
            configured.append(_MessageStartCallback(on_start))
        self._pending_counts[resolved_path] = (
            self._pending_counts.get(resolved_path, 0) + 1
        )
        self._mark_working(resolved_path, 1)
        try:
            result = await self._manager.send_message(
                runtime_id,
                {
                    "action": "request",
                    "prompt": prompt,
                    "interrupt": interrupt,
                    "config": {
                        "metadata": {"ursa_message_recipient": resolved_path}
                    },
                },
                callbacks=configured,
            )
        finally:
            self._pending_counts[resolved_path] = max(
                0, self._pending_counts.get(resolved_path, 1) - 1
            )
            self._mark_working(resolved_path, -1)
        if not isinstance(result, str):
            raise TypeError(
                f"Agent {resolved_path!r} returned a non-string result"
            )
        return result

    async def interrupt_active_agent(self, path: str) -> bool:
        """Cancel current work so all already-queued messages can proceed."""
        runtime_id, _resolved_path = self._resolve_active_path(path)
        result = await self._manager.send_message(
            runtime_id, {"action": "interrupt_queued"}
        )
        return bool(result)

    async def restart_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            await self._manager.restart_agent(self.agents[name].runtime_id)

    async def kill_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
        root_id = self._runtime_roots.get(runtime_id, runtime_id)
        hierarchy = self._hierarchies.get(root_id, {})
        runtime_ids = tuple(hierarchy.values()) or (runtime_id,)
        await asyncio.gather(
            *(
                self._manager.kill_agent(child_id)
                for child_id in reversed(runtime_ids)
            )
        )

    async def send_message(
        self, name: str, payload: Any, *, sender: str = "host"
    ) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
        await self._manager.send_message(runtime_id, payload, sender=sender)

    def _mark_working(self, path: str, delta: int) -> None:
        count = max(0, self._working_counts.get(path, 0) + delta)
        if count:
            self._working_counts[path] = count
        else:
            self._working_counts.pop(path, None)

    async def _handle_runtime_request(
        self,
        sender_runtime_id: str,
        action: str,
        payload: Mapping[str, Any],
        callbacks: Sequence[Any],
    ) -> Any:
        if action == "spawn":
            child_path = await self._spawn_runtime_subagent(
                sender_runtime_id,
                str(payload.get("name") or ""),
                payload.get("agent_type"),
            )
            child_id, child_path = self._resolve_path(
                sender_runtime_id, child_path
            )
            await self._notify_callbacks(
                callbacks, "on_agent_message_start", child_path
            )
            succeeded = False
            self._mark_working(child_path, 1)
            try:
                result = await self._manager.invoke(
                    child_id,
                    str(payload.get("message") or ""),
                    config={
                        "metadata": {
                            "ursa_message_recipient": child_path,
                            "ursa_message_sender": self._runtime_paths[
                                sender_runtime_id
                            ],
                        }
                    },
                    callbacks=callbacks,
                )
                succeeded = True
                return {"path": child_path, "response": result}
            finally:
                self._mark_working(child_path, -1)
                await self._notify_callbacks(
                    callbacks,
                    "on_agent_message_end",
                    child_path,
                    succeeded=succeeded,
                )
        if action == "list":
            root_id = self._runtime_roots[sender_runtime_id]
            return list(self._hierarchies[root_id])
        if action == "invoke":
            target_id, target_path = self._resolve_path(
                sender_runtime_id, str(payload.get("target") or "")
            )
            sender_path = self._runtime_paths[sender_runtime_id]
            if target_id == sender_runtime_id:
                raise ValueError("An agent cannot synchronously message itself")
            if sender_path.startswith(f"{target_path}/"):
                raise ValueError(
                    "Cannot synchronously message an ancestor; send work to "
                    "a child or sibling agent instead"
                )
            await self._notify_callbacks(
                callbacks, "on_agent_message_start", target_path
            )
            succeeded = False
            self._mark_working(target_path, 1)
            try:
                result = await self._manager.invoke(
                    target_id,
                    str(payload.get("message") or ""),
                    config={
                        "metadata": {
                            "ursa_message_recipient": target_path,
                            "ursa_message_sender": sender_path,
                        }
                    },
                    callbacks=callbacks,
                )
                succeeded = True
                return result
            finally:
                self._mark_working(target_path, -1)
                await self._notify_callbacks(
                    callbacks,
                    "on_agent_message_end",
                    target_path,
                    succeeded=succeeded,
                )
        raise ValueError(f"Unknown runtime request {action!r}")

    async def _spawn_runtime_subagent(
        self,
        parent_runtime_id: str,
        name: str,
        agent_type: Any = None,
    ) -> str:
        BaseAgent._validate_subagent_name(name)
        async with self._hierarchy_lock:
            root_id = self._runtime_roots[parent_runtime_id]
            hierarchy = self._hierarchies[root_id]
            parent_path = self._runtime_paths[parent_runtime_id]
            child_path = f"{parent_path}/{name}"
            if child_path in hierarchy:
                raise ValueError(f"Subagent name already exists: {name!r}")

            max_agents = self.config.subagents.max_agents
            max_depth = self.config.subagents.max_depth
            if max_agents == 0:
                raise SubagentLimitError(
                    "Subagents are disabled by subagents.max_agents=0"
                )
            if max_depth == 0:
                raise SubagentLimitError(
                    "Subagents are disabled by subagents.max_depth=0"
                )
            if max_agents is not None and len(hierarchy) - 1 >= max_agents:
                raise SubagentLimitError(
                    f"Subagent limit reached: subagents.max_agents={max_agents}"
                )
            depth = child_path.count("/")
            if max_depth is not None and depth > max_depth:
                raise SubagentLimitError(
                    "Subagent depth limit reached: "
                    f"subagents.max_depth={max_depth}"
                )

            parent_agent_name = self._runtime_agent_names[parent_runtime_id]
            agent_name = self._resolve_subagent_type(
                agent_type, parent_agent_name
            )
            runtime_id = f"{root_id}:sub:{child_path}"
            child_config = _resolve_host_secrets(
                self.config.model_copy(
                    update={
                        "llm_model": (
                            self.config.subagents.llm_model
                            or self.config.llm_model
                        ),
                        "emb_model": (
                            self.config.subagents.embedding_model
                            if self.config.subagents.embedding_model is not None
                            else self.config.emb_model
                        ),
                        "agent_name": None,
                        "thread_id": f"{self.thread_id}:{child_path}",
                        "agent_config": {
                            agent_name: dict(
                                self.config.agent_config.get(agent_name) or {}
                            )
                        },
                    }
                )
            )
            await self._manager.register(
                AgentSpec(
                    agent_id=runtime_id,
                    factory="ursa.runtime.agent_runtime:_ursa_agent_factory",
                    config={
                        "agent_name": agent_name,
                        "agent_path": child_path,
                        "ursa_config": _runner_config(child_config),
                    },
                )
            )
            self._manager.set_runtime_request_handler(
                runtime_id, self._handle_runtime_request
            )
            try:
                await self._manager.start_agent(runtime_id)
            except BaseException:
                await self._manager.unregister(runtime_id)
                raise
            hierarchy[child_path] = runtime_id
            self._dynamic_runtime_ids.add(runtime_id)
            self._runtime_paths[runtime_id] = child_path
            self._runtime_roots[runtime_id] = root_id
            self._runtime_agent_names[runtime_id] = agent_name
            return child_path

    def _resolve_subagent_type(self, requested: Any, inherited: str) -> str:
        """Validate a host-selected child type, defaulting to the parent."""
        if requested is None:
            return inherited
        if not isinstance(requested, str) or not requested.strip():
            raise ValueError("Subagent agent_type must be a non-blank string")
        selected = requested.strip()
        if selected not in self.agents:
            available = ", ".join(sorted(self.agents))
            raise ValueError(
                f"Unknown subagent agent_type {selected!r}; available types: "
                f"{available}"
            )
        return selected

    def _resolve_active_path(self, path: str) -> tuple[str, str]:
        root_id = self._active_root_id
        if root_id is None:
            raise RuntimeError("No active agent hierarchy")
        return self._resolve_path(root_id, path)

    def _resolve_path(
        self, sender_runtime_id: str, target: str
    ) -> tuple[str, str]:
        if not target.strip():
            raise ValueError("Agent path cannot be empty")
        root_id = self._runtime_roots[sender_runtime_id]
        hierarchy = self._hierarchies[root_id]
        sender_path = self._runtime_paths[sender_runtime_id]
        raw = target.strip().strip("/")
        if raw in {".", "self"}:
            resolved = sender_path
        elif raw == "root":
            resolved = "root"
        elif raw in {"..", "parent"}:
            if "/" not in sender_path:
                raise KeyError("The root agent has no parent")
            resolved = sender_path.rsplit("/", 1)[0]
        elif raw.startswith("root/"):
            resolved = raw
        else:
            child = f"{sender_path}/{raw}"
            sibling = (
                f"{sender_path.rsplit('/', 1)[0]}/{raw}"
                if "/" in sender_path
                else raw
            )
            if child in hierarchy:
                resolved = child
            elif sibling in hierarchy:
                resolved = sibling
            else:
                resolved = raw
        try:
            return hierarchy[resolved], resolved
        except KeyError:
            raise KeyError(
                f"Unknown agent path {target!r} from {sender_path!r}"
            ) from None

    @staticmethod
    async def _notify_callbacks(
        callbacks: Sequence[Any], method_name: str, *args: Any, **kwargs: Any
    ) -> None:
        for callback in callbacks:
            method = getattr(callback, method_name, None)
            if not callable(method):
                continue
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                await result

    async def reconfigure_model(
        self, model_name: str, inference_provider: str | None
    ) -> None:
        candidate = ChatModelConfig(
            model=model_name,
            inference_provider=inference_provider,
            max_completion_tokens=self.config.llm_model.max_completion_tokens,
        )
        await self.reconfigure_models(candidate, self.config.emb_model)

    async def reconfigure_models(
        self,
        chat_config: ChatModelConfig,
        embedding_config: EmbModelConfig | None,
    ) -> None:
        async with self._operation_lock:
            candidate = self.config.model_copy(
                update={
                    "llm_model": chat_config,
                    "emb_model": embedding_config,
                }
            )
            resolved = _resolve_host_secrets(candidate)
            for runtime_id in reversed(tuple(self._dynamic_runtime_ids)):
                await self._manager.unregister(runtime_id)
            self._dynamic_runtime_ids.clear()
            for name in tuple(self._registered):
                await self._manager.unregister(self.agents[name].runtime_id)
                self._registered.discard(name)
            self._runtime_paths.clear()
            self._runtime_roots.clear()
            self._runtime_agent_names.clear()
            self._hierarchies.clear()
            self._working_counts.clear()
            self._pending_counts.clear()
            self._active_root_id = None
            self._active_agent_ready.clear()
            self.config = resolved
            self.inference_provider = chat_config.inference_provider
            self.embedding_inference_provider = (
                embedding_config.inference_provider
                if embedding_config
                else None
            )

    async def aclose(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._aclose_once())
            self._close_task = task
        await asyncio.shield(task)

    async def _aclose_once(self) -> None:
        async with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_manager:
                await self._manager.aclose()
            else:
                for runtime_id in reversed(tuple(self._dynamic_runtime_ids)):
                    await self._manager.unregister(runtime_id)
                self._dynamic_runtime_ids.clear()
                for name in tuple(self._registered):
                    await self._manager.unregister(self.agents[name].runtime_id)
                    self._registered.discard(name)

    async def close(self) -> None:
        await self.aclose()

    def as_mcp_server(self, **kwargs: Any) -> FastMCP:
        from ursa import __version__ as ursa_version

        mcp = FastMCP(
            "URSA", version=ursa_version, on_duplicate="error", **kwargs
        )
        for name, agent in self.agents.items():
            mcp.tool(
                self._make_agent_tool(name),
                name=name,
                description=agent.description,
            )
        return mcp

    def _make_agent_tool(self, agent_name: str):
        async def call_agent(prompt: str) -> str:
            return await self.run_agent(agent_name, prompt)

        return call_agent
