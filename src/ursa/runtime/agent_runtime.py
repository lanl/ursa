"""Process-isolated URSA agent facade for interactive hosts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ursa import agents
from ursa.agents.base import AgentWithTools, BaseAgent
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    UrsaConfig,
    resolve_ursa_config,
)
from ursa.messaging.coordination import (
    MessageWait,
    check_capacity,
    resolve_message_routes,
    wait_dependency,
    wait_for_completion,
    wait_targets,
    wait_timeout,
    wake_waiters,
)
from ursa.messaging.messages import (
    agent_message_wait_result,
    attach_agent_message_instructions,
)
from ursa.security import enforce_model_group_policy
from ursa.util.async_utils import cancel_tasks, maybe_await
from ursa.util.has_optional_dep_group import has_optional_dep_group
from ursa.util.inference_providers import validate_model_provider
from ursa.util.mcp import (
    mcp_tool_specs_from_details,
    mcp_tools_from_specs,
    start_mcp_client,
)
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
    for field_name in ("llm_model", "emb_model"):
        model = getattr(resolved, field_name)
        if model is None or not isinstance(model.api_key, SecretReference):
            continue
        setattr(
            resolved,
            field_name,
            model.model_copy(update={"api_key": model.api_key.resolve()}),
        )

    for server_name, server in tuple(resolved.mcp_servers.items()):
        if isinstance(server, Mapping):
            headers = server.get("headers")
        else:
            headers = getattr(server, "headers", None)
        if not isinstance(headers, Mapping) or not headers:
            continue
        materialized = {}
        for name, value in headers.items():
            if isinstance(value, Mapping):
                value = SecretReference.maybe_validate(value)
            if not isinstance(value, SecretReference):
                materialized[name] = value
                continue
            secret = value.resolve()
            if secret is None:
                raise ValueError(
                    f"Secret for MCP server {server_name!r} is not set"
                )
            materialized[name] = secret.get_secret_value()
        if isinstance(server, BaseModel):
            resolved.mcp_servers[server_name] = server.model_copy(
                update={"headers": materialized}
            )
        else:
            resolved.mcp_servers[server_name] = {
                **server,
                "headers": materialized,
            }
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
    message_sender: str | None = None
    message_recipient: str | None = None
    additional_messages: tuple[tuple[str, str, str], ...] = ()
    on_start: Any = None


@dataclass(slots=True)
class _BufferedAgentMessage:
    """A routed message held until its busy recipient can accept it."""

    sender_path: str
    target_id: str
    target_path: str
    message: str
    callbacks: tuple[Any, ...]
    # User-originated deliveries wait for a response. Peer messages are
    # fire-and-forget and leave this unset.
    response: asyncio.Future[str | None] | None = None

    def envelope(self) -> dict[str, str]:
        """Retain the trusted route and exact body in either delivery path."""
        return {
            "sender": self.sender_path,
            "recipient": self.target_path,
            "content": self.message,
        }


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

    async def send_messages(
        self, targets: list[str], message: str
    ) -> list[Any]:
        result = await self.context.request_runtime(
            "invoke_many", targets=targets, message=message
        )
        if not isinstance(result, list):
            raise TypeError("The host returned invalid batched agent results")
        return result

    async def list_agents(self) -> list[str]:
        result = await self.context.request_runtime("list")
        return [str(path) for path in result]

    async def wait_for_agent(
        self,
        target: str | list[str],
        timeout_seconds: float = 120,
        *,
        wait_all: bool = False,
    ) -> str:
        return str(
            await self.context.request_runtime(
                "wait_for",
                target=target,
                timeout_seconds=timeout_seconds,
                wait_all=wait_all,
            )
        )

    async def stop_subagent(self, target: str) -> str:
        """Stop a descendant hierarchy subtree through the host."""
        return str(await self.context.request_runtime("stop", target=target))

    async def wake_waiters(self) -> None:
        """Wake this agent's host-managed waits after an inbox delivery."""
        await self.context.request_runtime("wake_waiters")


class _UrsaProcessAgent(_ProcessRuntimeBridge):
    """One fully constructed URSA agent living exclusively in its child."""

    def __init__(
        self,
        agent: Any,
        context: AgentContext,
        agent_path: str,
        tool_sources: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(context, agent_path)
        self.agent = agent
        self.tool_sources = dict(tool_sources or {})
        self.state: Any = None
        self._inbox: asyncio.Queue[_RuntimePrompt] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def ainvoke(
        self, inputs: Any, config: Mapping[str, Any] | None = None
    ) -> str:
        return await self._submit(str(inputs), config=config)

    async def on_message(self, message: Any) -> Any:
        """Accept queued user messages without blocking an active invoke."""
        # A wait_for call is deliberately hosted outside this process, so an
        # incoming message can always wake it even while its model invocation
        # is paused in the tool call.
        runtime = getattr(self.agent, "_process_runtime", None)
        if runtime is not None:
            await runtime.wake_waiters()
        payload = message.payload
        if not isinstance(payload, Mapping):
            return await self._submit(str(payload))
        action = str(payload.get("action") or "request")
        if action != "request":
            raise ValueError(f"Unknown runtime message action {action!r}")
        config = self.context.callback_config(payload.get("config"))

        async def on_start() -> None:
            await self.context.emit_callback(
                "on_agent_message_start", self.agent_path
            )

        succeeded = False
        try:
            result = await self._submit(
                str(payload.get("prompt") or ""),
                config=config,
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
        on_start: Any = None,
    ) -> str:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run_inbox(),
                name=f"ursa-runtime-inbox:{self.agent_path}",
            )
        response = asyncio.get_running_loop().create_future()
        configured = dict(config or {})
        metadata = configured.get("metadata")
        message_sender = None
        message_recipient = None
        additional_messages = []
        if isinstance(metadata, Mapping):
            sender = metadata.get("ursa_message_sender")
            recipient = metadata.get("ursa_message_recipient")
            if isinstance(sender, str) and isinstance(recipient, str):
                message_sender = sender
                message_recipient = recipient
        # Consume transport-only content here; do not copy it into every model
        # callback's metadata or forward it to LangChain's runnable config.
        for envelope in configured.pop("ursa_additional_messages", ()):
            if (
                not isinstance(envelope, Mapping)
                or not all(
                    isinstance(envelope.get(key), str)
                    for key in ("sender", "recipient", "content")
                )
                or envelope["recipient"] != self.agent_path
            ):
                raise ValueError("Invalid batched agent-message envelope")
            additional_messages.append((
                envelope["sender"],
                envelope["recipient"],
                envelope["content"],
            ))
        item = _RuntimePrompt(
            prompt=prompt,
            config=configured,
            response=response,
            request_id=self.context.active_request_id,
            message_sender=message_sender,
            message_recipient=message_recipient,
            additional_messages=tuple(additional_messages),
            on_start=on_start,
        )
        await self._inbox.put(item)
        return await response

    def _format_prompt(self, item: _RuntimePrompt) -> Any:
        """Format once, retaining every delivery envelope in a batched turn."""
        if not item.additional_messages:
            query = self.agent.format_query(item.prompt, state=self.state)
            if item.message_sender and item.message_recipient:
                return attach_agent_message_instructions(
                    query,
                    sender=item.message_sender,
                    recipient=item.message_recipient,
                )
            return query

        envelopes = [
            (
                item.message_sender or "user",
                item.message_recipient or self.agent_path,
                item.prompt,
            ),
            *item.additional_messages,
        ]
        combined = (
            "Messages received together, oldest first. Read all before acting:\n"
            + json.dumps(
                [
                    {
                        "sender": sender,
                        "recipient": recipient,
                        "content": content,
                    }
                    for sender, recipient, content in envelopes
                ],
                ensure_ascii=False,
            )
        )
        # Some agent types reset history or set task/query fields here. A
        # single call keeps those fields aware of the complete incoming batch.
        formatted = self.agent.format_query(combined, state=self.state)
        query = dict(formatted) if isinstance(formatted, Mapping) else {}
        messages = list(query.get("messages") or [])
        if (
            messages
            and isinstance(messages[-1], HumanMessage)
            and messages[-1].content == combined
        ):
            messages.pop()
        query["messages"] = messages
        for sender, recipient, content in envelopes:
            query["messages"].append(HumanMessage(content=content))
            query = dict(
                attach_agent_message_instructions(
                    query,
                    sender=sender,
                    recipient=recipient,
                )
            )
        return query

    async def _run_inbox(self) -> None:
        while True:
            item = await self._inbox.get()
            request_token = None
            # Formatting and copying are part of this request's failure domain;
            # neither may terminate the inbox worker and strand later requests.
            committed_state = self.state
            try:
                request_token = self.context._enter_request(
                    item.request_id or ""
                )
                committed_state = deepcopy(self.state)
                if item.on_start is not None:
                    await maybe_await(item.on_start())
                # Formatting can scan skills or read files. Keep the inbox and
                # heartbeat responsive while that synchronous work runs.
                query = await asyncio.to_thread(self._format_prompt, item)
                # Keep model cancellation separate from worker shutdown.
                self.state = await asyncio.create_task(
                    self.agent.ainvoke(query, config=item.config or None)
                )
                # Rendering failures must preserve the completed model state.
                committed_state = self.state
                if not item.response.done():
                    item.response.set_result(
                        str(self.agent.format_result(self.state))
                    )
            except asyncio.CancelledError:
                self.state = committed_state
                stopping = asyncio.current_task().cancelling()
                if not item.response.done():
                    if stopping:
                        item.response.cancel()
                    else:
                        item.response.set_exception(
                            RuntimeError("Agent invocation was cancelled")
                        )
                if stopping:
                    raise
            except Exception as exc:
                self.state = committed_state
                if not item.response.done():
                    item.response.set_exception(exc)
            finally:
                if request_token is not None:
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
            tool_name = str(getattr(tool, "name", None) or fallback_name)
            schema_json = await _tool_schema(tool)
            # MCP adapter tools retain the server's unmodified input schema as
            # a dict.  Keep it separately from the presentation schema so a
            # child can recreate the same MCP ``Tool`` without a second
            # list-tools request.
            raw_mcp_schema = getattr(tool, "args_schema", None)
            if not isinstance(raw_mcp_schema, Mapping):
                raw_mcp_schema = None
            details.append({
                "name": tool_name,
                "class_name": tool.__class__.__name__,
                "description": str(
                    getattr(tool, "description", None)
                    or "No description available."
                ).strip(),
                "schema": schema_json,
                "return_direct": getattr(tool, "return_direct", None),
                "mcp_server": str(self.tool_sources.get(tool_name, "")),
                "mcp_input_schema": (
                    dict(raw_mcp_schema)
                    if tool_name in self.tool_sources
                    and raw_mcp_schema is not None
                    else None
                ),
                "mcp_metadata": (
                    getattr(tool, "metadata", None)
                    if tool_name in self.tool_sources
                    else None
                ),
            })
        return details

    async def describe_mcp_tools(self) -> dict[str, Any]:
        """Describe only MCP tools when their discovery is known complete.

        A general tool listing cannot tell whether absent MCP provenance means
        the server returned no tools or that the agent does not support MCP.
        Cache only the former.
        """
        if not isinstance(self.agent, AgentWithTools):
            return {"complete": False, "tools": []}
        details = await self.describe_tools()
        return {
            "complete": True,
            "tools": [detail for detail in details if detail.get("mcp_server")],
        }

    # Preserve the process RPC names while sharing the model-facing bridge.
    send_subagent_message = _ProcessRuntimeBridge.send_message
    send_subagent_messages = _ProcessRuntimeBridge.send_messages
    list_subagents = _ProcessRuntimeBridge.list_agents
    wait_for_subagent = _ProcessRuntimeBridge.wait_for_agent

    async def aclose(self) -> None:
        if self._worker is not None:
            await cancel_tasks(self._worker)
        while not self._inbox.empty():
            item = self._inbox.get_nowait()
            if not item.response.done():
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
            cached_specs = config.get("mcp_tool_specs")
            if cached_specs is not None:
                try:
                    if not isinstance(cached_specs, list):
                        raise TypeError(
                            "Cached MCP tool specifications must be a list"
                        )
                    tools, tool_sources = mcp_tools_from_specs(
                        client, cached_specs
                    )
                    agent.add_tool(tools)
                except (TypeError, ValueError, KeyError):
                    # Preserve tool availability if a captured specification
                    # is malformed or incompatible with this MCP adapter.
                    # Normal cached starts still avoid discovery entirely.
                    LOGGER.warning(
                        "Cached MCP schemas are unusable for %s; rediscovering",
                        agent_path,
                        exc_info=True,
                    )
                    tool_sources = await agent.add_mcp_tools(client)
            else:
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
        self.started = False

    async def on_agent_message_start(self, *_args: Any, **_kwargs: Any) -> None:
        if self.started:
            return
        self.started = True
        await maybe_await(self.callback(None))


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
        self._manager = manager or AgentManager(runtime=self.config.runtime)
        self._registered: set[str] = set()
        self._mcp_tool_specs: list[dict[str, Any]] | None = None
        self._mcp_tool_specs_fingerprint: str | None = None
        self._mcp_tool_specs_lock = asyncio.Lock()
        self._dynamic_runtime_ids: set[str] = set()
        self._subagent_start_tasks: dict[str, asyncio.Task[None]] = {}
        self._runtime_paths: dict[str, str] = {}
        self._runtime_roots: dict[str, str] = {}
        self._runtime_agent_names: dict[str, str] = {}
        self._hierarchies: dict[str, dict[str, str]] = {}
        self._working_counts: dict[str, int] = {}
        self._background_agent_tasks: set[asyncio.Task[Any]] = set()
        self._background_teardown_tasks: set[asyncio.Task[None]] = set()
        self._stopping_subtrees: set[tuple[str, str]] = set()
        self._wait_dependencies: dict[str, set[MessageWait[str]]] = {}
        self._buffered_agent_messages: dict[
            str, deque[_BufferedAgentMessage]
        ] = {}
        self._active_root_id: str | None = None
        self._active_agent_ready = asyncio.Event()
        self._register_lock = asyncio.Lock()
        self._hierarchy_lock = asyncio.Lock()
        self._subagent_cleanup_lock = asyncio.Lock()
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
            runtime_id = self.agents[name].runtime_id
            await self._register_process(runtime_id, name, "root", self.config)
            self._runtime_paths[runtime_id] = "root"
            self._runtime_roots[runtime_id] = runtime_id
            self._runtime_agent_names[runtime_id] = name
            self._hierarchies.setdefault(runtime_id, {"root": runtime_id})
            self._registered.add(name)

    async def _register_process(
        self,
        runtime_id: str,
        name: str,
        path: str,
        config: UrsaConfig,
        mcp_tool_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        # multiprocessing copies each configuration. A shallow host copy avoids
        # copying TemporaryDirectory's cleanup finalizer while filtering options.
        config = config.model_copy(
            update={
                "agent_config": {
                    name: dict(config.agent_config.get(name) or {})
                }
            }
        )
        payload = {
            "agent_name": name,
            "agent_path": path,
            "ursa_config": _runner_config(config),
        }
        if mcp_tool_specs is not None:
            payload["mcp_tool_specs"] = mcp_tool_specs
        await self._manager.register(
            AgentSpec(
                agent_id=runtime_id,
                factory="ursa.runtime.agent_runtime:_ursa_agent_factory",
                config=payload,
            )
        )
        self._manager.set_runtime_request_handler(
            runtime_id, self._handle_runtime_request
        )

    async def get_agent(self, name: str) -> RuntimeAgent:
        async with self.use_agent(name) as agent:
            return agent

    @asynccontextmanager
    async def use_agent(self, name: str) -> AsyncGenerator[RuntimeAgent]:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
            await self._manager.start_agent(runtime_id)
            await self._capture_mcp_tool_specs(runtime_id)
            yield self.agents[name]

    async def run_agent(
        self,
        name: str,
        prompt: str,
        callbacks: Sequence[Any] | None = None,
    ) -> str:
        """Run one agent without blocking lifecycle control operations."""
        async with self.use_agent(name) as agent:
            runtime_id = agent.runtime_id
            self._active_root_id = runtime_id
            self._active_agent_ready.set()
        configured = tuple(callbacks or ())
        await self._notify_callbacks(
            configured,
            "on_hierarchy_message",
            "user",
            ("root",),
            prompt,
        )
        self._mark_working("root", 1)
        try:
            result = await self._manager.invoke(
                runtime_id,
                prompt,
                config={
                    "metadata": {
                        "ursa_message_sender": "user",
                        "ursa_message_recipient": "root",
                    }
                },
                callbacks=configured,
            )
        finally:
            self._mark_working("root", -1)
        return await self._record_user_reply(name, "root", result, configured)

    async def _capture_mcp_tool_specs(self, runtime_id: str) -> None:
        """Reuse root MCP schemas when starting equivalent child agents."""
        fingerprint = self._mcp_server_fingerprint()
        if not self.config.mcp_servers:
            return
        async with self._mcp_tool_specs_lock:
            if (
                self._mcp_tool_specs is not None
                and self._mcp_tool_specs_fingerprint == fingerprint
            ):
                return
            # A config change with an existing root must never let an old
            # schema through to children. Its next start will recapture.
            self._mcp_tool_specs = None
            self._mcp_tool_specs_fingerprint = None
            result = await self._manager.call(runtime_id, "describe_mcp_tools")
            if not isinstance(result, Mapping):
                raise TypeError("Agent returned invalid MCP tool descriptions")
            if result.get("complete") is not True:
                return
            details = result.get("tools")
            if not isinstance(details, list) or not all(
                isinstance(detail, Mapping) for detail in details
            ):
                raise TypeError("Agent returned invalid MCP tool descriptions")
            self._mcp_tool_specs = mcp_tool_specs_from_details(details)
            self._mcp_tool_specs_fingerprint = fingerprint

    def _mcp_server_fingerprint(self) -> str:
        """Return a host-only identity for the resolved MCP configuration."""
        payload: dict[str, Any] = {}
        for name, server in self.config.mcp_servers.items():
            dump = getattr(server, "model_dump", None)
            payload[name] = (
                dump(mode="json") if callable(dump) else repr(server)
            )
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _cached_mcp_tool_specs(self) -> list[dict[str, Any]] | None:
        """Return a cache entry only when it matches this server config."""
        if (
            self._mcp_tool_specs is not None
            and self._mcp_tool_specs_fingerprint
            == self._mcp_server_fingerprint()
        ):
            return self._mcp_tool_specs
        return None

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
        for path, runtime_id in self._hierarchies.get(root_id, {}).items():
            pending = len(self._buffered_agent_messages.get(path, ()))
            if self._is_stopping_path(root_id, path):
                state = "Stopping"
            elif runtime_id in self._subagent_start_tasks:
                state = "Starting"
            elif self._wait_dependencies.get(path):
                state = "Waiting"
            elif self._working_counts.get(path, 0):
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
        on_start: Any = None,
    ) -> str | None:
        """Deliver a user message through the same queue as peer messages.

        ``None`` means the message was consumed by a running ``wait_for`` or
        included in a batch whose reply belongs to another user turn. There
        is no separate response to render for that delivery.
        """
        runtime_id, resolved_path = self._resolve_active_path(path)
        configured = list(callbacks or ())
        message_start = None
        if on_start is not None:
            message_start = _MessageStartCallback(on_start)
            configured.append(message_start)
        await self._notify_callbacks(
            configured,
            "on_hierarchy_message",
            "user",
            (resolved_path,),
            prompt,
        )
        response = asyncio.get_running_loop().create_future()
        try:
            async with self._hierarchy_lock:
                root_id = self._runtime_roots.get(runtime_id)
                if (
                    root_id is None
                    or self._hierarchies.get(root_id, {}).get(resolved_path)
                    != runtime_id
                ):
                    raise RuntimeError(
                        f"Unknown active agent {resolved_path!r}"
                    )
                if self._is_stopping_path(root_id, resolved_path):
                    raise RuntimeError(f"Agent {resolved_path!r} is stopping")
                self._accept_message_target(
                    "user",
                    runtime_id,
                    resolved_path,
                    prompt,
                    configured,
                    response=response,
                )
            result = await response
        except BaseException:
            if not response.done():
                response.cancel()
            raise
        if result is None:
            if message_start is not None:
                await message_start.on_agent_message_start()
            return None
        return await self._record_user_reply(
            resolved_path, resolved_path, result, configured
        )

    async def _record_user_reply(
        self, name: str, path: str, result: Any, callbacks: Sequence[Any]
    ) -> str:
        """Validate and record a completed reply to the user."""
        if not isinstance(result, str):
            raise TypeError(f"Agent {name!r} returned a non-string result")
        await self._notify_callbacks(
            callbacks, "on_hierarchy_message", path, ("user",), result
        )
        return result

    async def restart_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            self._mcp_tool_specs = None
            self._mcp_tool_specs_fingerprint = None
            runtime_id = self.agents[name].runtime_id
            await self._manager.restart_agent(runtime_id)
            await self._capture_mcp_tool_specs(runtime_id)

    async def kill_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
            self._mcp_tool_specs = None
            self._mcp_tool_specs_fingerprint = None
        root_id = self._runtime_roots.get(runtime_id, runtime_id)
        hierarchy = self._hierarchies.get(root_id, {})
        runtime_ids = tuple(hierarchy.values()) or (runtime_id,)
        await self._cancel_subagent_starts(runtime_ids)
        await asyncio.gather(
            *(
                self._manager.kill_agent(child_id)
                for child_id in reversed(runtime_ids)
                if child_id == root_id or child_id in self._dynamic_runtime_ids
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
            self._flush_buffered_agent_messages(path)

    def _flush_buffered_agent_messages(self, path: str) -> None:
        """Schedule messages that were not consumed by an in-flight wait."""
        buffered = self._buffered_agent_messages.pop(path, None)
        if not buffered:
            return
        items = tuple(buffered)
        primary, additional = items[0], items[1:]
        root_id = self._runtime_roots.get(primary.target_id)
        if (
            self._closed
            or root_id is None
            or self._is_stopping_path(root_id, path)
        ):
            self._fail_buffered_messages(
                (primary, *additional),
                RuntimeError(f"Agent {path!r} is no longer available"),
            )
            return
        self._schedule_message_target(primary, additional=additional)

    def _take_buffered_agent_messages(
        self, path: str
    ) -> tuple[_BufferedAgentMessage, ...]:
        """Drain arrivals together so one busy turn cannot hide newer input."""
        buffered = self._buffered_agent_messages.pop(path, None) or ()
        for item in buffered:
            self._complete_in_band_delivery(item)
        return tuple(buffered)

    @staticmethod
    def _complete_in_band_delivery(item: _BufferedAgentMessage) -> None:
        """Settle a user message that was consumed by a wait tool result."""
        if item.response is not None and not item.response.done():
            item.response.set_result(None)

    @staticmethod
    def _fail_buffered_messages(
        items: Sequence[_BufferedAgentMessage], error: BaseException
    ) -> None:
        for item in items:
            if item.response is not None and not item.response.done():
                item.response.set_exception(error)

    def _fail_all_buffered_messages(self, error: BaseException) -> None:
        """Unblock TUI workers when a lifecycle operation drops delivery."""
        for buffered in self._buffered_agent_messages.values():
            self._fail_buffered_messages(tuple(buffered), error)
        self._buffered_agent_messages.clear()

    async def _handle_runtime_request(
        self,
        sender_runtime_id: str,
        action: str,
        payload: Mapping[str, Any],
        callbacks: Sequence[Any],
    ) -> Any:
        if action == "spawn":
            message = str(payload.get("message") or "")
            child_path = await self._spawn_runtime_subagent(
                sender_runtime_id,
                str(payload.get("name") or ""),
                payload.get("agent_type"),
            )
            child_id, child_path = self._resolve_path(
                sender_runtime_id, child_path
            )
            acknowledgements = await self._queue_message_targets(
                sender_runtime_id, [(child_id, child_path)], message, callbacks
            )
            return {"path": child_path, "response": acknowledgements[0]}
        if action == "list":
            root_id = self._runtime_roots[sender_runtime_id]
            return list(self._hierarchies[root_id])
        if action == "wait_for":
            return await self._wait_for_runtime_agent(
                sender_runtime_id,
                payload.get("target"),
                payload.get("timeout_seconds", 120),
                wait_all=bool(payload.get("wait_all", False)),
            )
        if action == "stop":
            return await self._stop_runtime_subagent(
                sender_runtime_id, str(payload.get("target") or "")
            )
        if action == "wake_waiters":
            self._wake_runtime_waiters(sender_runtime_id)
            return None
        if action in {"invoke", "invoke_many"}:
            targets = (
                [str(payload.get("target") or "")]
                if action == "invoke"
                else payload.get("targets")
            )
            if action == "invoke_many":
                if not isinstance(targets, list) or not targets:
                    raise ValueError(
                        "targets must be a non-empty list of agent paths"
                    )
                if not all(
                    isinstance(target, str) and target.strip()
                    for target in targets
                ):
                    raise ValueError(
                        "each target must be a non-empty agent path"
                    )

            resolved = self._resolve_message_targets(sender_runtime_id, targets)
            acknowledgements = await self._queue_message_targets(
                sender_runtime_id,
                resolved,
                str(payload.get("message") or ""),
                callbacks,
            )
            if action == "invoke":
                return "\n\n".join(acknowledgements)
            return [
                {"result": acknowledgement}
                for acknowledgement in acknowledgements
            ]
        raise ValueError(f"Unknown runtime request {action!r}")

    def _resolve_message_targets(
        self, sender_runtime_id: str, targets: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Expand send-only selectors and preflight every concrete route.

        The expansion happens before callbacks or delivery scheduling. It is
        deliberately not part of ``_resolve_path``: wait, stop, and spawn keep
        their exact path-only grammar.
        """
        return resolve_message_routes(
            self._runtime_paths[sender_runtime_id],
            targets,
            self._hierarchies[self._runtime_roots[sender_runtime_id]],
            lambda path: self._validate_message_target(sender_runtime_id, path),
        )

    async def _queue_message_targets(
        self,
        sender_runtime_id: str,
        resolved: Sequence[tuple[str, str]],
        message: str,
        callbacks: Sequence[Any],
    ) -> list[str]:
        """Log and accept an all-or-nothing set of concrete message routes."""
        sender_path = self._runtime_paths[sender_runtime_id]
        async with self._hierarchy_lock:
            root_id = self._runtime_roots[sender_runtime_id]
            for target_id, target_path in resolved:
                if self._hierarchies[root_id].get(target_path) != target_id:
                    raise KeyError(
                        f"Unknown agent path {target_path!r} from {sender_path!r}"
                    )
                if self._is_stopping_path(root_id, target_path):
                    raise ValueError(f"Agent {target_path!r} is stopping")
            acknowledgements = [
                self._accept_message_target(
                    sender_path, target_id, target_path, message, callbacks
                )
                for target_id, target_path in resolved
            ]
        # Delivery is now accepted in the host queue. Log the aggregate only
        # after the lock recheck succeeds, so rejected broadcasts never appear
        # in `/coordination` as if they were received.
        try:
            await self._notify_callbacks(
                callbacks,
                "on_hierarchy_message",
                sender_path,
                tuple(target_path for _target_id, target_path in resolved),
                message,
            )
        except Exception:
            LOGGER.exception("Could not record a delivered hierarchy message")
        return acknowledgements

    def _validate_message_target(
        self, sender_runtime_id: str, target: str
    ) -> tuple[str, str]:
        """Resolve a route before queueing a one-way message."""
        target_id, target_path = self._resolve_path(sender_runtime_id, target)
        root_id = self._runtime_roots[sender_runtime_id]
        if self._is_stopping_path(root_id, target_path):
            raise ValueError(f"Agent {target_path!r} is stopping")
        return target_id, target_path

    def _is_stopping_path(self, root_id: str, path: str) -> bool:
        return any(
            stopping_root == root_id
            and (path == stopping_path or path.startswith(f"{stopping_path}/"))
            for stopping_root, stopping_path in self._stopping_subtrees
        )

    async def _stop_runtime_subagent(
        self, sender_runtime_id: str, target: str
    ) -> str:
        """Recursively unregister a descendant subtree owned by the caller."""
        target_id, target_path = self._resolve_path(sender_runtime_id, target)
        sender_path = self._runtime_paths[sender_runtime_id]
        if target_id == sender_runtime_id or not target_path.startswith(
            f"{sender_path}/"
        ):
            raise ValueError("An agent can only stop one of its descendants")

        async with self._hierarchy_lock:
            root_id = self._runtime_roots[sender_runtime_id]
            hierarchy = self._hierarchies[root_id]
            subtree = [
                (path, runtime_id)
                for path, runtime_id in hierarchy.items()
                if path == target_path or path.startswith(f"{target_path}/")
            ]
            # A valid resolved target is necessarily in this hierarchy, but
            # retaining a defensive error makes a race recoverable to the tool.
            if not subtree:
                raise KeyError(
                    f"Unknown agent path {target!r} from {sender_path!r}"
                )
            if self._is_stopping_path(root_id, target_path):
                raise ValueError(f"Agent {target_path!r} is already stopping")
            self._stopping_subtrees.add((root_id, target_path))
            task_names = {
                f"ursa-message-{path}" for path, _runtime_id in subtree
            }
            for task in tuple(self._background_agent_tasks):
                if task.get_name() in task_names:
                    task.cancel()
            teardown = asyncio.create_task(
                self._teardown_runtime_subagent(root_id, target_path, subtree),
                name=f"ursa-stop-agent-{target_path}",
            )
            self._background_teardown_tasks.add(teardown)
            teardown.add_done_callback(self._teardown_task_done)
        return f"Stopped agent {target_path} and its subagents."

    async def _teardown_runtime_subagent(
        self,
        root_id: str,
        target_path: str,
        subtree: list[tuple[str, str]],
    ) -> None:
        """Kill an already-marked subtree without blocking its caller."""
        try:
            await self._cancel_subagent_starts(
                tuple(runtime_id for _path, runtime_id in subtree)
            )
            for _path, runtime_id in sorted(
                subtree, key=lambda item: item[0].count("/"), reverse=True
            ):
                await self._unregister_subagent(runtime_id)
        finally:
            async with self._hierarchy_lock:
                hierarchy = self._hierarchies.get(root_id)
                for path, runtime_id in subtree:
                    if (
                        hierarchy is not None
                        and hierarchy.get(path) == runtime_id
                    ):
                        hierarchy.pop(path, None)
                    self._dynamic_runtime_ids.discard(runtime_id)
                    self._runtime_paths.pop(runtime_id, None)
                    self._runtime_roots.pop(runtime_id, None)
                    self._runtime_agent_names.pop(runtime_id, None)
                    self._working_counts.pop(path, None)
                    buffered = self._buffered_agent_messages.pop(path, ())
                    self._fail_buffered_messages(
                        tuple(buffered),
                        RuntimeError(f"Agent {path!r} was stopped"),
                    )
                self._stopping_subtrees.discard((root_id, target_path))
                self._wake_all_waiters()

    def _teardown_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_teardown_tasks.discard(task)
        self._log_task_failure(task, "Asynchronous subagent teardown failed")

    @staticmethod
    def _log_task_failure(task: asyncio.Task[Any], message: str) -> None:
        """Observe a background task's failure without disrupting its host."""
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            LOGGER.error(
                message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _wait_for_runtime_agent(
        self,
        sender_runtime_id: str,
        target: Any,
        timeout_value: Any,
        *,
        wait_all: bool = False,
    ) -> str:
        """Wait interruptibly for any or all hierarchy members to become idle."""
        targets = wait_targets(target)
        resolved = [
            self._resolve_path(sender_runtime_id, item) for item in targets
        ]
        target_paths = [path for _target_id, path in resolved]
        if any(target_id == sender_runtime_id for target_id, _path in resolved):
            raise ValueError("An agent cannot wait for itself")
        sender_path = self._runtime_paths[sender_runtime_id]
        timeout = wait_timeout(timeout_value)
        if buffered := self._take_buffered_agent_messages(sender_path):
            if len(buffered) == 1:
                item = buffered[0]
                return agent_message_wait_result(
                    item.sender_path, item.target_path, item.message
                )
            return (
                f"Received {len(buffered)} messages while waiting. "
                "Each entry retains its sender, recipient, and exact content:\n"
                + json.dumps(
                    [item.envelope() for item in buffered],
                    ensure_ascii=False,
                )
            )

        waiter = MessageWait[str](targets=tuple(target_paths))
        self._wait_dependencies.setdefault(sender_path, set()).add(waiter)
        try:
            return await wait_for_completion(
                waiter,
                target_paths,
                lambda: self._completed_wait_targets(resolved, sender_path),
                timeout,
                wait_all,
            )
        finally:
            dependencies = self._wait_dependencies.get(sender_path)
            if dependencies is not None:
                dependencies.discard(waiter)
                if not dependencies:
                    self._wait_dependencies.pop(sender_path, None)

    def _completed_wait_targets(
        self, resolved: list[tuple[str, str]], waiting_path: str
    ) -> list[tuple[str, str | None]]:
        """Return idle targets or targets that would close a wait cycle."""
        completed = []
        for runtime_id, path in resolved:
            if runtime_id in self._subagent_start_tasks:
                continue
            if not self._working_counts.get(path, 0):
                completed.append((path, None))
            elif relation := wait_dependency(
                path,
                waiting_path,
                lambda agent: (
                    target
                    for waiter in self._wait_dependencies.get(agent, ())
                    for target in waiter.targets
                ),
            ):
                completed.append((path, relation))
        return completed

    def _wake_runtime_waiters(
        self, sender_runtime_id: str, *, delivery: str | None = None
    ) -> bool:
        return wake_waiters(
            self._wait_dependencies.get(
                self._runtime_paths.get(sender_runtime_id), ()
            ),
            delivery,
        )

    def _accept_message_target(
        self,
        sender_path: str,
        target_id: str,
        target_path: str,
        message: str,
        callbacks: Sequence[Any],
        *,
        response: asyncio.Future[str | None] | None = None,
    ) -> str:
        """Accept a message while the caller holds ``_hierarchy_lock``."""
        # A model paused inside wait_for cannot process a separate inbox turn
        # until its current invocation finishes. Deliver the first arriving
        # message as the tool result instead, so it can react immediately and
        # do not queue the same message for a duplicate later turn.
        if self._wake_runtime_waiters(
            target_id,
            delivery=agent_message_wait_result(
                sender_path, target_path, message
            ),
        ):
            if response is not None and not response.done():
                response.set_result(None)
            return f"Message queued for {target_path}."
        item = _BufferedAgentMessage(
            sender_path=sender_path,
            target_id=target_id,
            target_path=target_path,
            message=message,
            callbacks=tuple(callbacks),
            response=response,
        )
        if self._working_counts.get(target_path, 0):
            self._buffered_agent_messages.setdefault(
                target_path, deque()
            ).append(item)
            return f"Message queued for {target_path}."
        self._schedule_message_target(item)
        return f"Message queued for {target_path}."

    def _schedule_message_target(
        self,
        item: _BufferedAgentMessage,
        *,
        additional: Sequence[_BufferedAgentMessage] = (),
    ) -> None:
        """Start one turn for already-validated messages to the same agent."""
        self._mark_working(item.target_path, 1)
        items = (item, *additional)
        task_state = {"started": False}
        task = asyncio.create_task(
            self._invoke_message_target(items, task_state),
            name=f"ursa-message-{item.target_path}",
        )
        self._background_agent_tasks.add(task)
        task.add_done_callback(
            lambda done: self._agent_task_done(
                done,
                items=items,
                task_state=task_state,
            )
        )

    async def _invoke_message_target(
        self,
        items: tuple[_BufferedAgentMessage, ...],
        task_state: dict[str, bool],
    ) -> None:
        """Run one FIFO batch and give its reply to the newest user message."""
        task_state["started"] = True
        item, *additional = items
        deliveries = tuple(
            candidate for candidate in items if candidate.response is not None
        )
        response_item = deliveries[-1] if deliveries else item
        callbacks = response_item.callbacks
        response = response_item.response
        succeeded = False
        try:
            if deliveries:
                # Promote queued user rows without adding delegated spinners.
                for delivery in deliveries:
                    for callback in delivery.callbacks:
                        if isinstance(callback, _MessageStartCallback):
                            await callback.on_agent_message_start()
                for delivery in deliveries:
                    if delivery.response is not response:
                        self._complete_in_band_delivery(delivery)
            else:
                await self._notify_callbacks(
                    callbacks, "on_agent_message_start", item.target_path
                )
            result = await self._manager.invoke(
                item.target_id,
                item.message,
                config={
                    "metadata": {
                        "ursa_message_recipient": item.target_path,
                        "ursa_message_sender": item.sender_path,
                    },
                    "ursa_additional_messages": [
                        item.envelope() for item in additional
                    ],
                },
                callbacks=callbacks,
            )
            if response is not None and not response.done():
                response.set_result(result)
            succeeded = True
        except BaseException as exc:
            self._fail_buffered_messages(deliveries, exc)
            raise
        finally:
            self._mark_working(item.target_path, -1)
            if not deliveries:
                await self._notify_callbacks(
                    callbacks,
                    "on_agent_message_end",
                    item.target_path,
                    succeeded=succeeded,
                )

    def _agent_task_done(
        self,
        task: asyncio.Task[Any],
        *,
        items: tuple[_BufferedAgentMessage, ...],
        task_state: Mapping[str, bool],
    ) -> None:
        """Retrieve background failures so they only reach structured logs."""
        self._background_agent_tasks.discard(task)
        if not task_state["started"]:
            target_path = items[0].target_path
            self._fail_buffered_messages(
                items,
                RuntimeError(
                    f"Message delivery to {target_path!r} was cancelled before start"
                ),
            )
            self._mark_working(target_path, -1)
        self._log_task_failure(task, "Asynchronous agent message failed")

    @staticmethod
    async def _cancel_background_tasks(tasks: set[asyncio.Task[Any]]) -> None:
        await cancel_tasks(*tasks)
        tasks.clear()

    async def _spawn_runtime_subagent(
        self,
        parent_runtime_id: str,
        name: str,
        agent_type: Any = None,
    ) -> str:
        BaseAgent._validate_subagent_name(name)
        async with self._hierarchy_lock:
            if self._closed:
                raise RuntimeError("Subagent runtime is closed")
            root_id = self._runtime_roots[parent_runtime_id]
            hierarchy = self._hierarchies[root_id]
            parent_path = self._runtime_paths[parent_runtime_id]
            if self._is_stopping_path(root_id, parent_path):
                raise ValueError(f"Agent {parent_path!r} is stopping")
            child_path = f"{parent_path}/{name}"
            if child_path in hierarchy:
                raise ValueError(f"Subagent name already exists: {name!r}")

            check_capacity(
                len(hierarchy),
                child_path.count("/"),
                self.config.subagents.max_agents,
                self.config.subagents.max_depth,
            )

            parent_agent_name = self._runtime_agent_names[parent_runtime_id]
            agent_name = self._resolve_subagent_type(
                agent_type, parent_agent_name
            )
            cached_mcp_tool_specs = self._cached_mcp_tool_specs()
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
                        # Every hierarchy member writes to the root's durable
                        # agent store. Their thread IDs below remain distinct,
                        # so LangGraph keeps each agent's history isolated
                        # inside the shared checkpoint database.
                        "agent_name": self.config.agent_name,
                        "thread_id": f"{self.thread_id}:{child_path}",
                    }
                )
            )
            await self._register_process(
                runtime_id,
                agent_name,
                child_path,
                child_config,
                cached_mcp_tool_specs,
            )
            # Reserve both the name and capacity before yielding. Process
            # initialization runs outside the shared lock, so independent
            # starts and message routing can proceed concurrently.
            hierarchy[child_path] = runtime_id
            self._dynamic_runtime_ids.add(runtime_id)
            self._runtime_paths[runtime_id] = child_path
            self._runtime_roots[runtime_id] = root_id
            self._runtime_agent_names[runtime_id] = agent_name
            startup = asyncio.create_task(
                self._start_reserved_subagent(runtime_id),
                name=f"ursa-start-agent-{child_path}",
            )
            self._subagent_start_tasks[runtime_id] = startup
        try:
            await startup
        except asyncio.CancelledError:
            # Cancellation can occur before the startup task ever runs, in
            # which case its own finally block cannot release the reservation.
            await self._unregister_subagent(runtime_id, startup=startup)
            if self._subagent_start_tasks.get(runtime_id) is startup:
                self._subagent_start_tasks.pop(runtime_id, None)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            raise ValueError(
                f"Subagent {child_path!r} was stopped during startup"
            ) from None
        return child_path

    async def _start_reserved_subagent(self, runtime_id: str) -> None:
        """Start a reserved child, rolling back a failure or cancellation."""
        startup = asyncio.current_task()
        try:
            await self._manager.start_agent(runtime_id)
        except BaseException:
            await self._unregister_subagent(runtime_id, startup=startup)
            raise
        finally:
            if self._subagent_start_tasks.get(runtime_id) is startup:
                self._subagent_start_tasks.pop(runtime_id, None)

    async def _unregister_subagent(
        self, runtime_id: str, *, startup: asyncio.Task[None] | None = None
    ) -> None:
        """Unregister once when startup cancellation and subtree stop race."""
        async with self._subagent_cleanup_lock:
            if (
                startup is not None
                and self._subagent_start_tasks.get(runtime_id) is not startup
            ):
                return
            if runtime_id not in self._dynamic_runtime_ids:
                return
            root_id = self._runtime_roots[runtime_id]
            path = self._runtime_paths[runtime_id]
            try:
                await self._manager.unregister(runtime_id, force=True)
            finally:
                async with self._hierarchy_lock:
                    hierarchy = self._hierarchies.get(root_id, {})
                    if hierarchy.get(path) == runtime_id:
                        hierarchy.pop(path, None)
                    self._dynamic_runtime_ids.discard(runtime_id)
                    self._runtime_paths.pop(runtime_id, None)
                    self._runtime_roots.pop(runtime_id, None)
                    self._runtime_agent_names.pop(runtime_id, None)

    async def _cancel_subagent_starts(
        self, runtime_ids: Sequence[str] | None = None
    ) -> None:
        """Release startup lifecycle locks before killing or closing agents."""
        selected = set(runtime_ids) if runtime_ids is not None else None
        starts = tuple(
            (runtime_id, task)
            for runtime_id, task in self._subagent_start_tasks.items()
            if selected is None or runtime_id in selected
        )
        for _runtime_id, task in starts:
            if not task.cancelling():
                task.cancel()
        if starts:
            await asyncio.gather(
                *(task for _runtime_id, task in starts), return_exceptions=True
            )
        for runtime_id, task in starts:
            if task.cancelled():
                await self._unregister_subagent(runtime_id, startup=task)
            if self._subagent_start_tasks.get(runtime_id) is task:
                self._subagent_start_tasks.pop(runtime_id, None)

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
        runtime_id, resolved_path = self._resolve_path(root_id, path)
        if self._is_stopping_path(root_id, resolved_path):
            raise RuntimeError(f"Agent {resolved_path!r} is stopping")
        return runtime_id, resolved_path

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
            await maybe_await(method(*args, **kwargs))

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
            self._stopping_subtrees.update(
                (root_id, "root") for root_id in self._hierarchies
            )
            await self._cancel_subagent_starts()
            await self._cancel_background_tasks(self._background_agent_tasks)
            await self._cancel_background_tasks(self._background_teardown_tasks)
            await self._unregister_owned_agents()
            self._runtime_paths.clear()
            self._runtime_roots.clear()
            self._runtime_agent_names.clear()
            self._hierarchies.clear()
            self._working_counts.clear()
            self._fail_all_buffered_messages(
                RuntimeError("Agent runtime was reconfigured")
            )
            self._wake_all_waiters()
            self._stopping_subtrees.clear()
            self._active_root_id = None
            self._active_agent_ready.clear()
            self._mcp_tool_specs = None
            self._mcp_tool_specs_fingerprint = None
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
            self._wake_all_waiters()
            await self._cancel_subagent_starts()
            await self._cancel_background_tasks(self._background_agent_tasks)
            self._fail_all_buffered_messages(
                RuntimeError("Agent runtime closed")
            )
            await self._cancel_background_tasks(self._background_teardown_tasks)
            if self._owns_manager:
                await self._manager.aclose()
            else:
                await self._unregister_owned_agents()

    async def _unregister_owned_agents(self) -> None:
        for runtime_id in reversed(tuple(self._dynamic_runtime_ids)):
            await self._manager.unregister(runtime_id)
        self._dynamic_runtime_ids.clear()
        for name in tuple(self._registered):
            await self._manager.unregister(self.agents[name].runtime_id)
            self._registered.discard(name)

    def _wake_all_waiters(self) -> None:
        """Release host waits before hierarchy ownership is discarded."""
        for events in self._wait_dependencies.values():
            for waiter in tuple(events):
                waiter.event.set()
        self._wait_dependencies.clear()

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
