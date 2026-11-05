"""Process-isolated URSA agent facade for interactive hosts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP

from ursa import agents
from ursa.agents.base import AgentWithTools
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


class _UrsaProcessAgent:
    """One fully constructed URSA agent living exclusively in its child."""

    def __init__(
        self,
        agent: Any,
        tool_sources: Mapping[str, str] | None = None,
    ) -> None:
        self.agent = agent
        self.tool_sources = dict(tool_sources or {})
        self.state: Any = None

    async def ainvoke(
        self, inputs: Any, config: Mapping[str, Any] | None = None
    ) -> str:
        query = self.agent.format_query(str(inputs), state=self.state)
        self.state = await self.agent.ainvoke(query, config=config)
        return str(self.agent.format_result(self.state))

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

    async def aclose(self) -> None:
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
    del context
    ursa_config = config["ursa_config"]
    if not isinstance(ursa_config, UrsaConfig):
        ursa_config = UrsaConfig.model_validate(ursa_config)
    resolved = resolve_ursa_config(ursa_config)
    agent_name = str(config["agent_name"])
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
        resolved.llm_model.api_key = await asyncio.to_thread(
            resolved.llm_model.api_key.resolve
        )
    await asyncio.to_thread(validate_model_provider, resolved.llm_model, "chat")
    model = await asyncio.to_thread(resolved.llm_model.init_chat_model)
    enforce_model_group_policy(model, resolved.group)
    embedding = None
    if resolved.emb_model is not None:
        if isinstance(resolved.emb_model.api_key, SecretReference):
            resolved.emb_model.api_key = await asyncio.to_thread(
                resolved.emb_model.api_key.resolve
            )
        await asyncio.to_thread(
            validate_model_provider, resolved.emb_model, "embedding"
        )
        embedding = await asyncio.to_thread(resolved.emb_model.init_embedding)
        enforce_model_group_policy(embedding, resolved.group)

    agent_kwargs: dict[str, Any] = dict(
        resolved.agent_config.get(agent_name) or {}
    )
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
    return _UrsaProcessAgent(agent, tool_sources)


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


class AgentRuntime:
    """Host facade backed by one isolated process per configured agent."""

    def __init__(
        self,
        config: UrsaConfig,
        *,
        manager: AgentManager | None = None,
    ) -> None:
        self.config = resolve_ursa_config(config)
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
        self._register_lock = asyncio.Lock()
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
                        # multiprocessing serializes this independently for
                        # every child. Avoid deep-copying TemporaryDirectory's
                        # cleanup finalizer in the host.
                        "ursa_config": self.config.model_copy(
                            update={
                                "agent_config": {
                                    name: dict(
                                        self.config.agent_config.get(name) or {}
                                    )
                                }
                            },
                        ),
                    },
                )
            )
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
        result = await self._manager.invoke(
            runtime_id, prompt, callbacks=callbacks
        )
        if not isinstance(result, str):
            raise TypeError(f"Agent {name!r} returned a non-string result")
        return result

    async def restart_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            await self._manager.restart_agent(self.agents[name].runtime_id)

    async def kill_agent(self, name: str) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
        await self._manager.kill_agent(runtime_id)

    async def send_message(
        self, name: str, payload: Any, *, sender: str = "host"
    ) -> None:
        async with self._operation_lock:
            await self._ensure_registered(name)
            runtime_id = self.agents[name].runtime_id
        await self._manager.send_message(runtime_id, payload, sender=sender)

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
            resolved = resolve_ursa_config(candidate)
            for name in tuple(self._registered):
                await self._manager.unregister(self.agents[name].runtime_id)
                self._registered.discard(name)
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
