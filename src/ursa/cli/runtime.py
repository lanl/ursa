# ruff: noqa: TID251

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ursa import agents
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    UrsaConfig,
    resolve_ursa_config,
)
from ursa.security import (
    enforce_model_group_policy,
    group_agents_dir,
)
from ursa.util.has_extra import has_extra
from ursa.util.inference_providers import validate_model_provider

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from ursa.agents import BaseAgent

# Reserved name prefix of the throwaway agents backing unnamed sessions.
EPHEMERAL_AGENT_PREFIX = "tmp_"


def _shares_message_history(agent: Any) -> bool:
    """Whether ``agent`` reads and writes the session's shared message history.

    True only for checkpointed agents whose state schema has a ``messages``
    channel. Decided by class, not by runtime state, so it is also correct on
    an agent's first turn. Unknown schemas count as not sharing.
    """
    state_type = getattr(agent, "state_type", None)
    return getattr(
        agent, "checkpointer", None
    ) is not None and "messages" in getattr(state_type, "__annotations__", {})


def start_mcp_client(server_configs):
    """Load MCP clients only when an agent first needs them."""
    from ursa.util.mcp import start_mcp_client as factory

    return factory(server_configs)


@dataclass(init=False)
class AgentHITL:
    """Wrapper for BaseAgent to delay instantiation and async method calls"""

    _agent_class: Any = field(repr=False)
    config: dict = field(default_factory=dict)
    state: Any | None = None
    tool_sources: dict[str, str] = field(default_factory=dict, init=False)
    _agent: BaseAgent | None = field(default=None, init=False)
    _initialization_task: asyncio.Task[None] | None = field(
        default=None, init=False, repr=False
    )

    def __init__(
        self, agent_class: Any, config: dict | None = None, state: Any = None
    ):
        self._agent_class = agent_class
        self.config = {} if config is None else config
        self.state = state
        self.tool_sources = {}
        self._agent = None
        self._initialization_task = None

    @property
    def agent_class(self):
        """Import an agent implementation only when requested."""
        if isinstance(self._agent_class, str):
            self._agent_class = getattr(agents, self._agent_class)
        return self._agent_class

    async def instantiate(
        self,
        mcp_client: MultiServerMCPClient | None = None,
        finalizer: Callable[[Any], Awaitable[None]] | None = None,
        **kwargs,
    ):
        """Instantiate once, shared by all concurrent and cancelled waiters."""
        if self._agent is not None:
            return
        task = self._initialization_task
        if task is None:
            task = asyncio.create_task(
                self._instantiate_once(
                    mcp_client=mcp_client, finalizer=finalizer, **kwargs
                )
            )
            self._initialization_task = task
            task.add_done_callback(self._initialization_done)
        # A UI waiter may be cancelled when its modal closes. Initialization
        # is runtime-owned and must still publish or clean up its result.
        await asyncio.shield(task)

    def _initialization_done(self, task: asyncio.Task[None]) -> None:
        if self._initialization_task is task:
            self._initialization_task = None
        if not task.cancelled():
            # Retrieve a failure even if the last UI waiter was cancelled;
            # active waiters still receive the same exception from `await`.
            task.exception()

    async def _instantiate_once(
        self,
        mcp_client: MultiServerMCPClient | None = None,
        finalizer: Callable[[Any], Awaitable[None]] | None = None,
        **kwargs,
    ) -> None:
        from ursa.agents.base import AgentWithTools

        kwargs |= self.config

        def build_agent():
            try:
                agent = self.agent_class(**kwargs)
            except TypeError as exc:
                raise TypeError(
                    f"Failed to instantiate {self.agent_class.__name__} with "
                    f"config {self.config}. {exc}"
                ) from exc
            return agent

        agent = await asyncio.to_thread(build_agent)
        try:
            tool_sources: dict[str, str] = {}
            if mcp_client and isinstance(agent, AgentWithTools):
                tool_sources = await agent.add_mcp_tools(mcp_client)
            if finalizer is not None:
                await finalizer(agent)
        except BaseException:
            await self._close_agent(agent)
            raise
        self._agent = agent
        self.tool_sources = tool_sources

    @staticmethod
    async def _close_agent(agent: Any) -> None:
        async_close = getattr(agent, "aclose", None)
        if callable(async_close):
            try:
                await async_close()
            except Exception:
                logging.exception("Failed to close partially initialized agent")
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception:
                logging.exception("Failed to close partially initialized agent")

    async def wait_until_initialized(self) -> None:
        """Wait for runtime-owned initialization, if one is in flight."""
        if self._initialization_task is not None:
            await asyncio.shield(self._initialization_task)

    @property
    def description(self):
        if self._agent is None:
            return self.agent_class.__doc__
        return self._agent.__doc__

    async def __call__(
        self,
        prompt: str,
        last_agent_result: str | None = None,
        last_agent: Any | None = None,
        callbacks: Sequence[Any] | None = None,
    ) -> str:
        assert self._agent is not None, "Agent not yet instantiated"
        agent = self._agent

        # Hand the previous agent's output over only when it is not already in
        # this agent's context: both agents must share the checkpointed message
        # history for it to be redundant (it would otherwise be duplicated).
        # Compared by class so rebuilt agents (e.g. after a model switch) still
        # count as the same agent.
        if (
            last_agent_result is not None
            and type(last_agent) is not type(agent)
            and not (
                _shares_message_history(last_agent)
                and _shares_message_history(agent)
            )
        ):
            prompt = "\n".join([
                f"The last agent output was: {last_agent_result}\n\n",
                f"The user stated: {prompt}",
            ])

        # Setup the agents input state from it's current state and plain text input
        # then invoke the agent and extract a final message from it's new state.
        # For checkpointed agents the history already lives in the checkpoint, so
        # the cached copy is sent without its messages: re-sending them could
        # resurrect messages another agent has since summarized away.
        state = self.state
        if (
            state is not None
            and "messages" in state
            and getattr(agent, "checkpointer", None) is not None
        ):
            state = {**state, "messages": []}
        query = agent.format_query(prompt, state=state)

        invoke_config = None
        if callbacks:
            invoke_config = {"callbacks": list(callbacks)}

        new_state = await agent.ainvoke(query, config=invoke_config)
        msg = agent.format_result(new_state)
        self.state = new_state

        # Return only the result message
        return msg


def get_base_url(model: BaseChatModel) -> str | None:
    for attr in ["base_url", "api_base", "openai_api_base"]:
        if base_url := getattr(model, attr, None):
            return base_url
    logging.warning(f"Missing base_url for {model}")
    return None


class HITL:
    def __init__(self, config: UrsaConfig):
        self.inference_provider = config.llm_model.inference_provider
        self.embedding_inference_provider = (
            config.emb_model.inference_provider
            if config.emb_model is not None
            else None
        )
        self.config = resolve_ursa_config(config)
        self.thread_id = self.config.thread_id or "ursa"
        # expose workspace and init common attributes
        self.workspace = self.config.workspace
        self.config.workspace.mkdir(parents=True, exist_ok=True)

        self.agent_name = self.config.agent_name
        # Unnamed sessions run as a throwaway persistent agent. Its checkpoint
        # lives in the agent den (outside the workspace) so conversation history
        # survives agent rebuilds such as model switches, is shared by every
        # agent type exactly as it is for a named agent, and keeps the store,
        # metrics and experiences out of the workspace. It is deleted on close;
        # leftovers from a crash can be removed with `ursa delete-agent --name 'tmp_*'`.
        # `agent_name` stays the user's choice (None) so the UI does not show
        # the generated name.
        self._ephemeral_agent = self.agent_name is None
        self.session_agent_name = (
            f"{EPHEMERAL_AGENT_PREFIX}{uuid.uuid4().hex}"
            if self._ephemeral_agent
            else self.agent_name
        )
        self.group = self.config.group

        self.agents: dict[str, AgentHITL] = {}
        for agent_name, agent_class_name, deps in [
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
        ]:
            if deps is not None and not has_extra(deps):
                continue

            config = {
                "tokens_before_summarize": self.config.tokens_before_summarize
            }
            if agent_name in {"chat", "execute", "deep_review", "dsi"}:
                config["rag_tools"] = self.config.rag_tools
            self.agents[agent_name] = AgentHITL(
                agent_class=agent_class_name,
                config=config,
            )

        # Apply agent-specific configuration overrides
        for agent, agent_config in self.config.agent_config.items():
            assert agent in self.agents, (
                f"Unknown agent {agent}, Know agents: {','.join(self.agents.keys())}"
            )
            self.agents[agent].config.update(agent_config)
            logging.debug(
                f"Updated {agent} config: {self.agents[agent].config}"
            )

        self.last_agent_result = None
        self.last_agent = None
        self._runtime_checkpointers: list[AsyncSqliteSaver] = []
        self._dependencies_task: asyncio.Task | None = None
        self._transition_lock = asyncio.Lock()
        self._transition_serial_lock = asyncio.Lock()
        self._loads_allowed = asyncio.Event()
        self._loads_allowed.set()
        self._no_active_operations = asyncio.Event()
        self._no_active_operations.set()
        self._active_operations = 0
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @cached_property
    def model(self):
        validate_model_provider(self.config.llm_model, "chat")
        model = self.config.llm_model.init_chat_model()
        enforce_model_group_policy(model, self.group)
        return model

    @cached_property
    def embedding(self):
        if self.config.emb_model is None:
            return None
        validate_model_provider(self.config.emb_model, "embedding")
        embedding = self.config.emb_model.init_embedding()
        enforce_model_group_policy(embedding, self.group)
        return embedding

    @cached_property
    def mcp_client(self):
        return start_mcp_client(self.config.mcp_servers)

    def _agent_dependencies(self):
        return self.model, self.embedding, self.mcp_client

    def _dependencies_done(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            if self._dependencies_task is task:
                self._dependencies_task = None

    async def _get_checkpointer(
        self, checkpoint_path: Path
    ) -> AsyncSqliteSaver:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        checkpoint_path = checkpoint_path / "db" / "checkpointer.db"
        await asyncio.to_thread(
            checkpoint_path.parent.mkdir, parents=True, exist_ok=True
        )
        conn = await aiosqlite.connect(str(checkpoint_path))
        return AsyncSqliteSaver(conn)

    @asynccontextmanager
    async def _agent_operation(self) -> AsyncIterator[None]:
        """Lease runtime agent resources against close/reconfiguration."""
        while True:
            if self._closed:
                raise RuntimeError("HITL runtime is closed")
            await self._loads_allowed.wait()
            async with self._transition_lock:
                if self._closed:
                    raise RuntimeError("HITL runtime is closed")
                if not self._loads_allowed.is_set():
                    continue
                self._active_operations += 1
                self._no_active_operations.clear()
                break
        try:
            yield
        finally:
            async with self._transition_lock:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._no_active_operations.set()

    async def _begin_transition(self) -> None:
        await self._transition_serial_lock.acquire()
        try:
            async with self._transition_lock:
                self._loads_allowed.clear()
            await self._no_active_operations.wait()
        except BaseException:
            async with self._transition_lock:
                self._loads_allowed.set()
            self._transition_serial_lock.release()
            raise

    async def _end_transition(self) -> None:
        async with self._transition_lock:
            self._loads_allowed.set()
        self._transition_serial_lock.release()

    @asynccontextmanager
    async def _agent_transition(self) -> AsyncIterator[None]:
        await self._begin_transition()
        try:
            yield
        finally:
            await self._end_transition()

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    async def _run_runtime_task(self, coroutine: Awaitable[Any]) -> Any:
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self._consume_task_exception)
        return await asyncio.shield(task)

    async def _finalize_named_agent(self, built_agent: Any) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        # Custom agent classes from the config need not be persistent
        # BaseAgents; without a den there is nothing to checkpoint.
        if getattr(built_agent, "den", None) is None:
            return
        sync_checkpointer = built_agent.checkpointer
        async_checkpointer = await self._get_checkpointer(built_agent.den)
        try:
            if isinstance(sync_checkpointer, SqliteSaver):
                await asyncio.to_thread(sync_checkpointer.conn.close)
        except BaseException:
            await async_checkpointer.conn.close()
            await asyncio.to_thread(async_checkpointer.conn.join)
            raise
        built_agent.checkpointer = async_checkpointer
        self._runtime_checkpointers.append(async_checkpointer)

    async def _get_agent(self, name: str) -> AgentHITL:
        agent = self.agents[name]
        # First-use imports and connectivity checks must not block the TUI.
        # Share the initialized dependencies across concurrent agent requests.
        task = self._dependencies_task
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(self._agent_dependencies)
            )
            self._dependencies_task = task
            task.add_done_callback(self._dependencies_done)
        model, embedding, mcp_client = await asyncio.shield(task)
        if "rag_tools" in agent.config:
            agent.config.setdefault("rag_tool_embedding", embedding)
        await agent.instantiate(
            llm=model,
            workspace=self.workspace,
            agent_name=self.session_agent_name,
            group=self.group,
            mcp_client=mcp_client,
            finalizer=self._finalize_named_agent,
            thread_id=f"{self.thread_id}",
        )
        assert agent._agent is not None
        return agent

    async def get_agent(self, name: str):
        async with self.use_agent(name) as agent:
            return agent

    @asynccontextmanager
    async def use_agent(self, name: str) -> AsyncIterator[AgentHITL]:
        """Keep one agent alive for metadata extraction or execution setup."""
        async with self._agent_operation():
            yield await self._get_agent(name)

    async def run_agent(
        self,
        name: str,
        prompt: str,
        callbacks: Sequence[Any] | None = None,
    ) -> str:
        assert name in self.agents, f"Unknown agent {name}"
        async with self.use_agent(name) as agent:
            msg = await agent(
                prompt,
                last_agent_result=self.last_agent_result,
                last_agent=self.last_agent,
                callbacks=callbacks,
            )
            assert isinstance(msg, str)
            self.last_agent_result = msg
            self.last_agent = agent._agent
            return msg

    async def reconfigure_model(
        self, model_name: str, inference_provider: str | None
    ) -> None:
        """Replace the chat model and reset agents bound to the old model.

        This method owns rollback: if replacement fails, callers may assume
        the existing model configuration remains active.
        """
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Model name cannot be empty")
        if (
            inference_provider is not None
            and inference_provider not in self.config.inference_providers
        ):
            raise ValueError(
                f"Unknown inference provider '{inference_provider}'"
            )

        candidate = ChatModelConfig(
            model=model_name,
            inference_provider=inference_provider,
            max_completion_tokens=self.config.llm_model.max_completion_tokens,
        )
        resolved = candidate.resolve_inference_provider(
            self.config.inference_providers
        )
        await asyncio.to_thread(validate_model_provider, resolved, "chat")
        model = await asyncio.to_thread(resolved.init_chat_model)
        enforce_model_group_policy(model, self.group)

        async def apply_reconfiguration() -> None:
            async with self._agent_transition():
                if self._closed:
                    raise RuntimeError("HITL runtime is closed")
                await self._close_instantiated_agents()
                self.model = model
                self.config.llm_model = resolved
                self.inference_provider = inference_provider
                # The handoff is kept: like the history it survives a model
                # switch, and AgentHITL compares agents by class, so the
                # rebuilt agents still match their pre-switch counterparts.

        await self._run_runtime_task(apply_reconfiguration())

    async def reconfigure_models(
        self,
        chat_config: ChatModelConfig,
        embedding_config: EmbModelConfig | None,
    ) -> None:
        """Replace chat and embedding models after both validate successfully.

        This method owns rollback: if replacement fails, callers may assume
        the existing model configuration remains active.
        """
        for provider in (
            chat_config.inference_provider,
            embedding_config.inference_provider if embedding_config else None,
        ):
            if (
                provider is not None
                and provider not in self.config.inference_providers
            ):
                raise ValueError(f"Unknown inference provider '{provider}'")

        if not chat_config.model.strip():
            raise ValueError("Chat model cannot be empty")
        resolved_chat = chat_config.resolve_inference_provider(
            self.config.inference_providers
        )
        await asyncio.to_thread(validate_model_provider, resolved_chat, "chat")
        new_chat = await asyncio.to_thread(resolved_chat.init_chat_model)
        enforce_model_group_policy(new_chat, self.group)

        resolved_embedding = None
        new_embedding = None
        if embedding_config is not None:
            resolved_embedding = embedding_config.resolve_inference_provider(
                self.config.inference_providers
            )
            await asyncio.to_thread(
                validate_model_provider, resolved_embedding, "embedding"
            )
            new_embedding = await asyncio.to_thread(
                resolved_embedding.init_embedding
            )
            enforce_model_group_policy(new_embedding, self.group)

        async def apply_reconfiguration() -> None:
            async with self._agent_transition():
                if self._closed:
                    raise RuntimeError("HITL runtime is closed")
                await self._close_instantiated_agents()
                self.model = new_chat
                self.embedding = new_embedding
                self.config.llm_model = resolved_chat
                self.config.emb_model = resolved_embedding
                self.inference_provider = chat_config.inference_provider
                self.embedding_inference_provider = (
                    embedding_config.inference_provider
                    if embedding_config
                    else None
                )
                for wrapper in self.agents.values():
                    if "rag_tool_embedding" in wrapper.config:
                        wrapper.config["rag_tool_embedding"] = new_embedding

        await self._run_runtime_task(apply_reconfiguration())

    async def _close_instantiated_agents(self) -> None:
        """Close and reset agents so they bind to the configured model."""
        task = self._dependencies_task
        if task is not None:
            try:
                await asyncio.shield(task)
            except Exception:
                logging.exception(
                    "Dependency initialization failed during cleanup"
                )
            self._dependencies_task = None
        for wrapper in self.agents.values():
            try:
                await wrapper.wait_until_initialized()
            except Exception:
                logging.exception("Agent initialization failed during cleanup")
            agent = wrapper._agent
            if agent is None:
                continue
            try:
                await agent.aclose()
            except Exception:
                logging.exception("Failed to close async agent resources")
            try:
                await asyncio.to_thread(agent.close)
            except Exception:
                logging.exception("Failed to close sync agent resources")
            wrapper._agent = None
            wrapper.state = None

        for checkpointer in self._runtime_checkpointers:
            try:
                await checkpointer.conn.close()
                await asyncio.to_thread(checkpointer.conn.join)
            except Exception:
                logging.exception("Failed to close agent checkpointer")
        self._runtime_checkpointers.clear()

    async def aclose(self) -> None:
        """Close instantiated agents and runtime-owned persistence resources."""
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._aclose_once())
            self._close_task = task
            task.add_done_callback(self._close_done)
        await asyncio.shield(task)

    async def _aclose_once(self) -> None:
        async with self._agent_transition():
            if self._closed:
                return
            await self._close_instantiated_agents()
            self._closed = True
            await asyncio.to_thread(self._remove_ephemeral_agent)

    def _remove_ephemeral_agent(self) -> None:
        """Delete the throwaway agent of an unnamed session.

        Runs after every agent and checkpointer is closed. Only ever removes a
        directory this runtime generated: the flag must be set and the name
        must carry the reserved prefix.
        """
        name = self.session_agent_name
        if not self._ephemeral_agent or not name.startswith(
            EPHEMERAL_AGENT_PREFIX
        ):
            return
        path = group_agents_dir(self.group) / name
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            logging.exception("Failed to remove temporary agent %s", path)

    def _close_done(self, task: asyncio.Task[None]) -> None:
        if self._close_task is task:
            self._close_task = None
        self._consume_task_exception(task)

    async def close(self) -> None:
        """Compatibility alias for :meth:`aclose`."""
        await self.aclose()

    def as_mcp_server(self, **kwargs):
        from fastmcp import FastMCP

        from ursa import __version__ as ursa_version

        mcp = FastMCP(
            "URSA",
            version=ursa_version,
            on_duplicate="error",
            **kwargs,
        )

        # Add all agents
        for name, agent in self.agents.items():
            mcp.tool(
                self._make_agent_tool(name),
                name=name,
                description=agent.description,
            )

        return mcp

    def _make_agent_tool(self, agent_name: str):
        # Need to ensure the call_agent closure is correctly constructed
        async def call_agent(prompt: str) -> str:
            return await self.run_agent(agent_name, prompt)

        return call_agent
