# ruff: noqa: TID251

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from fastmcp import FastMCP
from langchain.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ursa import agents
from ursa.agents import BaseAgent
from ursa.agents.base import AgentWithTools
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    UrsaConfig,
    resolve_ursa_config,
)
from ursa.security import (
    enforce_model_group_policy,
)
from ursa.util.has_optional_dep_group import has_optional_dep_group
from ursa.util.mcp import start_mcp_client


@dataclass
class AgentHITL:
    """Wrapper for BaseAgent to delay instantiation and async method calls"""

    agent_class: Any
    config: dict = field(default_factory=dict)
    state: Any | None = None
    tool_sources: dict[str, str] = field(default_factory=dict, init=False)
    _agent: BaseAgent | None = field(default=None, init=False)

    async def instantiate(
        self, mcp_client: MultiServerMCPClient | None = None, **kwargs
    ):
        """Instantiate the underlying agent instance"""
        assert self._agent is None
        kwargs |= self.config
        try:
            self._agent = self.agent_class(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate {self.agent_class.__name__} with config "
                f"{self.config}. {exc}"
            ) from exc

        # Attach tools from MCP client
        if mcp_client and isinstance(self._agent, AgentWithTools):
            self.tool_sources = await self._agent.add_mcp_tools(mcp_client)

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

        # Inject the previous agent's response into the query
        if (last_agent_result is not None) and (last_agent != agent):
            prompt = "\n".join([
                f"The last agent output was: {last_agent_result}\n\n",
                f"The user stated: {prompt}",
            ])

        # Setup the agents input state from it's current state and plain text input
        # then invoke the agent and extract a final message from it's new state
        query = agent.format_query(prompt, state=self.state)

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
        self.group = self.config.group

        self.model: BaseChatModel = self.config.llm_model.init_chat_model()
        enforce_model_group_policy(self.model, self.group)

        self.embedding = (
            self.config.emb_model.init_embedding()
            if self.config.emb_model is not None
            else None
        )
        enforce_model_group_policy(self.embedding, self.group)

        self.mcp_client = start_mcp_client(self.config.mcp_servers)

        rag_tool_config = {
            "rag_tools": self.config.rag_tools,
            "rag_tool_embedding": self.embedding,
        }

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
            if deps is not None and not has_optional_dep_group(deps):
                continue

            config = {}
            if agent_name in {"chat", "execute", "deep_review", "dsi"}:
                config.update(rag_tool_config)
            self.agents[agent_name] = AgentHITL(
                agent_class=getattr(agents, agent_class_name),
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
        self._closed = False

    async def _get_checkpointer(
        self, checkpoint_path: Path
    ) -> AsyncSqliteSaver:
        checkpoint_path = checkpoint_path / "db" / "checkpointer.db"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(checkpoint_path))
        return AsyncSqliteSaver(conn)

    async def get_agent(self, name: str):
        if self._closed:
            raise RuntimeError("HITL runtime is closed")

        agent = self.agents[name]

        # Lazily instantiate the agents
        if agent._agent is None:
            await agent.instantiate(
                llm=self.model,
                workspace=self.workspace,
                agent_name=self.agent_name,
                group=self.group,
                mcp_client=self.mcp_client,
                thread_id=f"{self.thread_id}",
            )
            # Named agents are persistent. Replace their sync checkpointer with
            # an async one for HITL execution. Unnamed CLI sessions are
            # ephemeral and intentionally run without a checkpointer so they do
            # not leave checkpoint files in the workspace.
            if self.agent_name is not None:
                sync_checkpointer = agent._agent.checkpointer
                async_checkpointer = await self._get_checkpointer(
                    agent._agent.den
                )
                agent._agent.checkpointer = async_checkpointer
                self._runtime_checkpointers.append(async_checkpointer)
                if isinstance(sync_checkpointer, SqliteSaver):
                    sync_checkpointer.conn.close()

        assert agent._agent is not None
        return agent

    async def run_agent(
        self,
        name: str,
        prompt: str,
        callbacks: Sequence[Any] | None = None,
    ) -> str:
        assert name in self.agents, f"Unknown agent {name}"
        agent = await self.get_agent(name)
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
        """Replace the chat model and reset agents bound to the old model."""
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
        model = await asyncio.to_thread(resolved.init_chat_model)
        enforce_model_group_policy(model, self.group)

        await self._close_instantiated_agents()
        self.model = model
        self.config.llm_model = resolved
        self.inference_provider = inference_provider
        self.last_agent = None
        self.last_agent_result = None

    async def reconfigure_models(
        self,
        chat_model: str,
        chat_inference_provider: str | None,
        embedding_model: str | None,
        embedding_inference_provider: str | None,
    ) -> None:
        """Replace chat and embedding models after both validate successfully."""
        for provider in (
            chat_inference_provider,
            embedding_inference_provider,
        ):
            if (
                provider is not None
                and provider not in self.config.inference_providers
            ):
                raise ValueError(f"Unknown inference provider '{provider}'")

        chat_model = chat_model.strip()
        if not chat_model:
            raise ValueError("Chat model cannot be empty")
        chat_config = ChatModelConfig(
            model=chat_model,
            inference_provider=chat_inference_provider,
            max_completion_tokens=self.config.llm_model.max_completion_tokens,
        )
        resolved_chat = chat_config.resolve_inference_provider(
            self.config.inference_providers
        )
        new_chat = await asyncio.to_thread(resolved_chat.init_chat_model)
        enforce_model_group_policy(new_chat, self.group)

        resolved_embedding = None
        new_embedding = None
        if embedding_model and embedding_model.strip():
            embedding_config = EmbModelConfig(
                model=embedding_model.strip(),
                inference_provider=embedding_inference_provider,
            )
            resolved_embedding = embedding_config.resolve_inference_provider(
                self.config.inference_providers
            )
            new_embedding = await asyncio.to_thread(
                resolved_embedding.init_embedding
            )
            enforce_model_group_policy(new_embedding, self.group)

        await self._close_instantiated_agents()
        self.model = new_chat
        self.embedding = new_embedding
        self.config.llm_model = resolved_chat
        self.config.emb_model = resolved_embedding
        self.inference_provider = chat_inference_provider
        self.embedding_inference_provider = embedding_inference_provider
        for wrapper in self.agents.values():
            if "rag_tool_embedding" in wrapper.config:
                wrapper.config["rag_tool_embedding"] = new_embedding
        self.last_agent = None
        self.last_agent_result = None

    async def _close_instantiated_agents(self) -> None:
        """Close and reset agents so they bind to the configured model."""
        for wrapper in self.agents.values():
            agent = wrapper._agent
            if agent is None:
                continue
            try:
                await agent.aclose()
            except Exception:
                logging.exception("Failed to close async agent resources")
            try:
                agent.close()
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
        if self._closed:
            return
        self._closed = True

        await self._close_instantiated_agents()

    async def close(self) -> None:
        """Compatibility alias for :meth:`aclose`."""
        await self.aclose()

    def as_mcp_server(self, **kwargs):
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
