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
from ursa.cli.config import UrsaConfig
from ursa.security import (
    enforce_group_base_url_policy,
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
        self.config = config
        self.thread_id = config.thread_id or "ursa"
        # expose workspace and init common attributes
        self.workspace = self.config.workspace
        self.config.workspace.mkdir(parents=True, exist_ok=True)

        agent_overrides = dict(config.agent_config or {})

        self.agent_name = self.config.agent_name
        self.group = self.config.group

        enforce_group_base_url_policy(
            self.config.llm_model.base_url, self.group
        )
        self.model: BaseChatModel = self.config.llm_model.init_chat_model()
        enforce_model_group_policy(self.model, self.group)
        if self.config.emb_model:
            enforce_group_base_url_policy(
                self.config.emb_model.base_url, self.group
            )
        self.embedding = (
            self.config.emb_model.init_embedding()
            if self.config.emb_model
            else None
        )
        enforce_model_group_policy(self.embedding, self.group)

        self.mcp_client = start_mcp_client(self.config.mcp_servers)
        if base_url := getattr(self.config.llm_model, "base_url"):
            if model_base_url := get_base_url(self.model):
                if base_url != model_base_url:
                    logging.error(
                        f"Model base url ({model_base_url}) and config ({base_url}) do not match"
                    )

        if self.embedding:
            if base_url := getattr(self.config.emb_model, "base_url"):
                if model_base_url := get_base_url(self.model):
                    if base_url != model_base_url:
                        logging.error(
                            f"Model base url ({model_base_url}) and config ({base_url}) do not match"
                        )

        rag_tool_config = {
            "rag_tools": self.config.rag_tools,
            "rag_tool_embedding": self.embedding,
        }

        self.agents: dict[str, AgentHITL] = {}
        self.agents["chat"] = AgentHITL(
            agent_class=agents.ChatAgent,
            config={"use_web": self.config.use_web, **rag_tool_config},
        )
        self.agents["arxiv"] = AgentHITL(agent_class=agents.ArxivAgent)
        if has_optional_dep_group("dsi"):
            self.agents["dsi"] = AgentHITL(
                agent_class=agents.DSIAgent,
                config=dict(rag_tool_config),
            )
        self.agents["execute"] = AgentHITL(
            agent_class=agents.ExecutionAgent,
            config={
                "use_web": self.config.use_web,
                **rag_tool_config,
            },
        )
        self.agents["deep_review"] = AgentHITL(
            agent_class=agents.DeepReviewAgent,
            config={"use_web": self.config.use_web, **rag_tool_config},
        )
        self.agents["hypothesize"] = AgentHITL(
            agent_class=agents.HypothesizerAgent
        )
        self.agents["plan"] = AgentHITL(agent_class=agents.PlanningAgent)
        self.agents["prompt"] = AgentHITL(
            agent_class=agents.PromptingAgent,
            config={"use_web": self.config.use_web},
        )
        self.agents["web"] = AgentHITL(agent_class=agents.WebSearchAgent)

        if has_optional_dep_group("lammps"):
            self.agents["lammps"] = AgentHITL(agent_class=agents.LammpsAgent)

        # Apply agent-specific configuration overrides
        for agent, agent_config in agent_overrides.items():
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

    async def aclose(self) -> None:
        """Close instantiated agents and runtime-owned persistence resources."""
        if self._closed:
            return
        self._closed = True

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

        for checkpointer in self._runtime_checkpointers:
            try:
                await checkpointer.conn.close()
                await asyncio.to_thread(checkpointer.conn.join)
            except Exception:
                logging.exception("Failed to close agent checkpointer")
        self._runtime_checkpointers.clear()

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
