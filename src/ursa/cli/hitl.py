import asyncio
import os
import platform
import threading
from cmd import Cmd
from dataclasses import dataclass, field
from typing import Any, Optional

import aiosqlite
from fastmcp import FastMCP
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

from ursa.agents import (
    ArxivAgent,
    BaseAgent,
    ChatAgent,
    ExecutionAgent,
    HypothesizerAgent,
    PlanningAgent,
    RecallAgent,
    WebSearchAgent,
)
from ursa.agents.base import AgentWithTools
from ursa.cli.config import UrsaConfig
from ursa.util.mcp import start_mcp_client
from ursa.util.memory_logger import AgentMemory

ursa_banner = r"""
  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/
"""


@dataclass
class AgentHITL:
    """Wrapper for BaseAgent to delay instantiation and async method calls"""

    agent_class: Any
    config: dict = field(default_factory=dict)
    state: Any | None = None
    _agent: BaseAgent | None = field(default=None, init=False)

    async def instantiate(
        self, mcp_client: MultiServerMCPClient | None = None, **kwargs
    ):
        """Instantiate the underlying agent instance"""
        assert self._agent is None
        kwargs |= self.config
        self._agent = self.agent_class(**kwargs)

        # Attach tools from MCP client
        if mcp_client and isinstance(self._agent, AgentWithTools):
            await self._agent.add_mcp_tools(mcp_client)

    @property
    def description(self):
        if self._agent is None:
            return self.agent_class.__doc__
        return self._agent.__doc__

    async def __call__(
        self, prompt: str, last_agent_result: str | None = None
    ) -> str:
        assert self._agent is not None, "Agent not yet instantiated"
        agent = self._agent

        # Inject the previous agent's response into the query
        if last_agent_result is not None:
            prompt = "\n".join([
                f"The last agent output was: {last_agent_result}",
                f"The user stated: {prompt}",
            ])

        # Setup the agents input state from it's current state and plain text input
        # then invoke the agent and extract a final message from it's new state
        query = agent.format_query(prompt, state=self.state)

        new_state = await agent.ainvoke(query)
        msg = agent.format_result(new_state)
        self.state = new_state

        # Return only the result message
        return msg


class HITL:
    def __init__(self, config: UrsaConfig):
        self.config = config
        self.thread_id = "cli"
        # expose workspace and init common attributes
        self.workspace = self.config.workspace
        self.safe_codes = None
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        self.model = init_chat_model(**self.config.llm_model.kwargs)
        self.embedding = (
            init_embeddings(**self.config.emb_model.kwargs)
            if self.config.emb_model
            else None
        )
        self.mcp_client = start_mcp_client(self.config.mcp_servers)
        self.memory = (
            AgentMemory(
                embedding_model=self.embedding,
                path=str(self.workspace / "memory"),
            )
            if self.embedding
            else None
        )

        self.agents: dict[str, AgentHITL] = {}
        self.agents["chat"] = AgentHITL(agent_class=ChatAgent)
        self.agents["arxiv"] = AgentHITL(agent_class=ArxivAgent)
        self.agents["execute"] = AgentHITL(
            agent_class=ExecutionAgent,
            config={"agent_memory": self.memory},
        )
        self.agents["hypothesize"] = AgentHITL(agent_class=HypothesizerAgent)
        self.agents["plan"] = AgentHITL(agent_class=PlanningAgent)
        self.agents["web"] = AgentHITL(agent_class=WebSearchAgent)

        if self.memory is not None:
            self.agents["recall"] = AgentHITL(
                agent_class=RecallAgent,
                config={"memory": self.memory},
            )

        self.last_agent_result = None

    async def _get_checkpointer(
        self, name: str = "checkpoint"
    ) -> AsyncSqliteSaver:
        checkpoint_path = (self.config.workspace / name).with_suffix(".db")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(checkpoint_path))
        return AsyncSqliteSaver(conn)

    async def get_agent(self, name: str):
        agent = self.agents[name]

        # Lazily instantiate the agents
        if agent._agent is None:
            checkpointer = await self._get_checkpointer(name)
            await agent.instantiate(
                llm=self.model,
                workspace=self.workspace,
                checkpointer=checkpointer,
                mcp_client=self.mcp_client,
                thread_id=f"{self.thread_id}_{name}",
            )

        assert agent._agent is not None
        return agent

    async def run_agent(self, name: str, prompt: str) -> str:
        assert name in self.agents, f"Unknown agent {name}"
        agent = await self.get_agent(name)
        msg = await agent(prompt, last_agent_result=self.last_agent_result)
        assert isinstance(msg, str)
        self.last_agent_result = msg
        return msg

    def as_mcp_server(self, **kwargs):
        from ursa import __version__ as ursa_version

        mcp = FastMCP(
            "URSA",
            version=ursa_version,
            on_duplicate_tools="error",
            on_duplicate_prompts="error",
            on_duplicate_resources="error",
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


class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


class UrsaRepl(Cmd):
    exit_message: str = "[dim]Exiting ursa..."
    _help_message: str = "[dim]For help, type: ? or help. Exit with Ctrl+d."
    prompt: str = "ursa> "

    def __init__(self, hitl: HITL, **kwargs):
        super().__init__(**kwargs)
        self.hitl = hitl
        self.ursa_loop = AsyncLoopThread()
        self.console = Console(
            file=self.stdout,
            theme=Theme({
                "success": "green",
                "error": "bold red",
                "dim": "grey50",
                "warn": "yellow",
                "emph": "bold cyan",
            }),
        )

    def __getattribute__(self, name: str) -> Any:
        # Dynamically add do_agent methods
        if name.startswith("do_"):
            agent_name = name.removeprefix("do_")
            if agent_name in self.hitl.agents.keys():

                def run_agent(prompt):
                    return self.run_agent(agent_name, prompt)

                run_agent.__doc__ = self.hitl.agents[agent_name].description
                return run_agent

        return super().__getattribute__(name)

    def get_names(self) -> list[str]:
        names = super().get_names()
        for name in self.hitl.agents.keys():
            names.append(f"do_{name}")
        return names

    def run_agent(self, name: str, prompt: str | None = None):
        if not prompt:
            prompt = input(f"{name}: ")
        with self.console.status("Generating response"):
            result = self.hitl.run_agent(name, prompt)
            result = self.ursa_loop.submit(result)

        assert isinstance(result, str)
        self.show(result)

    def show(self, msg: str, markdown: bool = True, **kwargs):
        self.console.print(Markdown(msg) if markdown else msg, **kwargs)

    def default(self, prompt: str):
        self.run_agent("chat", prompt)

    def postcmd(self, stop: bool, line: str):
        print(file=self.stdout)
        return stop

    def do_exit(self, _: str):
        """Exit shell."""
        self.show(self.exit_message, markdown=False)
        return True

    def do_EOF(self, _: str):
        """Exit on Ctrl+D."""
        self.show("\n" + self.exit_message, markdown=False)
        return True

    def do_clear(self, _: str):
        """Clear the screen. Same as pressing Ctrl+L."""
        os.system("cls" if platform.system() == "Windows" else "clear")

    def emptyline(self):
        """Do nothing when an empty line is entered"""
        pass

    def run(self):
        """Handle Ctrl+C to avoid quitting the program"""
        # Print intro only once.
        self.show(f"[magenta]{ursa_banner}", markdown=False, highlight=False)
        self.show(self._help_message, markdown=False)

        while True:
            try:
                self.cmdloop()
                break  # Allows breaking out of loop if EOF is triggered.
            except KeyboardInterrupt:
                print(
                    "\n(Interrupted) Press Ctrl+D to exit or continue typing."
                )

    def do_models(self, _: str):
        """List models and base urls"""
        llm_provider, llm_name = get_provider_and_model(
            self.hitl.config.llm_model.model
        )
        self.show(
            f"[dim]*[/] LLM: [emph]{llm_name} "
            f"[dim]{self.hitl.config.llm_model.base_url or llm_provider}",
            markdown=False,
        )

        emb_provider, emb_name = (
            get_provider_and_model(self.hitl.config.emb_model.model)
            if self.hitl.config.emb_model
            else ("None", "None")
        )
        if not emb_provider:
            emb_provider = self.hitl.config.emb_model.base_url
        self.show(
            f"[dim]*[/] Embedding Model: [emph]{emb_name} [dim]{emb_provider}",
            markdown=False,
        )


def get_provider_and_model(model_str: Optional[str]):
    if model_str is None:
        return "none", "none"

    if ":" in model_str:
        provider, model = model_str.split(":", 1)
    else:
        provider = "openai"
        model = model_str

    return provider, model


# TODO:
# * Add option to swap models in REPL
# * Add option for seed setting via flags
# * Name change: --llm-model-name -> llm
# * Name change: --emb-model-name -> emb
