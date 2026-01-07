import logging
import asyncio
import threading
import inspect
import os
import platform
from cmd import Cmd
from typing import Any, Optional
from dataclasses import dataclass, field

import aiosqlite
import httpx
from langchain.chat_models import init_chat_model
from mcp.server.fastmcp import FastMCP
from langchain.embeddings import init_embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from typer import Typer

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
from ursa.config import UrsaConfig
from ursa.prompt_library.chatter_prompts import get_chatter_system_prompt
from ursa.util.mcp import start_mcp_client
from ursa.util.memory_logger import AgentMemory

ursa_banner = r"""
  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/
"""


def init_model_kwargs(cfg):
    kwargs = {k: v for k, v in cfg.items() if v is not None}
    if kwargs.pop("ssl_verify", None):
        kwargs["http_client"] = httpx.Client(verify=False)
    return kwargs


@dataclass
class AgentHITL:
    agent_class: Any
    config: dict = field(default_factory=dict)
    state: Any | None = None
    _agent: BaseAgent | None = field(default=None, init=False)

    def instantiate(self, **kwargs):
        """Instantiate the underlying agent instance"""
        assert self._agent is None
        kwargs |= self.config
        self._agent = self.agent_class(**kwargs)

    @property
    def description(self):
        if self._agent is None:
            return self.agent_class.__doc__
        return self._agent.__doc__

    async def __call__(self, prompt: str) -> str:
        assert self._agent is not None, "Agent not yet instantiated"
        agent = self._agent

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
        self.model = init_chat_model(**init_model_kwargs(self.config.llm_model))
        self.embedding = init_embeddings(
            **init_model_kwargs(self.config.emb_model)
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
        self.agents["execute"] = AgentHITL(agent_class=ExecutionAgent)
        self.agents["hypothesize"] = AgentHITL(agent_class=HypothesizerAgent)
        self.agents["plan"] = AgentHITL(agent_class=PlanningAgent)
        self.agents["web"] = AgentHITL(agent_class=WebSearchAgent)
        self.agents["recall"] = AgentHITL(agent_class=RecallAgent)

        self.last_agent_result = ""
        self.arxiv_state = []
        self.executor_state = {}
        self.hypothesizer_state = {}
        self.planner_state = {}
        self.websearcher_state = []

    def update_last_agent_result(self, result: str):
        self.last_agent_result = result

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
            agent.instantiate(
                llm=self.model,
                checkpointer=checkpointer,
                thread_id=f"{self.thread_id}_{name}",
            )

        assert agent._agent is not None
        return agent

    # Properties for REPL usage
    @property
    def llm_model_name(self) -> str:
        return self.config.llm_model.model

    @property
    def emb_model_name(self) -> str:
        return self.config.emb_model.model

    @property
    def llm_base_url(self) -> Optional[str]:
        return self.config.llm_model.base_url

    @property
    def emb_base_url(self) -> Optional[str]:
        return self.config.emb_model.base_url

    async def run_agent(self, name: str, prompt: str) -> str:
        assert name in self.agents, f"Unknown agent {name}"
        agent = await self.get_agent(name)
        msg = await agent(prompt)
        assert isinstance(msg, str)
        self.last_agent_result = msg
        return msg

    async def arxiv_agent(self) -> ArxivAgent:
        return ArxivAgent(llm=self.model)

    async def chatter(self) -> ChatAgent:
        checkpointer = await self._get_checkpointer("chatter")
        chat = ChatAgent(
            llm=self.model,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_chatter",
        )
        return chat

    async def executor(self) -> ExecutionAgent:
        checkpointer = await self._get_checkpointer("executor")
        executor = ExecutionAgent(
            llm=self.model,
            checkpointer=checkpointer,
            agent_memory=self.memory,
            thread_id=self.thread_id + "_executor",
            safe_codes=self.safe_codes,
        )
        tools = await self.mcp_client.get_tools()
        executor.add_mcp_tool(tools)
        return executor

    async def hypothesizer(self) -> HypothesizerAgent:
        checkpointer = await self._get_checkpointer("hypothesizer")
        return HypothesizerAgent(
            llm=self.model,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_hypothesizer",
        )

    async def planner(self) -> PlanningAgent:
        checkpointer = await self._get_checkpointer("planner")
        return PlanningAgent(
            llm=self.model,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_planner",
        )

    async def websearcher(self) -> WebSearchAgent:
        checkpointer = await self._get_checkpointer("websearcher")
        return WebSearchAgent(
            llm=self.model,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_websearch",
        )

    async def rememberer(self) -> RecallAgent:
        return RecallAgent(llm=self.model, memory=self.memory)

    async def run_arxiv(self, prompt: str) -> str:
        message: BaseMessage = await self.model.ainvoke(
            f"The user stated {prompt}. Generate between 1 and 8 words for a search query to address the users need. Return only the words to search.",
        )
        llm_search_query = message.content
        print("Searching ArXiv for ", llm_search_query)

        if isinstance(llm_search_query, str):
            arxiv_agent = await self.arxiv_agent
            arxiv_result = await arxiv_agent.ainvoke(
                arxiv_search_query=llm_search_query,
                context=prompt,
            )
            self.arxiv_state.append(arxiv_result)
            self.update_last_agent_result(arxiv_result)
            return f"[ArXiv Agent Output]:\n {self.last_agent_result}"
        else:
            raise RuntimeError("Unexpected error while running ArxivAgent!")

    async def run_executor(self, prompt: str) -> str:
        executor = await self.executor
        if "messages" in self.executor_state and isinstance(
            self.executor_state["messages"], list
        ):
            self.executor_state["messages"].append(
                HumanMessage(
                    f"The last agent output was: {self.last_agent_result}\n"
                    f"The user stated: {prompt}"
                )
            )
            executor_state = await executor.ainvoke(
                self.executor_state,
            )
            self.executor_state = executor_state

            if isinstance(
                content := executor_state["messages"][-1].content, str
            ):
                self.update_last_agent_result(content)
            else:
                raise TypeError(
                    f"content is supposed to have type str! Instead, it is {content}"
                )
        else:
            self.executor_state = dict(
                workspace=self.workspace,
                messages=[
                    HumanMessage(
                        f"The last agent output was: {self.last_agent_result}\n The user stated: {prompt}"
                    )
                ],
            )
            self.executor_state = await executor.ainvoke(
                self.executor_state,
            )
            self.update_last_agent_result(
                self.executor_state["messages"][-1].content
            )
        return f"[Executor Agent Output]:\n {self.last_agent_result}"

    async def run_rememberer(self, prompt: str) -> str:
        rememberer = await self.rememberer
        memory_output = await rememberer.ainvoke(prompt)
        return f"[Rememberer Output]:\n {memory_output}"

    async def run_chatter(self, prompt: str) -> str:
        chatter = await self.chatter
        assert chatter is not None
        self.chatter_state["messages"].append(
            HumanMessage(
                content=f"The last agent output was: {self.last_agent_result}\n The user stated: {prompt}"
            )
        )
        self.chatter_state = await chatter.ainvoke(self.chatter_state)
        chat_output = self.chatter_state["messages"][-1]

        if not isinstance(chat_output.content, str):
            raise TypeError(
                f"chat_output is not a str! Instead, it is: {type(chat_output.content)}."
            )

        self.update_last_agent_result(chat_output.content)
        return f"{self.last_agent_result}"

    async def run_planner(self, prompt: str) -> str:
        planner = await self.planner
        self.planner_state.setdefault("messages", [])
        self.planner_state["messages"].append(
            HumanMessage(
                f"The last agent output was: {self.last_agent_result}\n"
                f"The user stated: {prompt}"
            )
        )
        self.planner_state = await planner.ainvoke(self.planner_state)
        self.update_last_agent_result(str(self.planner_state))
        return f"[Planner Agent Output]:\n {self.last_agent_result}"

    async def run_websearcher(self, prompt: str) -> str:
        message: BaseMessage = await self.model.ainvoke(
            f"The user stated {prompt}. Generate between 1 and 8 words for a search query to address the users need. Return only the words to search.",
        )
        llm_search_query = message.content
        print("Searching Web for ", llm_search_query)
        if isinstance(llm_search_query, str):
            websearcher = await self.websearcher
            web_result = await websearcher.ainvoke(
                query=llm_search_query,
                context=prompt,
            )
            self.websearcher_state.append(web_result)
            self.update_last_agent_result(web_result)
            return f"[WebSearch Agent Output]:\n {self.last_agent_result}"
        else:
            raise RuntimeError("Unexpected error while running WebSearchAgent!")

    def as_mcp_server(self, **kwargs):
        mcp = FastMCP("URSA", **kwargs)

        # Add all agents
        for name, agent in self.agents.items():

            def call_agent(prompt: str) -> str:
                return self.run_agent(name, prompt)

            mcp.add_tool(call_agent, description=agent.description)

        return mcp


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
    exit_message: str = "\n[dim]Exiting ursa..."
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
        self.show(self.exit_message, markdown=False)
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
            self.hitl.llm_model_name
        )
        self.show(
            f"[dim]*[/] LLM: [emph]{llm_name} "
            f"[dim]{self.hitl.llm_base_url or llm_provider}",
            markdown=False,
        )

        emb_provider, emb_name = get_provider_and_model(
            self.hitl.emb_model_name
        )
        self.show(
            f"[dim]*[/] Embedding Model: [emph]{emb_name} "
            f"[dim]{self.hitl.emb_base_url or emb_provider}",
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
