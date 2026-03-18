from math import sqrt
from pathlib import Path
from typing import Iterator

import pytest
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents import ExecutionAgent


@pytest.fixture(autouse=True)
def stub_execution_tools(monkeypatch):
    """Replace external tools with lightweight stubs for deterministic testing."""

    @tool
    def fake_run_command(query: str, runtime: ToolRuntime) -> str:
        """Return a placeholder response instead of executing shell commands."""
        return "STDOUT:\nstubbed output\nSTDERR:\n"

    @tool
    def fake_run_web_search(
        prompt: str,
        query: str,
        runtime: ToolRuntime,
        max_results: int = 3,
    ) -> str:
        """Return a deterministic web search payload for testing."""
        return f"[stubbed web search] {query}"

    @tool
    def fake_run_arxiv_search(
        prompt: str,
        query: str,
        runtime: ToolRuntime,
        max_results: int = 3,
    ) -> str:
        """Return a deterministic arXiv search payload for testing."""
        return f"[stubbed arxiv search] {query}"

    @tool
    def fake_run_osti_search(
        prompt: str,
        query: str,
        runtime: ToolRuntime,
        max_results: int = 3,
    ) -> str:
        """Return a deterministic OSTI search payload for testing."""
        return f"[stubbed osti search] {query}"

    monkeypatch.setattr(
        "ursa.agents.execution_agent.run_command", fake_run_command
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.run_web_search", fake_run_web_search
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.run_arxiv_search", fake_run_arxiv_search
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.run_osti_search", fake_run_osti_search
    )


class ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


@pytest.fixture
def chat_model():
    return ToolReadyFakeChatModel(messages=_message_stream("ok"))


@pytest.mark.asyncio
async def test_execution_agent_ainvoke_returns_ai_message(
    chat_model, tmpdir: Path
):
    execution_agent = ExecutionAgent(llm=chat_model, workspace=tmpdir)
    workspace = tmpdir / ".ursa"
    inputs = {
        "messages": [
            HumanMessage(
                content=(
                    "Acknowledge this instruction with a brief response "
                    "without calling any tools."
                )
            )
        ],
        "workspace": workspace,
    }

    result = await execution_agent.ainvoke(inputs)

    assert "messages" in result
    assert any(isinstance(msg, HumanMessage) for msg in result["messages"])
    ai_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage)
    ]
    assert ai_messages
    assert any((message.content or "").strip() for message in ai_messages)
    assert (
        isinstance(execution_agent.workspace, Path)
        and execution_agent.workspace.exists()
    )


@pytest.mark.asyncio
async def test_execution_agent_invokes_extra_tool(chat_model, tmpdir: Path):
    @tool
    def do_magic(a: int, b: int) -> float:
        """Return the hypotenuse for the provided right-triangle legs."""
        return sqrt(a**2 + b**2)

    execution_agent = ExecutionAgent(
        llm=chat_model,
        extra_tools=[do_magic],
        workspace=tmpdir,
    )
    workspace = tmpdir / ".ursa_with_tool"
    prompt = "List every tool you have access to and provide the names only."
    inputs = {
        "messages": [HumanMessage(content=prompt)],
        "workspace": workspace,
    }

    result = await execution_agent.ainvoke(inputs)

    assert "messages" in result
    tool_names = list(execution_agent.tools.keys())
    assert "fake_run_command" in tool_names
    assert "do_magic" in tool_names
    ai_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage)
    ]
    assert ai_messages
    assert isinstance(result["messages"][-1], AIMessage)
    assert (
        isinstance(execution_agent.workspace, Path)
        and execution_agent.workspace.exists()
    )


def test_safe_codes_in_store(chat_model, tmpdir):
    execution_agent = ExecutionAgent(
        llm=chat_model,
        workspace=tmpdir,
    )
    assert len(execution_agent.safe_codes) > 0
    store = execution_agent.storage
    safe_codes = [
        item.key
        for item in store.search(("workspace", "safe_codes"), limit=1000)
    ]
    assert set(safe_codes) == set(execution_agent.safe_codes)
