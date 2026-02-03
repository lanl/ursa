from math import sqrt
from pathlib import Path
from typing import Iterator

import pytest
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

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


def test_write_code_edits_are_considered_in_safety_check(
    tmpdir: Path, monkeypatch
):
    execution_agent = ExecutionAgent(
        llm=ToolReadyFakeChatModel(messages=_message_stream("[YES] allowed")),
        workspace=tmpdir,
    )

    runtime = Runtime(
        context=execution_agent.context, store=execution_agent.storage
    )

    write_call = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "write-1",
                "name": "write_code",
                "args": {"code": "print('tracked')", "filename": "tracked.py"},
                "type": "tool_call",
            }
        ],
    )

    execution_agent.tool_node.invoke(
        {"messages": [write_call]}, runtime=runtime
    )

    captured = {}

    def fake_prompt(query, safe_codes, edited_files):
        captured["files"] = edited_files
        return "prompt"

    monkeypatch.setattr(execution_agent, "get_safety_prompt", fake_prompt)

    command_call = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "run-1",
                "name": "run_command",
                "args": {"query": "ls"},
                "type": "tool_call",
            }
        ],
    )
    state = {
        "messages": [HumanMessage(content="start"), command_call],
        "current_progress": "",
        "workspace": execution_agent.workspace,
        "symlinkdir": {},
        "model": execution_agent.llm,
    }

    result = execution_agent.safety_check(state, runtime)

    assert captured["files"], "safety prompt receives recorded edits"
    assert all(isinstance(entry, str) for entry in captured["files"])
    assert "tracked.py" in captured["files"]
    # safe command should leave messages unchanged
    assert result["messages"] == state["messages"]
