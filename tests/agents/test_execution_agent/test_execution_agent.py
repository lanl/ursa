from collections.abc import Iterator
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ursa.agents import ExecutionAgent
from ursa.agents.execution_agent import ReviewAssessment
from ursa.util import Checkpointer


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


class SplitBehaviorFakeChatModel(GenericFakeChatModel):
    """Emit plain responses; bind_tools() returns a separate tool-calling fake model."""

    def bind_tools(self, tools, **kwargs):
        return ToolCallingFakeChatModel(messages=_tool_call_message_stream())


class ToolCallingFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


def _tool_call_message_stream() -> Iterator[AIMessage]:
    while True:
        yield AIMessage(
            content="tool-bound-response",
            tool_calls=[
                {
                    "name": "fake_run_command",
                    "args": {"query": "pwd"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )


@pytest.fixture
def chat_model():
    return ToolReadyFakeChatModel(messages=_message_stream("ok"))


def test_execution_agent_captures_current_invoke_request(
    chat_model, tmpdir: Path
):
    execution_agent = ExecutionAgent(llm=chat_model, workspace=tmpdir)

    normalized = execution_agent._normalize_inputs("current request")

    assert normalized["current_user_request"] == "current request"
    assert (
        execution_agent._get_current_user_request({
            "messages": [
                HumanMessage(content="old persisted request"),
                AIMessage(content="old response"),
                HumanMessage(content="current request"),
            ],
            "current_user_request": normalized["current_user_request"],
            "symlinkdir": {},
        })
        == "current request"
    )


def test_execution_agent_review_uses_current_invoke_request(
    chat_model, monkeypatch, tmpdir: Path
):
    captured = {}

    def fake_get_review_prompt(user_request: str) -> str:
        captured["user_request"] = user_request
        return f"REVIEW::{user_request}"

    def fake_invoke_structured(*args, **kwargs):
        captured["review_messages"] = args[2]
        return ReviewAssessment(is_complete=True, reason="done")

    monkeypatch.setattr(
        "ursa.agents.execution_agent.get_review_prompt",
        fake_get_review_prompt,
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.invoke_structured",
        fake_invoke_structured,
    )

    execution_agent = ExecutionAgent(llm=chat_model, workspace=tmpdir)
    result = execution_agent.review_work({
        "messages": [
            HumanMessage(content="old persisted request"),
            AIMessage(content="old response"),
            HumanMessage(content="new invocation request"),
            AIMessage(content="new response"),
        ],
        "current_user_request": "new invocation request",
        "symlinkdir": {},
    })

    assert captured["user_request"] == "new invocation request"
    assert captured["review_messages"][-1].content == (
        "REVIEW::new invocation request"
    )
    assert result["review"].is_complete is True


def test_execution_agent_persistent_thread_reviews_latest_invoke_request(
    chat_model, monkeypatch, tmpdir: Path
):
    reviewed_requests = []

    def fake_get_review_prompt(user_request: str) -> str:
        reviewed_requests.append(user_request)
        return f"REVIEW::{user_request}"

    def fake_invoke_structured(*args, **kwargs):
        return ReviewAssessment(is_complete=True, reason="done")

    monkeypatch.setattr(
        "ursa.agents.execution_agent.get_review_prompt",
        fake_get_review_prompt,
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.invoke_structured",
        fake_invoke_structured,
    )

    execution_agent = ExecutionAgent(
        llm=chat_model,
        workspace=tmpdir,
        checkpointer=Checkpointer.from_workspace(Path(tmpdir) / "checkpoints"),
    )
    try:
        execution_agent.invoke("first persisted request")
        execution_agent.invoke("second persisted request")
    finally:
        execution_agent.close()

    assert reviewed_requests == [
        "first persisted request",
        "second persisted request",
    ]


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


def test_execution_agent_keeps_tool_calls_out_of_summary_and_recap(
    tmpdir: Path,
):
    execution_agent = ExecutionAgent(
        llm=SplitBehaviorFakeChatModel(
            messages=_message_stream("base-response")
        ),
        workspace=tmpdir,
        tokens_before_summarize=99999,
        messages_to_keep=1,
    )

    _ = execution_agent.compiled_graph

    recap_result = execution_agent.recap({
        "messages": [
            SystemMessage(content="system"),
            HumanMessage(content="hi"),
        ],
        "symlinkdir": {},
    })
    assert not recap_result["messages"][1].tool_calls

    query_result = execution_agent.query_executor(
        {"messages": [HumanMessage(content="hi")], "symlinkdir": {}},
        runtime=SimpleNamespace(context=execution_agent.context),
    )
    assert query_result["messages"].tool_calls
    assert query_result["messages"].tool_calls[0]["name"] == "fake_run_command"


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


def test_patch_dangling_tool_calls_emits_events(
    chat_model,
    monkeypatch,
    capsys,
    tmpdir,
):
    events = []

    def capture_event(event_name, payload, config=None):
        events.append((event_name, payload))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        capture_event,
    )

    execution_agent = ExecutionAgent(llm=chat_model, workspace=tmpdir)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_tool",
                        "args": {},
                        "id": "call-1",
                    }
                ],
            )
        ],
        "symlinkdir": {},
    }

    new_state, summarized = execution_agent._patch_dangling(
        state,
        summarized=False,
        config={"callbacks": []},
    )

    assert summarized is True
    assert any(isinstance(msg, ToolMessage) for msg in new_state["messages"])
    assert events
    assert events[0][1]["agent"] == "ExecutionAgent"
    assert events[0][1]["stage"] == "dangling_tool_calls"
    assert events[0][1]["tool_call_ids"] == ["call-1"]
    assert "[Dangling Tool Call Warning]" not in capsys.readouterr().out
