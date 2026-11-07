from collections.abc import Iterator
from contextlib import closing
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

from tests.agents.utils import (
    ToolReadyFakeChatModel,
    message_stream,
    tool_call_message,
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

    def search_tool(source, label):
        def search(
            prompt: str,
            query: str,
            runtime: ToolRuntime,
            max_results: int = 3,
        ) -> str:
            return f"[stubbed {source} search] {query}"

        search.__name__ = f"fake_run_{source}_search"
        search.__doc__ = (
            f"Return a deterministic {label} search payload for testing."
        )
        return tool(search)

    monkeypatch.setattr(
        "ursa.agents.execution_agent.run_command", fake_run_command
    )
    for source, label in (("web", "web"), ("arxiv", "arXiv"), ("osti", "OSTI")):
        monkeypatch.setattr(
            f"ursa.agents.execution_agent.run_{source}_search",
            search_tool(source, label),
        )


class SplitBehaviorFakeChatModel(GenericFakeChatModel):
    """Emit plain responses; bind_tools() returns a separate tool-calling fake model."""

    def bind_tools(self, tools, **kwargs):
        return ToolReadyFakeChatModel(messages=_tool_call_message_stream())


def _tool_call_message_stream() -> Iterator[AIMessage]:
    while True:
        yield tool_call_message(
            "fake_run_command",
            {"query": "pwd"},
            "call-1",
            content="tool-bound-response",
            type="tool_call",
        )


@pytest.fixture
def chat_model():
    return ToolReadyFakeChatModel(messages=message_stream("ok"))


@pytest.fixture
def make_agent(chat_model, tmpdir):
    def create(**options):
        return ExecutionAgent(llm=chat_model, workspace=tmpdir, **options)

    return create


@pytest.fixture
def review_capture(monkeypatch):
    """Capture review prompts and model inputs with a successful assessment."""
    captured = SimpleNamespace(requests=[], messages=[])

    def fake_get_review_prompt(user_request: str) -> str:
        captured.requests.append(user_request)
        return f"REVIEW::{user_request}"

    def fake_invoke_structured(*args, **kwargs):
        captured.messages.append(args[2])
        return ReviewAssessment(is_complete=True, reason="done")

    monkeypatch.setattr(
        "ursa.agents.execution_agent.get_review_prompt",
        fake_get_review_prompt,
    )
    monkeypatch.setattr(
        "ursa.agents.execution_agent.invoke_structured",
        fake_invoke_structured,
    )

    return captured


def test_execution_agent_captures_current_invoke_request(make_agent):
    execution_agent = make_agent()

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
    make_agent, review_capture
):
    execution_agent = make_agent()
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

    assert review_capture.requests[-1] == "new invocation request"
    assert review_capture.messages[-1][-1].content == (
        "REVIEW::new invocation request"
    )
    assert result["review"].is_complete is True


def test_execution_agent_persistent_thread_reviews_latest_invoke_request(
    make_agent, review_capture, tmpdir: Path
):
    execution_agent = make_agent(
        checkpointer=Checkpointer.from_workspace(Path(tmpdir) / "checkpoints")
    )
    with closing(execution_agent):
        execution_agent.invoke("first persisted request")
        execution_agent.invoke("second persisted request")

    assert review_capture.requests == [
        "first persisted request",
        "second persisted request",
    ]


async def test_execution_agent_ainvoke_returns_ai_message(
    make_agent, tmpdir: Path
):
    execution_agent = make_agent()
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


async def test_execution_agent_invokes_extra_tool(make_agent, tmpdir: Path):
    @tool
    def do_magic(a: int, b: int) -> float:
        """Return the hypotenuse for the provided right-triangle legs."""
        return sqrt(a**2 + b**2)

    execution_agent = make_agent(extra_tools=[do_magic])
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
            messages=message_stream("base-response")
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


def test_safe_codes_in_store(make_agent):
    execution_agent = make_agent()
    assert len(execution_agent.safe_codes) > 0
    store = execution_agent.storage
    safe_codes = [
        item.key
        for item in store.search(("workspace", "safe_codes"), limit=1000)
    ]
    assert set(safe_codes) == set(execution_agent.safe_codes)


def test_patch_dangling_tool_calls_emits_events(
    make_agent, monkeypatch, capsys
):
    events = []

    def capture_event(event_name, payload, config=None):
        events.append((event_name, payload))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        capture_event,
    )

    execution_agent = make_agent()
    state = {
        "messages": [tool_call_message("fake_tool", {}, "call-1")],
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
