"""Acceptance tests for the 296 summarization repair.

Pre-registered before the fix: a summarized history must never end on
the assistant summary (the request shape Claude 4.6 and newer reject),
for messages_to_keep=0 and for the tool-tail absorption edge case, with
the summary landing as framed human-role context.
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.agents.utils import recording_model, tool_call_message
from ursa.agents.chat_agent import ChatAgent


def _agent(log: list, keep: int, workspace) -> ChatAgent:
    return ChatAgent(
        llm=recording_model(log, "summary of the chat"),
        workspace=workspace,
        tokens_before_summarize=1,
        messages_to_keep=keep,
    )


def _long(text: str) -> str:
    return text + " filler" * 50


def test_keep_zero_summary_is_framed_human_context(tmp_path):
    log: list = []
    agent = _agent(log, keep=0, workspace=tmp_path)
    state = {
        "messages": [
            SystemMessage("sys prompt"),
            HumanMessage(_long("first question")),
            AIMessage(_long("first answer")),
            HumanMessage(_long("second question")),
        ]
    }

    new_state, changed = agent._summarize_context(state)

    assert changed is True
    assert not isinstance(new_state["messages"][-1], AIMessage), (
        "messages_to_keep=0 must not leave the history ending on the "
        "assistant summary (the request shape Claude 4.6+ rejects)"
    )
    summary_message = new_state["messages"][1]
    assert isinstance(summary_message, HumanMessage), (
        "the summary must land as human-role context"
    )
    assert summary_message.content.startswith(
        "[Summary of the earlier conversation]"
    )
    assert "summary of the chat" in summary_message.content


def test_absorbed_tool_tail_still_ends_human(tmp_path):
    # The narrow case from issue 296: the kept tail is only tool results
    # whose calls were summarized away; dangling-pair preservation moves
    # them into the summarized block, so the summary becomes the final
    # message even with a nonzero messages_to_keep.
    log: list = []
    agent = _agent(log, keep=1, workspace=tmp_path)
    state = {
        "messages": [
            SystemMessage("sys prompt"),
            HumanMessage(_long("question")),
            tool_call_message(
                "some_tool",
                {},
                "call-1",
                content=_long("calling a tool"),
                type="tool_call",
            ),
            ToolMessage(content=_long("tool output"), tool_call_id="call-1"),
        ]
    }

    new_state, changed = agent._summarize_context(state)

    assert changed is True
    assert not isinstance(new_state["messages"][-1], AIMessage), (
        "an absorbed tool tail must not leave the assistant summary as "
        "the final message"
    )


def test_nonzero_keep_preserves_the_kept_tail(tmp_path):
    # Guard (green before and after the fix): a normal nonzero
    # messages_to_keep keeps the requested tail behind the summary.
    log: list = []
    agent = _agent(log, keep=2, workspace=tmp_path)
    state = {
        "messages": [
            SystemMessage("sys prompt"),
            HumanMessage(_long("first question")),
            AIMessage(_long("first answer")),
            HumanMessage(_long("second question")),
            AIMessage(_long("second answer")),
            HumanMessage(_long("third question")),
        ]
    }

    new_state, changed = agent._summarize_context(state)

    assert changed is True
    kept = new_state["messages"][-2:]
    assert [(type(m), m.content) for m in kept] == [
        (AIMessage, _long("second answer")),
        (HumanMessage, _long("third question")),
    ]
    assert len(new_state["messages"]) == 4
    assert isinstance(new_state["messages"][0], SystemMessage)


def test_post_summarization_request_is_not_assistant_final(tmp_path):
    # The issue's request-level repro: with messages_to_keep=0 the next
    # model call after summarization must not end on the assistant
    # summary.
    log: list = []
    agent = _agent(log, keep=0, workspace=tmp_path)

    agent.invoke({
        "messages": [
            HumanMessage(_long("first question")),
            AIMessage(_long("first answer")),
            HumanMessage(_long("second question")),
        ]
    })

    assert len(log) >= 2, "expected a summarizer call and a chat call"
    post_summarization_request = log[-1]
    assert not isinstance(post_summarization_request[-1], AIMessage), (
        "post-summarization request ends on "
        f"{type(post_summarization_request[-1]).__name__}"
    )
