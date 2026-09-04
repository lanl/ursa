"""Acceptance tests for the 296 summarization repair.

Pre-registered before the fix: a summarized history must never end on
the assistant summary (the request shape Claude 4.6 and newer reject),
for messages_to_keep=0 and for the tool-tail absorption edge case, with
the summary landing as framed human-role context.
"""

from collections.abc import Iterator

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ursa.agents.chat_agent import ChatAgent


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        )


def _recording_model(log: list) -> GenericFakeChatModel:
    class _Recorder(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            log.append(list(messages))
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    return _Recorder(messages=_message_stream("summary of the chat"))


def _agent(log: list, keep: int, workspace) -> ChatAgent:
    return ChatAgent(
        llm=_recording_model(log),
        workspace=workspace,
        tokens_before_summarize=1,
        messages_to_keep=keep,
    )


def _long(text: str) -> str:
    return text + " filler" * 50


def test_keep_zero_state_does_not_end_on_assistant(tmp_path):
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


def test_summary_lands_as_framed_human_message(tmp_path):
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
            AIMessage(
                _long("calling a tool"),
                tool_calls=[
                    {
                        "name": "some_tool",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
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
