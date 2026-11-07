"""Acceptance tests for the #295 summarizer repair.

Pre-registered before the fix: ``_summarize_context`` called
``self.tool_llm``, which only ``AgentWithTools`` assigns, so a plain
``BaseAgent`` subclass that reaches summarization (``BasicChatAgent``,
``PromptingAgent``) raised ``AttributeError`` the first time a
conversation crossed ``tokens_before_summarize``. The summarizer must
fall back to ``llm`` on plain agents, keep preferring ``tool_llm`` when
an agent has one, and keep sending the summarized span as plain text
even when the history carries tool calls, which is the mixed-history
case raised on the issue. The kept tail is left as it was; that boundary
is pinned too.
"""

from contextlib import closing

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.agents.utils import (
    RecordingChatModel,
    assert_requests_provider_valid,
    tool_call_message,
)
from ursa.agents.base import _SUMMARIZE_INSTRUCTIONS
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.prompting_agent import PromptingAgent, PromptingState

_PAD = "pad " * 60


def _long_history(pairs: int = 4) -> list:
    history = []
    for index in range(pairs):
        history.append(HumanMessage(f"q{index} {_PAD}"))
        history.append(AIMessage(f"a{index} {_PAD}"))
    history.append(HumanMessage("latest question"))
    return history


def _tool_history(rounds: int = 3) -> list:
    # What a tool-using agent leaves behind in a persisted thread that a
    # plain agent later picks up by name.
    history = []
    for index in range(rounds):
        history.append(HumanMessage(f"task {index} {_PAD}"))
        history.append(
            tool_call_message(
                "run_command", {"query": f"ls {index}"}, f"call_{index}"
            )
        )
        history.append(
            ToolMessage(
                content=f"output {index} {_PAD}", tool_call_id=f"call_{index}"
            )
        )
        history.append(AIMessage(f"done {index} {_PAD}"))
    history.append(HumanMessage("latest question"))
    return history


def _roles(messages) -> list[str]:
    return [message.type for message in messages]


def _summarizer_requests(calls) -> list:
    # The summarizer sends its instructions as the system message and the
    # rendered transcript in one human message.
    return [
        call
        for call in calls
        if len(call) == 2
        and isinstance(call[0], SystemMessage)
        and str(call[0].content) == _SUMMARIZE_INSTRUCTIONS
        and isinstance(call[1], HumanMessage)
    ]


def _plain_agent(llm, workspace, agent_class=BasicChatAgent, keep: int = 2):
    return agent_class(
        llm=llm,
        workspace=workspace,
        tokens_before_summarize=1,
        messages_to_keep=keep,
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_plain_chat_agent_summarizes_without_a_tool_model(
    tmp_path, explicit_none
):
    # Regression for #295: BasicChatAgent has no tool_llm, and the first
    # summarization raised AttributeError before any request was sent.
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path)
    if explicit_none:
        agent.tool_llm = None
    with closing(agent):
        result = agent.invoke({"messages": _long_history()})

    assert_requests_provider_valid(llm.calls)
    assert len(_summarizer_requests(llm.calls)) == 1
    assert _roles(llm.calls[-1]) == ["human", "ai", "human"]
    assert str(result["messages"][0].content).startswith(
        "[Summary of the earlier conversation]"
    )


def test_plain_prompting_agent_summarizes_instead_of_crashing(tmp_path):
    # PromptingAgent is the other plain BaseAgent subclass that reaches
    # prepare_messages_context with summarization enabled.
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path, agent_class=PromptingAgent)
    with closing(agent):
        agent.invoke(PromptingState(messages=_long_history(), approved=False))

    assert_requests_provider_valid(llm.calls)
    assert len(_summarizer_requests(llm.calls)) == 1
    assert _roles(llm.calls[-1]) == ["system", "human", "ai", "human"]


def test_mixed_tool_history_summarizes_as_plain_text_on_a_plain_agent(
    tmp_path,
):
    # The mixed-history case from #295: a thread persisted by a tool-using
    # agent is summarized by a plain agent. The summarized span must go
    # over as rendered text, with no tool-call structure in the
    # summarizer request.
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path)
    with closing(agent):
        agent.invoke({"messages": _tool_history()})

    assert_requests_provider_valid(llm.calls)
    (summarizer_request,) = _summarizer_requests(llm.calls)
    transcript = str(summarizer_request[1].content)
    assert "[tool call call_0] run_command" in transcript
    assert "[tool result call_0]" in transcript
    assert not any(isinstance(m, ToolMessage) for m in summarizer_request)
    assert not any(getattr(m, "tool_calls", None) for m in summarizer_request)


def test_mixed_tool_history_kept_tail_is_left_as_is(tmp_path):
    # Boundary of the fix: only the summarized span is rendered to text.
    # The kept tail (messages_to_keep) still carries whatever tool calls
    # and results it held, and goes to the plain agent's llm unchanged.
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path, keep=4)
    with closing(agent):
        agent.invoke({"messages": _tool_history()})

    (summarizer_request,) = _summarizer_requests(llm.calls)
    assert not any(isinstance(m, ToolMessage) for m in summarizer_request)
    assert _roles(llm.calls[-1]) == ["human", "ai", "tool", "ai", "human"]


def test_mixed_tool_history_summarizer_request_passes_the_anthropic_converter(
    tmp_path,
):
    # Anthropic rejects tool_use/tool_result blocks in a request that
    # declares no tools, which is the failure the issue thread confirmed.
    # Once rendered, the summarizer request must format with none.
    chat_models = pytest.importorskip("langchain_anthropic.chat_models")
    format_messages = getattr(chat_models, "_format_messages", None)
    if format_messages is None:
        pytest.skip("langchain-anthropic no longer exposes _format_messages")
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path)
    with closing(agent):
        agent.invoke({"messages": _tool_history()})

    (summarizer_request,) = _summarizer_requests(llm.calls)
    _system, formatted = format_messages(summarizer_request)
    block_types = [
        block.get("type")
        for message in formatted
        for block in (
            message["content"] if isinstance(message["content"], list) else []
        )
    ]
    assert "tool_use" not in block_types
    assert "tool_result" not in block_types


@pytest.mark.parametrize("agent_class", [ChatAgent, BasicChatAgent])
def test_agent_prefers_an_explicit_tool_llm(tmp_path, agent_class):
    # Both tool-bound ChatAgent and plain agents honor an explicit tool_llm.
    # The fallback to llm applies only when tool_llm is absent or None.
    llm = RecordingChatModel()
    agent = _plain_agent(llm, tmp_path, agent_class=agent_class)
    summarizer = RecordingChatModel()
    agent.tool_llm = summarizer
    with closing(agent):
        agent.invoke({"messages": _long_history()})

    assert len(_summarizer_requests(summarizer.calls)) == 1
    assert _summarizer_requests(llm.calls) == []
    if agent_class is ChatAgent:
        assert_requests_provider_valid(llm.calls)
