"""Apply model context thresholds consistently to chat and execution agents."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.execution_agent import ExecutionAgent


@pytest.mark.parametrize(
    "agent_type", [BasicChatAgent, ChatAgent, ExecutionAgent]
)
@pytest.mark.parametrize(
    "window,tokens,expected",
    [
        (1000, 699, False),
        (1000, 700, True),
        (1000, 701, True),
        (None, 499, False),
        (None, 500, True),
    ],
)
def test_summary_starts_at_threshold(
    agent_type, window, tokens, expected, chat_model, tmp_path, monkeypatch
):
    chat_model.profile = {"max_input_tokens": window} if window else None
    agent = agent_type(
        llm=chat_model,
        workspace=tmp_path,
        tokens_before_summarize=500,
        messages_to_keep=0,
    )
    monkeypatch.setattr(
        "ursa.agents.base.count_tokens_approximately", lambda *_, **__: tokens
    )
    state = {"messages": [HumanMessage("First"), HumanMessage("Second")]}

    result, changed = agent._summarize_context(state)

    assert changed is expected
    if expected:
        assert result["messages"][0].content.startswith(
            "[Summary of the earlier conversation]"
        )
    else:
        assert result == state


@pytest.mark.parametrize("agent_type", [BasicChatAgent, ChatAgent])
def test_threshold_counts_system_prompt_and_tools(
    agent_type, chat_model, tmp_path
):
    chat_model.profile = {"max_input_tokens": 1000}
    agent = agent_type(llm=chat_model, workspace=tmp_path, messages_to_keep=0)
    # Exercise system overhead alone, then the tool-capable agent's schemas.
    system = "context " * 400 if agent_type is BasicChatAgent else "system"
    messages = [
        SystemMessage(system),
        HumanMessage("First"),
        HumanMessage("Next"),
    ]
    tools = list(getattr(agent, "tools", {}).values())
    assert count_tokens_approximately(messages[1:]) < 700
    assert count_tokens_approximately(messages, tools=tools) >= 700

    result, changed = agent._summarize_context({"messages": messages})

    assert changed
    assert result["messages"][0] == messages[0]
