"""Acceptance tests for the leading-system-prompt repair in _sanitize_history.

Pre-registered before the fix: the sanitizer added in #339 removes
unanswered trailing prompts by keeping only the last human message after
the last reply. A history with no reply yet makes the whole list the tail,
so the leading system prompt never reaches the model on the first turn of
a conversation, and a history holding only a system prompt becomes an
empty request. The guards pin the cleanups #339 intended: duplicate
trailing prompts collapse, and mid-list system messages are dropped.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.agents.utils import RecordingChatModel
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent


def _roles(messages) -> list[str]:
    return [m.type for m in messages]


def _sanitized(agent, messages):
    new_state, _changed = agent._sanitize_history({"messages": list(messages)})
    return new_state["messages"]


def test_first_turn_request_keeps_the_chat_system_prompt(tmp_path):
    # Regression from #339: the first request of every ChatAgent
    # conversation went to the model without its system prompt.
    llm = RecordingChatModel()
    agent = ChatAgent(llm=llm, workspace=tmp_path)
    try:
        agent.invoke(agent.format_query("hello"))
    finally:
        agent.close()

    assert _roles(llm.calls[0]) == ["system", "human"]


def test_first_call_keeps_a_caller_supplied_system_message(tmp_path):
    llm = RecordingChatModel()
    agent = BasicChatAgent(llm=llm, workspace=tmp_path)
    try:
        agent.invoke({
            "messages": [
                SystemMessage("custom instructions"),
                HumanMessage("hi"),
            ]
        })
    finally:
        agent.close()

    assert _roles(llm.calls[0]) == ["system", "human"]
    assert llm.calls[0][0].content == "custom instructions"


def test_first_turn_duplicate_prompt_collapses_and_keeps_the_system_prompt(
    tmp_path,
):
    # A caller that keeps its own state and retries a failed first turn
    # hands the agent [system, human, human], because format_query appends
    # in place. The duplicate must go; the prompt must stay.
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=tmp_path)
    try:
        out = _sanitized(
            agent,
            [
                SystemMessage("s"),
                HumanMessage("first try"),
                HumanMessage("retry"),
            ],
        )
    finally:
        agent.close()

    assert _roles(out) == ["system", "human"]
    assert out[-1].content == "retry"


def test_duplicate_trailing_prompts_still_collapse_after_a_reply(tmp_path):
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=tmp_path)
    try:
        out = _sanitized(
            agent,
            [
                SystemMessage("s"),
                HumanMessage("one"),
                AIMessage("reply"),
                HumanMessage("two"),
                HumanMessage("two again"),
            ],
        )
    finally:
        agent.close()

    assert _roles(out) == ["system", "human", "ai", "human"]
    assert out[-1].content == "two again"


def test_mid_list_system_messages_are_still_dropped(tmp_path):
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=tmp_path)
    try:
        out = _sanitized(
            agent,
            [
                SystemMessage("s"),
                HumanMessage("one"),
                AIMessage("reply"),
                SystemMessage("s"),
                HumanMessage("two"),
            ],
        )
    finally:
        agent.close()

    assert _roles(out) == ["system", "human", "ai", "human"]


def test_history_without_a_system_prompt_keeps_the_last_prompt_only(tmp_path):
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=tmp_path)
    try:
        out = _sanitized(
            agent, [HumanMessage("first try"), HumanMessage("retry")]
        )
    finally:
        agent.close()

    assert _roles(out) == ["human"]
    assert out[0].content == "retry"


def test_system_only_history_is_left_alone(tmp_path):
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=tmp_path)
    try:
        out = _sanitized(agent, [SystemMessage("s")])
    finally:
        agent.close()

    assert _roles(out) == ["system"]
