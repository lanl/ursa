"""Acceptance tests for the leading-system-prompt repair in _sanitize_history.

Pre-registered before the fix: the sanitizer added in #339 removes
unanswered trailing prompts by keeping only the last human message after
the last reply. A history with no reply yet makes the whole list the tail,
so the leading system prompt never reaches the model on the first turn of
a conversation, and a history holding only a system prompt becomes an
empty request. The guards pin the cleanups #339 intended: duplicate
trailing prompts collapse, and mid-list system messages are dropped.
"""

from contextlib import closing

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.agents.utils import RecordingChatModel
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent


def _roles(messages) -> list[str]:
    return [m.type for m in messages]


def _sanitized(workspace, messages):
    agent = BasicChatAgent(llm=RecordingChatModel(), workspace=workspace)
    with closing(agent):
        new_state, _changed = agent._sanitize_history({
            "messages": list(messages)
        })
        return new_state["messages"]


def test_first_turn_request_keeps_the_chat_system_prompt(tmp_path):
    # Regression from #339: the first request of every ChatAgent
    # conversation went to the model without its system prompt.
    llm = RecordingChatModel()
    agent = ChatAgent(llm=llm, workspace=tmp_path)
    with closing(agent):
        agent.invoke(agent.format_query("hello"))

    assert _roles(llm.calls[0]) == ["system", "human"]


def test_first_call_keeps_a_caller_supplied_system_message(tmp_path):
    llm = RecordingChatModel()
    agent = BasicChatAgent(llm=llm, workspace=tmp_path)
    with closing(agent):
        agent.invoke({
            "messages": [
                SystemMessage("custom instructions"),
                HumanMessage("hi"),
            ]
        })

    assert _roles(llm.calls[0]) == ["system", "human"]
    assert llm.calls[0][0].content == "custom instructions"


def test_first_turn_duplicate_prompt_collapses_and_keeps_the_system_prompt(
    tmp_path,
):
    # A caller that keeps its own state and retries a failed first turn
    # hands the agent [system, human, human], because format_query appends
    # in place. The duplicate must go; the prompt must stay.
    out = _sanitized(
        tmp_path,
        [
            SystemMessage("s"),
            HumanMessage("first try"),
            HumanMessage("retry"),
        ],
    )

    assert _roles(out) == ["system", "human"]
    assert out[-1].content == "retry"


def test_duplicate_trailing_prompts_still_collapse_after_a_reply(tmp_path):
    out = _sanitized(
        tmp_path,
        [
            SystemMessage("s"),
            HumanMessage("one"),
            AIMessage("reply"),
            HumanMessage("two"),
            HumanMessage("two again"),
        ],
    )

    assert _roles(out) == ["system", "human", "ai", "human"]
    assert out[-1].content == "two again"


def test_mid_list_system_messages_are_still_dropped(tmp_path):
    out = _sanitized(
        tmp_path,
        [
            SystemMessage("s"),
            HumanMessage("one"),
            AIMessage("reply"),
            SystemMessage("s"),
            HumanMessage("two"),
        ],
    )

    assert _roles(out) == ["system", "human", "ai", "human"]


def test_history_without_a_system_prompt_keeps_the_last_prompt_only(tmp_path):
    out = _sanitized(
        tmp_path, [HumanMessage("first try"), HumanMessage("retry")]
    )

    assert _roles(out) == ["human"]
    assert out[0].content == "retry"


def test_system_only_history_is_left_alone(tmp_path):
    out = _sanitized(tmp_path, [SystemMessage("s")])

    assert _roles(out) == ["system"]
