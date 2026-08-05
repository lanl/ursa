"""Cross-agent regression coverage for provider-valid message sequences.

Follow-up to the planner prefill bug: every request an agent sends must
be non-empty and must not end on an assistant turn, or providers without
assistant-message prefill support reject it with a 400.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.agents.utils import (
    RecordingChatModel,
    assert_requests_provider_valid,
)
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.deep_review_agent import DeepReviewAgent
from ursa.agents.execution_agent import ExecutionAgent
from ursa.agents.planning_agent import PlanningAgent
from ursa.agents.prompting_agent import PromptingAgent


def _plan_factory(schema):
    return schema(
        steps=[
            {
                "name": "Add numbers",
                "description": "Add 1 and 2.",
                "requires_code": False,
                "expected_outputs": ["sum"],
                "success_criteria": ["sum equals 3"],
            }
        ]
    )


async def test_basic_chat_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = BasicChatAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("What is URSA?"))

    assert_requests_provider_valid(llm.calls)


async def test_chat_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = ChatAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("What is URSA?"))

    assert_requests_provider_valid(llm.calls)


async def test_prompting_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = PromptingAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("Refine this prompt: hello"))

    assert_requests_provider_valid(llm.calls)


async def test_planning_agent_role_sequences(tmpdir):
    llm = RecordingChatModel(structured_factory=_plan_factory)
    agent = PlanningAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke({"messages": [HumanMessage(content="make a plan")]})

    assert_requests_provider_valid(llm.calls)


@pytest.mark.xfail(
    reason=(
        "DeepReviewAgent appends a fresh SystemMessage per debate phase into "
        "the accumulated history, so requests from the second phase onward "
        "carry mid-conversation system messages, which langchain-anthropic "
        "rejects; upstream issue pending"
    ),
    strict=True,
)
async def test_deep_review_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = DeepReviewAgent(llm=llm, workspace=tmpdir, max_iterations=1)

    await agent.ainvoke({"question": "How can cooling usage be reduced?"})

    assert_requests_provider_valid(llm.calls)


def test_invariant_rejects_assistant_final_requests():
    good = [SystemMessage(content="s"), HumanMessage(content="h")]
    bad = [SystemMessage(content="s"), AIMessage(content="a")]

    assert_requests_provider_valid([good])

    with pytest.raises(AssertionError, match="assistant turn"):
        assert_requests_provider_valid([good, bad])


def test_invariant_rejects_empty_requests():
    with pytest.raises(AssertionError, match="no LLM calls"):
        assert_requests_provider_valid([])

    with pytest.raises(AssertionError, match="empty message list"):
        assert_requests_provider_valid([[]])


def test_invariant_rejects_mid_conversation_system_messages():
    leading_prefix_ok = [
        SystemMessage(content="s1"),
        SystemMessage(content="s2"),
        HumanMessage(content="h"),
    ]
    mid_list_system = [
        SystemMessage(content="s1"),
        HumanMessage(content="h"),
        SystemMessage(content="s2"),
        HumanMessage(content="h2"),
    ]

    assert_requests_provider_valid([leading_prefix_ok])

    with pytest.raises(AssertionError, match="after the leading"):
        assert_requests_provider_valid([mid_list_system])


async def test_execution_agent_role_sequences(tmpdir):
    llm = RecordingChatModel(
        structured_factory=lambda schema: schema(
            is_complete=True, reason="Work complete."
        )
    )
    agent = ExecutionAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke("say hello")

    assert_requests_provider_valid(llm.calls)
