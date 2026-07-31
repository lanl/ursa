from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from ursa.agents.planning_agent import Plan, PlanningAgent


class FakePlanningChatModel(GenericFakeChatModel):
    messages: Iterator[AIMessage | str] = Field(
        default_factory=lambda: iter([AIMessage(content="unused")])
    )

    def model_copy(self, update=None):
        return self

    def with_structured_output(self, schema, **kwargs):
        class _Runner:
            def invoke(self, messages):
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

        return _Runner()


async def test_planning_agent_creates_structured_plan(tmpdir):
    planning_agent = PlanningAgent(
        llm=FakePlanningChatModel(),
        workspace=tmpdir,
        max_reflection_steps=0,
    )

    prompt = "Outline a concise plan for adding the numbers 1 and 2 together."
    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content=prompt)],
        "reflection_steps": 0,
    })

    assert "plan" in result
    plan = result["plan"]
    assert isinstance(plan, Plan)
    assert len(plan.steps) > 0, "expected at least one plan step"
    assert isinstance(str(plan), str)

    assert "messages" in result
    assert result["messages"], "agent should return at least one message"
    assert getattr(result["messages"][-1], "content", None)


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


class EmptyReflectionFakeChatModel(GenericFakeChatModel):
    def __init__(self, plan: Plan):
        super().__init__(messages=_message_stream(""))
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "structured_invocations", 0)

    def with_structured_output(self, schema):
        model = self

        class StructuredOutput:
            def invoke(self, messages):
                object.__setattr__(
                    model,
                    "structured_invocations",
                    model.structured_invocations + 1,
                )
                if model.structured_invocations > 1:
                    raise AssertionError(
                        "empty reflection should terminate planning without regenerating"
                    )
                return model.plan

        return StructuredOutput()


@pytest.mark.asyncio
async def test_planning_agent_treats_empty_reflection_as_approval(tmpdir):
    plan = Plan.model_validate({
        "steps": [
            {
                "name": "Single step",
                "description": "Do one thing",
                "requires_code": False,
                "expected_outputs": ["done"],
                "success_criteria": ["it is done"],
            }
        ]
    })
    planning_agent = PlanningAgent(
        llm=EmptyReflectionFakeChatModel(plan),
        workspace=tmpdir,
        max_reflection_steps=1,
    )

    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content="make a plan")]
    })

    assert result["plan"] == plan
    assert result["messages"][-1].content == "[APPROVED]"


class RecordingPlannerChatModel(GenericFakeChatModel):
    """Fake chat model that records the messages of every planner LLM call."""

    messages: Iterator[AIMessage | str] = Field(
        default_factory=lambda: iter([AIMessage(content="unused")])
    )
    calls: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="[APPROVED]"))
            ]
        )

    def model_copy(self, update=None):
        return self

    def with_structured_output(self, schema, **kwargs):
        model = self

        class StructuredOutput:
            def invoke(self, messages, config=None):
                model.calls.append(list(messages))
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

        return StructuredOutput()


@pytest.mark.asyncio
async def test_planner_requests_never_end_with_assistant_message(tmpdir):
    """Anthropic removed assistant-message prefill with Claude 4.6, so any
    request whose final message is an AI message now fails with a 400 error.
    Every LLM call the planner makes must end with a non-AI message.
    """
    llm = RecordingPlannerChatModel()
    planning_agent = PlanningAgent(
        llm=llm, workspace=tmpdir, max_reflection_steps=1
    )

    await planning_agent.ainvoke({
        "messages": [HumanMessage(content="make a plan")]
    })

    assert len(llm.calls) >= 2, "expected generation and reflection calls"
    for call in llm.calls:
        assert not isinstance(call[-1], AIMessage), (
            "planner sent a request ending with an AI message, which "
            "models without assistant-message prefill support reject"
        )


@pytest.mark.asyncio
async def test_planner_accumulates_messages_across_nodes(tmpdir):
    """Planner state must append messages rather than replace them, so the
    reflection step sees the original request alongside the drafted plan.
    """
    llm = RecordingPlannerChatModel()
    planning_agent = PlanningAgent(
        llm=llm, workspace=tmpdir, max_reflection_steps=1
    )

    prompt = "make a plan"
    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content=prompt)]
    })

    contents = [msg.content for msg in result["messages"]]
    assert contents[0] == prompt, "original request should be preserved"
    assert any(isinstance(msg, AIMessage) for msg in result["messages"]), (
        "drafted plan message should be retained in state"
    )
