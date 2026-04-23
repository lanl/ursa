from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents.planning_agent import Plan, PlanningAgent


async def test_planning_agent_creates_structured_plan(chat_model, tmpdir):
    planning_agent = PlanningAgent(
        llm=chat_model.model_copy(update={"max_tokens": 4000}),
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
    plan = Plan.model_validate(
        {
            "steps": [
                {
                    "name": "Single step",
                    "description": "Do one thing",
                    "requires_code": False,
                    "expected_outputs": ["done"],
                    "success_criteria": ["it is done"],
                }
            ]
        }
    )
    planning_agent = PlanningAgent(
        llm=EmptyReflectionFakeChatModel(plan),
        workspace=tmpdir,
        max_reflection_steps=1,
    )

    result = await planning_agent.ainvoke(
        {"messages": [HumanMessage(content="make a plan")]}
    )

    assert result["plan"] == plan
    assert result["messages"][-1].content == "[APPROVED]"
