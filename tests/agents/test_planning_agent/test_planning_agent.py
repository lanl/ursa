from langchain_core.messages import HumanMessage

from ursa.agents.planning_agent import Plan, PlanningAgent


class FakePlanningChatModel:
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
