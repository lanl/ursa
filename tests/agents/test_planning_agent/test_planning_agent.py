from pathlib import Path

from langchain_core.messages import HumanMessage

from ursa.agents import PlanningAgent


async def test_planning_agent_creates_structured_plan(chat_model, tmpdir: Path):
    planning_agent = PlanningAgent(
        llm=chat_model.model_copy(update={"max_tokens": 4000}),
        workspace=tmpdir / ".ursa",
        max_reflection_steps=0,
    )

    prompt = "Outline a concise plan for adding the numbers 1 and 2 together."
    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content=prompt)],
        "reflection_steps": 0,
    })

    assert "plan_steps" in result
    assert isinstance(result["plan_steps"], list)
    assert result["plan_steps"], "expected at least one plan step"

    for step in result["plan_steps"]:
        assert isinstance(step, dict)
        for key in (
            "name",
            "description",
            "requires_code",
            "expected_outputs",
            "success_criteria",
        ):
            assert key in step

    assert "messages" in result
    assert result["messages"], "agent should return at least one message"
    assert getattr(result["messages"][-1], "content", None)
