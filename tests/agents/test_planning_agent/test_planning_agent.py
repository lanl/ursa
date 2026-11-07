from collections.abc import Iterator
from unittest.mock import Mock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import Field

from tests.agents.utils import RecordingChatModel, simple_plan
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
                return simple_plan(schema)

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
    plan_factory = Mock(return_value=plan)
    planning_agent = PlanningAgent(
        llm=RecordingChatModel(response="", structured_factory=plan_factory),
        workspace=tmpdir,
        max_reflection_steps=1,
    )

    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content="make a plan")]
    })

    assert result["plan"] == plan
    assert result["messages"][-1].content == "[APPROVED]"
    plan_factory.assert_called_once_with(Plan)


class RecordingPlanningChatModel(GenericFakeChatModel):
    def __init__(self, plan: Plan, reviews: list[str]):
        super().__init__(
            messages=iter(AIMessage(content=review) for review in reviews)
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "generation_messages", [])
        object.__setattr__(self, "reflection_messages", [])

    def with_structured_output(self, schema, **kwargs):
        model = self

        class StructuredOutput:
            def invoke(self, messages):
                model.generation_messages.append(list(messages))
                return model.plan

        return StructuredOutput()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.reflection_messages.append(list(messages))
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


async def test_planner_preserves_history_without_assistant_prefill(tmpdir):
    """Retain the request and drafted plan while sending provider-valid calls."""
    llm = RecordingChatModel(
        response="[APPROVED]", structured_factory=simple_plan
    )
    planning_agent = PlanningAgent(
        llm=llm, workspace=tmpdir, max_reflection_steps=1
    )

    prompt = "make a plan"
    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content=prompt)]
    })

    assert len(llm.calls) >= 2, "expected generation and reflection calls"
    for call in llm.calls:
        assert not isinstance(call[-1], AIMessage), (
            "planner sent a request ending with an AI message, which "
            "models without assistant-message prefill support reject"
        )

    contents = [msg.content for msg in result["messages"]]
    assert contents[0] == prompt, "original request should be preserved"
    assert any(isinstance(msg, AIMessage) for msg in result["messages"]), (
        "drafted plan message should be retained in state"
    )


async def test_planning_agent_uses_fresh_user_requests_for_reflection(tmpdir):
    plan = Plan.model_validate({
        "steps": [
            {
                "name": "Run experiment",
                "description": "Collect and analyze observations.",
                "requires_code": False,
                "expected_outputs": ["results"],
                "success_criteria": ["results answer the question"],
            }
        ]
    })
    model = RecordingPlanningChatModel(
        plan,
        reviews=["Add a validation check.", "[APPROVED]"],
    )
    planning_agent = PlanningAgent(
        llm=model,
        workspace=tmpdir,
        max_reflection_steps=2,
    )

    result = await planning_agent.ainvoke({
        "messages": [HumanMessage(content="Design an experiment")]
    })

    assert result["plan"] == plan
    assert len(model.generation_messages) == 2
    assert len(model.reflection_messages) == 2
    for messages in model.reflection_messages:
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[-1], HumanMessage)
        assert "Design an experiment" in messages[-1].content
        assert plan.model_dump_json(indent=2) in messages[-1].content

    revision_request = model.generation_messages[1]
    assert isinstance(revision_request[0], SystemMessage)
    assert isinstance(revision_request[-1], HumanMessage)
    assert "Add a validation check." in revision_request[-1].content
