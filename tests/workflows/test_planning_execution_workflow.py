from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from ursa.agents.planning_agent import Plan, PlanStep
from ursa.workflows.planning_execution_workflow import PlanningExecutorWorkflow


class StubPlanner:
    def __init__(self, plan: Plan):
        self.prompts: list[str] = []
        self.plan = plan
        self.invoke_kwargs: list[dict] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return {"plan": self.plan}

    def format_query(self, prompt: str):
        self.prompts.append(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    async def _ainvoke(self, _state, **kwargs):
        self.invoke_kwargs.append(kwargs)
        return {"plan": self.plan}


class StubExecutor:
    def __init__(self):
        self.prompts: list[str] = []
        self.step_counter = 0
        self.invoke_kwargs: list[dict] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        self.step_counter += 1
        return {
            "messages": [AIMessage(content=f"exec-step-{self.step_counter}")]
        }

    def format_query(self, prompt: str):
        self.prompts.append(prompt)
        return {"messages": [HumanMessage(content=prompt)]}

    async def _ainvoke(self, _state, **kwargs):
        self.invoke_kwargs.append(kwargs)
        self.step_counter += 1
        return {
            "messages": [AIMessage(content=f"exec-step-{self.step_counter}")]
        }

    def format_result(self, state):
        return state["messages"][-1].text


def _plan_with_two_steps() -> Plan:
    return Plan(
        steps=[
            PlanStep(
                name="Step 1",
                description="Initial work",
                requires_code=False,
                expected_outputs=["artifact-1"],
                success_criteria=["criteria-1"],
            ),
            PlanStep(
                name="Step 2",
                description="Follow-on work",
                requires_code=False,
                expected_outputs=["artifact-2"],
                success_criteria=["criteria-2"],
            ),
        ]
    )


def test_planning_execution_workflow_planner_executor_handoff(monkeypatch):
    monkeypatch.setattr(
        "ursa.workflows.planning_execution_workflow.render_plan_steps_rich",
        lambda _steps: None,
    )

    planner = StubPlanner(plan=_plan_with_two_steps())
    executor = StubExecutor()
    workflow = PlanningExecutorWorkflow(
        planner=planner,
        executor=executor,
        workspace="unused-in-test",
    )

    result = workflow.invoke("Solve this task")

    assert len(planner.prompts) == 1
    assert planner.prompts[0] == "Solve this task"
    assert (
        planner.invoke_kwargs[0]["configurable"]["checkpoint_ns"] == "planner"
    )
    assert len(executor.prompts) == 2
    assert "Previous-step summary" not in executor.prompts[0]
    assert "Previous-step summary:\nexec-step-1" in executor.prompts[1]
    assert (
        executor.invoke_kwargs[0]["configurable"]["checkpoint_ns"] == "executor"
    )
    assert result == "exec-step-2"
