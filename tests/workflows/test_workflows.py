from types import SimpleNamespace

from langchain_core.messages import AIMessage

from ursa.workflows import PlanningExecutorWorkflow, SimulationUseWorkflow


class _FakePlanner:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        step = SimpleNamespace(
            name="Step 1",
            description="Do a thing",
            requires_code=False,
            expected_outputs=["output"],
            success_criteria=["criterion"],
        )
        return {"plan": SimpleNamespace(steps=[step])}


class _FakeExecutor:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return {"messages": [AIMessage(content="done")]}


def test_planning_executor_workflow_handles_string_input(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.workflows.planning_execution_workflow.render_plan_steps_rich",
        lambda _: None,
    )
    planner = _FakePlanner()
    executor = _FakeExecutor()
    workflow = PlanningExecutorWorkflow(
        planner=planner,
        executor=executor,
        workspace=tmp_path,
    )

    result = workflow.invoke("Solve this")

    assert result == "done"
    assert planner.prompts
    assert "Solve this" in planner.prompts[0]
    assert executor.prompts


def test_simulation_use_workflow_handles_string_input_and_tool_schema(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.workflows.simulation_use_workflow.render_plan_steps_rich",
        lambda _: None,
    )
    planner = _FakePlanner()
    executor = _FakeExecutor()
    workflow = SimulationUseWorkflow(
        planner=planner,
        executor=executor,
        workspace=tmp_path,
        tool_description="tool desc",
        tool_schema="custom schema",
    )

    result = workflow.invoke("Run simulation sweep")

    assert result == "done"
    assert workflow.tool_schema == "custom schema"
    assert planner.prompts
    assert "Run simulation sweep" in planner.prompts[0]
    assert executor.prompts
