from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from ursa.workflows import SimulationUseWorkflow


def test_deprecated_simulation_workflow_preserves_schema_prompt_and_result():
    class Planner:
        calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            return {"plan": SimpleNamespace(steps=["run simulator"])}

    class Executor:
        calls = []

        def invoke(self, prompt):
            self.calls.append(prompt)
            return {"messages": [AIMessage(content="simulation complete")]}

    planner = Planner()
    executor = Executor()
    with pytest.warns(DeprecationWarning, match="PlanningExecutionAgent"):
        workflow = SimulationUseWorkflow(
            planner=planner,
            executor=executor,
            workspace="workspace",
            tool_description="Run the DCOPF simulator.",
        )

    result = workflow.invoke("sweep loads")

    assert result == "simulation complete"
    assert "CodeExecutionDescriptor" in planner.calls[0]
    assert "Run the DCOPF simulator." in planner.calls[0]
    assert "Run the DCOPF simulator." in executor.calls[0]
