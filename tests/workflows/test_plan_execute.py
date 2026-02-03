import pytest
from pathlib import Path

from examples.two_agent_examples.plan_execute.plan_execute_from_yaml import (
    setup_agents,
)


@pytest.mark.asyncio
async def test_setup_agents_minimal(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    model = object()

    planner_instances = []
    executor_instances = []

    class DummyPlanner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            planner_instances.append(self)

    class DummyExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mcp_client = None
            executor_instances.append(self)

        async def add_mcp_tools(self, client):
            self.mcp_client = client

    monkeypatch.setattr(
        "examples.two_agent_examples.plan_execute.plan_execute_from_yaml.PlanningAgent",
        DummyPlanner,
    )
    monkeypatch.setattr(
        "examples.two_agent_examples.plan_execute.plan_execute_from_yaml.ExecutionAgent",
        DummyExecutor,
    )

    captured_mcp_configs = []
    dummy_client = object()

    def fake_start_mcp_client(config):
        captured_mcp_configs.append(config)
        return dummy_client

    monkeypatch.setattr(
        "examples.two_agent_examples.plan_execute.plan_execute_from_yaml.start_mcp_client",
        fake_start_mcp_client,
    )

    thread_id, planner_bundle, executor_bundle = await setup_agents(
        str(workspace),
        model,
        {"alpha": {"transport": "stdio"}},
    )

    assert thread_id == workspace.name
    assert len(planner_instances) == 1
    assert len(executor_instances) == 1

    planner = planner_instances[0]
    executor = executor_instances[0]

    assert planner.kwargs["llm"] is model
    assert planner_bundle[0] is planner
    assert planner_bundle[1] is planner.kwargs["checkpointer"]

    assert executor.kwargs["llm"] is model
    assert executor_bundle[0] is executor
    assert executor_bundle[1] is executor.kwargs["checkpointer"]

    planner_db_path = planner_bundle[2]
    executor_db_path = executor_bundle[2]

    assert isinstance(planner_db_path, Path)
    assert isinstance(executor_db_path, Path)
    assert planner_db_path.name == "planner_checkpoint.db"
    assert executor_db_path.name == "executor_checkpoint.db"
    assert (
        planner_db_path.parent
        == executor_db_path.parent
        == workspace / "checkpoints"
    )
    assert planner_db_path.parent.exists()

    assert captured_mcp_configs == [{"alpha": {"transport": "stdio"}}]
    assert executor.mcp_client is dummy_client
