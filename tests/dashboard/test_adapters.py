from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from ursa.cli.callbacks import HITLLogEventHandler
from ursa.workflows.planning_execution_workflow import PlanningExecutorWorkflow
from ursa_dashboard.adapters import (
    BaseAgentInProcessAdapter,
    DirectInvokeAdapter,
    RunContext,
)


def _run_context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="run-1",
        agent_id="agent-1",
        workspace_dir=tmp_path,
    )


def test_base_agent_adapter_attaches_cli_callback_handler(tmp_path: Path):
    captured: dict[str, object] = {}

    class FakeAgent:
        def invoke(self, inputs, config=None):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"text": "done"}

        def format_result(self, result):
            return result["text"]

    adapter = BaseAgentInProcessAdapter(
        lambda _workspace, _inputs: FakeAgent(),
        supports_streaming=False,
    )

    result = adapter.invoke(
        ctx=_run_context(tmp_path),
        inputs="hello",
        sink=SimpleNamespace(emit=lambda event: None),
    )

    assert result == "done"
    assert captured["inputs"] == "hello"
    config = captured["config"]
    assert isinstance(config, dict)
    callbacks = config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], HITLLogEventHandler)
    assert callbacks[0].workspace == tmp_path.resolve()


def test_direct_invoke_adapter_skips_config_for_simple_agents(tmp_path: Path):
    captured: dict[str, object] = {}

    class FakeDemoAgent:
        def invoke(self, inputs):
            captured["inputs"] = inputs
            return "done"

    adapter = DirectInvokeAdapter(lambda _workspace, _inputs: FakeDemoAgent())

    result = adapter.invoke(
        ctx=_run_context(tmp_path),
        inputs="hello",
        sink=SimpleNamespace(emit=lambda event: None),
    )

    assert result == "done"
    assert captured == {"inputs": "hello"}


def test_direct_invoke_adapter_attaches_cli_handler_when_supported(
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    class FakeWorkflow:
        def invoke(self, inputs, *, config=None):
            captured["inputs"] = inputs
            captured["config"] = config
            return "done"

    adapter = DirectInvokeAdapter(lambda _workspace, _inputs: FakeWorkflow())

    result = adapter.invoke(
        ctx=_run_context(tmp_path),
        inputs="hello",
        sink=SimpleNamespace(emit=lambda event: None),
    )

    assert result == "done"
    assert captured["inputs"] == "hello"
    config = captured["config"]
    assert isinstance(config, dict)
    callbacks = config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], HITLLogEventHandler)


def test_planning_executor_workflow_propagates_callbacks(
    tmp_path: Path,
    monkeypatch,
):
    planner_calls: list[dict | None] = []
    executor_calls: list[dict | None] = []

    class FakePlanner:
        checkpointer = None

        def invoke(self, prompt, *, config=None):
            planner_calls.append(config)
            return {"plan": SimpleNamespace(steps=["step one", "step two"])}

    class FakeExecutor:
        checkpointer = None

        def invoke(self, prompt, *, config=None):
            executor_calls.append(config)
            return {"messages": [SimpleNamespace(text=f"done for {prompt}")]}

    class DummyConsole:
        def status(self, *_args, **_kwargs):
            return nullcontext()

        def print(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "ursa.workflows.planning_execution_workflow.console",
        DummyConsole(),
    )
    monkeypatch.setattr(
        "ursa.workflows.planning_execution_workflow.render_plan_steps_rich",
        lambda steps: None,
    )

    workflow = PlanningExecutorWorkflow(
        planner=FakePlanner(),
        executor=FakeExecutor(),
        workspace=tmp_path,
    )
    callbacks = [object()]

    result = workflow.invoke("solve this", config={"callbacks": callbacks})

    assert "done for" in result
    assert planner_calls == [{"callbacks": callbacks}]
    assert executor_calls == [{"callbacks": callbacks}] * 2
