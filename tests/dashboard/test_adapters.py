from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from ursa.cli.callbacks import AgentLogEventHandler
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


@pytest.fixture(params=[BaseAgentInProcessAdapter, DirectInvokeAdapter])
def adapter_factory(request):
    def create(agent):
        options = (
            {"supports_streaming": False}
            if request.param is BaseAgentInProcessAdapter
            else {}
        )
        return request.param(lambda _workspace, _inputs: agent, **options)

    return create


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("formatted", [False, True])
def test_adapters_attach_cli_callbacks_and_format_results(
    tmp_path, adapter_factory, asynchronous, formatted
):
    captured = {}
    lifecycle = []

    def invoke(inputs, *, config=None):
        captured.update(inputs=inputs, config=config)
        return {"text": "done"} if formatted else "done"

    async def ainvoke(inputs, *, config=None):
        return invoke(inputs, config=config)

    async def aclose():
        lifecycle.append("aclose")

    async def setup(_agent, ctx, inputs):
        assert ctx.workspace_dir == tmp_path and inputs == "hello"
        lifecycle.append("setup")

    agent = SimpleNamespace(
        invoke=invoke, aclose=aclose, close=lambda: lifecycle.append("close")
    )
    if asynchronous:
        agent.ainvoke = ainvoke
    if formatted:
        agent.format_result = lambda result: result["text"]
    adapter = adapter_factory(agent)
    adapter.set_setup_hook(setup)
    assert (
        adapter.invoke(ctx=_run_context(tmp_path), inputs="hello", sink=None)
        == "done"
    )
    assert captured["inputs"] == "hello"
    assert isinstance(captured["config"], dict)
    callbacks = captured["config"]["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], AgentLogEventHandler)
    assert callbacks[0].workspace == tmp_path.resolve()
    assert lifecycle == (
        ["setup", "aclose", "close"] if asynchronous else ["setup"]
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_adapters_skip_config_for_simple_agents(
    tmp_path, adapter_factory, asynchronous
):
    captured = {}

    def invoke(inputs):
        captured["inputs"] = inputs
        return "done"

    async def ainvoke(inputs):
        return invoke(inputs)

    agent = SimpleNamespace(invoke=invoke)
    if asynchronous:
        agent.ainvoke = ainvoke
    adapter = adapter_factory(agent)
    assert (
        adapter.invoke(ctx=_run_context(tmp_path), inputs="hello", sink=None)
        == "done"
    )
    assert captured == {"inputs": "hello"}


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
