from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ursa.cli.callbacks import AgentLogEventHandler
from ursa_dashboard import registry
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


def test_planning_execution_registry_uses_one_base_agent_runtime(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakePlanningExecutionAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_lazy_class(path: str):
        captured["class_path"] = path
        return FakePlanningExecutionAgent

    monkeypatch.setattr(registry, "_lazy_class", fake_lazy_class)

    entry = registry.REGISTRY["planning_executor_workflow"]
    model = object()
    adapter = entry.build_adapter(model, {"max_reflection_steps": 0})

    assert isinstance(adapter, BaseAgentInProcessAdapter)
    agent = adapter._agent_factory(tmp_path, "solve this")
    assert isinstance(agent, FakePlanningExecutionAgent)
    assert captured["class_path"] == (
        "ursa.workflows.planning_execution_workflow.PlanningExecutionAgent"
    )
    assert captured["llm"] is model
    assert captured["workspace"] == str(tmp_path)
    assert captured["max_reflection_steps"] == 0
    assert "planner" not in captured
    assert "executor" not in captured


def test_hypothesis_symposium_registry_builds_workflow_adapter(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakeHypothesisSymposiumWorkflow:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_lazy_class(path: str):
        captured["class_path"] = path
        return FakeHypothesisSymposiumWorkflow

    monkeypatch.setattr(registry, "_lazy_class", fake_lazy_class)

    entry = registry.REGISTRY["hypothesis_symposium_agent"]
    model = object()
    adapter = entry.build_adapter(
        model,
        {
            "max_hypotheses": 3,
            "revision_rounds": 2,
            "experience_filename": "hypothesis_space.md",
        },
    )

    assert isinstance(adapter, BaseAgentInProcessAdapter)
    agent = adapter._agent_factory(tmp_path, "why is the sky blue?")
    assert isinstance(agent, FakeHypothesisSymposiumWorkflow)
    assert captured["class_path"] == (
        "ursa.workflows.hypothesis_symposium.HypothesisSymposiumWorkflow"
    )
    assert captured["llm"] is model
    # Like the other workflow entries, the factory passes the workspace through;
    # BaseWorkflow.__init__(**kwargs) accepts and ignores it.
    assert captured["workspace"] == str(tmp_path)
    assert captured["max_hypotheses"] == 3
    assert captured["revision_rounds"] == 2
    assert captured["experience_filename"] == "hypothesis_space.md"


def test_hypothesis_symposium_registry_spec_is_well_formed():
    entry = registry.REGISTRY["hypothesis_symposium_agent"]
    spec = entry.spec

    assert spec.agent_id == "hypothesis_symposium_agent"
    assert spec.capabilities.produces_artifacts is True
    # UI passes a single prompt string, normalized to the workflow's query.
    assert entry.build_inputs({"prompt": "hello"}) == "hello"

    param_names = {p.name for p in spec.parameters}
    assert {"prompt", "max_hypotheses", "revision_rounds"} <= param_names


def test_hypothesis_orchestrator_registry_builds_workflow_adapter(
    tmp_path: Path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakeHypothesisOrchestratorWorkflow:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_lazy_class(path: str):
        captured["class_path"] = path
        return FakeHypothesisOrchestratorWorkflow

    monkeypatch.setattr(registry, "_lazy_class", fake_lazy_class)

    entry = registry.REGISTRY["hypothesis_orchestrator_agent"]
    model = object()
    adapter = entry.build_adapter(
        model,
        {
            "max_hypotheses": 4,
            "experience_filename": "hypothesis_space.md",
        },
    )

    assert isinstance(adapter, BaseAgentInProcessAdapter)
    agent = adapter._agent_factory(tmp_path, "why is the sky blue?")
    assert isinstance(agent, FakeHypothesisOrchestratorWorkflow)
    assert captured["class_path"] == (
        "ursa.workflows.hypothesis_orchestrator.HypothesisOrchestratorWorkflow"
    )
    assert captured["llm"] is model
    # Workspace is threaded through so the workflow scopes its sub-agents.
    assert captured["workspace"] == str(tmp_path)
    assert captured["max_hypotheses"] == 4
    assert captured["experience_filename"] == "hypothesis_space.md"


def test_hypothesis_orchestrator_registry_spec_is_well_formed():
    entry = registry.REGISTRY["hypothesis_orchestrator_agent"]
    spec = entry.spec

    assert spec.agent_id == "hypothesis_orchestrator_agent"
    assert spec.capabilities.produces_artifacts is True
    assert entry.build_inputs({"prompt": "hello"}) == "hello"

    param_names = {p.name for p in spec.parameters}
    assert {"prompt", "max_hypotheses", "experience_filename"} <= param_names
