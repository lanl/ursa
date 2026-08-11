from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ursa.cli.callbacks import HITLLogEventHandler
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
