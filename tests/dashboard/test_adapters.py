from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ursa.cli.callbacks import HITLLogEventHandler
from ursa.workflows.planning_execution_workflow import PlanningExecutorWorkflow
from ursa_dashboard import adapters
from ursa_dashboard.adapters import (
    BaseAgentInProcessAdapter,
    DirectInvokeAdapter,
    RunContext,
)


@pytest.fixture(
    params=[
        DirectInvokeAdapter,
        partial(BaseAgentInProcessAdapter, supports_streaming=False),
    ]
)
def adapter_factory(request):
    return request.param


def _run_context(tmp_path: Path) -> RunContext:
    return RunContext(
        run_id="run-1",
        agent_id="agent-1",
        workspace_dir=tmp_path,
    )


@pytest.mark.parametrize("workflow", [False, True])
def test_adapter_attaches_cli_callback_handler(
    tmp_path: Path, adapter_factory, workflow
):
    captured: dict[str, object] = {}

    class FakeAgent:
        def invoke(self, inputs, config=None):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"text": "done"}

        def format_result(self, result):
            return result["text"]

    class FakeWorkflow:
        def invoke(self, inputs, *, config=None):
            captured["inputs"] = inputs
            captured["config"] = config
            return "done"

    agent = FakeWorkflow() if workflow else FakeAgent()
    adapter = adapter_factory(lambda _workspace, _inputs: agent)

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


def test_adapter_skips_config_for_simple_agents(
    tmp_path: Path, adapter_factory
):
    captured: dict[str, object] = {}

    class FakeDemoAgent:
        def invoke(self, inputs):
            captured["inputs"] = inputs
            return "done"

    adapter = adapter_factory(lambda _workspace, _inputs: FakeDemoAgent())

    result = adapter.invoke(
        ctx=_run_context(tmp_path),
        inputs="hello",
        sink=SimpleNamespace(emit=lambda event: None),
    )

    assert result == "done"
    assert captured == {"inputs": "hello"}


@pytest.mark.parametrize("failure", [None, "setup", "invoke"])
async def test_async_adapter_setup_and_cleanup(
    tmp_path, adapter_factory, monkeypatch, failure
):
    calls = []
    resource = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        adapters,
        "_ensure_async_checkpointer",
        AsyncMock(return_value=[resource]),
    )

    async def setup(agent, ctx, inputs):
        calls.append("setup")
        assert ctx.workspace_dir == tmp_path and inputs == "hello"
        if failure == "setup":
            raise RuntimeError("setup")

    async def invoke(inputs, **kwargs):
        calls.append("invoke")
        assert isinstance(kwargs["config"]["callbacks"][0], HITLLogEventHandler)
        if failure == "invoke":
            raise RuntimeError("invoke")
        return {"text": inputs}

    agent = SimpleNamespace(ainvoke=invoke, format_result=lambda r: r["text"])
    adapter = adapter_factory(lambda *_: agent)
    adapter.set_setup_hook(setup)
    with (
        pytest.raises(RuntimeError, match=failure) if failure else nullcontext()
    ):
        assert (
            await adapter.ainvoke(
                ctx=_run_context(tmp_path), inputs="hello", sink=None
            )
            == "hello"
        )
    assert calls == (["setup"] if failure == "setup" else ["setup", "invoke"])
    resource.close.assert_awaited_once()


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
