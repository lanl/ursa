from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from tests.composite_helpers import CompositeFakeModel, request_text
from ursa.agents.hypothesizer_agent import HypothesizerAgent
from ursa.util import Checkpointer
from ursa.workflows.think_plan_execute import (
    ThinkPlanningExecutionAgent,
    think_plan_execute_workflow,
)

INITIAL_HYPOTHESIS_SPACE = """# Hypothesis Space

### H1: A testable causal mechanism

- **Relative likelihood:** 0.6
- **Recommended next evidence:** Inspect the system.
"""

UPDATED_HYPOTHESIS_SPACE = """# Hypothesis Space

### H1: The causal mechanism is supported by execution

- **Relative likelihood:** 0.8
- **Evidence for:** Both execution steps produced supporting results.
"""


class CountingCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self.llm_starts: list[dict | None] = []

    def on_llm_start(self, serialized, prompts, **kwargs):  # noqa: ANN001
        self.llm_starts.append(kwargs.get("metadata"))


def think_model(
    *,
    initial: str = INITIAL_HYPOTHESIS_SPACE,
    updated: str = UPDATED_HYPOTHESIS_SPACE,
) -> CompositeFakeModel:
    return CompositeFakeModel(
        messages=iter([
            AIMessage(content=initial),
            AIMessage(content="step-1-work"),
            AIMessage(content="step-1-summary"),
            AIMessage(content="step-2-work"),
            AIMessage(content="step-2-summary"),
            AIMessage(content=updated),
        ])
    )


def test_think_plan_execute_uses_ursa_hypothesizer_before_and_after_plan(
    tmp_path,
):
    model = think_model()
    agent = ThinkPlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    state = agent.invoke("Investigate the system")

    assert isinstance(agent.hypothesizer_agent, HypothesizerAgent)
    assert state["hypothesis"] == UPDATED_HYPOTHESIS_SPACE.strip()
    assert state["hypothesis_space_markdown"] == (
        UPDATED_HYPOTHESIS_SPACE.strip()
    )
    assert state["step_results"] == [
        "step-1-summary",
        "step-2-summary",
    ]
    assert len(state["revision_history"]) == 2
    assert agent.format_result(state) == UPDATED_HYPOTHESIS_SPACE.strip()
    assert agent.llm.bound is model
    assert {name for name, _graph in agent.compiled_graph.get_subgraphs()} == {
        "hypothesizer",
        "planner",
        "executor",
    }

    initial_hypothesis_request = request_text(model.plain_requests[0])
    assert "Investigate the system" in initial_hypothesis_request
    assert "Initialize a hypothesis space before planning" in (
        initial_hypothesis_request
    )

    plan_request = next(
        request_text(messages)
        for schema, messages in model.structured_requests
        if schema == "Plan"
    )
    assert "Working hypothesis:" in plan_request
    assert "H1: A testable causal mechanism" in plan_request

    final_hypothesis_request = request_text(model.plain_requests[-1])
    assert "Execution produced the following evidence and results" in (
        final_hypothesis_request
    )
    assert "Step 1: step-1-summary" in final_hypothesis_request
    assert "Step 2: step-2-summary" in final_hypothesis_request
    assert "H1: A testable causal mechanism" in final_hypothesis_request

    artifact = tmp_path / "experiences" / "hypothesis_space.md"
    assert artifact.read_text(encoding="utf-8").strip() == (
        UPDATED_HYPOTHESIS_SPACE.strip()
    )
    agent.close()


@pytest.mark.asyncio
async def test_think_plan_execute_supports_async_nested_persistence(tmp_path):
    checkpointer = await Checkpointer.async_from_workspace(tmp_path)
    agent = ThinkPlanningExecutionAgent(
        llm=think_model(),
        workspace=tmp_path,
        checkpointer=checkpointer,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    state = await agent.ainvoke(
        "Investigate asynchronously",
        config={"configurable": {"thread_id": "async-think-thread"}},
    )

    assert agent.format_result(state) == UPDATED_HYPOTHESIS_SPACE.strip()
    assert state["step_results"] == [
        "step-1-summary",
        "step-2-summary",
    ]
    cursor = await checkpointer.conn.execute(
        "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id = ?",
        ("async-think-thread",),
    )
    assert {row[0] for row in await cursor.fetchall()} == {
        "",
        "hypothesizer",
        "planner",
        "executor",
    }
    await agent.aclose()
    await checkpointer.conn.close()


def test_follow_up_updates_existing_space_then_replans_and_updates_again(
    tmp_path,
):
    first_initial = INITIAL_HYPOTHESIS_SPACE
    first_updated = UPDATED_HYPOTHESIS_SPACE
    follow_up_initial = """# Hypothesis Space

### H1: Follow-up evidence weakens the original mechanism

- **Relative likelihood:** 0.45
"""
    follow_up_updated = """# Hypothesis Space

### H1: Follow-up execution supports a revised mechanism

- **Relative likelihood:** 0.7
"""
    model = CompositeFakeModel(
        messages=iter([
            AIMessage(content=first_initial),
            AIMessage(content="first-step-1-work"),
            AIMessage(content="first-step-1-summary"),
            AIMessage(content="first-step-2-work"),
            AIMessage(content="first-step-2-summary"),
            AIMessage(content=first_updated),
            AIMessage(content=follow_up_initial),
            AIMessage(content="follow-up-step-1-work"),
            AIMessage(content="follow-up-step-1-summary"),
            AIMessage(content="follow-up-step-2-work"),
            AIMessage(content="follow-up-step-2-summary"),
            AIMessage(content=follow_up_updated),
        ])
    )
    checkpointer = Checkpointer.from_workspace(tmp_path)
    agent = ThinkPlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        checkpointer=checkpointer,
        thread_id="continuing-investigation",
        max_reflection_steps=0,
        enable_metrics=False,
    )

    first = agent.invoke("Investigate the original system behavior")
    second = agent.invoke("Now account for the newly observed sensor drift")

    assert first["query"] == "Investigate the original system behavior"
    assert second["query"] == "Investigate the original system behavior"
    assert second["task"] == ("Now account for the newly observed sensor drift")
    assert second["hypothesis_space_markdown"] == (follow_up_updated.strip())
    assert second["step_results"] == [
        "follow-up-step-1-summary",
        "follow-up-step-2-summary",
    ]
    assert len(second["revision_history"]) == 4

    follow_up_hypothesis_request = request_text(model.plain_requests[6])
    assert "newly observed sensor drift" in follow_up_hypothesis_request
    assert "Treat this as a follow-up request" in follow_up_hypothesis_request
    assert "execution supports a revised mechanism" not in (
        follow_up_hypothesis_request
    )
    assert "causal mechanism is supported by execution" in (
        follow_up_hypothesis_request
    )

    plan_requests = [
        request_text(messages)
        for schema, messages in model.structured_requests
        if schema == "Plan"
    ]
    assert len(plan_requests) == 2
    assert "Follow-up evidence weakens" in plan_requests[1]
    assert "newly observed sensor drift" in plan_requests[1]

    final_update_request = request_text(model.plain_requests[-1])
    assert "follow-up-step-1-summary" in final_update_request
    assert "follow-up-step-2-summary" in final_update_request
    assert "Follow-up evidence weakens" in final_update_request
    agent.close()


def test_think_factory_returns_the_parent_agent_not_an_orphan_graph(tmp_path):
    model = CompositeFakeModel()

    agent = think_plan_execute_workflow(
        model,
        workspace=tmp_path,
        max_reflection_steps=0,
        enable_metrics=False,
    )

    assert isinstance(agent, ThinkPlanningExecutionAgent)
    assert agent.llm.bound is model
    assert agent.compiled_graph is not None
    assert agent.thread_id
    agent.close()


def test_parent_callbacks_and_telemetry_propagate_to_native_nodes(tmp_path):
    model = think_model()
    callback = CountingCallback()
    agent = ThinkPlanningExecutionAgent(
        llm=model,
        workspace=tmp_path,
        max_reflection_steps=0,
        enable_metrics=True,
        autosave_metrics=True,
    )

    agent.invoke(
        "Investigate callbacks",
        config={"callbacks": [callback]},
    )

    # Two hypothesizer updates + two executor agent calls + two recaps. The
    # fake structured planner adapter records requests without an LLM hook.
    assert len(callback.llm_starts) >= 6
    node_names = {
        metadata.get("langgraph_node")
        for metadata in callback.llm_starts
        if metadata is not None
    }
    assert "update_hypothesis_space" in node_names
    assert "agent" in node_names
    assert "recap" in node_names

    metrics_files = list(Path(agent.telemetry.output_dir).glob("*.json"))
    assert metrics_files
    payload = json.loads(
        max(metrics_files, key=lambda path: path.stat().st_mtime).read_text()
    )
    telemetry_nodes = {
        event.get("metadata", {}).get("langgraph_node")
        for event in payload["llm_events"]
    }
    assert {"update_hypothesis_space", "agent", "recap"} <= (telemetry_nodes)
    agent.close()
