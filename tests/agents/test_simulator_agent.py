from __future__ import annotations

from langchain_core.messages import AIMessage

from tests.composite_helpers import CompositeFakeModel, request_text
from ursa.experimental.agents.simulator_agent import (
    SimulatorAgent,
    documenter_prompt,
    runner_prompt,
)
from ursa.util import Checkpointer


def _model() -> CompositeFakeModel:
    return CompositeFakeModel(
        messages=iter([
            AIMessage(content="documentation ready"),
            AIMessage(content="simulation complete"),
            AIMessage(content="final recap"),
        ])
    )


def test_simulator_uses_documenter_and_runner_child_subgraphs(tmp_path):
    model = _model()
    agent = SimulatorAgent(
        model,
        workspace=tmp_path,
        enable_metrics=False,
    )

    assert {name for name, _ in agent.compiled_graph.get_subgraphs()} == {
        "_documenter",
        "_runner",
    }
    assert agent.agent_nodes["_documenter"] is agent.documenter
    assert agent.agent_nodes["_runner"] is agent.runner

    state = agent.invoke("Run the simulation")

    assert state["goal"] == "Run the simulation"
    assert state["messages"][-1].text == "final recap"
    assert documenter_prompt.strip() in request_text(model.plain_requests[0])
    assert runner_prompt.strip() in request_text(model.plain_requests[1])
    agent.close()


async def test_simulator_child_subgraphs_support_async_invocation(tmp_path):
    agent = SimulatorAgent(
        _model(),
        workspace=tmp_path,
        enable_metrics=False,
    )

    state = await agent.ainvoke("Run asynchronously")

    assert state["goal"] == "Run asynchronously"
    assert state["messages"][-1].text == "final recap"
    await agent.aclose()


def test_simulator_child_transcripts_are_isolated_across_persistent_runs(
    tmp_path,
):
    model = CompositeFakeModel(
        messages=iter([
            AIMessage(content="first documentation"),
            AIMessage(content="first simulation"),
            AIMessage(content="first recap"),
            AIMessage(content="second documentation"),
            AIMessage(content="second simulation"),
            AIMessage(content="second recap"),
        ])
    )
    agent = SimulatorAgent(
        model,
        workspace=tmp_path,
        checkpointer=Checkpointer.from_workspace(tmp_path),
        enable_metrics=False,
    )
    config = {"configurable": {"thread_id": "simulator-thread"}}

    agent.invoke("First simulation", config=config)
    agent.invoke("Second simulation", config=config)

    second_documenter_request = request_text(model.plain_requests[3])
    second_runner_request = request_text(model.plain_requests[4])
    assert second_documenter_request.count("first documentation") == 1
    assert second_documenter_request.count("Second simulation") == 1
    assert "First simulation" not in second_runner_request
    assert second_runner_request.count("Second simulation") == 1
    agent.close()
