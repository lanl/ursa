from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from tests.composite_helpers import CompositeFakeModel, request_text
from ursa.agents import SimulatorAgent
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


def test_simulator_agent_import():
    """The simulator is available through the regular agents package."""
    from ursa.agents import SimulatorAgent
    from ursa.agents.base import BaseAgent
    from ursa.agents.simulator_agent import SimulatorAgent as SimulatorClass

    assert SimulatorAgent is SimulatorClass
    assert issubclass(SimulatorAgent, BaseAgent)


@pytest.mark.parametrize(
    "config,expected_path",
    [
        ({}, "../simulator_docs"),
        ({"docs_path": "custom_docs"}, "custom_docs"),
        ({"docs_path": Path("custom_docs")}, Path("custom_docs")),
    ],
)
def test_simulator_rag_documentation_path(
    monkeypatch, tmp_path, chat_model, embedding_model, config, expected_path
):
    import ursa.agents.simulator_agent as simulator
    import ursa.security as security

    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        simulator, "ChatAgent", Mock(side_effect=lambda **_: Mock())
    )
    rag_class = Mock()
    monkeypatch.setattr(simulator, "RAGAgent", rag_class)

    agent = simulator.SimulatorAgent(
        llm=chat_model,
        embedding=embedding_model,
        workspace=tmp_path,
        thread_id="simulation-test",
        checkpointer=None,
        **config,
    )

    assert rag_class.call_args.kwargs["database_path"] == expected_path
    assert rag_class.call_args.kwargs["embedding"] is embedding_model
    assert agent.rag_agent is rag_class.return_value
    agent.documenter.add_tool.assert_called_once()
    agent.runner.add_tool.assert_called_once()


def test_simulator_without_embeddings_does_not_initialize_rag(
    monkeypatch, tmp_path, chat_model
):
    import ursa.agents.simulator_agent as simulator
    import ursa.security as security

    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        simulator, "ChatAgent", Mock(side_effect=lambda **_: Mock())
    )
    rag_class = Mock()
    monkeypatch.setattr(simulator, "RAGAgent", rag_class)

    agent = simulator.SimulatorAgent(
        llm=chat_model,
        workspace=tmp_path,
        thread_id="simulation-test",
        checkpointer=None,
        docs_path="custom_docs",
    )

    rag_class.assert_not_called()
    assert (
        agent.doc_rag("documentation")
        == "Tool Failed: No RAG database available."
    )
