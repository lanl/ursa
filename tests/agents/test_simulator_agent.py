from pathlib import Path
from unittest.mock import Mock

import pytest


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
        ({"database_path": "custom_docs"}, "custom_docs"),
        ({"database_path": Path("custom_docs")}, Path("custom_docs")),
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
        database_path="custom_docs",
    )

    rag_class.assert_not_called()
    assert (
        agent.doc_rag("documentation")
        == "Tool Failed: No RAG database available."
    )
