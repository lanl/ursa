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
    "config,expected_path,with_embeddings",
    [
        ({}, "../simulator_docs", True),
        ({"docs_path": "custom_docs"}, "custom_docs", True),
        ({"docs_path": Path("custom_docs")}, Path("custom_docs"), True),
        ({"docs_path": "custom_docs"}, None, False),
    ],
)
def test_simulator_rag_configuration(
    monkeypatch,
    tmp_path,
    chat_model,
    embedding_model,
    config,
    expected_path,
    with_embeddings,
):
    import ursa.agents.simulator_agent as simulator
    import ursa.security as security

    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        simulator, "ChatAgent", Mock(side_effect=lambda **_: Mock())
    )
    rag_class = Mock()
    monkeypatch.setattr(simulator, "RAGAgent", rag_class)

    embedding = {"embedding": embedding_model} if with_embeddings else {}
    agent = simulator.SimulatorAgent(
        llm=chat_model,
        **embedding,
        workspace=tmp_path,
        thread_id="simulation-test",
        checkpointer=None,
        **config,
    )

    if with_embeddings:
        assert rag_class.call_args.kwargs["database_path"] == expected_path
        assert rag_class.call_args.kwargs["embedding"] is embedding_model
        assert agent.rag_agent is rag_class.return_value
        agent.documenter.add_tool.assert_called_once()
        agent.runner.add_tool.assert_called_once()
    else:
        rag_class.assert_not_called()
        assert (
            agent.doc_rag("documentation")
            == "Tool Failed: No RAG database available."
        )
