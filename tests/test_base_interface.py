import importlib
import inspect
from pathlib import Path

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ursa.agents.base import BaseAgent
from ursa.util.memory_logger import AgentMemory


def load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


DEFAULT_QUERY = "What do you do?"
MODEL_QUERY = {
    "ursa.agents.acquisition_agents": {
        "query": "sky blue",
        "context": "Why is the sky blue?",
    },
}


@pytest.fixture(
    params=[
        "ursa.agents.acquisition_agents.ArxivAgent",
        "ursa.agents.acquisition_agents.WebSearchAgent",
        "ursa.agents.acquisition_agents.OSTIAgent",
        "ursa.agents.arxiv_agent.ArxivAgentLegacy",
        "ursa.agents.chat_agent.ChatAgent",
        "ursa.agents.code_review_agent.CodeReviewAgent",
        "ursa.agents.execution_agent.ExecutionAgent",
        "ursa.agents.hypothesizer_agent.HypothesizerAgent",
        "ursa.agents.mp_agent.MaterialsProjectAgent",
        "ursa.agents.planning_agent.PlanningAgent",
        "ursa.agents.rag_agent.RAGAgent",
        "ursa.agents.recall_agent.RecallAgent",
        "ursa.agents.websearch_agent.WebSearchAgentLegacy",
    ],
    ids=lambda agent_import: agent_import.rsplit(".", 1)[-1],
)
def agent_instance(request, tmpdir: Path, chat_model, embedding_model):
    agent_class = load_class(request.param)
    sig = inspect.signature(agent_class.__init__)

    kwargs = {}
    kwargs["llm"] = chat_model

    if request.param == "ursa.agents.recall_agent.RecallAgent":
        kwargs["memory"] = AgentMemory(embedding_model, Path(tmpdir / "memory"))

    kwargs["workspace"] = Path(tmpdir / ".ursa")
    for name, param in list(sig.parameters.items())[1:]:
        if name in ["embedding", "rag_embedding"]:
            kwargs[name] = embedding_model

    agent = agent_class(**kwargs)
    assert isinstance(agent, BaseAgent)

    # Will display on failed tests
    try:
        agent.compiled_graph.get_graph().print_ascii()
    except Exception as err:
        print(f"Failed to create graph: {err}")

    return agent


def test_interface(agent_instance):
    assert isinstance(agent_instance, BaseAgent)
    g = agent_instance.build_graph()
    assert isinstance(g, StateGraph)
    gc = agent_instance.compiled_graph
    assert isinstance(gc, CompiledStateGraph)
