import importlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from ursa.agents.base import AgentContext, AgentWithTools, BaseAgent
from ursa.tools.write_code_tool import write_code
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
        "ursa.agents.chat_agent.ChatAgent",
        "ursa.agents.code_review_agent.CodeReviewAgent",
        "ursa.agents.execution_agent.ExecutionAgent",
        "ursa.agents.hypothesizer_agent.HypothesizerAgent",
        "ursa.agents.mp_agent.MaterialsProjectAgent",
        "ursa.agents.planning_agent.PlanningAgent",
        "ursa.agents.rag_agent.RAGAgent",
        "ursa.agents.recall_agent.RecallAgent",
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


def _make_tool(name: str):
    @tool(name)
    def simple_tool(message: str) -> str:
        """Echo the provided message."""
        return message

    return simple_tool


class DummyAgentWithTools(AgentWithTools, BaseAgent):
    def __init__(self, llm, tools=None, handle_tool_errors=True, **kwargs):
        self.build_graph_calls = 0
        super().__init__(
            llm=llm,
            tools=tools,
            handle_tool_errors=handle_tool_errors,
            **kwargs,
        )

    def _noop(self, state):
        return state

    def _build_graph(self):
        self.build_graph_calls += 1
        self.add_node(self._noop, "noop")
        self.graph.set_entry_point("noop")
        self.graph.set_finish_point("noop")


def test_agent_with_tools_defaults_to_empty_tool_dict(chat_model, tmp_path):
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=None,
        workspace=tmp_path,
    )
    assert agent.tools == {}


def test_agent_with_tools_initializes_tool_mapping(chat_model, tmp_path):
    alpha = _make_tool("alpha")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha],
        workspace=tmp_path,
    )
    assert agent.tools == {"alpha": alpha}


def test_agent_with_tools_add_tool_updates_mapping(chat_model, tmp_path):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha],
        workspace=tmp_path,
    )
    agent.add_tool(beta)
    assert agent.tools == {"alpha": alpha, "beta": beta}


def test_agent_with_tools_remove_tool_accepts_single_name(chat_model, tmp_path):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha, beta],
        workspace=tmp_path,
    )
    agent.remove_tool("alpha")
    assert agent.tools == {"beta": beta}


def test_agent_with_tools_remove_tool_accepts_list(chat_model, tmp_path):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    gamma = _make_tool("gamma")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha, beta, gamma],
        workspace=tmp_path,
    )
    agent.remove_tool(["alpha", "gamma"])
    assert agent.tools == {"beta": beta}


def test_agent_with_tools_setter_updates_mapping_and_rebuilds_graph(
    chat_model, tmp_path
):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha],
        workspace=tmp_path,
    )
    initial_calls = agent.build_graph_calls
    agent.tools = {"beta": beta}
    assert agent.tools == {"beta": beta}
    assert agent.build_graph_calls == initial_calls + 1
    assert isinstance(agent.tool_node, ToolNode)


@pytest.mark.parametrize("handle_tool_errors", [True, "Custom error message"])
def test_handle_tool_errors_propagation(
    chat_model, tmp_path, handle_tool_errors
):
    """Verify that handle_tool_errors is propagated to ToolNode."""
    # Dummy tool used only for adding to the agent; no need to run it
    alpha = _make_tool("alpha")
    agent = DummyAgentWithTools(
        llm=chat_model,
        tools=[alpha],
        workspace=tmp_path,
        handle_tool_errors=handle_tool_errors,
    )
    # The ToolNode internally stores the error handling strategy
    assert agent.tool_node._handle_tool_errors == handle_tool_errors


@pytest.mark.asyncio
async def test_agent_with_tools_add_mcp_tools_adds_all(chat_model, tmp_path):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    client = AsyncMock()
    client.get_tools.return_value = [alpha, beta]

    agent = DummyAgentWithTools(llm=chat_model, workspace=tmp_path)
    await agent.add_mcp_tools(client)

    assert agent.tools == {"alpha": alpha, "beta": beta}
    client.get_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_with_tools_add_mcp_tools_filters_by_name(
    chat_model, tmp_path
):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    client = AsyncMock()
    client.get_tools.return_value = [alpha, beta]

    agent = DummyAgentWithTools(llm=chat_model, workspace=tmp_path)
    await agent.add_mcp_tools(client, tool_name="beta")

    assert agent.tools == {"beta": beta}


def test_tool_runtime_preserved_for_tool_node(chat_model, tmp_path: Path):
    class WriteToolAgent(AgentWithTools, BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(llm=chat_model, tools=[write_code], **kwargs)

        def _inject_tool_call(self, state, runtime: Runtime[AgentContext]):
            messages = list(state["messages"])
            assert isinstance(runtime.context, AgentContext)
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call_write_code",
                            "name": "write_code",
                            "args": {
                                "code": "print('runtime ok')",
                                "filename": "runtime_check.py",
                            },
                        }
                    ],
                )
            )
            return {"messages": messages}

        def _build_graph(self):
            self.add_node(self._inject_tool_call, "inject")
            self.add_node(self.tool_node, "tools")
            self.graph.set_entry_point("inject")
            self.graph.add_edge("inject", "tools")
            self.graph.set_finish_point("tools")

    agent = WriteToolAgent(workspace=tmp_path)
    result = agent.invoke("trigger tool")

    written_file = Path(tmp_path, "runtime_check.py")
    assert written_file.exists()
    assert written_file.read_text(encoding="utf-8") == "print('runtime ok')"
    assert any(
        getattr(msg, "tool_call_id", "") == "tool_call_write_code"
        for msg in result["messages"]
    )
