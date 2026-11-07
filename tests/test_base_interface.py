import importlib
import inspect
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from tests.agents.utils import tool_call_message
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
        "ursa.agents.execution_agent.ExecutionAgent",
        "ursa.agents.deep_review_agent.DeepReviewAgent",
        "ursa.agents.hypothesizer_agent.HypothesizerAgent",
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

    if request.param == "ursa.agents.rag_agent.RAGAgent":
        kwargs["embedding"] = embedding_model

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
    def __init__(self, handle_tool_errors=True, **kwargs):
        self.build_graph_calls = 0
        super().__init__(handle_tool_errors=handle_tool_errors, **kwargs)

    def _noop(self, state):
        return state

    def _build_graph(self):
        self.build_graph_calls += 1
        self.add_node(self._noop, "noop")
        self.graph.set_entry_point("noop")
        self.graph.set_finish_point("noop")


@pytest.fixture
def tool_agent(chat_model, tmp_path):
    def create(**kwargs):
        return DummyAgentWithTools(llm=chat_model, workspace=tmp_path, **kwargs)

    return create


def test_agent_with_tools_defaults_to_empty_tool_dict(tool_agent):
    agent = tool_agent(tools=None)
    assert agent.tools == {}


def test_agent_with_tools_initializes_tool_mapping(tool_agent):
    alpha = _make_tool("alpha")
    agent = tool_agent(tools=[alpha])
    assert agent.tools == {"alpha": alpha}


def test_agent_with_tools_add_tool_updates_mapping(tool_agent):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = tool_agent(tools=[alpha])
    agent.add_tool(beta)
    assert agent.tools == {"alpha": alpha, "beta": beta}


def test_agent_with_tools_remove_tool_accepts_single_name(tool_agent):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = tool_agent(tools=[alpha, beta])
    agent.remove_tool("alpha")
    assert agent.tools == {"beta": beta}


def test_agent_with_tools_remove_tool_accepts_list(tool_agent):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    gamma = _make_tool("gamma")
    agent = tool_agent(tools=[alpha, beta, gamma])
    agent.remove_tool(["alpha", "gamma"])
    assert agent.tools == {"beta": beta}


def test_agent_with_tools_setter_updates_mapping_and_rebuilds_graph(tool_agent):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    agent = tool_agent(tools=[alpha])
    initial_calls = agent.build_graph_calls
    agent.tools = {"beta": beta}
    assert agent.tools == {"beta": beta}
    assert agent.build_graph_calls == initial_calls + 1
    assert isinstance(agent.tool_node, ToolNode)


@pytest.mark.parametrize("handle_tool_errors", [True, "Custom error message"])
def test_handle_tool_errors_propagation(tool_agent, handle_tool_errors):
    """Verify that handle_tool_errors is propagated to ToolNode."""
    # Dummy tool used only for adding to the agent; no need to run it
    alpha = _make_tool("alpha")
    agent = tool_agent(tools=[alpha], handle_tool_errors=handle_tool_errors)
    # The ToolNode internally stores the error handling strategy
    assert agent.tool_node._handle_tool_errors == handle_tool_errors


@pytest.mark.parametrize("tool_name", [None, "beta"])
async def test_agent_with_tools_adds_requested_mcp_tools(tool_agent, tool_name):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    client = AsyncMock()
    client.get_tools.return_value = [alpha, beta]

    agent = tool_agent()
    options = {} if tool_name is None else {"tool_name": tool_name}
    await agent.add_mcp_tools(client, **options)

    expected = {"beta": beta} if tool_name else {"alpha": alpha, "beta": beta}
    assert agent.tools == expected
    if tool_name is None:
        client.get_tools.assert_awaited_once()


async def test_agent_with_tools_applies_mcp_tools_off_event_loop(
    tool_agent, monkeypatch
):
    alpha = _make_tool("alpha")
    client = AsyncMock()
    client.get_tools.return_value = [alpha]
    agent = tool_agent()
    event_loop_thread = threading.get_ident()
    apply_threads = []
    original_add_tool = agent.add_tool

    def recording_add_tool(tools):
        apply_threads.append(threading.get_ident())
        original_add_tool(tools)

    monkeypatch.setattr(agent, "add_tool", recording_add_tool)

    await agent.add_mcp_tools(client)

    assert apply_threads
    assert apply_threads[0] != event_loop_thread


async def test_agent_with_tools_returns_mcp_tool_provenance_without_mutation(
    tool_agent,
):
    alpha = _make_tool("alpha")
    beta = _make_tool("beta")
    client = AsyncMock()
    client.connections = {"laboratory": {}, "archive": {}}

    async def get_tools(*, server_name):
        return [alpha] if server_name == "laboratory" else [beta]

    client.get_tools.side_effect = get_tools
    agent = tool_agent()

    sources = await agent.add_mcp_tools(client)

    assert sources == {"alpha": "laboratory", "beta": "archive"}
    assert agent.tools["alpha"].metadata is None
    assert agent.tools["beta"].metadata is None


def test_tool_runtime_preserved_for_tool_node(chat_model, tmp_path: Path):
    class WriteToolAgent(AgentWithTools, BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(llm=chat_model, tools=[write_code], **kwargs)

        def _inject_tool_call(self, state, runtime: Runtime[AgentContext]):
            messages = list(state["messages"])
            assert isinstance(runtime.context, AgentContext)
            messages.append(
                tool_call_message(
                    "write_code",
                    {
                        "code": "print('runtime ok')",
                        "filename": "runtime_check.py",
                    },
                    "tool_call_write_code",
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
