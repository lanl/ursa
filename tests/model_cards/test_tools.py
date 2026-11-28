"""Model-card tools run inside a LangGraph ToolNode, the way agents call them."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ursa.agents.base import AgentContext
from ursa.model_cards.tools import model_card_tools

from . import cards


def run_tools(calls, workspace, catalog_dir):
    graph = StateGraph(MessagesState, context_schema=AgentContext)
    graph.add_node("tools", ToolNode(model_card_tools()))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{index}"}
            for index, (name, args) in enumerate(calls)
        ],
    )
    result = graph.compile().invoke(
        {"messages": [message]},
        config={"configurable": {"model_catalog_dir": str(catalog_dir)}},
        context=AgentContext(
            llm=None,
            workspace=workspace,
            agent_name=None,
            group="default",
            den=workspace,
        ),
    )
    return [m.content for m in result["messages"][1:]]


def test_model_card_tools_share_one_catalog(tmp_path):
    (tmp_path / "toy_mlip.yaml").write_text(
        cards.NATIVE_CARD_YAML, encoding="utf-8"
    )
    catalog = tmp_path / "catalog"

    (parsed,) = run_tools(
        [("parse_model_card", {"source": "toy_mlip.yaml"})], tmp_path, catalog
    )
    assert "ToyMLIP" in parsed and "eV/Å" in parsed
    assert list(catalog.glob("*.json"))

    found, described, ranked, checked, missing = run_tools(
        [
            ("search_model_cards", {"query": "interatomic potential"}),
            ("describe_model_card", {"model": "ToyMLIP"}),
            (
                "recommend_models",
                {
                    "request": "I need an interatomic potential for organic crystals."
                },
            ),
            (
                "check_model_inputs",
                {
                    "model": "ToyMLIP",
                    "arguments_json": '{"positions": [0, 0, 0]}',
                },
            ),
            ("describe_model_card", {"model": "NoSuchModel"}),
        ],
        tmp_path,
        catalog,
    )
    assert "ToyMLIP" in found
    assert "structure" in described and "ASE Atoms" in described
    assert "1. **ToyMLIP**" in ranked
    assert '"unknown": [\n    "positions"' in checked
    assert '"missing": [\n    "structure"' in checked
    assert "No catalog model matches" in missing


def test_parse_model_card_reports_unreadable_sources(tmp_path):
    (message,) = run_tools(
        [("parse_model_card", {"source": "missing_card.md"})],
        tmp_path,
        tmp_path / "catalog",
    )
    assert message.startswith("Could not read a model card")
