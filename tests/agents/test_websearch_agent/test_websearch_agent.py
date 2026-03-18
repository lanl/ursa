from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, ToolMessage

from ursa.agents import WebSearchAgentLegacy


async def test_websearch_agent_legacy_websearch_flow(
    chat_model, monkeypatch, tmpdir
):
    """Ensure the legacy websearch agent wires the search tool into the graph."""
    query = "Who won the 2025 International Chopin Competition?"
    search_url = "https://example.com/chopin-2025"

    # Prevent real network access during the test.
    monkeypatch.setattr(
        WebSearchAgentLegacy,
        "_check_for_internet",
        lambda self, url="http://www.lanl.gov", timeout=2: True,
    )

    class FakeResponse:
        def __init__(self, content: bytes):
            self.content = content
            self.text = str(content)

    monkeypatch.setattr(
        "ursa.agents.websearch_agent.requests.get",
        lambda url, timeout=2: FakeResponse(
            b"<html><body><p>Mock content for testing.</p></body></html>"
        ),
    )

    monkeypatch.setattr(
        "langchain_community.utilities.duckduckgo_search.DuckDuckGoSearchAPIWrapper.results",
        lambda self, q, max_results, source=None: [
            {
                "title": "Mock Chopin Coverage",
                "href": search_url,
                "body": "Summary of the winner and teachers.",
            }
        ],
    )

    class FakeReactAgent:
        def __init__(self):
            self.invocations = 0

        def invoke(self, state):
            self.invocations += 1
            tool_call_id = "tool_call_1"
            tool_request = AIMessage(
                content="Searching for official announcement.",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "process_content",
                        "args": {
                            "url": search_url,
                            "context": "competition winner details",
                        },
                        "type": "tool_call",
                    }
                ],
            )
            tool_result = ToolMessage(
                content="Winner: Jane Doe. Teachers: John Smith and Alice Brown.",
                tool_call_id=tool_call_id,
            )
            researcher_summary = AIMessage(
                content="Collected the winner and teacher information from the announcement."
            )
            return {
                "messages": [tool_request, tool_result, researcher_summary],
                "urls_visited": [search_url],
            }

    fake_react_agent = FakeReactAgent()
    monkeypatch.setattr(
        "ursa.agents.websearch_agent.create_agent",
        lambda *args, **kwargs: fake_react_agent,
    )

    agent = WebSearchAgentLegacy(llm=chat_model, workspace=tmpdir)
    inputs = {
        "messages": [HumanMessage(content=query)],
        "model": chat_model,
        "websearch_query": query,
        "urls_visited": [],
        "max_websearch_steps": 0,
    }

    # Run once via ainvoke to satisfy the async API contract.
    await agent.ainvoke(inputs)
    assert fake_react_agent.invocations >= 1

    # Collect a second run with astream so we can inspect intermediate node outputs.
    create_react_state = None
    response_state = None
    async for step in agent.compiled_graph.astream(inputs):
        create_react_state = step.get("_create_react") or create_react_state
        response_state = step.get("_response_node") or response_state

    assert create_react_state is not None
    assert search_url in create_react_state["urls_visited"]

    tool_messages = create_react_state["messages"]
    assert any(
        getattr(msg, "tool_calls", [])
        and msg.tool_calls[0]["args"].get("url") == search_url
        for msg in tool_messages
    )

    assert response_state is not None
    final_messages = response_state["messages"]
    assert final_messages and isinstance(final_messages[0], str)
