from collections.abc import Sequence

import pytest

from ursa.agents.hypothesizer_agent import HypothesizerAgent


class DummySearchTool:
    """Minimal stand-in for DuckDuckGo search that records queries."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def text(
        self, query: str, backend: str = "duckduckgo"
    ) -> list[dict[str, str]]:
        self.queries.append((query, backend))
        idx = len(self.queries)
        return [
            {
                "link": f"https://example.com/result-{idx}",
                "title": f"Result {idx}",
                "snippet": f"Snippet for query {idx}",
            }
        ]


@pytest.mark.asyncio
async def test_hypothesizer_agent_ainvoke(
    chat_model,
    monkeypatch: pytest.MonkeyPatch,
    tmpdir,
) -> None:
    dummy_search = DummySearchTool()
    monkeypatch.setattr(
        "ursa.agents.hypothesizer_agent.DDGS",
        lambda: dummy_search,
    )
    monkeypatch.chdir(tmpdir)

    agent = HypothesizerAgent(llm=chat_model, workspace=tmpdir)
    initial_state = {
        "question": "How can we reduce the cooling energy usage in edge data centers?",
        "current_iteration": 0,
        "max_iterations": 1,
        "agent1_solution": [],
        "agent2_critiques": [],
        "agent3_perspectives": [],
        "solution": "",
        "summary_report": "",
        "visited_sites": set(),
    }

    result = await agent.ainvoke(initial_state)

    assert isinstance(result["agent1_solution"], Sequence)
    assert isinstance(result["agent2_critiques"], Sequence)
    assert isinstance(result["agent3_perspectives"], Sequence)
    assert len(result["agent1_solution"]) >= 1
    assert len(result["agent2_critiques"]) >= 1
    assert len(result["agent3_perspectives"]) >= 1
    assert isinstance(result["solution"], str)
    assert isinstance(result["summary_report"], str)
    if result["summary_report"].strip():
        assert "\\documentclass" in result["summary_report"]
    assert result["current_iteration"] == 1
    assert len(dummy_search.queries) == 3
    assert all(backend == "duckduckgo" for _, backend in dummy_search.queries)
    assert result["visited_sites"] == {
        "https://example.com/result-1",
        "https://example.com/result-2",
        "https://example.com/result-3",
    }
    assert isinstance(result["question_search_query"], str)

    generated_logs = list(agent.workspace.glob("iteration_details_*.txt"))
    assert generated_logs, "Expected iteration history files to be written"
