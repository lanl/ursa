from collections.abc import Sequence

import pytest
from ddgs.exceptions import DDGSException

from ursa.agents.hypothesizer_agent import HypothesizerAgent


class DummySearchTool:
    """Minimal stand-in for DuckDuckGo search that records queries."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def text(
        self,
        query: str,
        backend: str = "duckduckgo",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        self.queries.append((query, backend))
        idx = len(self.queries)
        return [
            {
                "href": f"https://example.com/result-{idx}",
                "title": f"Result {idx}",
                "snippet": f"Snippet for query {idx}",
            }
        ]


class FallbackSearchTool:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []

    def text(
        self,
        query: str,
        backend: str = "duckduckgo",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        self.queries.append((query, backend))
        if "evidence" in query or "alternatives" in query:
            raise DDGSException("No results found.")
        return [
            {
                "href": f"https://example.com/{len(self.queries)}",
                "title": "Recovered result",
                "snippet": f"Recovered via {query}",
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


@pytest.mark.asyncio
async def test_hypothesizer_search_falls_back_when_ddgs_returns_no_results(
    chat_model,
    monkeypatch: pytest.MonkeyPatch,
    tmpdir,
) -> None:
    dummy_search = FallbackSearchTool()
    monkeypatch.setattr(
        "ursa.agents.hypothesizer_agent.DDGS",
        lambda: dummy_search,
    )

    agent = HypothesizerAgent(llm=chat_model, workspace=tmpdir)
    initial_state = {
        "question": "How can we improve jet design verification?",
        "question_search_query": "jet design verification",
        "current_iteration": 0,
        "max_iterations": 1,
        "agent1_solution": ["Initial answer"],
        "agent2_critiques": [],
        "agent3_perspectives": [],
        "solution": "",
        "summary_report": "",
        "visited_sites": set(),
    }

    critique = await agent.agent2_critique(initial_state)

    assert critique["agent2_critiques"]
    assert critique["visited_sites"]
    assert (
        "jet design verification evidence",
        "duckduckgo,google",
    ) in dummy_search.queries
    assert (
        "jet design verification performance",
        "duckduckgo,google",
    ) in dummy_search.queries
