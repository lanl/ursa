from pathlib import Path

import pytest

from ursa.agents.arxiv_agent import ArxivAgentLegacy


@pytest.mark.asyncio
async def test_arxiv_agent_legacy_fetches_local_papers_without_network(
    chat_model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "ursa.agents.arxiv_agent.requests.get",
        lambda *args, **kwargs: pytest.fail("requests.get should not be called"),
    )

    agent = ArxivAgentLegacy(
        llm=chat_model,
        summarize=False,
        process_images=False,
        download_papers=False,
        workspace=tmp_path,
        database_path="papers",
        summaries_path="summaries",
        vectorstore_path="vectors",
    )

    local_pdf = agent.database_path / "2401.01234.pdf"
    local_pdf.parent.mkdir(parents=True, exist_ok=True)
    local_pdf.write_bytes(b"")

    query = "quantum error correction codes"
    context = "Identify recent progress in near-term experiments."
    result = await agent.ainvoke({"query": query, "context": context})

    assert result["query"] == query
    assert result["context"] == context
    assert isinstance(result["papers"], list)
    assert any(
        paper["arxiv_id"] == "2401.01234" and paper["full_text"] == ""
        for paper in result["papers"]
    )
