from pathlib import Path

from ursa.agents.acquisition_agents import ArxivAgent


async def test_agent(chat_model, tmpdir):
    tmpdir_path = Path(tmpdir)
    agent = ArxivAgent(
        llm=chat_model,
        database_path=str(tmpdir_path / "papers"),
        summaries_path=str(tmpdir_path / "summaries"),
        vectorstore_path=str(tmpdir_path / "vectors"),
        workspace=tmpdir,
    )
    input = {
        "context": "What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
        "query": "Experimental Constraints on neutron star radius",
    }

    result = await agent.ainvoke(input)
    assert "final_summary" in result


HITS = [
    {
        "id": "a",
        "arxiv_id": "a",
        "title": "Cooking pasta",
        "abstract": "Boil water and add salt.",
    },
    {
        "id": "b",
        "arxiv_id": "b",
        "title": "Neutron star radius",
        "abstract": "Constraints on neutron star radius from NICER.",
    },
    {
        "id": "c",
        "arxiv_id": "c",
        "title": "Galaxy clusters",
        "abstract": "Dark matter in clusters.",
    },
    {"id": "e", "title": "Search error", "error": "boom"},
]
CTX = "constraints on neutron star radius"


def _agent(tmp_path, chat_model, **kw):
    return ArxivAgent(
        llm=chat_model,
        database_path=str(tmp_path / "p"),
        summaries_path=str(tmp_path / "s"),
        workspace=str(tmp_path),
        **kw,
    )


async def test_rerank_fast(tmp_path, chat_model):
    agent = _agent(tmp_path, chat_model, rerank="fast")
    out = await agent._arerank(list(HITS), CTX)
    assert out[0]["id"] == "b"
    assert out[-1]["id"] == "e"  # error hit goes last


async def test_rerank_embedding(tmp_path, chat_model):
    from langchain_core.embeddings import FakeEmbeddings

    agent = _agent(
        tmp_path,
        chat_model,
        rerank="embedding",
        rag_embedding=FakeEmbeddings(size=8),
    )
    out = await agent._arerank(list(HITS), CTX)
    assert {h["id"] for h in out} == {h["id"] for h in HITS}
    assert out[-1]["id"] == "e"


async def test_rerank_llm_and_fallback(tmp_path, chat_model):
    from langchain_core.messages import AIMessage

    class Stub:
        def __init__(self, text):
            self.text = text

        async def ainvoke(self, *a, **k):
            return AIMessage(content=self.text)

    agent = _agent(tmp_path, chat_model, rerank="llm")
    agent.llm = Stub("Ranking: [1, 0, 1, 9]")
    out = await agent._arerank(list(HITS), CTX)
    assert [h["id"] for h in out] == ["b", "a", "c", "e"]

    agent.llm = Stub("no idea")  # unparsable -> original order
    out = await agent._arerank(list(HITS), CTX)
    assert [h["id"] for h in out] == [h["id"] for h in HITS]


def test_rerank_validation(tmp_path, chat_model):
    import pytest

    with pytest.raises(ValueError):
        _agent(tmp_path, chat_model, rerank="bogus")
    with pytest.raises(ValueError):
        _agent(tmp_path, chat_model, rerank="embedding")
    assert _agent(tmp_path, chat_model).fetch_k == 3
    assert _agent(tmp_path, chat_model, rerank="fast").fetch_k == 10


async def test_rerank_embedding_separate_from_rag(tmp_path, chat_model):
    from langchain_core.embeddings import FakeEmbeddings

    agent = _agent(
        tmp_path,
        chat_model,
        rerank="embedding",
        rerank_embedding=FakeEmbeddings(size=8),
    )
    assert agent.rag_embedding is None  # summarization path unchanged
    out = await agent._arerank(list(HITS), CTX)
    assert out[-1]["id"] == "e"


def test_arxiv_search_query_requires_every_term():
    from ursa.agents.acquisition_agents import _arxiv_search_query as q

    assert q("neutron transport") == "all:neutron AND all:transport"
    assert (
        q('"data reduction" neutron!') == 'all:"data reduction" AND all:neutron'
    )
    assert q("   ") == "all:"
