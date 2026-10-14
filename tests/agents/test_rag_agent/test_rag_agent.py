from pathlib import Path

import pytest

from ursa.agents import RAGAgent


async def test_rag_agent_retrieves_contextual_documents(
    chat_model, embedding_model, monkeypatch, capsys, tmpdir
):
    events = []

    def capture_event(event_name, payload, config=None):
        events.append((event_name, payload))

    monkeypatch.setattr(
        "ursa.util.events.dispatch_custom_event",
        capture_event,
    )

    workspace = Path(tmpdir)
    database_dir = workspace / "database"
    summaries_dir = workspace / "summaries"
    vectors_dir = workspace / "vectors"

    for path in (database_dir, summaries_dir, vectors_dir):
        path.mkdir(parents=True, exist_ok=True)

    (database_dir / "mechanical_entanglement.pdf").write_bytes(b"%PDF-1.4\n")

    def fakePDFLoader(path_name):
        doc_text = (
            "Quantum entanglement between mechanical resonators enables "
            "ultra-sensitive force detection in cryogenic setups."
        )
        return doc_text

    monkeypatch.setattr(
        "ursa.agents.rag_agent.read_text_from_file",
        fakePDFLoader,
    )

    agent = RAGAgent(
        llm=chat_model,
        embedding=embedding_model,
        workspace=tmpdir,
        database_path="database",
        summaries_path="summaries",
        vectorstore_path="vectors",
        return_k=1,
        chunk_size=256,
        chunk_overlap=0,
    )

    query = "Explain quantum entanglement between mechanical resonators."
    result = await agent.ainvoke({"context": query, "query": query})

    assert "summary" in result
    assert isinstance(result["summary"], str)

    rag_metadata = result.get("rag_metadata")
    assert rag_metadata is not None
    assert rag_metadata["num_results"] > 0
    assert rag_metadata["k"] == agent.return_k
    assert rag_metadata["relevance_scores"]

    summary_file = summaries_dir / "RAG_summary.txt"
    assert summary_file.exists()
    assert summary_file.read_text() == result["summary"]

    manifest_path = vectors_dir / "_ingested_ids.txt"
    assert manifest_path.exists()

    payloads = [payload for _, payload in events]
    stages = [payload["stage"] for payload in payloads]
    assert "read_docs" in stages
    assert "ingest_docs" in stages
    assert "retrieve" in stages
    assert "summarize" in stages
    assert "retrieve_result" in stages
    assert all(payload["agent"] == "RAGAgent" for payload in payloads)

    stdout = capsys.readouterr().out
    assert "[RAG Agent]" not in stdout
    assert "RAG failed due to:" not in stdout


def test_rag_agent_requires_explicit_embedding(chat_model, tmpdir):
    with pytest.raises(
        ValueError, match="requires an explicit embedding model"
    ):
        RAGAgent(
            llm=chat_model,
            workspace=tmpdir,
            database_path="database",
            summaries_path="summaries",
            vectorstore_path="vectors",
        )


def test_maybe_tqdm_skips_progress_bar_for_empty_work():
    """No progress bar is constructed when there is nothing to process.

    Queries always traverse the read/ingest nodes (the RAG graph is a fixed
    linear chain), so without this guard every query would emit a useless "0/0"
    bar. Returning the iterable untouched also avoids constructing tqdm's global
    write lock in the common query case.
    """
    from ursa.agents import rag_agent as rag_agent_module

    calls = []

    def fake_tqdm(*args, **kwargs):  # pragma: no cover - must not be called
        calls.append(kwargs)
        raise AssertionError("tqdm must not be constructed for empty work")

    original = rag_agent_module.tqdm
    rag_agent_module.tqdm = fake_tqdm
    try:
        result = rag_agent_module._maybe_tqdm([], total=0, desc="nothing")
        assert list(result) == []
        assert calls == []
    finally:
        rag_agent_module.tqdm = original


def test_maybe_tqdm_wraps_progress_bar_when_work_exists():
    """A progress bar IS used (with desc/total forwarded) when work exists."""
    from ursa.agents import rag_agent as rag_agent_module

    captured = {}

    def fake_tqdm(iterable, **kwargs):
        captured.update(kwargs)
        return iterable

    original = rag_agent_module.tqdm
    rag_agent_module.tqdm = fake_tqdm
    try:
        items = [("a", "1"), ("b", "2")]
        result = rag_agent_module._maybe_tqdm(
            items, total=len(items), desc="RAG parsing text"
        )
        assert list(result) == items
        assert captured["total"] == 2
        assert captured["desc"] == "RAG parsing text"
    finally:
        rag_agent_module.tqdm = original


def test_maybe_tqdm_preserves_all_items_when_wrapping():
    """Gating must not drop or reorder work items (real tqdm, lazy zip)."""
    from ursa.agents.rag_agent import _maybe_tqdm

    texts = ["t0", "t1", "t2"]
    ids = ["i0", "i1", "i2"]
    wrapped = _maybe_tqdm(
        zip(texts, ids), total=len(texts), desc="RAG Ingesting", disable=True
    )
    assert list(wrapped) == [("t0", "i0"), ("t1", "i1"), ("t2", "i2")]
