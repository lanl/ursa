from pathlib import Path

from ursa.agents import RAGAgent


async def test_rag_agent_retrieves_contextual_documents(
    chat_model, embedding_model, monkeypatch, tmpdir
):
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


async def test_rag_agent_extension_filtering(chat_model, embedding_model, tmpdir):
    workspace = Path(tmpdir)
    database_dir = workspace / "database"
    summaries_dir = workspace / "summaries"
    vectors_dir = workspace / "vectors"
    for path in (database_dir, summaries_dir, vectors_dir):
        path.mkdir(parents=True, exist_ok=True)

    (database_dir / "report.txt").write_text(
        "Critical mineral supply chain report content."
    )
    (database_dir / "scratch.py").write_text(
        "print('dev script that should be excluded')"
    )

    agent = RAGAgent(
        llm=chat_model,
        embedding=embedding_model,
        workspace=tmpdir,
        database_path="database",
        summaries_path="summaries",
        vectorstore_path="vectors",
        include_extensions={".txt"},
        max_docs_per_ingest=10,
        min_chars=5,
    )

    state = agent._read_docs_node({"context": "critical minerals"})
    doc_ids = state.get("doc_ids") or []
    assert any(doc_id.endswith("report.txt") for doc_id in doc_ids)
    assert not any(doc_id.endswith("scratch.py") for doc_id in doc_ids)
