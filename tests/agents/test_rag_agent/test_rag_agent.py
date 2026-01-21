from pathlib import Path

from langchain_core.documents import Document

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

    doc_text = (
        "Quantum entanglement between mechanical resonators enables "
        "ultra-sensitive force detection in cryogenic setups."
    )

    class _FakePDFLoader:
        def __init__(self, file_path: str):
            self.file_path = file_path

        def load(self):
            return [Document(page_content=doc_text)]

    monkeypatch.setattr(
        "ursa.agents.rag_agent.PyPDFLoader",
        _FakePDFLoader,
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
