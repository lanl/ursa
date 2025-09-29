from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from ursa.agents import RAGAgent


def test_rag_agent():
    rag_output = Path("workspace") / "rag-agent"
    summary_dir = rag_output / "summary"
    vectorstore_dir = rag_output / "db"
    summary_file = summary_dir / "RAG_summary.txt"

    agent = RAGAgent(
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        database_path="tests/tiny-corpus",
        summaries_path=str(summary_dir),
        vectorstore_path=str(vectorstore_dir),
    )
    agent.run(context=("What is AIBD?"))

    assert (summary_dir / "RAG_summary.txt").exists()
    assert vectorstore_dir.exists()
    assert (
        "attraction indian buffet distribution"
        in summary_file.read_text().lower()
    )
