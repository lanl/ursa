from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from ursa.agents import RAGAgent
from ursa.observability.timing import render_session_summary


def test_rag_agent():
    rag_output = Path("workspace") / "rag-agent"
    summary_dir = rag_output / "summary"
    vectorstore_dir = rag_output / "db"
    summary_file = summary_dir / "RAG_summary.txt"

    agent = RAGAgent(
        llm=init_chat_model(model="openai:gpt-5-nano"),
        embedding=init_embeddings(model="ollama:nomic-embed-text"),
        database_path="tests/tiny-corpus",
        summaries_path=str(summary_dir),
        vectorstore_path=str(vectorstore_dir),
    )
    agent.invoke(context="What is AIBD?")
    render_session_summary(agent.thread_id)

    assert (summary_dir / "RAG_summary.txt").exists()
    assert vectorstore_dir.exists()
    assert (
        "attraction indian buffet distribution"
        in summary_file.read_text().lower()
    )
