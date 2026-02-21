from langchain_core.documents import Document

from ursa.agents.cmm_chunker import CMMChunker
from ursa.agents.cmm_embeddings import parse_embedding_model_spec
from ursa.agents.cmm_query_classifier import CMMQueryClassifier
from ursa.agents.cmm_taxonomy import (
    detect_commodity_tags,
    detect_subdomain_tags,
    extract_temporal_indicators,
)
from ursa.agents.cmm_vectorstore import ChromaBM25VectorStore


def test_parse_embedding_model_spec_variants():
    assert parse_embedding_model_spec("openai:text-embedding-3-large:1024") == (
        "openai",
        "text-embedding-3-large",
        1024,
    )
    assert parse_embedding_model_spec("openai:text-embedding-3-small") == (
        "openai",
        "text-embedding-3-small",
        None,
    )
    assert parse_embedding_model_spec("local:BAAI/bge-large-en-v1.5") == (
        "local",
        "BAAI/bge-large-en-v1.5",
        None,
    )


def test_chunker_preserves_markdown_tables_and_metadata():
    text = """
# Lithium Supply
Q4 2024 update for lithium carbonate market.

| region | demand |
| --- | --- |
| NA | 120 |
| EU | 95 |
"""
    chunker = CMMChunker(max_tokens=120, overlap_tokens=20, min_tokens=5)
    docs = chunker.chunk_document(
        text,
        metadata={"source_doc_id": "doc-1", "source_doc_title": "demo"},
    )

    assert docs
    assert any(doc.metadata["chunk_type"] == "table" for doc in docs)
    assert any("LI" in doc.metadata["commodity_tags"] for doc in docs)
    assert any(doc.metadata["temporal_indicator"] for doc in docs)


def test_taxonomy_taggers_detect_domain_hints():
    query = "Compare lithium and cobalt policy impacts on 2025 trade flows."
    commodities = detect_commodity_tags(query)
    subdomains = detect_subdomain_tags(query)
    temporal = extract_temporal_indicators(query)

    assert "LI" in commodities
    assert "CO" in commodities
    assert "G-PR" in subdomains
    assert "Q-TF" in subdomains
    assert "2025" in temporal


def test_query_classifier_adapts_retrieval():
    classifier = CMMQueryClassifier()
    profile = classifier.classify(
        "Compare lithium and cobalt supply shocks in 2025."
    )

    assert profile.query_type in {"comparative", "multi_hop"}
    assert profile.retrieval_k >= 20
    assert profile.return_k >= 5
    assert "commodity_tags" in profile.filters


def test_hybrid_rrf_fusion_prefers_cross_signal_docs():
    store = ChromaBM25VectorStore.__new__(ChromaBM25VectorStore)

    doc_a = Document(page_content="A", metadata={"chunk_id": "A"})
    doc_b = Document(page_content="B", metadata={"chunk_id": "B"})
    doc_c = Document(page_content="C", metadata={"chunk_id": "C"})

    store._dense_search = lambda query, k: [(doc_a, 0.9), (doc_b, 0.8)]
    store._bm25_search = lambda query, k: [(doc_b, 2.0), (doc_c, 1.0)]

    results = ChromaBM25VectorStore.hybrid_search(
        store,
        query="demo",
        k=3,
        alpha=0.7,
        filters=None,
    )

    ordered_ids = [doc.metadata["chunk_id"] for doc, _ in results]
    assert ordered_ids[0] == "B"
    assert set(ordered_ids) == {"A", "B", "C"}
