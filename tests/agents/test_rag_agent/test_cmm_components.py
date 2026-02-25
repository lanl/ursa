from langchain_core.documents import Document

from ursa.agents.cmm_chunker import CMMChunker
from ursa.agents.cmm_embeddings import parse_embedding_model_spec
from ursa.agents.cmm_query_classifier import CMMQueryClassifier, QueryProfile
from ursa.agents.cmm_taxonomy import (
    detect_commodity_tags,
    detect_subdomain_tags,
    extract_temporal_indicators,
)
from ursa.agents.cmm_vectorstore import (
    ChromaBM25VectorStore,
    CMMVectorStoreBase,
)
from ursa.agents.rag_agent import RAGAgent


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


class _StubVectorStore(CMMVectorStoreBase):
    def __init__(self):
        self.calls = []

    def add_documents(self, documents: list[Document]) -> None:
        del documents

    def hybrid_search(self, query: str, k: int, alpha: float, filters: dict | None):
        del query, k, alpha
        self.calls.append(filters)
        if filters:
            return []
        return [
            (Document(page_content="fallback-hit", metadata={"chunk_id": "doc-1"}), 0.9)
        ]

    def delete_collection(self) -> None:
        return None

    def count(self) -> int:
        return 0


class _StubClassifier:
    def classify(self, query: str) -> QueryProfile:
        del query
        return QueryProfile(
            query_type="general",
            commodity_hints=["LREE"],
            subdomain_hints=[],
            temporal_hints=[],
            retrieval_k=20,
            return_k=5,
            alpha=0.7,
            filters={"commodity_tags": ["LREE"]},
        )


def test_rag_retrieval_falls_back_when_filters_return_empty():
    agent = RAGAgent.__new__(RAGAgent)
    agent.legacy_mode = False
    agent.vectorstore = _StubVectorStore()
    agent.classifier = _StubClassifier()
    agent._adaptive_retrieval_k = True
    agent._adaptive_return_k = True
    agent.retrieval_k = 20
    agent.return_k = 5
    agent.hybrid_alpha = 0.7
    agent.use_reranker = False
    agent.vectorstore_backend = "chroma"

    results, params = RAGAgent._retrieve(agent, "lanthanum yttrium ndfeb")

    assert len(results) == 1
    assert params["filter_fallback_used"] is True
    assert agent.vectorstore.calls[0] == {"commodity_tags": ["LREE"]}
    assert agent.vectorstore.calls[1] is None
