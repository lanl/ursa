from __future__ import annotations

import os
import re
import statistics
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, TypedDict

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain.embeddings import init_embeddings as init_lc_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from ursa.agents.base import BaseAgent
from ursa.agents.cmm_chunker import CMMChunker
from ursa.agents.cmm_embeddings import (
    CMMEmbeddingsBase,
    LangChainEmbeddingsAdapter,
)
from ursa.agents.cmm_embeddings import (
    init_embeddings as init_cmm_embeddings,
)
from ursa.agents.cmm_query_classifier import CMMQueryClassifier
from ursa.agents.cmm_reranker import CMMRerankerBase, init_reranker
from ursa.agents.cmm_vectorstore import CMMVectorStoreBase, init_vectorstore
from ursa.util.parse import (
    OFFICE_EXTENSIONS,
    SPECIAL_TEXT_FILENAMES,
    TEXT_EXTENSIONS,
    read_text_from_file,
)

MIN_CHARS = 30


class RAGMetadata(TypedDict):
    k: int
    num_results: int
    relevance_scores: list[float]
    query_type: str
    retrieval_k: int
    backend: str


class RAGState(TypedDict, total=False):
    context: str
    doc_texts: list[str]
    doc_ids: list[str]
    summary: str
    rag_metadata: RAGMetadata


def remove_surrogates(text: str) -> str:
    return re.sub(r"[\ud800-\udfff]", "", text)


class RAGAgent(BaseAgent[RAGState]):
    state_type = RAGState

    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings | str | None = None,
        embedding_dimensions: int | None = None,
        retrieval_k: int = 20,
        return_k: int = 5,
        vectorstore_backend: str | None = None,
        hybrid_alpha: float | None = None,
        use_reranker: bool = False,
        reranker_provider: str = "none",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        database_path: str | Path = "database",
        summaries_path: str | Path = "database",
        vectorstore_path: str | Path = "vectorstore",
        include_extensions: set[str] | None = None,
        exclude_extensions: set[str] | None = None,
        max_docs_per_ingest: int | None = None,
        min_chars: int = MIN_CHARS,
        **kwargs: Any,
    ):
        super().__init__(llm, **kwargs)
        self._vs_lock = Lock()

        self.retrieval_k = max(1, int(retrieval_k))
        self.return_k = max(1, int(return_k))
        self._adaptive_retrieval_k = self.retrieval_k == 20
        self._adaptive_return_k = self.return_k == 5
        self.hybrid_alpha = float(hybrid_alpha or os.getenv("CMM_HYBRID_ALPHA", "0.7"))
        self.use_reranker = bool(use_reranker)
        self.reranker_provider = reranker_provider
        self.vectorstore_backend = (
            vectorstore_backend
            or os.getenv("CMM_VECTORSTORE_BACKEND", "chroma")
        ).lower()
        self.legacy_mode = (
            os.getenv("URSA_RAG_LEGACY_MODE", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_path = self._resolve_path(database_path)
        self.summaries_path = self._resolve_path(summaries_path)
        self.vectorstore_path = self._resolve_path(vectorstore_path)
        self.include_extensions = (
            self._normalize_extensions(include_extensions)
            if include_extensions is not None
            else None
        )
        self.exclude_extensions = self._normalize_extensions(exclude_extensions)
        self.max_docs_per_ingest = max_docs_per_ingest
        self.min_chars = max(1, int(min_chars))

        self.embedding = self._init_embedding(embedding, embedding_dimensions)
        self.chunker = CMMChunker(
            max_tokens=max(64, chunk_size // 2),
            overlap_tokens=max(0, chunk_overlap // 4),
            min_tokens=max(20, min(chunk_size // 6, 120)),
        )
        self.classifier = CMMQueryClassifier()

        provider = reranker_provider or os.getenv(
            "CMM_RERANKER_PROVIDER", "none"
        )
        self.reranker: CMMRerankerBase = (
            init_reranker(provider) if self.use_reranker else init_reranker("none")
        )

        self.vectorstore_path.mkdir(exist_ok=True, parents=True)
        self._ingested_manifest = self._load_manifest_ids()

        if self.legacy_mode:
            self.vectorstore = self._open_legacy_vectorstore()
        else:
            self.vectorstore = init_vectorstore(
                backend=self.vectorstore_backend,
                persist_directory=self.vectorstore_path,
                embedding_model=self.embedding,
                collection_name=os.getenv(
                    "CMM_VECTORSTORE_COLLECTION", "cmm_chunks"
                ),
            )

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.vectorstore_path, "_ingested_ids.txt")

    @property
    def manifest_exists(self) -> bool:
        return os.path.exists(self.manifest_path)

    def _resolve_path(self, value: str | Path) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return self.workspace / p

    def _normalize_extensions(self, values: Iterable[str] | None) -> set[str]:
        if not values:
            return set()
        normalized = set()
        for value in values:
            ext = self._normalize_extension(value)
            if ext:
                normalized.add(ext)
        return normalized

    def _normalize_extension(self, value: str) -> str:
        v = value.strip().lower()
        if not v:
            return ""
        return v if v.startswith(".") else f".{v}"

    def _init_embedding(
        self,
        embedding: Embeddings | str | None,
        embedding_dimensions: int | None,
    ) -> CMMEmbeddingsBase:
        if isinstance(embedding, str):
            return init_cmm_embeddings(
                embedding,
                dimensions=embedding_dimensions,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        if isinstance(embedding, Embeddings):
            return LangChainEmbeddingsAdapter(
                embedding,
                embedding_dim=embedding_dimensions,
            )

        model = os.getenv("CMM_EMBEDDING_MODEL", "openai:text-embedding-3-large")
        if self.legacy_mode:
            legacy = init_lc_embeddings("openai:text-embedding-3-small")
            return LangChainEmbeddingsAdapter(legacy, embedding_dim=1536)

        return init_cmm_embeddings(
            model,
            dimensions=embedding_dimensions
            or _safe_int(os.getenv("CMM_EMBEDDING_DIMENSIONS")),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def _open_legacy_vectorstore(self) -> Chroma:
        return Chroma(
            persist_directory=str(self.vectorstore_path),
            embedding_function=self.embedding,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _load_manifest_ids(self) -> set[str]:
        if not self.manifest_exists:
            return set()
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def _paper_exists_in_vectorstore(self, doc_id: str) -> bool:
        return doc_id in self._ingested_manifest

    def _mark_paper_ingested(self, doc_id: str) -> None:
        if doc_id in self._ingested_manifest:
            return
        self._ingested_manifest.add(doc_id)
        with open(self.manifest_path, "a", encoding="utf-8") as handle:
            handle.write(f"{doc_id}\n")

    def _read_docs_node(self, state: RAGState) -> RAGState:
        print("[RAG Agent] Reading Documents....")
        new_state = state.copy()

        custom_extensions = {
            self._normalize_extension(item)
            for item in os.environ.get("URSA_TEXT_EXTENSIONS", "").split(",")
            if item.strip()
        }
        custom_readable_files = {
            item.strip().lower()
            for item in os.environ.get(
                "URSA_SPECIAL_TEXT_FILENAMES", ""
            ).split(",")
            if item.strip()
        }

        base_dir = Path(self.database_path)
        ingestible_paths: list[Path] = []

        for path in base_dir.rglob("*"):
            if not path.is_file():
                continue

            ext = path.suffix.lower()
            file_name = path.name.lower()

            base_ingestible = (
                ext == ".pdf"
                or ext in TEXT_EXTENSIONS
                or ext in custom_extensions
                or file_name in SPECIAL_TEXT_FILENAMES
                or file_name in custom_readable_files
                or ext in OFFICE_EXTENSIONS
            )
            if not base_ingestible:
                continue

            if ext in self.exclude_extensions:
                continue

            if self.include_extensions is not None:
                if (
                    ext not in self.include_extensions
                    and file_name not in SPECIAL_TEXT_FILENAMES
                    and file_name not in custom_readable_files
                ):
                    continue

            ingestible_paths.append(path)

        candidates: list[tuple[Path, str]] = []
        for path in ingestible_paths:
            doc_id = str(path)
            if not self._paper_exists_in_vectorstore(doc_id):
                candidates.append((path, doc_id))

        if self.max_docs_per_ingest is not None and self.max_docs_per_ingest > 0:
            candidates = candidates[: self.max_docs_per_ingest]

        papers: list[str] = []
        doc_ids: list[str] = []
        for path, doc_id in tqdm(candidates, desc="RAG parsing text"):
            full_text = read_text_from_file(path)
            if len(full_text) < self.min_chars:
                continue
            papers.append(full_text)
            doc_ids.append(doc_id)

        new_state["doc_texts"] = papers
        new_state["doc_ids"] = doc_ids
        return new_state

    def _build_docs_for_ingest(self, paper: str, doc_id: str) -> list[Document]:
        cleaned_text = remove_surrogates(paper)

        if self.legacy_mode:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            docs = splitter.create_documents(
                [cleaned_text], metadatas=[{"id": doc_id}]
            )
            for i, doc in enumerate(docs):
                doc.metadata.setdefault("source_doc_id", doc_id)
                doc.metadata.setdefault("chunk_index", i)
            return docs

        title = Path(doc_id).name
        docs = self.chunker.chunk_document(
            cleaned_text,
            metadata={"source_doc_id": doc_id, "source_doc_title": title},
        )
        for i, doc in enumerate(docs):
            if "chunk_id" not in doc.metadata:
                chunk_id = f"{doc_id}::{doc.metadata.get('chunk_index', i)}"
                doc.metadata["chunk_id"] = chunk_id
            doc.metadata.setdefault("id", doc_id)
        return docs

    def _ingest_docs_node(self, state: RAGState) -> RAGState:
        if "doc_texts" not in state:
            raise RuntimeError("Unexpected error: doc_texts not in state!")
        if "doc_ids" not in state:
            raise RuntimeError("Unexpected error: doc_ids not in state!")

        batch_docs: list[Document] = []
        ingest_ids: list[str] = []
        for paper, doc_id in tqdm(
            zip(state["doc_texts"], state["doc_ids"]),
            total=len(state["doc_texts"]),
            desc="RAG Ingesting",
        ):
            docs = self._build_docs_for_ingest(paper, doc_id)
            if docs:
                batch_docs.extend(docs)
                ingest_ids.append(doc_id)

        if batch_docs:
            print("[RAG Agent] Ingesting Documents Into RAG Database....")
            with self._vs_lock:
                if self.legacy_mode:
                    assert isinstance(self.vectorstore, Chroma)
                    ids = [
                        str(doc.metadata.get("chunk_id") or f"chunk::{i}")
                        for i, doc in enumerate(batch_docs)
                    ]
                    self.vectorstore.add_documents(batch_docs, ids=ids)
                else:
                    assert isinstance(self.vectorstore, CMMVectorStoreBase)
                    self.vectorstore.add_documents(batch_docs)

                for doc_id in ingest_ids:
                    self._mark_paper_ingested(doc_id)

        return state

    def _retrieve(self, query: str) -> tuple[list[tuple[Document, float]], dict[str, Any]]:
        if self.legacy_mode:
            assert isinstance(self.vectorstore, Chroma)
            retrieved = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.return_k,
            )
            params = {
                "query_type": "legacy",
                "retrieval_k": self.return_k,
                "return_k": self.return_k,
                "alpha": self.hybrid_alpha,
                "backend": "legacy-chroma",
            }
            return retrieved, params

        assert isinstance(self.vectorstore, CMMVectorStoreBase)
        profile = self.classifier.classify(query)
        retrieval_k = max(1, int(profile.retrieval_k))
        return_k = max(1, int(profile.return_k))
        alpha = float(profile.alpha)
        effective_retrieval_k = (
            retrieval_k if self._adaptive_retrieval_k else self.retrieval_k
        )
        effective_return_k = (
            return_k if self._adaptive_return_k else self.return_k
        )

        dense_sparse = self.vectorstore.hybrid_search(
            query=query,
            k=effective_retrieval_k,
            alpha=self.hybrid_alpha if self.hybrid_alpha is not None else alpha,
            filters=profile.filters,
        )
        top_docs = dense_sparse[:effective_return_k]

        if self.use_reranker:
            top_docs = self.reranker.rerank(
                query=query,
                documents=top_docs,
                top_k=effective_return_k,
            )

        params = {
            "query_type": profile.query_type,
            "retrieval_k": effective_retrieval_k,
            "return_k": effective_return_k,
            "alpha": self.hybrid_alpha if self.hybrid_alpha is not None else alpha,
            "backend": self.vectorstore_backend,
        }
        return top_docs, params

    def _retrieve_and_summarize_node(self, state: RAGState) -> RAGState:
        print("[RAG Agent] Retrieving Contextually Relevant Information...")
        if "context" not in state:
            raise RuntimeError("Unexpected error: context not in state!")

        prompt = ChatPromptTemplate.from_template(
            """
You are a scientific assistant responsible for summarizing extracts from
research papers in the context of: {context}

Summarize the retrieved scientific content below.
Cite source IDs when relevant: {source_ids}

{retrieved_content}
"""
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            results, params = self._retrieve(state["context"])
            relevance_scores = [float(score) for _, score in results]
        except Exception as exc:
            print(f"RAG failed due to: {exc}")
            return {**state, "summary": ""}

        source_ids_list: list[str] = []
        for doc, _ in results:
            source_id = (
                doc.metadata.get("source_doc_id")
                or doc.metadata.get("id")
                or doc.metadata.get("source")
            )
            if source_id and source_id not in source_ids_list:
                source_ids_list.append(str(source_id))
        source_ids = ", ".join(source_ids_list)

        retrieved_content = (
            "\n\n".join(doc.page_content for doc, _ in results)
            if results
            else ""
        )

        print("[RAG Agent] Summarizing Retrieved Information...")
        rag_summary = chain.invoke(
            {
                "retrieved_content": retrieved_content,
                "context": state["context"],
                "source_ids": source_ids,
            }
        )

        os.makedirs(self.summaries_path, exist_ok=True)
        with open(
            os.path.join(self.summaries_path, "RAG_summary.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(rag_summary)

        if relevance_scores:
            print(f"\nMax Relevance Score: {max(relevance_scores):.4f}")
            print(f"Min Relevance Score: {min(relevance_scores):.4f}")
            median = statistics.median(relevance_scores)
            print(f"Median Relevance Score: {median:.4f}\n")
        else:
            print("\nNo RAG results retrieved (score list empty).\n")

        return {
            **state,
            "summary": rag_summary,
            "rag_metadata": {
                "k": params["return_k"],
                "num_results": len(results),
                "relevance_scores": relevance_scores,
                "query_type": params["query_type"],
                "retrieval_k": params["retrieval_k"],
                "backend": params["backend"],
            },
        }

    def _build_graph(self) -> None:
        self.add_node(self._read_docs_node)
        self.add_node(self._ingest_docs_node)
        self.add_node(self._retrieve_and_summarize_node)

        self.graph.add_edge("_read_docs_node", "_ingest_docs_node")
        self.graph.add_edge(
            "_ingest_docs_node", "_retrieve_and_summarize_node"
        )

        self.graph.set_entry_point("_read_docs_node")
        self.graph.set_finish_point("_retrieve_and_summarize_node")


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
