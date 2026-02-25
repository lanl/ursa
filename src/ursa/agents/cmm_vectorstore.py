from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ursa.agents.cmm_embeddings import CMMEmbeddingsBase

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore[assignment]


class CMMVectorStoreBase(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        k: int,
        alpha: float,
        filters: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


_LIST_METADATA_FIELDS = {"commodity_tags", "subdomain_tags"}


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = "|".join(str(item) for item in value)
        else:
            cleaned[key] = str(value)
    return cleaned


def _restore_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    restored = dict(metadata)
    for key in _LIST_METADATA_FIELDS:
        value = restored.get(key)
        if isinstance(value, str):
            if not value.strip():
                restored[key] = []
            elif "|" in value:
                restored[key] = [item for item in value.split("|") if item]
            else:
                restored[key] = [value]
    return restored


def _match_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True

    def intersects(meta_val: Any, expected: list[str]) -> bool:
        if meta_val is None:
            return False
        if isinstance(meta_val, list):
            return bool(set(map(str, meta_val)).intersection(set(expected)))
        return str(meta_val) in expected

    for field in ("commodity_tags", "subdomain_tags", "sensitivity_level"):
        expected = filters.get(field)
        if expected and not intersects(metadata.get(field), expected):
            return False

    gte = filters.get("temporal_indicator_gte")
    lte = filters.get("temporal_indicator_lte")
    if gte or lte:
        temporal = str(metadata.get("temporal_indicator", ""))
        year = temporal[:4] if "-Q" in temporal else temporal
        if gte and (not year or year < str(gte)):
            return False
        if lte and (not year or year > str(lte)):
            return False

    return True


class ChromaBM25VectorStore(CMMVectorStoreBase):
    def __init__(
        self,
        *,
        persist_directory: str | Path,
        embedding_model: CMMEmbeddingsBase,
        collection_name: str = "cmm_chunks",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._chroma = Chroma(
            collection_name=collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._docs_by_id: dict[str, Document] = {}
        self._bm25 = None
        self._tokenized_docs: list[list[str]] = []
        self._bm25_ids: list[str] = []
        self._rebuild_bm25_index()

    def _collection_ids(self) -> list[str]:
        res = self._chroma._collection.get(include=[])
        ids = res.get("ids", [])
        if ids and isinstance(ids[0], list):
            return [i for sub in ids for i in sub]
        return list(ids)

    def _rebuild_bm25_index(self) -> None:
        self._docs_by_id = {}
        self._tokenized_docs = []
        self._bm25_ids = []

        res = self._chroma._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        if ids and isinstance(ids[0], list):
            ids = [i for sub in ids for i in sub]
        for cid, text, meta in zip(ids, docs, metas):
            metadata = _restore_metadata(dict(meta or {}))
            metadata.setdefault("chunk_id", cid)
            doc = Document(page_content=text or "", metadata=metadata)
            self._docs_by_id[cid] = doc
            self._bm25_ids.append(cid)
            self._tokenized_docs.append(_tokenize(doc.page_content))

        if BM25Okapi is not None and self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)
        else:
            self._bm25 = None

    def add_documents(self, documents: list[Document]) -> None:
        chroma_docs: list[Document] = []
        ids = []
        for i, doc in enumerate(documents):
            chunk_id = (
                doc.metadata.get("chunk_id")
                or f"{doc.metadata.get('source_doc_id', 'doc')}::"
                f"{doc.metadata.get('chunk_index', i)}::{i}"
            )
            metadata = _sanitize_metadata(dict(doc.metadata))
            metadata["chunk_id"] = str(chunk_id)
            chroma_docs.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )
            ids.append(str(chunk_id))
        try:
            self._chroma.add_documents(chroma_docs, ids=ids)
        except Exception as exc:
            print(
                "[ChromaBM25VectorStore] Batch insert failed; falling back to"
                f" per-document insert. error={exc}"
            )
            skipped = 0
            for doc, chunk_id in zip(chroma_docs, ids):
                try:
                    self._chroma.add_documents([doc], ids=[chunk_id])
                except Exception:
                    skipped += 1
            if skipped:
                print(
                    "[ChromaBM25VectorStore] Skipped"
                    f" {skipped} chunk(s) during fallback insert."
                )
        self._rebuild_bm25_index()

    def _dense_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        dense = self._chroma.similarity_search_with_relevance_scores(query, k=k)
        out: list[tuple[Document, float]] = []
        for i, (doc, score) in enumerate(dense):
            metadata = _restore_metadata(dict(doc.metadata or {}))
            metadata.setdefault(
                "chunk_id",
                f"{metadata.get('source_doc_id', 'dense')}::{metadata.get('chunk_index', i)}::{i}",
            )
            out.append(
                (
                    Document(page_content=doc.page_content, metadata=metadata),
                    float(score),
                )
            )
        return out

    def _bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self._bm25_ids:
            return []

        query_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = list(self._bm25.get_scores(query_tokens))
        else:
            qset = set(query_tokens)
            scores = []
            for toks in self._tokenized_docs:
                scores.append(float(len(qset.intersection(set(toks)))))

        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )[:k]
        results: list[tuple[Document, float]] = []
        for idx, score in ranked:
            cid = self._bm25_ids[idx]
            doc = self._docs_by_id[cid]
            results.append((doc, float(score)))
        return results

    def hybrid_search(
        self,
        query: str,
        k: int,
        alpha: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        retrieval_k = max(k, 20)
        dense = self._dense_search(query, retrieval_k)
        sparse = self._bm25_search(query, retrieval_k)
        k_rrf = 60.0
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense, start=1):
            cid = str(doc.metadata.get("chunk_id"))
            docs[cid] = doc
            scores[cid] += alpha * (1.0 / (k_rrf + rank))

        for rank, (doc, _) in enumerate(sparse, start=1):
            cid = str(doc.metadata.get("chunk_id"))
            docs[cid] = doc
            scores[cid] += (1.0 - alpha) * (1.0 / (k_rrf + rank))

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: list[tuple[Document, float]] = []
        for cid, score in fused:
            doc = docs[cid]
            if _match_filter(doc.metadata, filters or {}):
                results.append((doc, score))
            if len(results) >= k:
                break
        return results

    def delete_collection(self) -> None:
        ids = self._collection_ids()
        if ids:
            self._chroma._collection.delete(ids=ids)
        self._rebuild_bm25_index()

    def count(self) -> int:
        return int(self._chroma._collection.count())


class WeaviateVectorStore(CMMVectorStoreBase):
    def __init__(
        self,
        *,
        embedding_model: CMMEmbeddingsBase,
        collection_name: str = "CMMChunk",
        weaviate_url: str | None = None,
        weaviate_api_key: str | None = None,
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.weaviate_url = weaviate_url or os.getenv("CMM_WEAVIATE_URL")
        self.weaviate_api_key = weaviate_api_key or os.getenv(
            "CMM_WEAVIATE_API_KEY"
        )
        if not self.weaviate_url or not self.weaviate_api_key:
            raise ValueError(
                "CMM_WEAVIATE_URL and CMM_WEAVIATE_API_KEY are required"
            )
        try:
            import weaviate
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "weaviate-client package is required for WeaviateVectorStore"
            ) from exc

        self._weaviate = weaviate
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(self.weaviate_api_key),
        )

    def _collection(self):
        return self.client.collections.get(self.collection_name)

    def add_documents(self, documents: list[Document]) -> None:
        coll = self._collection()
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_model.embed_documents(texts)
        with coll.batch.dynamic() as batch:
            for doc, vector in zip(documents, vectors):
                data = {"text": doc.page_content}
                data.update(doc.metadata)
                batch.add_object(properties=data, vector=vector)

    def hybrid_search(
        self,
        query: str,
        k: int,
        alpha: float = 0.7,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        coll = self._collection()
        response = coll.query.hybrid(query=query, alpha=alpha, limit=k)
        out = []
        for obj in response.objects:
            props = dict(obj.properties)
            text = str(props.pop("text", ""))
            score = float(getattr(obj.metadata, "score", 0.0))
            if _match_filter(props, filters or {}):
                out.append((Document(page_content=text, metadata=props), score))
        return out

    def delete_collection(self) -> None:
        try:
            self.client.collections.delete(self.collection_name)
        except Exception:
            # Collection may not exist yet.
            return

    def count(self) -> int:
        try:
            coll = self._collection()
            return int(coll.aggregate.over_all(total_count=True).total_count)
        except Exception:
            return 0


def init_vectorstore(
    *,
    backend: str | None = None,
    persist_directory: str | Path,
    embedding_model: CMMEmbeddingsBase,
    collection_name: str = "cmm_chunks",
) -> CMMVectorStoreBase:
    selected = (
        backend or os.getenv("CMM_VECTORSTORE_BACKEND", "chroma")
    ).lower()
    if selected == "chroma":
        return ChromaBM25VectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )
    if selected == "weaviate":
        return WeaviateVectorStore(
            embedding_model=embedding_model,
            collection_name=collection_name,
        )
    raise ValueError(f"Unsupported vectorstore backend '{selected}'")
