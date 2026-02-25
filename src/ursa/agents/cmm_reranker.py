from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.documents import Document


class CMMRerankerBase(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        ...


class NoOpReranker(CMMRerankerBase):
    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        del query
        return documents[:top_k]


class CohereReranker(CMMRerankerBase):
    def __init__(self, model: str = "rerank-english-v3.0"):
        try:
            import cohere
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("cohere package is required for CohereReranker") from exc
        self.model = model
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is required for CohereReranker")
        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if not documents:
            return []
        texts = [doc.page_content for doc, _ in documents]
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=min(top_k, len(documents)),
        )
        reranked: list[tuple[Document, float]] = []
        for item in response.results:
            doc, retrieval_score = documents[item.index]
            doc.metadata["retrieval_score"] = retrieval_score
            reranked.append((doc, float(item.relevance_score)))
        return reranked


class LocalCrossEncoderReranker(CMMRerankerBase):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for LocalCrossEncoderReranker"
            ) from exc
        self.batch_size = batch_size
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc, _ in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored = []
        for (doc, retrieval_score), rank_score in zip(documents, scores):
            doc.metadata["retrieval_score"] = retrieval_score
            scored.append((doc, float(rank_score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def init_reranker(provider: str | None = None) -> CMMRerankerBase:
    selected = (provider or os.getenv("CMM_RERANKER_PROVIDER", "none")).lower()
    if selected in {"none", "false", "off"}:
        return NoOpReranker()
    if selected == "cohere":
        return CohereReranker()
    if selected == "local":
        model = os.getenv("CMM_RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
        return LocalCrossEncoderReranker(model_name=model)
    raise ValueError(f"Unsupported reranker provider '{selected}'")
