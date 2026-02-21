from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from langchain.embeddings import Embeddings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional import guard
    OpenAI = None  # type: ignore[assignment]


def _default_dim_for_model(model: str) -> int:
    if model == "text-embedding-3-large":
        return 3072
    if model == "text-embedding-3-small":
        return 1536
    return 1024


class CMMEmbeddingsBase(Embeddings, ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...


class LangChainEmbeddingsAdapter(CMMEmbeddingsBase):
    def __init__(self, inner: Embeddings, embedding_dim: int | None = None):
        self.inner = inner
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim:
            return self._embedding_dim
        probe = self.inner.embed_query("dimension probe")
        return len(probe)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


class OpenAIEmbeddings(CMMEmbeddingsBase):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
        batch_size: int = 100,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIEmbeddings"
            )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddings")
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    @property
    def embedding_dim(self) -> int:
        return self.dimensions or _default_dim_for_model(self.model)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


class LocalEmbeddings(CMMEmbeddingsBase):
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddings"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name_or_path, **kwargs)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        prefixed = (
            "Represent this sentence for searching relevant passages: " + text
        )
        vector = self.model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return list(vector)


def parse_embedding_model_spec(spec: str) -> tuple[str, str, int | None]:
    if spec.startswith("openai:"):
        tail = spec.split("openai:", 1)[1]
        if ":" in tail:
            maybe_model, maybe_dim = tail.rsplit(":", 1)
            if maybe_dim.isdigit():
                return ("openai", maybe_model, int(maybe_dim))
        return ("openai", tail, None)

    if spec.startswith("local:"):
        model_path = spec.split("local:", 1)[1]
        return ("local", model_path, None)

    if Path(spec).exists() or "/" in spec:
        return ("local", spec, None)

    if ":" in spec:
        provider, model = spec.split(":", 1)
        if provider in {"openai", "local"}:
            return (provider, model, None)

    return ("openai", spec, None)


def init_embeddings(
    model_spec: str,
    *,
    dimensions: int | None = None,
    batch_size: int = 100,
    api_key: str | None = None,
    base_url: str | None = None,
) -> CMMEmbeddingsBase:
    provider, model, parsed_dim = parse_embedding_model_spec(model_spec)
    dim = dimensions if dimensions is not None else parsed_dim

    if provider == "openai":
        return OpenAIEmbeddings(
            model=model,
            dimensions=dim,
            batch_size=batch_size,
            api_key=api_key,
            base_url=base_url,
        )
    if provider == "local":
        return LocalEmbeddings(model_name_or_path=model, batch_size=32)
    raise ValueError(f"Unsupported embedding provider '{provider}'")
