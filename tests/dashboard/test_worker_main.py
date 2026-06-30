from __future__ import annotations

import os

from ursa_dashboard.worker_main import _init_embedding, _init_llm


def test_dashboard_worker_omits_null_base_url_for_provider_default(
    monkeypatch,
):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    _init_llm({"model": "openai:gpt-test", "api_key_env": None})

    assert captured["model"] == "openai:gpt-test"
    assert "base_url" not in captured
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


def test_dashboard_worker_omits_blank_base_url_for_provider_default(
    monkeypatch,
):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    _init_llm({
        "model": "openai:gpt-test",
        "base_url": "   ",
        "api_key_env": None,
    })

    assert "base_url" not in captured
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


def test_dashboard_worker_passes_configured_base_url(monkeypatch):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    _init_llm({
        "model": "openai:gpt-test",
        "base_url": " https://models.example.org/v1 ",
        "api_key_env": None,
    })

    assert captured["base_url"] == "https://models.example.org/v1"
    assert os.environ["OPENAI_BASE_URL"] == ("https://models.example.org/v1")
    assert os.environ["OPENAI_API_BASE"] == ("https://models.example.org/v1")


def test_dashboard_worker_returns_none_without_embedding_model():
    assert _init_embedding({}) is None
    assert _init_embedding({"model": "disabled"}) is None


def test_dashboard_worker_initializes_embedding_model(monkeypatch):
    captured: dict = {}

    def fake_init_embeddings(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.embeddings.init_embeddings", fake_init_embeddings
    )
    monkeypatch.setenv("SAFE_EMBEDDING_KEY", "test-secret")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    _init_embedding({
        "model": "openai:text-embedding-3-large",
        "base_url": " https://models.example.org/v1 ",
        "api_key_env": "SAFE_EMBEDDING_KEY",
        "model_kwargs": {"dimensions": 1024},
    })

    assert captured == {
        "model": "openai:text-embedding-3-large",
        "api_key": "test-secret",
        "base_url": "https://models.example.org/v1",
        "dimensions": 1024,
    }
    assert os.environ["OPENAI_BASE_URL"] == "https://models.example.org/v1"
    assert os.environ["OPENAI_API_BASE"] == "https://models.example.org/v1"
