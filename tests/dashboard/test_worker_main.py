from __future__ import annotations

import io
import os

import pytest

import ursa_dashboard.worker_main as worker_main
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
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


def test_dashboard_worker_prepends_openai_for_bare_model_without_provider(
    monkeypatch,
):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )

    _init_llm({"model": "gpt-test", "api_key_env": None})

    # No provider prefix and no model_provider kwarg -> default to openai:.
    assert captured["model"] == "openai:gpt-test"
    assert "model_provider" not in captured


def test_dashboard_worker_keeps_bare_model_when_model_provider_given(
    monkeypatch,
):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )

    _init_llm({
        "model": "gpt-test",
        "api_key_env": None,
        "model_kwargs": {"model_provider": "openai"},
    })

    # An explicit model_provider must NOT trigger the "openai:" prepend,
    # otherwise the endpoint receives a doubly-qualified "openai:gpt-test".
    assert captured["model"] == "gpt-test"
    assert captured["model_provider"] == "openai"


def test_dashboard_worker_leaves_prefixed_model_with_provider_untouched(
    monkeypatch,
):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )

    _init_llm({
        "model": "openai:gpt-test",
        "api_key_env": None,
        "model_kwargs": {"model_provider": "openai"},
    })

    # A genuinely double-specified config is left as-is (allowed to error
    # downstream); we do not silently rewrite the user's input.
    assert captured["model"] == "openai:gpt-test"
    assert captured["model_provider"] == "openai"


def test_dashboard_worker_returns_none_without_embedding_model():
    assert _init_embedding({}) is None
    assert _init_embedding({"model": "disabled"}) is None


def test_dashboard_worker_embedding_keeps_bare_model_with_provider(monkeypatch):
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
        "model": "text-embedding-custom",
        "api_key_env": "SAFE_EMBEDDING_KEY",
        "model_kwargs": {"model_provider": "openai"},
    })

    assert captured["model"] == "text-embedding-custom"
    assert captured["model_provider"] == "openai"


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
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


def test_dashboard_worker_uses_override_without_exporting_secret(monkeypatch):
    captured: dict = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _init_llm(
        {"model": "openai:gpt-test", "api_key_env": "MISSING_KEY"},
        api_key_override="pipe-secret",
    )

    assert captured["api_key"] == "pipe-secret"
    assert "OPENAI_API_KEY" not in os.environ


def test_dashboard_worker_reads_secret_payload_from_stdin(monkeypatch):
    class FakeStdin:
        buffer = io.BytesIO(
            b'{"llm_api_key":"pipe-secret","embedding_api_key":null}\n'
        )

    monkeypatch.setattr(worker_main.sys, "stdin", FakeStdin())

    assert worker_main._read_secrets_stdin() == {
        "llm_api_key": "pipe-secret",
        "embedding_api_key": None,
    }


def test_dashboard_worker_rejects_literal_key_config(monkeypatch):
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **_kwargs: object()
    )

    with pytest.raises(ValueError, match="Literal LLM API keys"):
        _init_llm({"model": "openai:gpt-test", "api_key": "secret"})


def test_worker_error_redaction_removes_full_secret() -> None:
    assert (
        worker_main._redact_secrets(
            "provider rejected pipe-secret", ["pipe-secret"]
        )
        == "provider rejected [REDACTED]"
    )
