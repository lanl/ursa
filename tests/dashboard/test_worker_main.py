from __future__ import annotations

import io
import os
from unittest.mock import Mock, create_autospec

import pytest

import ursa_dashboard.worker_main as worker_main
from ursa.runtime import AgentSpec
from ursa_dashboard.worker_main import _init_embedding, _init_llm


@pytest.fixture
def capture_model(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def install(kind):
        factory = Mock(return_value=object())
        path = (
            "langchain.chat_models.init_chat_model"
            if kind == "chat"
            else "langchain.embeddings.init_embeddings"
        )
        monkeypatch.setattr(path, factory)
        return factory

    return install


@pytest.mark.parametrize(
    "base_url",
    [{}, {"base_url": "   "}, {"base_url": " https://models.example.org/v1 "}],
)
def test_dashboard_worker_normalizes_base_url_without_exporting_it(
    capture_model, base_url
):
    model_factory = capture_model("chat")
    _init_llm({"model": "openai:gpt-test", "api_key_env": None, **base_url})
    expected = {"model": "openai:gpt-test"}
    if base_url.get("base_url", "").strip():
        expected["base_url"] = "https://models.example.org/v1"
    model_factory.assert_called_once_with(**expected)
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        ("gpt-test", {}, "openai:gpt-test"),
        ("gpt-test", {"model_provider": "openai"}, "gpt-test"),
        ("openai:gpt-test", {"model_provider": "openai"}, "openai:gpt-test"),
    ],
)
def test_dashboard_worker_qualifies_models_only_without_explicit_provider(
    capture_model, model, provider, expected
):
    model_factory = capture_model("chat")
    # An explicit provider must not add a second prefix to a bare model;
    # already-prefixed user input is preserved, even with a provider.
    _init_llm({"model": model, "api_key_env": None, "model_kwargs": provider})
    model_factory.assert_called_once_with(model=expected, **provider)


def test_dashboard_worker_returns_none_without_embedding_model():
    assert _init_embedding({}) is None
    assert _init_embedding({"model": "disabled"}) is None


def test_dashboard_worker_embedding_keeps_bare_model_with_provider(
    monkeypatch, capture_model
):
    model_factory = capture_model("embedding")
    monkeypatch.setenv("SAFE_EMBEDDING_KEY", "test-secret")

    _init_embedding({
        "model": "text-embedding-custom",
        "api_key_env": "SAFE_EMBEDDING_KEY",
        "model_kwargs": {"model_provider": "openai"},
    })

    assert model_factory.call_args.kwargs["model"] == "text-embedding-custom"
    assert model_factory.call_args.kwargs["model_provider"] == "openai"


def test_dashboard_worker_initializes_embedding_model(
    monkeypatch, capture_model
):
    model_factory = capture_model("embedding")
    monkeypatch.setenv("SAFE_EMBEDDING_KEY", "test-secret")

    _init_embedding({
        "model": "openai:text-embedding-3-large",
        "base_url": " https://models.example.org/v1 ",
        "api_key_env": "SAFE_EMBEDDING_KEY",
        "model_kwargs": {"dimensions": 1024},
    })

    assert model_factory.call_args.kwargs == {
        "model": "openai:text-embedding-3-large",
        "api_key": "test-secret",
        "base_url": "https://models.example.org/v1",
        "dimensions": 1024,
    }
    assert "OPENAI_BASE_URL" not in os.environ
    assert "OPENAI_API_BASE" not in os.environ


def test_dashboard_worker_uses_override_without_exporting_secret(
    monkeypatch, capture_model
):
    model_factory = capture_model("chat")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _init_llm(
        {"model": "openai:gpt-test", "api_key_env": "MISSING_KEY"},
        api_key_override="pipe-secret",
    )

    assert model_factory.call_args.kwargs["api_key"] == "pipe-secret"
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


async def test_dashboard_worker_uses_shared_agent_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    manager = create_autospec(worker_main.AgentManager, instance=True)
    manager.invoke.return_value = "managed result"
    monkeypatch.setattr(worker_main, "AgentManager", lambda: manager)
    config = {
        "agent_id": "chat_agent",
        "run_id": "run-1",
        "workspace_dir": str(tmp_path),
        "agent_init": {},
        "llm": {"model": "disabled"},
        "embedding": {},
        "mcp": {},
        "credentials": {"llm_api_key": "pipe-secret"},
    }

    result = await worker_main._run_with_shared_agent_manager(config, "hello")

    spec = manager.register.await_args.args[0]
    manager.register.assert_awaited_once_with(spec)
    assert isinstance(spec, AgentSpec)
    assert result == "managed result"
    manager.invoke.assert_awaited_once_with("dashboard-run", "hello")
    manager.aclose.assert_awaited_once_with()
    assert (
        spec.factory == "ursa_dashboard.worker_main:_dashboard_process_factory"
    )
    assert spec.config == config


async def test_dashboard_worker_runs_demo_through_shared_agent_manager(
    tmp_path,
) -> None:
    result = await worker_main._run_with_shared_agent_manager(
        {
            "agent_id": "demo_quick",
            "run_id": "run-1",
            "workspace_dir": str(tmp_path),
            "agent_init": {},
            "llm": {"model": "disabled"},
            "embedding": {},
            "mcp": {},
            "credentials": {},
        },
        "hello",
    )

    assert "DemoQuickAgent finished" in result
    assert (tmp_path / "artifacts" / "demo_output.txt").is_file()
