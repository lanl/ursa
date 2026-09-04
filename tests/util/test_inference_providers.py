from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ursa.cli.config import InferenceProviderConfig, ModelConfig
from ursa.util import inference_providers
from ursa.util.inference_providers import ProviderModel
from ursa.util.secrets import SecretReference


@pytest.fixture(autouse=True)
def clear_model_provider_caches():
    inference_providers._list_provider_models.cache_clear()
    inference_providers.supported_model_providers.cache_clear()
    yield
    inference_providers._list_provider_models.cache_clear()
    inference_providers.supported_model_providers.cache_clear()


class FakeOpenAI:
    models = SimpleNamespace()
    kwargs = None

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_list_provider_models_resolves_secret_and_preserves_metadata(
    monkeypatch,
):
    monkeypatch.setattr(
        inference_providers,
        "_client_type",
        lambda module, name: FakeOpenAI,
    )
    monkeypatch.setattr(
        inference_providers,
        "build_httpx_client",
        lambda *, verify: f"client:{verify}",
    )
    FakeOpenAI.models.list = lambda: SimpleNamespace(
        data=[
            SimpleNamespace(
                model_dump=lambda mode: {
                    "id": "chat-model",
                    "owned_by": "acme",
                    "created": 42,
                    "type": "chat",
                }
            )
        ]
    )

    models = inference_providers.list_provider_models(
        InferenceProviderConfig(
            base_url="https://models.example/v1",
            api_key=SecretStr("secret"),
            ssl_verify=False,
        )
    )

    assert models == [
        ProviderModel(
            name="chat-model",
            model_provider="openai",
            type="chat",
            metadata={"owned_by": "acme", "created": 42},
        )
    ]
    assert FakeOpenAI.kwargs == {
        "api_key": "secret",
        "base_url": "https://models.example/v1",
        "http_client": "client:False",
    }


def test_list_provider_models_resolves_secret_reference(monkeypatch):
    monkeypatch.setenv("MODELS_API_KEY", "from-env")
    monkeypatch.setattr(
        inference_providers,
        "_client_type",
        lambda module, name: FakeOpenAI,
    )
    monkeypatch.setattr(
        inference_providers, "build_httpx_client", lambda **_kwargs: object()
    )
    FakeOpenAI.models.list = lambda: SimpleNamespace(data=[])

    inference_providers.list_provider_models(
        InferenceProviderConfig(api_key=SecretReference(env="MODELS_API_KEY"))
    )

    assert FakeOpenAI.kwargs["api_key"] == "from-env"


def test_list_provider_models_rejects_missing_secret(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    config = InferenceProviderConfig(
        api_key=SecretReference(env="MISSING_API_KEY")
    )

    with pytest.raises(ValueError, match="API key is missing"):
        inference_providers.list_provider_models(config)


def test_validate_model_provider_accepts_advertised_model(monkeypatch):
    monkeypatch.setattr(
        inference_providers,
        "list_provider_models",
        lambda _provider: [ProviderModel("gpt-test", "openai")],
    )

    assert (
        inference_providers.validate_model_provider(
            ModelConfig(
                model="gpt-test",
                model_provider="openai",
                api_key=SecretStr("secret"),
            ),
            "chat",
        )
        is None
    )


def test_validate_model_provider_rejects_unadvertised_model(monkeypatch):
    monkeypatch.setattr(
        inference_providers,
        "list_provider_models",
        lambda _provider: [ProviderModel("another-model", "openai")],
    )

    with pytest.raises(ValueError, match="gpt-test.*not available"):
        inference_providers.validate_model_provider(
            ModelConfig(
                model="gpt-test",
                model_provider="openai",
                api_key=SecretStr("secret"),
            ),
            "chat",
        )


def test_validate_model_provider_config_only_checks_connectivity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inference_providers,
        "list_provider_models",
        lambda provider: calls.append(provider) or [],
    )
    provider = InferenceProviderConfig(api_key=SecretStr("secret"))

    assert inference_providers.validate_model_provider(provider, "chat") is None
    assert calls == [provider]


@pytest.mark.parametrize("provider", inference_providers._MODEL_LISTERS)
def test_list_provider_models_dispatches_by_model_provider(
    monkeypatch, provider
):
    calls = []
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        provider,
        lambda config: calls.append(config) or [],
    )
    config = ModelConfig(model="test", model_provider=provider)

    assert inference_providers.list_provider_models(config) == []
    assert calls == [config]


def test_list_provider_models_dispatches_litellm(monkeypatch):
    calls = []
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        "litellm",
        lambda config: calls.append(config) or [],
    )
    config = ModelConfig(model="test", model_provider="litellm")

    assert inference_providers.list_provider_models(config) == []
    assert calls == [config]


def test_list_provider_models_caches_equivalent_provider_requests(monkeypatch):
    calls = []
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        "openai",
        lambda config: calls.append(config) or [{"id": "gpt-test"}],
    )
    chat = ModelConfig(
        model="gpt-test",
        model_provider="openai",
        base_url="https://models.example/v1",
    )
    embedding = ModelConfig(
        model="text-embedding-test",
        model_provider="openai",
        base_url="https://models.example/v1",
    )

    assert inference_providers.list_provider_models(chat)
    assert inference_providers.list_provider_models(embedding)
    assert calls == [chat]


def test_list_provider_models_uses_inference_provider_model_provider(
    monkeypatch,
):
    calls = []
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        "anthropic",
        lambda config: calls.append(config) or [],
    )
    config = InferenceProviderConfig(model_provider="anthropic")

    assert inference_providers.list_provider_models(config) == []
    assert calls == [config]


def test_litellm_models_can_have_different_model_providers(monkeypatch):
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        "litellm",
        lambda _config: [
            {"id": "openai/gpt-test", "litellm_provider": "openai"},
            {
                "id": "anthropic/claude-test",
                "litellm_provider": "anthropic",
            },
        ],
    )
    config = ModelConfig(model="test", model_provider="litellm")

    assert inference_providers.list_provider_models(config) == [
        ProviderModel(
            "openai/gpt-test",
            "openai",
            metadata={"litellm_provider": "openai"},
        ),
        ProviderModel(
            "anthropic/claude-test",
            "anthropic",
            metadata={"litellm_provider": "anthropic"},
        ),
    ]


def test_model_provider_is_inferred_from_model_name(monkeypatch):
    monkeypatch.setitem(
        inference_providers._MODEL_LISTERS,
        "litellm",
        lambda _config: [{"id": "claude-test", "owned_by": "gateway"}],
    )
    config = ModelConfig(model="test", model_provider="litellm")

    assert inference_providers.list_provider_models(config) == [
        ProviderModel(
            "claude-test",
            "anthropic",
            metadata={"owned_by": "gateway"},
        )
    ]


def test_ollama_listing_does_not_require_a_key(monkeypatch):
    captured = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def list(self):
            return SimpleNamespace(models=[{"model": "llama-test"}])

    monkeypatch.setattr(
        inference_providers,
        "_client_type",
        lambda module, name: FakeOllama,
    )
    config = ModelConfig(
        model="llama-test",
        model_provider="ollama",
        base_url="http://localhost:11434",
        ssl_verify=False,
    )

    assert inference_providers.list_provider_models(config) == [
        ProviderModel("llama-test", "ollama")
    ]
    assert captured == {"host": "http://localhost:11434", "verify": False}


def test_validate_model_provider_accepts_google_qualified_model_name(
    monkeypatch,
):
    monkeypatch.setattr(
        inference_providers,
        "list_provider_models",
        lambda _provider: [ProviderModel("models/gemini-test", "google_genai")],
    )
    config = ModelConfig(model="gemini-test", model_provider="google_genai")

    assert inference_providers.validate_model_provider(config, "chat") is None


def test_supported_model_providers_are_installed_langchain_builtins(
    monkeypatch,
):
    from langchain.chat_models.base import _BUILTIN_PROVIDERS

    installed = {"langchain_openai", "langchain_anthropic"}
    monkeypatch.setattr(
        inference_providers,
        "find_spec",
        lambda module: object() if module in installed else None,
    )
    inference_providers.supported_model_providers.cache_clear()
    providers = inference_providers.supported_model_providers()

    assert providers == tuple(
        provider
        for provider, (
            module,
            _class_name,
            _creator,
        ) in _BUILTIN_PROVIDERS.items()
        if module.partition(".")[0] in installed
    )
    inference_providers.supported_model_providers.cache_clear()


def test_supported_embedding_providers_use_embedding_registry(monkeypatch):
    from langchain.embeddings.base import _BUILTIN_PROVIDERS

    installed = {
        module.partition(".")[0] for module, _, _ in _BUILTIN_PROVIDERS.values()
    }
    monkeypatch.setattr(
        inference_providers,
        "find_spec",
        lambda module: object() if module in installed else None,
    )
    inference_providers.supported_model_providers.cache_clear()

    providers = inference_providers.supported_model_providers("embedding")

    assert providers == tuple(_BUILTIN_PROVIDERS)
    inference_providers.supported_model_providers.cache_clear()


def test_sort_provider_models_prioritizes_models_for_picker_type():
    models = [
        ProviderModel("gpt-4-0613"),
        ProviderModel("text-embedding-3-large"),
        ProviderModel("gpt-4-realtime"),
        ProviderModel("gpt-4"),
        ProviderModel("embed-small"),
        ProviderModel("whisper-1"),
        ProviderModel("tts-1"),
        ProviderModel("sora-2"),
    ]

    assert [
        model.name
        for model in inference_providers.sort_provider_models(models, "chat")
    ] == [
        "gpt-4",
        "gpt-4-0613",
        "embed-small",
        "text-embedding-3-large",
        "tts-1",
        "sora-2",
        "whisper-1",
        "gpt-4-realtime",
    ]
    assert [
        model.name
        for model in inference_providers.sort_provider_models(
            models, "embedding"
        )
    ] == [
        "embed-small",
        "text-embedding-3-large",
        "gpt-4",
        "gpt-4-0613",
        "tts-1",
        "sora-2",
        "whisper-1",
        "gpt-4-realtime",
    ]


def test_sort_provider_models_prefers_recent_metadata_before_name_length():
    models = [
        ProviderModel("gpt-4", metadata={"created": 100}),
        ProviderModel("gpt-5-long-snapshot", metadata={"created": 200}),
        ProviderModel("gpt-5", metadata={"created_at": "2026-01-01T00:00:00Z"}),
    ]

    assert [
        model.name
        for model in inference_providers.sort_provider_models(models, "chat")
    ] == ["gpt-5", "gpt-5-long-snapshot", "gpt-4"]
