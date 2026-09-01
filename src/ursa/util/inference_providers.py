"""Inspect inference providers without importing their SDKs eagerly."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
from importlib.util import find_spec
from typing import Any, Literal
from urllib.parse import urlparse

from ursa.cli.config import InferenceProviderConfig, ModelConfig
from ursa.util.http import build_httpx_client

ProviderConfig = InferenceProviderConfig | ModelConfig
ModelLister = Callable[[ProviderConfig], Iterable[Any]]


@dataclass(frozen=True)
class ProviderModel:
    """A model advertised by an inference endpoint."""

    name: str
    model_provider: str | None = None
    type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ProviderRequest:
    fingerprint: tuple[Any, ...]
    config: ProviderConfig = field(compare=False, hash=False, repr=False)


@lru_cache(maxsize=2)
def supported_model_providers(
    model_type: Literal["chat", "embedding"] = "chat",
) -> tuple[str, ...]:
    """Return built-in providers whose LangChain integration is installed."""
    if model_type == "embedding":
        from langchain.embeddings.base import _BUILTIN_PROVIDERS
    else:
        from langchain.chat_models.base import _BUILTIN_PROVIDERS

    return tuple(
        provider
        for provider, (
            module,
            _class_name,
            _creator,
        ) in _BUILTIN_PROVIDERS.items()
        if find_spec(module.partition(".")[0]) is not None
    )


def sort_provider_models(
    models: Iterable[ProviderModel],
    model_type: Literal["chat", "embedding"],
) -> list[ProviderModel]:
    """Rank models for a chat or embedding model picker."""

    def recency(model: ProviderModel) -> float:
        value = next(
            (
                model.metadata.get(key)
                for key in ("created", "created_at", "updated_at")
                if model.metadata.get(key) is not None
            ),
            None,
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass
        return 0

    def priority(model: ProviderModel) -> tuple[bool, bool, float, int, str]:
        name = model.name.lower()
        embedding = model.type == "embedding" or any(
            marker in name for marker in ("embed", "text-embedding")
        )
        deprioritized = any(
            marker in name
            for marker in ("whisper", "live", "realtime", "tts", "sora")
        )
        wrong_type = embedding != (model_type == "embedding")
        return deprioritized, wrong_type, -recency(model), len(name), name

    return sorted(models, key=priority)


def _client_type(module: str, name: str) -> type:
    """Load a provider SDK only when its model-list endpoint is used."""
    return getattr(importlib.import_module(module), name)


def _secret(config: ProviderConfig, *, required: bool = True) -> str | None:
    reference = config.api_key
    value = reference.get_secret_value() if reference is not None else None
    if required and not value:
        raise ValueError("Inference provider API key is missing")
    return value


def _sdk_kwargs(config: ProviderConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": _secret(config),
        "http_client": build_httpx_client(verify=config.ssl_verify),
    }
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return kwargs


def _list_openai(config: ProviderConfig) -> Iterable[Any]:
    kwargs = _sdk_kwargs(config)
    with _client_type("openai", "OpenAI")(**kwargs) as client:
        return list(client.models.list().data)


def _list_azure_openai(config: ProviderConfig) -> Iterable[Any]:
    kwargs = _sdk_kwargs(config)
    extra = config.model_extra or {}
    kwargs.update({
        key: extra[key]
        for key in ("azure_endpoint", "api_version")
        if key in extra
    })
    if "azure_endpoint" in kwargs:
        kwargs.pop("base_url", None)
    with _client_type("openai", "AzureOpenAI")(**kwargs) as client:
        return list(client.models.list().data)


def _list_anthropic(config: ProviderConfig) -> Iterable[Any]:
    with _client_type("anthropic", "Anthropic")(
        **_sdk_kwargs(config)
    ) as client:
        return list(client.models.list().data)


def _list_google(config: ProviderConfig) -> Iterable[Any]:
    with _client_type("google.genai", "Client")(
        api_key=_secret(config)
    ) as client:
        return list(client.models.list())


def _list_ollama(config: ProviderConfig) -> Iterable[Any]:
    client = _client_type("ollama", "Client")(
        host=config.base_url, verify=config.ssl_verify
    )
    return client.list().models


_MODEL_LISTERS: dict[str, ModelLister] = {
    "anthropic": _list_anthropic,
    "azure_openai": _list_azure_openai,
    "google_genai": _list_google,
    "litellm": _list_openai,
    "ollama": _list_ollama,
    "openai": _list_openai,
}


def model_listing_provider(config: ProviderConfig) -> str:
    """Return the API protocol used to list a provider's models."""
    model_provider = getattr(config, "model_provider", None)
    if isinstance(model_provider, str) and model_provider:
        return model_provider
    parsed = urlparse(config.base_url or "")
    endpoint = f"{parsed.hostname or ''}:{parsed.port or ''}".lower()
    if "anthropic" in endpoint:
        return "anthropic"
    if "googleapis" in endpoint:
        return "google_genai"
    if parsed.port == 11434 or "ollama" in endpoint:
        return "ollama"
    if "azure" in endpoint:
        return "azure_openai"
    return "openai"


def _as_mapping(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="python")
    if isinstance(model, dict):
        return dict(model)
    return {
        key: getattr(model, key)
        for key in (
            "id",
            "name",
            "model",
            "owned_by",
            "provider",
            "model_provider",
            "litellm_provider",
            "type",
            "model_type",
        )
        if getattr(model, key, None) is not None
    }


def _returned_model_provider(
    raw: Mapping[str, Any],
    name: str,
    fallback: str,
    supported: frozenset[str],
) -> str | None:
    from langchain.chat_models.base import _attempt_infer_model_provider

    candidates = (
        raw.get("model_provider"),
        raw.get("litellm_provider"),
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        if candidate in supported:
            return candidate
    inferred = _attempt_infer_model_provider(name.rsplit("/", 1)[-1])
    if inferred in supported:
        return inferred
    return fallback if fallback in supported else None


def _provider_model(
    model: Any, fallback: str, supported: frozenset[str]
) -> ProviderModel:
    raw = _as_mapping(model)
    name = raw.get("id") or raw.get("name") or raw.get("model")
    if not isinstance(name, str) or not name:
        raise ValueError("Inference provider returned a model without a name")
    model_type = raw.get("type") or raw.get("model_type")
    metadata = {
        key: value
        for key, value in raw.items()
        if key not in {"id", "name", "model", "type", "model_type"}
        and value is not None
        and isinstance(value, (str, int, float, bool, list, dict))
    }
    return ProviderModel(
        name=name,
        model_provider=_returned_model_provider(raw, name, fallback, supported),
        type=model_type if isinstance(model_type, str) else None,
        metadata=metadata,
    )


@lru_cache(maxsize=16)
def _list_provider_models(
    request: _ProviderRequest,
) -> tuple[ProviderModel, ...]:
    provider = model_listing_provider(request.config)
    lister = _MODEL_LISTERS.get(provider, _list_openai)
    supported = frozenset(supported_model_providers())
    return tuple(
        _provider_model(model, provider, supported)
        for model in lister(request.config)
    )


def list_provider_models(
    inference_provider: ProviderConfig,
) -> list[ProviderModel]:
    """Return a cached model catalog advertised by an inference endpoint."""
    secret = _secret(inference_provider, required=False)
    secret_fingerprint = (
        sha256(secret.encode()).digest() if secret is not None else None
    )
    request = _ProviderRequest(
        fingerprint=(
            model_listing_provider(inference_provider),
            inference_provider.base_url,
            inference_provider.ssl_verify,
            secret_fingerprint,
            repr(inference_provider.model_extra),
        ),
        config=inference_provider,
    )
    return list(_list_provider_models(request))


def _matches_model(configured: str, advertised: str) -> bool:
    return (
        configured == advertised or configured == advertised.rsplit("/", 1)[-1]
    )


def validate_model_provider(
    inference_provider: ProviderConfig,
    model_type: Literal["chat", "embedding"],
) -> None:
    """Validate connectivity and model presence with a model-list request."""
    try:
        models = list_provider_models(inference_provider)
        if isinstance(inference_provider, ModelConfig) and not any(
            _matches_model(inference_provider.model, model.name)
            for model in models
        ):
            raise ValueError(
                f"Model '{inference_provider.model}' is not available from "
                "the inference provider"
            )
    except Exception as exc:
        provider = (
            getattr(inference_provider, "inference_provider", None)
            or getattr(inference_provider, "model_provider", None)
            or inference_provider.base_url
            or "direct configuration"
        )
        model = getattr(inference_provider, "model", "configured endpoint")
        raise ValueError(
            f"Unable to validate {model_type} model '{model}' with provider "
            f"'{provider}': {exc}. Check the model name, provider endpoint, "
            "and API credentials."
        ) from exc
