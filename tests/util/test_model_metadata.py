"""Exercise discovery through real provider clients with mocked HTTP responses."""

import httpx
import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ursa.cli.config import ChatModelConfig
from ursa.util import model_metadata
from ursa.util.model_context import (
    model_context_window,
    summarization_threshold,
)
from ursa.util.model_metadata import load_model_context, refresh_ollama_context


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    data = {
        "openai": {
            "models": {
                "gpt-6-astra": {"limit": {"context": 1050000}},
                "gpt-6.1-terra": {"limit": {"context": 1050000}},
            }
        }
    }
    requests = []

    def handler(request):
        requests.append(request)
        assert str(request.url) == "https://models.dev/api.json"
        assert "Authorization" not in request.headers
        assert "X-Tenant" not in request.headers
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        model_metadata,
        "build_httpx_client",
        lambda **kwargs: httpx.Client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    model_metadata._openai_catalog.cache_clear()
    yield data, requests
    model_metadata._openai_catalog.cache_clear()


def openai_model(
    handler, base_url="https://gateway.example/proxy/v1", **kwargs
):
    return ChatOpenAI(
        api_key="test-key",
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def ollama_model(handler, **kwargs):
    return ChatOllama(
        model="gemma4",
        base_url="http://ollama.example:11434",
        client_kwargs={
            "transport": httpx.MockTransport(handler),
            "headers": {"X-Test": "retained"},
        },
        **kwargs,
    )


def unexpected_request(request):
    pytest.fail(f"Unexpected metadata request: {request.url.path}")


def test_litellm_alias_uses_smallest_deployment_and_preserves_client_settings(
    monkeypatch,
):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/proxy/v1/model/info"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["X-Tenant"] == "science"
        assert request.extensions["timeout"]["read"] == 3.0
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "model_name": "other",
                        "model_info": {"max_input_tokens": 900000},
                    },
                    {
                        "model_name": "gpt-4o",
                        "litellm_params": {"model": "internal/backend"},
                        "model_info": {
                            "max_input_tokens": 65536,
                            "max_tokens": 4096,
                        },
                    },
                    {
                        "model_name": "gpt-4o",
                        "model_info": {"max_input_tokens": 32768},
                    },
                ]
            },
        )

    model = openai_model(
        handler, model="gpt-4o", default_headers={"X-Tenant": "science"}
    )
    assert model_context_window(model) == 128000  # Local catalog is overridden.
    monkeypatch.setattr("ursa.cli.config.init_chat_model", lambda **_: model)
    initialized = ChatModelConfig(model="gpt-4o").init_chat_model()

    assert initialized is model
    assert model_context_window(model.bind(extra_headers={})) == 32768
    assert summarization_threshold(model) == 22938
    assert model._ursa_context_source == "Endpoint /proxy/v1/model/info"
    assert len(requests) == 1
    # Neither the TUI lookup nor the trigger repeats endpoint requests.
    assert model_context_window(model) == 32768
    assert len(requests) == 1


@pytest.mark.parametrize(
    "path,row",
    [
        (
            "/proxy/model/info",
            {"model_name": "alias", "model_info": {"max_input_tokens": 8192}},
        ),
        (
            "/proxy/model_group/info",
            {"model_group": "alias", "max_input_tokens": 8192},
        ),
        ("/proxy/v1/models", {"id": "alias", "context_length": 8192}),
    ],
)
def test_compatible_metadata_routes_keep_reverse_proxy_prefix(path, row):
    def handler(request):
        if request.url.path == path:
            return httpx.Response(200, json={"data": [row]})
        return httpx.Response(404)

    model = openai_model(handler, model="alias")
    load_model_context(model)
    assert model_context_window(model) == 8192
    assert model._ursa_context_source == f"Endpoint {path}"


def test_unversioned_base_can_discover_versioned_litellm_metadata():
    def handler(request):
        if request.url.path == "/proxy/v1/model/info":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "alias",
                            "model_info": {"max_input_tokens": 8192},
                        }
                    ]
                },
            )
        return httpx.Response(404)

    model = openai_model(
        handler, base_url="https://gateway.example/proxy", model="alias"
    )
    load_model_context(model)
    assert model_context_window(model) == 8192


@pytest.mark.parametrize(
    "payload",
    [
        {
            "data": [
                {"id": "gpt-4o", "max_tokens": 8192, "max_output_tokens": 8192}
            ]
        },
        {"data": [{"id": "gpt-4o", "max_input_tokens": True}]},
        {"data": [{"id": "gpt-4o", "max_input_tokens": -1}]},
        {"data": [{"id": "gpt-4o", "max_input_tokens": "8192"}]},
        {
            "data": [
                {
                    "model_name": "other",
                    "litellm_params": {"model": "gpt-4o"},
                    "model_info": {"max_input_tokens": 8192},
                }
            ]
        },
        {
            "data": [
                {
                    "model_name": "gpt-4o",
                    "model_info": {"max_input_tokens": 8192},
                },
                {"model_name": "gpt-4o", "model_info": {}},
            ]
        },
        {"data": None},
        ["invalid"],
    ],
)
def test_unknown_custom_limits_do_not_use_public_catalog_or_output_cap(payload):
    model = openai_model(
        lambda _: httpx.Response(200, json=payload), model="gpt-4o"
    )
    load_model_context(model)
    assert model_context_window(model) is None
    assert summarization_threshold(model) == 50000


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_optional_metadata_failure_keeps_model_usable(status):
    model = openai_model(lambda _: httpx.Response(status), model="alias")
    load_model_context(model)
    assert summarization_threshold(model) == 50000
    if status in {401, 403}:
        assert f"HTTP {status}" in model._ursa_context_source


def test_timeout_is_bounded_without_retries():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("timeout", request=request)

    model = openai_model(handler, model="alias")
    load_model_context(model)
    assert len(requests) == 1
    assert model_context_window(model) is None


def test_explicit_profile_skips_endpoint_discovery(monkeypatch):
    model = openai_model(
        unexpected_request, model="alias", profile={"max_input_tokens": 16384}
    )
    monkeypatch.setattr("ursa.cli.config.init_chat_model", lambda **_: model)
    ChatModelConfig(
        model="alias", profile={"max_input_tokens": 16384}
    ).init_chat_model()
    assert model_context_window(model) == 16384
    assert model._ursa_context_source == "Configured profile"


@pytest.mark.parametrize(
    "name,window",
    [
        ("gpt-6-astra", 1050000),
        ("gpt-6.1-terra", 1050000),
        ("gpt-5.4", 1050000),
        ("gpt-6-astra-unknown", None),
    ],
)
def test_official_openai_local_profile_or_current_catalog(name, window):
    model = ChatOpenAI(
        model=name,
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(unexpected_request)
        ),
    )
    load_model_context(model)
    assert model_context_window(model) == window


def test_new_openai_models_use_one_anonymous_catalog_request(catalog):
    data, requests = catalog
    data["openai"]["models"] = {
        "new-model-a": {"limit": {"context": 32768}},
        "new-model-b": {"limit": {"context": 128000, "input": 100000}},
    }
    for name, expected in [("new-model-a", 32768), ("new-model-b", 128000)]:
        model = openai_model(
            unexpected_request,
            base_url="https://api.openai.com/v1",
            model=name,
            default_headers={"X-Tenant": "private-tenant"},
        )
        load_model_context(model)
        assert model_context_window(model) == expected
        assert model._ursa_context_source == "models.dev OpenAI catalog"
    assert len(requests) == 1


def test_known_or_custom_models_do_not_fetch_public_catalog(catalog):
    _, requests = catalog
    model = openai_model(
        unexpected_request, base_url="https://api.openai.com/v1", model="gpt-4o"
    )
    load_model_context(model)
    model = openai_model(
        lambda _: httpx.Response(404), model="unknown-private-alias"
    )
    load_model_context(model)
    assert not requests


def test_catalog_failure_uses_configured_fallback_and_is_cached(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("timeout", request=request)

    monkeypatch.setattr(
        model_metadata,
        "build_httpx_client",
        lambda **kwargs: httpx.Client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    for name in ("new-model-a", "new-model-b"):
        model = openai_model(
            unexpected_request, base_url="https://api.openai.com/v1", model=name
        )
        load_model_context(model)
        assert summarization_threshold(model, fallback=12000) == 12000
    assert len(calls) == 1
    assert calls[0].extensions["timeout"]["read"] == 3.0


@pytest.mark.parametrize(
    "limit",
    [
        {"context": True},
        {"context": -1},
        {"context": "8192"},
        {"output": 8192},
        None,
    ],
)
def test_invalid_public_catalog_limits_use_fallback(catalog, limit):
    data, _ = catalog
    data["openai"]["models"] = {"new-model": {"limit": limit}}
    model = openai_model(
        unexpected_request,
        base_url="https://api.openai.com/v1",
        model="new-model",
    )
    load_model_context(model)
    assert model_context_window(model) is None


def test_ollama_runtime_allocation_not_architecture_maximum():
    def handler(request):
        assert request.headers["X-Test"] == "retained"
        assert request.extensions["timeout"]["read"] == 3.0
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"model_info": {"gemma.context_length": 131072}}
            )
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "other:latest", "context_length": 16384},
                    {"name": "gemma4:latest", "context_length": 4096},
                ]
            },
        )

    model = ollama_model(handler)
    load_model_context(model)
    assert model_context_window(model) == 4096
    assert summarization_threshold(model.bind(extra_headers={})) == 2867
    assert model._ursa_context_source == "Ollama /api/ps"


def test_ollama_modelfile_context_and_explicit_num_ctx():
    def handler(request):
        assert request.url.path == "/api/show"
        return httpx.Response(
            200, json={"parameters": 'stop "end"\nnum_ctx    32768\n'}
        )

    model = ollama_model(handler)
    load_model_context(model)
    refresh_ollama_context(model)
    assert model_context_window(model) == 32768
    assert model._ursa_context_source == "Ollama model parameters"

    model = ollama_model(unexpected_request, num_ctx=8192)
    load_model_context(model)
    refresh_ollama_context(model)
    assert model_context_window(model) == 8192


def test_ollama_discovers_allocation_after_model_loads():
    running = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"model_info": {"gemma.context_length": 131072}}
            )
        return httpx.Response(200, json={"models": running})

    model = ollama_model(handler)
    load_model_context(model)
    assert model_context_window(model) is None
    running.append({"model": "gemma4:latest", "context_length": 8192})
    refresh_ollama_context(model)
    assert model_context_window(model) == 8192


def test_ollama_unavailable_metadata_does_not_prevent_initialization():
    model = ollama_model(lambda _: httpx.Response(404))
    load_model_context(model)
    assert model_context_window(model) is None
