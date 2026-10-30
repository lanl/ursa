"""Best-effort context discovery at model initialization, never during rendering."""

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel

from ursa.util.http import build_httpx_client
from ursa.util.model_context import model_context_window

_TIMEOUT = 3.0


def _positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


@lru_cache(maxsize=1)
def _openai_catalog() -> dict[str, int]:
    """Fetch the public catalog once per process, without provider credentials."""
    try:
        with build_httpx_client(timeout=_TIMEOUT) as client:
            response = client.get("https://models.dev/api.json")
            response.raise_for_status()
            models = response.json()["openai"]["models"]
        windows = {}
        for name, info in models.items():
            if not isinstance(info, Mapping):
                continue
            limit = info.get("limit")
            if not isinstance(limit, Mapping):
                continue
            window = _positive_int(limit.get("context")) or _positive_int(
                limit.get("input")
            )
            if window is not None:
                windows[name] = window
        return windows
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _set_context(model: BaseChatModel, window: int | None, source: str) -> None:
    profile = dict(model.profile or {})
    profile.pop("max_input_tokens", None)
    if window is not None:
        profile["max_input_tokens"] = window
    model.profile = profile or None
    model._ursa_context_source = source


def _advertised_window(payload: object, name: str) -> int | None:
    """Match the public alias, never a proxy's internal backend model name."""
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("data"), list
    ):
        return None
    windows = []
    for row in payload["data"]:
        if not isinstance(row, Mapping) or name not in (
            row.get("model_name"),
            row.get("model_group"),
            row.get("id"),
        ):
            continue
        info = row.get("model_info") or row
        if not isinstance(info, Mapping):
            return None
        window = next(
            (
                value
                for key in (
                    "max_input_tokens",
                    "context_length",
                    "context_window",
                    "max_model_len",
                )
                if (value := _positive_int(info.get(key))) is not None
            ),
            None,
        )
        # All deployments behind an alias must supply a limit. Use the smallest.
        if window is None:
            return None
        windows.append(window)
    return min(windows) if windows else None


def _openai_endpoint_context(model: BaseChatModel) -> None:
    from openai import APIConnectionError, APIError, APIStatusError

    client = model.root_client.with_options(timeout=_TIMEOUT, max_retries=0)
    base = client.base_url.path.rstrip("/")
    prefix = base.removesuffix("/v1")
    paths = dict.fromkeys([
        f"{base}/model/info",
        f"{prefix}/model/info",
        f"{prefix}/v1/model/info",
        f"{prefix}/model_group/info",
        f"{base}/models",
    ])
    denied = set()
    failure = "Endpoint context metadata unavailable"
    for path in paths:
        # Preserve reverse-proxy prefixes and the configured client's auth/TLS.
        url = str(client.base_url.copy_with(path=path))
        try:
            payload = client.get(url, cast_to=dict[str, Any])
        except APIConnectionError:
            failure = "Endpoint metadata connection failed or timed out"
            break  # Do not retry an unreachable server at every candidate path.
        except APIStatusError as exc:
            if exc.status_code in {401, 403}:
                denied.add(exc.status_code)
            continue
        except (APIError, ValueError):
            continue  # Optional metadata endpoints may be absent or forbidden.
        if (
            window := _advertised_window(payload, model.model_name)
        ) is not None:
            _set_context(model, window, f"Endpoint {path}")
            return
    # A public catalog cannot establish a custom deployment's actual window.
    if denied:
        failure += " (HTTP " + "/".join(map(str, sorted(denied))) + ")"
    _set_context(model, None, failure)


def refresh_ollama_context(
    model: object, *, inspect_parameters: bool = False
) -> None:
    """Read the allocated window; an unloaded model's architecture is not enough."""
    from langchain_ollama import ChatOllama

    if not isinstance(model, ChatOllama) or model.num_ctx is not None:
        return
    if getattr(model, "_ursa_context_source", None) in {
        "Configured profile",
        "Ollama model parameters",
    }:
        return
    # Use the existing transport to retain headers, TLS and host configuration.
    client = model._client._client
    if inspect_parameters:
        try:
            response = client.post(
                "/api/show", json={"model": model.model}, timeout=_TIMEOUT
            )
            response.raise_for_status()
            parameters = response.json().get("parameters", "")
            match = re.search(r"(?m)^\s*num_ctx\s+(\d+)\s*$", parameters)
            if match and (window := _positive_int(int(match[1]))) is not None:
                _set_context(model, window, "Ollama model parameters")
                return
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            pass
    try:
        response = client.get("/api/ps", timeout=_TIMEOUT)
        response.raise_for_status()
        name = model.model
        if ":" not in name.rsplit("/", 1)[-1]:
            name += ":latest"
        for row in response.json().get("models", []):
            if name in (row.get("name"), row.get("model")):
                if (
                    window := _positive_int(row.get("context_length"))
                ) is not None:
                    _set_context(model, window, "Ollama /api/ps")
                    return
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    _set_context(
        model,
        None,
        "Ollama runtime context unavailable; set num_ctx or run the model",
    )


def load_model_context(
    model: object, *, explicit_profile: bool = False
) -> None:
    """Enrich the runtime profile once; explicit configuration takes precedence."""
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    if not isinstance(model, BaseChatModel):
        return
    if explicit_profile and model_context_window(model) is not None:
        model._ursa_context_source = "Configured profile"
        return
    if isinstance(model, ChatOllama):
        refresh_ollama_context(model, inspect_parameters=True)
    elif isinstance(model, ChatOpenAI):
        if model.root_client.base_url.host == "api.openai.com":
            if model_context_window(model) is None:
                window = _openai_catalog().get(model.model_name)
                if window is not None:
                    _set_context(model, window, "models.dev OpenAI catalog")
                else:
                    model._ursa_context_source = (
                        "OpenAI model missing from available catalogs"
                    )
        else:
            _openai_endpoint_context(model)
