from __future__ import annotations

import importlib
import json
import os

# needed for SSL / PKI verifications, if the user needs that
import ssl
from typing import Any

import httpx
from langchain.chat_models import init_chat_model

"""
llm_factory.py

Utilities for constructing LangChain chat models for URSA runners using a YAML config.

This centralizes:
- provider alias resolution (e.g., "openai:gpt-5" or "my_endpoint:openai/gpt-oss-120b")
- auth and base_url wiring from cfg.models.providers
- merging of per-run defaults + optional YAML profiles + per-agent overrides
- safe(ish) logging banners that avoid printing secrets verbatim

Goal: any URSA program (plan/execute, hypothesizer, etc.) can share the same model
configuration behavior and get consistent logging and overrides.
"""


# ---------------------------------------------------------------------
# Secret masking / sanitization for logs
# ---------------------------------------------------------------------
_SECRET_KEY_SUBSTRS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "bearer",
)


def _looks_like_secret_key(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in _SECRET_KEY_SUBSTRS)


def _mask_secret(value: str, keep_start: int = 6, keep_end: int = 4) -> str:
    """
    Mask a secret-like string, keeping only the beginning and end.
    Example: sk-proj-abc123456789xyz -> sk-proj-...9xyz
    """
    if not isinstance(value, str):
        return value
    if len(value) <= keep_start + keep_end + 3:
        return "…"
    return f"{value[:keep_start]}...{value[-keep_end:]}"


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion to something json.dumps can handle."""
    # Primitives are fine
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Common containers
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    # Fallback: readable repr (keeps type info)
    return f"<{obj.__class__.__name__}>"


def _sanitize_for_logging(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _looks_like_secret_key(str(k)):
                out[k] = _mask_secret(v) if isinstance(v, str) else "..."
            else:
                out[k] = _sanitize_for_logging(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_logging(v) for v in obj]
    return _json_safe(obj)


# ---------------------------------------------------------------------
# Dict merge + YAML param resolution
# ---------------------------------------------------------------------
def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return a new dict.
    - dict + dict => deep merge
    - otherwise => override wins
    """
    base = dict(base or {})
    override = dict(override or {})
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_llm_kwargs_for_agent(
    models_cfg: dict | None, agent_name: str | None
) -> dict:
    """
    Given the YAML `models:` dict, compute merged kwargs for init_chat_model(...)
    for a specific agent ('planner', 'executor', etc.).

    Merge order (later wins):
      1) {} (empty)
      2) models.defaults.params (optional)
      3) models.profiles[defaults.profile] (optional)
      4) models.agents[agent_name].profile (optional; merges that profile on top)
      5) models.agents[agent_name].params (optional)
    """
    models_cfg = models_cfg or {}
    profiles = models_cfg.get("profiles") or {}
    defaults = models_cfg.get("defaults") or {}
    agents = models_cfg.get("agents") or {}

    merged: dict = {}
    merged = _deep_merge_dicts(merged, defaults.get("params") or {})

    default_profile_name = defaults.get("profile")
    if default_profile_name and default_profile_name in profiles:
        merged = _deep_merge_dicts(merged, profiles[default_profile_name] or {})

    if agent_name and isinstance(agents, dict) and agent_name in agents:
        a = agents.get(agent_name) or {}
        agent_profile_name = a.get("profile")
        if agent_profile_name and agent_profile_name in profiles:
            merged = _deep_merge_dicts(
                merged, profiles[agent_profile_name] or {}
            )
        merged = _deep_merge_dicts(merged, a.get("params") or {})

    return merged


# ---------------------------------------------------------------------
# Provider / model string resolution
# ---------------------------------------------------------------------
def _resolve_model_choice(
    model_choice: str, models_cfg: dict
) -> tuple[str, str, dict]:
    """
    Accepts strings like:
      - 'openai:gpt-5.2'
      - 'my_endpoint:openai/gpt-oss-120b'

    Looks up per-provider settings from cfg.models.providers.

    Returns:
      (model_provider, pure_model, provider_extra_kwargs_for_init)

    where:
      - model_provider is a LangChain provider string (e.g., "openai")
      - pure_model is the model name passed as `model=...`
      - provider_extra_kwargs may include base_url/api_key
    """
    if ":" in model_choice:
        alias, pure_model = model_choice.split(":", 1)
    else:
        alias, pure_model = "openai", model_choice  # back-compat default

    providers = (models_cfg or {}).get("providers", {})
    prov = providers.get(alias, {})

    model_provider = prov.get("model_provider", alias)

    api_key = None
    if prov.get("api_key_env"):
        api_key = os.getenv(prov["api_key_env"])
    if not api_key and prov.get("token_loader"):
        mod, fn = prov["token_loader"].rsplit(".", 1)
        api_key = getattr(importlib.import_module(mod), fn)()

    provider_extra = {}
    if prov.get("base_url"):
        provider_extra["base_url"] = prov["base_url"]
    if api_key:
        provider_extra["api_key"] = api_key

    return model_provider, pure_model, provider_extra


# ---------------------------------------------------------------------
# Logging banner
# ---------------------------------------------------------------------
def _print_llm_init_banner(
    *,
    agent_name: str | None,
    provider: str,
    model_name: str,
    provider_extra: dict,
    llm_kwargs: dict,
    model_obj: Any = None,
    console: Any | None = None,
) -> None:
    """
    Print a friendly summary of the init_chat_model(...) configuration.

    If `console` is a rich Console, we render Panels. Otherwise we print plain text.
    """
    who = agent_name or "llm"
    safe_provider_extra = _sanitize_for_logging(provider_extra or {})
    safe_llm_kwargs = _sanitize_for_logging(llm_kwargs or {})

    text = (
        f"LLM init ({who})\n"
        f"provider: {provider}\n"
        f"model: {model_name}\n\n"
        f"provider kwargs: {json.dumps(safe_provider_extra, indent=2)}\n\n"
        f"llm kwargs (merged): {json.dumps(safe_llm_kwargs, indent=2)}"
    )

    if console is not None:
        try:
            from rich.panel import Panel
            from rich.text import Text

            console.print(
                Panel.fit(
                    Text.from_markup(text.replace("\n", "\n")),
                    border_style="cyan",
                )
            )
        except Exception:
            print(text)
    else:
        print(text)

    # Best-effort readback from the LangChain model object
    if model_obj is not None:
        readback = {}
        for attr in (
            "model_name",
            "model",
            "reasoning",
            "temperature",
            "max_completion_tokens",
            "max_tokens",
        ):
            if hasattr(model_obj, attr):
                try:
                    readback[attr] = getattr(model_obj, attr)
                except Exception:
                    pass

        for attr in ("model_kwargs", "kwargs"):
            if hasattr(model_obj, attr):
                try:
                    readback[attr] = getattr(model_obj, attr)
                except Exception:
                    pass

        if readback:
            rb_text = (
                "LLM readback (best-effort from LangChain object)\n"
                + json.dumps(_sanitize_for_logging(readback), indent=2)
            )
            if console is not None:
                try:
                    from rich.panel import Panel
                    from rich.text import Text

                    console.print(
                        Panel.fit(
                            Text.from_markup(rb_text), border_style="green"
                        )
                    )
                except Exception:
                    print(rb_text)
            else:
                print(rb_text)

    # If reasoning effort was requested, highlight it
    effort = None
    try:
        effort = (llm_kwargs or {}).get("reasoning", {}).get("effort")
    except Exception:
        effort = None

    if effort:
        msg = (
            f"Reasoning effort requested: {effort}\n"
            "Note: This confirms what we sent to init_chat_model; actual enforcement is provider-side."
        )
        if console is not None:
            try:
                from rich.panel import Panel

                console.print(Panel.fit(msg, border_style="yellow"))
            except Exception:
                print(msg)
        else:
            print(msg)


def _maybe_add_system_trust_httpx_clients(
    provider: str, provider_extra: dict
) -> dict:
    """
    If we're using OpenAI-compatible provider + running on macOS corporate PKI,
    use system trust store via truststore to avoid certifi/conda OpenSSL issues.
    """
    if provider != "openai":
        return provider_extra

    # Don't override if caller already provided custom clients
    if "http_client" in provider_extra or "http_async_client" in provider_extra:
        return provider_extra

    try:
        import truststore  # pip install truststore
    except Exception:
        return provider_extra

    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # Provide both sync + async so invoke() and ainvoke() both work.
    provider_extra = dict(provider_extra or {})
    provider_extra["http_client"] = httpx.Client(verify=ctx, trust_env=False)
    provider_extra["http_async_client"] = httpx.AsyncClient(
        verify=ctx, trust_env=False
    )
    return provider_extra


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def resolve_model_choice(
    model_choice: str, models_cfg: dict
) -> tuple[str, str, dict]:
    """
    Public wrapper around the internal model/provider resolver.

    Accepts strings like 'openai:gpt-5.2' or 'my_endpoint:openai/gpt-oss-120b'
    and returns:
      (model_provider, pure_model, provider_extra_kwargs)

    provider_extra_kwargs may include base_url/api_key.
    """
    return _resolve_model_choice(model_choice, models_cfg)


def setup_llm(
    *,
    model_choice: str,
    models_cfg: dict | None = None,
    agent_name: str | None = None,
    base_llm_kwargs: dict | None = None,
    console: Any | None = None,
):
    """
    Build a LangChain chat model via init_chat_model(...), applying YAML-driven params.

    - `model_choice`: e.g. "openai:gpt-5" or "my_endpoint:openai/gpt-oss-120b"
    - `models_cfg`: cfg.models dict
    - `agent_name`: "planner" / "executor" / "hypothesizer" etc. (applies per-agent overrides)
    - `base_llm_kwargs`: default kwargs that apply before YAML overrides
    - `console`: optional rich console for pretty banners

    Behavior matches your runner:
      base defaults < YAML overrides
    """
    models_cfg = models_cfg or {}

    provider, pure_model, provider_extra = _resolve_model_choice(
        model_choice, models_cfg
    )
    provider_extra = _maybe_add_system_trust_httpx_clients(
        provider, provider_extra
    )

    # Preserve your existing hardcoded defaults by default
    default_base = {
        "max_completion_tokens": 10000,
        "max_retries": 2,
    }
    base = _deep_merge_dicts(default_base, base_llm_kwargs or {})

    yaml_llm_kwargs = _resolve_llm_kwargs_for_agent(models_cfg, agent_name)
    llm_kwargs = _deep_merge_dicts(base, yaml_llm_kwargs)

    model = init_chat_model(
        model=pure_model,
        model_provider=provider,
        **llm_kwargs,
        **(provider_extra or {}),
    )

    _print_llm_init_banner(
        agent_name=agent_name,
        provider=provider,
        model_name=pure_model,
        provider_extra=provider_extra,
        llm_kwargs=llm_kwargs,
        model_obj=model,
        console=console,
    )

    return model
