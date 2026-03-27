from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model

"""
llm_factory.py

Utilities for constructing LangChain chat models for URSA runners using a YAML config.

This centralizes:
- provider alias resolution (e.g., "openai:gpt-5" or "my_endpoint:openai/gpt-oss-120b")
- auth and base_url wiring from cfg.models.providers
- merging of per-run defaults + optional YAML profiles + per-agent overrides
- best-effort logging banners that redact known secret-like fields

Goal: any URSA program (plan/execute, hypothesizer, etc.) can share the same model
configuration behavior and get consistent logging and overrides.
"""


# ---------------------------------------------------------------------
# Secret masking / sanitization for logs
# ---------------------------------------------------------------------
def _looks_like_secret_key(name: str) -> bool:
    n = name.lower().replace("-", "_").replace(" ", "_")

    exact = {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "passwd",
        "authorization",
        "private_key",
    }
    if n in exact:
        return True

    return (
        n.endswith("_api_key")
        or n.endswith("_token")
        or n.endswith("_secret")
        or n.endswith("_password")
    )


def _mask_secret(value: Any) -> str:
    """
    Fully redact secret-like values for logging.
    """
    return "[REDACTED]"


def _sanitize_for_logging(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _looks_like_secret_key(str(k)):
                out[k] = _mask_secret(v)
            else:
                out[k] = _sanitize_for_logging(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_for_logging(v) for v in obj]
    return obj


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    model_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "model_provider": self.provider,
            **self.model_extra,
        }

    def __str__(self) -> str:
        safe_model_extra = _sanitize_for_logging(self.model_extra or {})

        text = (
            f"provider: {self.provider}\n"
            f"model: {self.model_name}\n\n"
            f"model kwargs: {json.dumps(safe_model_extra, indent=2)}"
        )

        effort = None
        try:
            effort = (self.model_extra or {}).get("reasoning", {}).get("effort")
        except Exception:
            effort = None

        if effort:
            text += (
                f"\n\nReasoning effort requested: {effort}\n"
                "Note: This confirms what we sent to init_chat_model; actual enforcement is provider-side."
            )

        return text


# ---------------------------------------------------------------------
# Logging banner
# ---------------------------------------------------------------------
def llm_init_banner(
    llm: ModelConfig,
    agent_name: str | None = None,
    model_obj: Any = None,
) -> str:
    who = agent_name or "llm"
    text = f"LLM init ({who})\n{llm}"

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
            text += (
                "\n\nLLM readback (best-effort from LangChain object)\n"
                + json.dumps(_sanitize_for_logging(readback), indent=2)
            )

    return text


def _print_llm_init_banner(
    llm: ModelConfig,
    *,
    agent_name: str | None = None,
    model_obj: Any = None,
    console: Any | None = None,
) -> None:
    """
    Render the formatted LLM init banner to either rich console output or plain text.
    """
    banner_text = llm_init_banner(
        llm=llm,
        agent_name=agent_name,
        model_obj=model_obj,
    )

    if console is not None:
        try:
            from rich.panel import Panel
            from rich.text import Text

            # Split the sections into separate panels for readability.
            parts = banner_text.split("\n\n")

            if parts:
                console.print(
                    Panel.fit(
                        Text.from_markup(
                            parts[0]
                            + ("\n\n" + parts[1] if len(parts) > 1 else "")
                        ),
                        border_style="cyan",
                    )
                )

            for part in parts[2:]:
                border = "green"
                if part.startswith("Reasoning effort requested:"):
                    border = "yellow"
                console.print(
                    Panel.fit(
                        Text.from_markup(part),
                        border_style=border,
                    )
                )
        except Exception:
            print(banner_text)
    else:
        print(banner_text)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def setup_llm(
    *,
    model_config,
    agent_name: str | None = None,
    console: Any | None = None,
):
    """
    Build a LangChain chat model from a resolved ModelConfig.
    """
    model = init_chat_model(**model_config.kwargs)

    _print_llm_init_banner(
        model_config,
        agent_name=agent_name,
        model_obj=model,
        console=console,
    )

    return model
