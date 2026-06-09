from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ursa.cli.config import UrsaConfig
from ursa.security import enforce_group_base_url_policy

from .storage import read_json, utc_now, write_json


class LLMSettings(BaseModel):
    model: str = "openai:gpt-5.2"
    base_url: str | None = None

    # Security: we intentionally do *not* store an API key in settings.json.
    # Instead, we store the *name* of an environment variable that contains
    # the key. The worker copies that value into OPENAI_API_KEY at runtime.
    api_key_env_var: str | None = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable that contains the LLM API key (the secret is not stored).",
    )

    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to langchain.chat_models.init_chat_model.",
    )

    @field_validator("model_kwargs")
    @classmethod
    def _validate_model_kwargs(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("llm.model_kwargs must be a JSON object")
        return v

    @field_validator("api_key_env_var")
    @classmethod
    def _validate_api_key_env_var(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if v == "":
            return None
        # Conservative env-var name validation.
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            raise ValueError(
                "api_key_env_var must be a valid environment variable name"
            )
        return v


class RunnerSettings(BaseModel):
    timeout_seconds: int | None = None


class MCPSettings(BaseModel):
    """Configuration for MCP servers whose tools should be attached to agents.

    The value of `servers` is passed to `ursa.util.mcp.start_mcp_client()`.
    """

    enabled: bool = True
    servers: dict[str, Any] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def _validate_servers(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(
                "mcp.servers must be an object mapping server_name -> server_config"
            )
        # Light validation: keys are server names; values must be objects.
        for name, cfg in v.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("mcp.servers keys must be non-empty strings")
            if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", name):
                raise ValueError(f"Invalid MCP server name: {name!r}")
            if not isinstance(cfg, dict):
                raise ValueError(f"mcp.servers[{name!r}] must be an object")
        return v


class ToolSettings(BaseModel):
    """Dashboard-level tools attached to new tool-capable agent runs."""

    rag_tools: list[str] = Field(default_factory=list)

    @field_validator("rag_tools", mode="before")
    @classmethod
    def _normalize_rag_tools(cls, v: Any) -> list[str]:
        if v is None:
            return []
        from ursa.rag.persistence import normalize_rag_tool_names

        return normalize_rag_tool_names(v)


class UISettings(BaseModel):
    theme: str = "system"  # e.g. dark/light mode
    stdout_buffer_lines: int = Field(default=20_000, ge=5_000, le=100_000_000)


class GlobalSettings(BaseModel):
    """Global settings that apply to new runs only."""

    updated_at: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    ui: UISettings = Field(default_factory=UISettings)


def dashboard_llm_patch_from_ursa_config(path: str | Path) -> dict[str, Any]:
    """Return a dashboard settings patch from a CLI-style URSA config.

    The dashboard intentionally stores only non-secret LLM settings. CLI
    ``llm_model.api_key_env`` is mapped to dashboard ``llm.api_key_env_var``;
    raw API keys are rejected so they are not persisted in settings.json.
    Additional ``llm_model`` fields accepted by the CLI config are passed
    through as dashboard ``llm.model_kwargs`` except for fields that have a
    first-class dashboard setting.
    """

    cfg = UrsaConfig.from_file(Path(path))
    llm_cfg = cfg.llm_model

    patch: dict[str, Any] = {"model": llm_cfg.model}
    if llm_cfg.base_url is not None:
        patch["base_url"] = llm_cfg.base_url
    if llm_cfg.api_key_env is not None:
        patch["api_key_env_var"] = llm_cfg.api_key_env
    if llm_cfg.max_completion_tokens is not None:
        patch["max_tokens"] = llm_cfg.max_completion_tokens

    model_kwargs: dict[str, Any] = {}
    for key, value in (llm_cfg.model_extra or {}).items():
        if value is None:
            continue
        if key == "api_key":
            raise ValueError(
                "Dashboard config does not store raw llm_model.api_key; "
                "use llm_model.api_key_env instead."
            )
        if key == "temperature":
            patch["temperature"] = value
            continue
        if key == "model_kwargs":
            if not isinstance(value, dict):
                raise ValueError("llm_model.model_kwargs must be an object")
            model_kwargs.update(value)
            continue
        # CLI configs may include provider-specific kwargs such as timeout,
        # seed, or use_responses_api. The dashboard worker forwards these via
        # init_chat_model(**model_kwargs).
        model_kwargs[key] = value

    if model_kwargs:
        patch["model_kwargs"] = model_kwargs

    # Validate against the dashboard settings schema before returning, so CLI
    # errors fail early instead of being deferred until the first run.
    LLMSettings.model_validate(patch)
    return {"llm": patch}


def merge_global_settings_patch(
    current: GlobalSettings, patch_obj: dict[str, Any]
) -> GlobalSettings:
    """Return settings with the dashboard PATCH deep-merge semantics."""

    merged = current.model_dump(mode="json")

    # Important: our PATCH endpoint uses deep-merge semantics so callers can
    # update individual nested fields. However, for some objects we want
    # *replace* semantics so deletions are respected.
    REPLACE_PATHS = {"mcp.servers", "llm.model_kwargs"}

    def deep_merge(
        dst: dict[str, Any], src: dict[str, Any], path: str = ""
    ) -> dict[str, Any]:
        for k, v in src.items():
            p = f"{path}.{k}" if path else str(k)

            # Replace semantics for specific paths (e.g. mcp.servers).
            if p in REPLACE_PATHS and isinstance(v, dict):
                dst[k] = v
                continue

            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                dst[k] = deep_merge(dst[k], v, p)
            else:
                dst[k] = v
        return dst

    return GlobalSettings.model_validate(deep_merge(merged, patch_obj))


def apply_dashboard_config(
    settings_store: "SettingsStore", path: str | Path, *, group: str
) -> GlobalSettings:
    """Apply a CLI-style YAML/JSON config to dashboard global settings.

    The resulting effective endpoint is validated against the selected group
    before it is persisted, matching the startup check performed by the
    dashboard app.
    """

    patch = dashboard_llm_patch_from_ursa_config(path)
    settings = merge_global_settings_patch(settings_store.load(), patch)
    enforce_group_base_url_policy(settings.llm.base_url, group)
    settings_store.save(settings)
    return settings


class SettingsStore:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.path = self.workspace_root / "_meta" / "settings.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> GlobalSettings:
        if not self.path.exists():
            s = GlobalSettings(updated_at=utc_now())
            self.save(s)
            return s
        data = read_json(self.path)
        return GlobalSettings.model_validate(data)

    def save(self, settings: GlobalSettings) -> None:
        settings.updated_at = utc_now()
        write_json(self.path, settings.model_dump(mode="json"))

    def patch(self, patch_obj: dict[str, Any]) -> GlobalSettings:
        new_settings = merge_global_settings_patch(self.load(), patch_obj)
        self.save(new_settings)
        return new_settings


class AuthConfig(BaseModel):
    mode: str = Field(default="local", description="local or remote")
    token: str | None = Field(
        default=None, description="Bearer token required in remote mode"
    )
    cors_origins: list[str] = Field(default_factory=list)

    @classmethod
    def from_env(cls) -> "AuthConfig":
        mode = os.environ.get("URSA_DASHBOARD_MODE")
        if not mode:
            mode = (
                "remote"
                if os.environ.get("URSA_DASHBOARD_REMOTE")
                in {"1", "true", "TRUE", "yes"}
                else "local"
            )
        token = os.environ.get("URSA_DASHBOARD_TOKEN")
        cors = os.environ.get("URSA_DASHBOARD_CORS_ORIGINS", "").strip()
        cors_origins = (
            [o.strip() for o in cors.split(",") if o.strip()] if cors else []
        )
        return cls(mode=mode, token=token, cors_origins=cors_origins)
