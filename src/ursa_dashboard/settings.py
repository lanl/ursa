from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

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

    max_tokens: int = 25000
    temperature: float = 0.2

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


class GlobalSettings(BaseModel):
    """Global settings that apply to new runs only."""

    updated_at: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)


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
        current = self.load()
        merged = current.model_dump(mode="json")

        # Important: our PATCH endpoint uses deep-merge semantics so callers can
        # update individual nested fields. However, for some objects we want
        # *replace* semantics so deletions are respected.
        #
        # Example: the UI sends the full desired `mcp.servers` mapping.
        # If we deep-merge, removed servers will never be deleted from disk.
        REPLACE_PATHS = {"mcp.servers"}

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

        merged = deep_merge(merged, patch_obj)
        new_settings = GlobalSettings.model_validate(merged)
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
