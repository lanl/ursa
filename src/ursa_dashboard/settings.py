from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator

from ursa.cli.config import (
    UrsaConfig,
    deep_interp_env,
    load_config_file,
    resolve_ursa_config,
    system_config_paths,
)
from ursa.security import enforce_group_base_url_policy
from ursa.util.crossplatform import user_config_paths

from .credentials import assert_no_raw_api_key, credential_target
from .storage import read_json, utc_now, write_json


def _qualified_model_name(model_config) -> str:
    if model_config.model_provider is None:
        return model_config.model
    return f"{model_config.model_provider}:{model_config.model}"


class LLMSettings(BaseModel):
    model: str = "openai:gpt-5.5"
    base_url: str | None = None
    inference_provider: str | None = None

    # Security: settings contain only a credential source/reference. Stored
    # keys live in the OS credential store; environment variables remain an
    # explicit compatibility option.
    api_key_env: str | None = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable that contains the LLM API key (the secret is not stored).",
    )
    credential_source: Literal["environment", "stored", "keyring", "none"] = (
        "stored"
    )
    api_key_keyring: str | None = None
    credential_id: str | None = None
    credential_target: str | None = None

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
        assert_no_raw_api_key(v, context="llm.model_kwargs")
        return v

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if v == "":
            return None
        # Conservative env-var name validation.
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            raise ValueError(
                "api_key_env must be a valid environment variable name"
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


class EmbeddingSettings(BaseModel):
    """Dashboard-level embedding model settings for RAG features."""

    model: str | None = None
    base_url: str | None = None
    inference_provider: str | None = None
    api_key_env: str | None = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable that contains the embedding API key (the secret is not stored).",
    )
    credential_source: Literal[
        "environment", "stored", "keyring", "llm", "none"
    ] = "stored"
    api_key_keyring: str | None = None
    credential_id: str | None = None
    credential_target: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_kwargs")
    @classmethod
    def _validate_model_kwargs(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("embedding.model_kwargs must be a JSON object")
        assert_no_raw_api_key(v, context="embedding.model_kwargs")
        return v

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if v == "":
            return None
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", v):
            raise ValueError(
                "api_key_env must be a valid environment variable name"
            )
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
    walkthrough_seen: bool = False


class GlobalSettings(BaseModel):
    """Global settings that apply to new runs only."""

    updated_at: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    ui: UISettings = Field(default_factory=UISettings)


def dashboard_environment_config_layer() -> dict[str, Any]:
    """Return sparse standard URSA environment-variable overrides."""
    from ursa.cli import build_parser

    parsed = build_parser().parse_args(args=[], defaults=False)
    values = parsed.as_dict()
    return {
        key: deepcopy(value)
        for key, value in values.items()
        if key in UrsaConfig.model_fields
    }


def _dashboard_settings_config_layer(
    settings: GlobalSettings,
) -> dict[str, Any]:
    """Translate the dashboard-owned model settings into an URSA layer."""

    def model_layer(config: LLMSettings | EmbeddingSettings) -> dict[str, Any]:
        model = str(config.model or "").strip()
        values: dict[str, Any] = {
            "model": model,
            **deepcopy(config.model_kwargs),
        }
        values["inference_provider"] = config.inference_provider
        if not config.inference_provider and config.base_url:
            values["base_url"] = config.base_url
        if config.credential_source == "environment" and config.api_key_env:
            values["api_key"] = {"env": config.api_key_env}
        elif config.credential_source == "keyring" and config.api_key_keyring:
            values["api_key"] = {"keyring": config.api_key_keyring}
        return values

    layer: dict[str, Any] = {
        "llm_model": model_layer(settings.llm),
        "mcp_servers": deepcopy(settings.mcp.servers),
        "rag_tools": list(settings.tools.rag_tools),
    }
    layer["emb_model"] = (
        model_layer(settings.embedding) if settings.embedding.model else None
    )
    return layer


def _resolve_dashboard_model_config(
    config: UrsaConfig, *, group: str
) -> UrsaConfig:
    """Resolve model/provider inheritance without creating a CLI workspace."""
    providers = {
        name: provider.resolve(name)
        for name, provider in config.inference_providers.items()
    }

    def resolve_model(model):
        if model is None:
            return None
        if model.inference_provider is not None:
            # A higher-precedence named provider clears a lower direct URL.
            # Do not let that cleared lower field block provider inheritance.
            model = model.model_copy()
            model.__pydantic_fields_set__ = model.model_fields_set - {
                "base_url"
            }
        return model.resolve_inference_provider(providers)

    llm = resolve_model(config.llm_model)
    embedding = (
        resolve_model(config.emb_model)
        if config.emb_model is not None
        else None
    )
    enforce_group_base_url_policy(llm.base_url, group)
    if embedding is not None:
        enforce_group_base_url_policy(embedding.base_url, group)
    return config.model_copy(
        update={
            "group": group,
            "inference_providers": providers,
            "llm_model": llm,
            "emb_model": embedding,
        }
    )


def _apply_ursa_models_to_dashboard_settings(
    settings: GlobalSettings, config: UrsaConfig
) -> GlobalSettings:
    """Overlay effective URSA model values without persisting secrets."""
    data = settings.model_dump(mode="python")

    def apply_model(
        section: dict[str, Any], model_config, *, embedding: bool
    ) -> None:
        section["model"] = _qualified_model_name(model_config)
        section["base_url"] = model_config.base_url
        section["inference_provider"] = model_config.inference_provider
        kwargs = deepcopy(model_config.model_extra or {})
        if not embedding and model_config.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = model_config.max_completion_tokens
        section["model_kwargs"] = kwargs
        if model_config.api_key_env:
            section["api_key_env"] = model_config.api_key_env
            section["credential_source"] = "environment"
            section["credential_id"] = None
            section["credential_target"] = None
            section["api_key_keyring"] = None
        elif getattr(model_config.api_key, "keyring", None):
            section["credential_source"] = "keyring"
            section["api_key_env"] = None
            section["api_key_keyring"] = model_config.api_key.keyring
            section["credential_id"] = None
            section["credential_target"] = credential_target(section)
        elif (
            "api_key" in model_config.model_fields_set
            and model_config.api_key is None
        ):
            section["credential_source"] = "none"
            section["api_key_env"] = None
            section["api_key_keyring"] = None
            section["credential_id"] = None
            section["credential_target"] = None

    apply_model(data["llm"], config.llm_model, embedding=False)
    if config.emb_model is None:
        data["embedding"]["model"] = None
        data["embedding"]["base_url"] = None
        data["embedding"]["inference_provider"] = None
        data["embedding"]["model_kwargs"] = {}
    else:
        apply_model(data["embedding"], config.emb_model, embedding=True)
    config_dump = config.model_dump(
        mode="json", context={"include_defaults": False}
    )
    data["mcp"]["servers"] = config_dump.get("mcp_servers") or {}
    data["tools"]["rag_tools"] = list(config.rag_tools or [])
    return GlobalSettings.model_validate(data)


class DashboardConfigResolver:
    """Resolve dashboard settings in the normal URSA precedence hierarchy."""

    def __init__(
        self,
        *,
        group: str,
        explicit_config: str | Path | None = None,
    ) -> None:
        self.group = group
        explicit_path = (
            Path(explicit_config).expanduser()
            if explicit_config is not None
            else None
        )
        self.explicit_path = explicit_path

        def load_existing(paths: list[Path]) -> list[dict[str, Any]]:
            return [
                load_config_file(path.expanduser())
                for path in paths
                if path.expanduser() != explicit_path
                and path.expanduser().is_file()
            ]

        self.system_layers = load_existing(system_config_paths())
        self.user_layers = load_existing(user_config_paths())
        self.environment_layer = dashboard_environment_config_layer()
        self.explicit_layer = (
            load_config_file(explicit_path)
            if explicit_path is not None
            else None
        )

    def resolve(
        self,
        settings: GlobalSettings,
        *,
        user_override: tuple[Path, dict[str, Any]] | None = None,
    ) -> tuple[GlobalSettings, UrsaConfig]:
        user_layers = self.user_layers
        explicit_layer = self.explicit_layer
        if user_override:
            path, data = user_override
            data = deep_interp_env(data)
            user_layers = [
                deepcopy(data)
                if item.expanduser() == path
                else load_config_file(item.expanduser())
                for item in user_config_paths()
                if item.expanduser() != self.explicit_path
                and (item.expanduser() == path or item.expanduser().is_file())
            ]
            if path == self.explicit_path:
                explicit_layer = data
        layers = [
            *self.system_layers,
            _dashboard_settings_config_layer(settings),
            *user_layers,
            self.environment_layer,
        ]
        if explicit_layer is not None:
            layers.append(explicit_layer)
        # Selecting a named provider in a higher layer selects that provider's
        # credential too; a stale dashboard env/keyring reference must not win.
        catalog = (
            UrsaConfig()
            .model_merge(*[
                {"inference_providers": layer["inference_providers"]}
                for layer in layers
                if "inference_providers" in layer
            ])
            .inference_providers
        )
        layers = deepcopy(layers)
        for layer in layers:
            for name in ("llm_model", "emb_model"):
                model = layer.get(name)
                if (
                    isinstance(model, dict)
                    and model.get("inference_provider")
                    and "api_key" not in model
                    and "api_key_env" not in model
                ):
                    provider = catalog.get(model["inference_provider"])
                    if provider is not None:
                        model["api_key"] = provider.resolve(
                            model["inference_provider"]
                        ).api_key
        # Merge every sparse source before validating. A lower-priority
        # dashboard preference may legitimately reference a provider declared
        # by a later user, environment, or launch layer. UrsaConfig.model_merge
        # also treats an explicit ``emb_model: null`` as a replacement, so the
        # dashboard can disable an embedding model inherited from the system
        # configuration without validating incomplete intermediate states.
        config = UrsaConfig().model_merge(*layers)
        config = _resolve_dashboard_model_config(config, group=self.group)
        return _apply_ursa_models_to_dashboard_settings(
            settings, config
        ), config


def dashboard_llm_patch_from_ursa_config(
    path: str | Path,
    *,
    group: str,
    current: GlobalSettings | None = None,
) -> dict[str, Any]:
    """Return a dashboard settings patch from a CLI-style URSA config.

    The dashboard intentionally stores only non-secret LLM settings.
    raw API keys are rejected so they are not persisted in settings.json.
    Additional ``llm_model`` fields accepted by the CLI config are passed
    through as dashboard ``llm.model_kwargs`` except for fields that have a
    first-class dashboard setting.
    """

    cfg = UrsaConfig.from_file(Path(path))
    llm_cfg = cfg.llm_model
    if isinstance(llm_cfg.api_key, SecretStr):
        raise ValueError(
            "Dashboard config does not store raw llm_model.api_key; "
            "use an environment or dashboard credential instead."
        )
    if (
        current is not None
        and llm_cfg.inference_provider is None
        and "base_url" not in llm_cfg.model_fields_set
    ):
        llm_cfg = llm_cfg.model_copy(update={"base_url": current.llm.base_url})
    emb_cfg = cfg.emb_model
    if emb_cfg is not None and isinstance(emb_cfg.api_key, SecretStr):
        raise ValueError(
            "Dashboard config does not store raw emb_model.api_key; "
            "use an environment or dashboard credential instead."
        )
    if (
        current is not None
        and emb_cfg is not None
        and emb_cfg.inference_provider is None
        and "base_url" not in emb_cfg.model_fields_set
    ):
        emb_cfg = emb_cfg.model_copy(
            update={"base_url": current.embedding.base_url}
        )
    cfg = cfg.model_copy(
        update={"group": group, "llm_model": llm_cfg, "emb_model": emb_cfg}
    )
    cfg = resolve_ursa_config(cfg)
    llm_cfg = cfg.llm_model
    llm_api_key_env = llm_cfg.api_key_env

    patch: dict[str, Any] = {"model": _qualified_model_name(llm_cfg)}
    if llm_cfg.inference_provider is not None:
        patch["inference_provider"] = llm_cfg.inference_provider
    if llm_cfg.base_url is not None:
        patch["base_url"] = llm_cfg.base_url
    if llm_api_key_env is not None:
        patch["api_key_env"] = llm_api_key_env
        patch["credential_source"] = "environment"
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

    out: dict[str, Any] = {"llm": patch}

    emb_cfg = cfg.emb_model
    if emb_cfg is not None:
        emb_api_key_env = emb_cfg.api_key_env
        emb_patch: dict[str, Any] = {"model": _qualified_model_name(emb_cfg)}
        if emb_cfg.inference_provider is not None:
            emb_patch["inference_provider"] = emb_cfg.inference_provider
        if emb_cfg.base_url is not None:
            emb_patch["base_url"] = emb_cfg.base_url
        if emb_api_key_env is not None:
            emb_patch["api_key_env"] = emb_api_key_env
            emb_patch["credential_source"] = "environment"

        emb_model_kwargs: dict[str, Any] = {}
        for key, value in (emb_cfg.model_extra or {}).items():
            if value is None:
                continue
            if key == "api_key":
                raise ValueError(
                    "Dashboard config does not store raw emb_model.api_key; "
                    "use emb_model.api_key_env instead."
                )
            if key == "model_kwargs":
                if not isinstance(value, dict):
                    raise ValueError("emb_model.model_kwargs must be an object")
                emb_model_kwargs.update(value)
                continue
            emb_model_kwargs[key] = value

        if emb_model_kwargs:
            emb_patch["model_kwargs"] = emb_model_kwargs

        EmbeddingSettings.model_validate(emb_patch)
        out["embedding"] = emb_patch

    return out


def merge_global_settings_patch(
    current: GlobalSettings, patch_obj: dict[str, Any]
) -> GlobalSettings:
    """Return settings with the dashboard PATCH deep-merge semantics."""

    merged = current.model_dump(mode="json")

    # Important: our PATCH endpoint uses deep-merge semantics so callers can
    # update individual nested fields. However, for some objects we want
    # *replace* semantics so deletions are respected.
    REPLACE_PATHS = {
        "mcp.servers",
        "llm.model_kwargs",
        "embedding.model_kwargs",
    }

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

    current = settings_store.load()
    patch = dashboard_llm_patch_from_ursa_config(
        path, group=group, current=current
    )
    settings = merge_global_settings_patch(current, patch)
    enforce_group_base_url_policy(settings.llm.base_url, group)
    if settings.embedding.model:
        enforce_group_base_url_policy(settings.embedding.base_url, group)
    settings_store.save(settings)
    return settings


class SettingsStore:
    def __init__(self, dashboard_root: Path):
        self.dashboard_root = dashboard_root
        self.path = self.dashboard_root / "_meta" / "settings.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> GlobalSettings:
        if not self.path.exists():
            s = GlobalSettings(updated_at=utc_now())
            self.save(s)
            return s
        data = read_json(self.path)
        # Settings created before secure storage existed used api_key_env.
        # Preserve that behavior while letting fresh installs default to the
        # operating system credential store.
        for section in ("llm", "embedding"):
            section_data = data.get(section)
            if (
                isinstance(section_data, dict)
                and "credential_source" not in section_data
            ):
                section_data["credential_source"] = (
                    "environment" if section_data.get("api_key_env") else "none"
                )
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
