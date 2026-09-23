"""Edit the normal URSA user config without exposing or persisting API keys."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, SecretStr

from ursa.cli.config import UrsaConfig, deep_interp_env
from ursa.security import enforce_group_base_url_policy
from ursa.util.crossplatform import user_config_paths

from .credentials import CredentialStore, assert_no_raw_api_key

STARTER_CONFIG = {
    "inference_providers": {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": {"env": "OPENAI_API_KEY"},
        }
    },
    "llm_model": {
        "model": "openai:gpt-5.6-terra",
        "inference_provider": "openai",
        "reasoning": {"effort": "medium"},
    },
    "emb_model": {
        "model": "openai:text-embedding-3-large",
        "inference_provider": "openai",
    },
}


class ConfigConflictError(ValueError):
    pass


class ProviderEdit(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    base_url: str | None = None
    model_provider: str | None = None
    credential_mode: Literal["preserve", "environment", "keyring", "none"] = (
        "preserve"
    )
    api_key_env: str | None = None
    api_key: SecretStr | None = None


class DefaultModelEdit(BaseModel):
    model: str = Field(min_length=1, max_length=512)
    inference_provider: str | None = None
    base_url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class UserConfigEdit(BaseModel):
    revision: str
    providers: list[ProviderEdit] = Field(max_length=100)
    llm_model: DefaultModelEdit
    emb_model: DefaultModelEdit | None = None


def default_config_path() -> Path:
    paths = user_config_paths()
    return next(
        (path for path in reversed(paths) if path.is_file()),
        paths[-1] if paths else Path.home() / ".config/ursa/config.yaml",
    )


def _revision(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_url(value: str | None) -> str | None:
    value = (value or "").strip() or None
    if value is not None:
        parsed = urlsplit(deep_interp_env(value))
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError(
                "Base URL must be a complete http:// or https:// URL."
            )
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Keep credentials, query strings, and fragments out of the Base URL."
            )
    return value


def _model_view(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    data = deepcopy(raw)
    model = str(data.pop("model", ""))
    protocol = data.pop("model_provider", None)
    if protocol and ":" not in model:
        model = f"{protocol}:{model}"
    provider = data.pop("inference_provider", None)
    base_url = data.pop("base_url", None)
    # Existing model-specific credentials stay on the server, including literals.
    data.pop("api_key", None)
    data.pop("api_key_env", None)
    assert_no_raw_api_key(data, context="model options")
    return {
        "model": model,
        "inference_provider": provider,
        "base_url": base_url,
        "options": data,
    }


class UserConfigStore:
    def __init__(
        self,
        *,
        group: str,
        credential_store: CredentialStore,
        path: Path | None = None,
    ):
        self.path = path or default_config_path()
        self.group = group
        self.credentials = credential_store
        self.lock = RLock()

    def _read(self) -> tuple[bytes, dict[str, Any]]:
        content = self.path.read_bytes() if self.path.exists() else b""
        try:
            data = (
                yaml.safe_load(content) if content else deepcopy(STARTER_CONFIG)
            )
        except yaml.YAMLError as exc:
            raise ValueError(
                "The existing config is not valid YAML. Correct it before editing defaults."
            ) from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("The user config must be a YAML mapping.")
        return content, data

    def describe(self) -> dict[str, Any]:
        with self.lock:
            content, data = self._read()
            providers = []
            for name, raw in (data.get("inference_providers") or {}).items():
                ref = raw.get("api_key")
                env = (
                    ref.get("env")
                    if isinstance(ref, dict)
                    else raw.get("api_key_env")
                )
                providers.append({
                    "name": name,
                    "base_url": raw.get("base_url"),
                    "model_provider": raw.get("model_provider"),
                    "credential_mode": "preserve",
                    "api_key_env": env,
                    "credential_description": f"Environment: {env}"
                    if env
                    else "Existing key (moves to secure storage on Update)"
                    if isinstance(ref, str)
                    else "Configured key (kept unchanged)"
                    if ref
                    else "No API key configured",
                })
            return {
                "path": str(self.path),
                "exists": self.path.exists(),
                "revision": _revision(content),
                "providers": providers,
                "llm_model": _model_view(data.get("llm_model"))
                or _model_view(STARTER_CONFIG["llm_model"]),
                "emb_model": _model_view(data.get("emb_model")),
            }

    def prepare(
        self, request: UserConfigEdit
    ) -> tuple[dict[str, Any], dict[str, str]]:
        content, data = self._read()
        if request.revision != _revision(content):
            raise ConfigConflictError(
                "The config changed since you opened it. Reload the editor before saving."
            )
        providers = data.setdefault("inference_providers", {})
        pending_secrets: dict[str, str] = {}
        seen = set()
        for edit in request.providers:
            if edit.name in seen:
                raise ValueError("Provider names must be unique.")
            seen.add(edit.name)
            previous = deepcopy(providers.get(edit.name) or {})
            base_url = _validate_url(edit.base_url)
            if base_url != (previous.get("base_url") or None):
                enforce_group_base_url_policy(
                    deep_interp_env(base_url), self.group
                )
            if (
                (previous.get("base_url") or None) != base_url
                and edit.credential_mode == "preserve"
                and (previous.get("api_key") or previous.get("api_key_env"))
            ):
                raise ValueError(
                    f"Choose the API key source again for '{edit.name}' after changing its endpoint."
                )
            previous["base_url"] = base_url
            if edit.model_provider:
                previous["model_provider"] = edit.model_provider
            else:
                previous.pop("model_provider", None)
            if edit.credential_mode != "preserve":
                previous.pop("api_key_env", None)
                if edit.credential_mode == "environment":
                    env = (edit.api_key_env or "").strip()
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
                        raise ValueError(
                            "Enter a valid API key environment variable name."
                        )
                    previous["api_key"] = {"env": env}
                elif edit.credential_mode == "keyring":
                    value = (
                        edit.api_key.get_secret_value() if edit.api_key else ""
                    )
                    if not value.strip() or len(value) > 65_536:
                        raise ValueError("Enter an API key to save securely.")
                    # Unique, endpoint-specific references allow old sessions to
                    # retain their original key and allow safe config rollback.
                    key_id = f"provider:{edit.name}:{uuid.uuid4().hex}"
                    previous["api_key"] = {"keyring": key_id}
                    pending_secrets[key_id] = value
                else:
                    previous["api_key"] = None
            if isinstance(previous.get("api_key"), str):
                key_id = f"provider:{edit.name}:{uuid.uuid4().hex}"
                pending_secrets[key_id] = deep_interp_env(previous["api_key"])
                previous["api_key"] = {"keyring": key_id}
            providers[edit.name] = previous

        for field, edit in (
            ("llm_model", request.llm_model),
            ("emb_model", request.emb_model),
        ):
            if edit is None:
                data[field] = None
                continue
            options = deepcopy(edit.options)
            assert_no_raw_api_key(options, context="model options")
            if {
                "model",
                "model_provider",
                "base_url",
                "inference_provider",
                "api_key",
                "api_key_env",
            } & options.keys():
                raise ValueError(
                    "Use the model and provider fields above for endpoint and credential settings."
                )
            previous = data.get(field) or {}
            model = {**options, "model": edit.model.strip()}
            if not model["model"]:
                raise ValueError("Enter a model name.")
            provider = (edit.inference_provider or "").strip() or None
            if provider:
                model["inference_provider"] = provider
            else:
                model["inference_provider"] = None
                model["base_url"] = _validate_url(edit.base_url)
            # Preserve model-specific credentials only while retaining the same
            # provider/endpoint. Selecting a different provider uses its key.
            if provider == previous.get("inference_provider") and model.get(
                "base_url"
            ) == previous.get("base_url"):
                for key in ("api_key", "api_key_env"):
                    if key in previous:
                        model[key] = previous[key]
            data[field] = model
            if isinstance(model.get("api_key"), str):
                key_id = f"model:{field}:{uuid.uuid4().hex}"
                pending_secrets[key_id] = deep_interp_env(model["api_key"])
                model["api_key"] = {"keyring": key_id}
        return data, pending_secrets

    def save(self, request: UserConfigEdit, validate) -> dict[str, Any]:
        with self.lock:
            original, _ = self._read()
            data, secrets = self.prepare(request)
            validate(data)
            payload = yaml.safe_dump(
                data, sort_keys=False, allow_unicode=True
            ).encode()
            # Keep dotfile symlinks intact when a user's config is managed in
            # another folder. Atomically replace the target, not the symlink.
            destination = self.path.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            saved_keys = []
            temporary = None
            backup = None
            try:
                for name, value in secrets.items():
                    self.credentials.set_secret(name, value)
                    saved_keys.append(name)
                if original:
                    fd, name = tempfile.mkstemp(
                        prefix="config.backup-",
                        suffix=".yaml",
                        dir=destination.parent,
                    )
                    backup = Path(name)
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(original)
                fd, name = tempfile.mkstemp(
                    prefix=".config-", suffix=".yaml", dir=destination.parent
                )
                temporary = Path(name)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Detect external edits that occurred during validation/keyring IO.
                current = self.path.read_bytes() if self.path.exists() else b""
                if current != original:
                    raise ConfigConflictError(
                        "The config changed while saving. Reload before trying again."
                    )
                os.replace(temporary, destination)
            except Exception:
                for name in saved_keys:
                    with contextlib.suppress(Exception):
                        self.credentials.delete_secret(name)
                raise
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            result = self.describe()
            result["backup_path"] = str(backup) if backup else None
            return result


async def probe_model(
    config: UrsaConfig, kind: Literal["chat", "embedding"]
) -> None:
    """Send a minimal real request; never return provider output or secrets."""
    model = config.llm_model if kind == "chat" else config.emb_model
    if model is None:
        raise ValueError("No embedding model is configured.")
    updates = {"timeout": 20, "max_retries": 0}
    if kind == "chat":
        if model.model_provider in {"openai", "azure_openai"}:
            updates["max_completion_tokens"] = 256
        elif model.model_provider == "anthropic":
            updates["max_tokens"] = 256
        elif model.model_provider == "google_genai":
            updates["max_output_tokens"] = 256
        elif model.model_provider == "ollama":
            updates["num_predict"] = 256
    model = model.model_copy(update=updates)
    if kind == "chat":
        llm = await asyncio.to_thread(model.init_chat_model)
        await asyncio.wait_for(
            llm.ainvoke("Reply with just the word OK."), timeout=30
        )
    else:
        embedding = await asyncio.to_thread(model.init_embedding)
        await asyncio.wait_for(
            embedding.aembed_query("URSA connection test"), timeout=30
        )
