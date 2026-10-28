"""Configuration-file boundary for guided setup."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from contextlib import contextmanager, suppress
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from keyring.errors import PasswordDeleteError
from pydantic import SecretStr

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
    deep_interp_env,
    load_config_file,
    save_config_file,
)
from ursa.util import get_keyring
from ursa.util.mcp import ServerParameters
from ursa.util.secrets import KEYRING_SERVICE, SecretReference, SecretTemplate

# The compact layout keeps this setup-specific boundary proportional to the
# canonical config machinery it delegates to.

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_SENSITIVE_NAME = re.compile(r"auth|token|secret|password|api.key", re.I)
_ENV_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SETUP_DEFAULT_FIELDS = frozenset(
    "python_version extras extra_packages inference_providers "
    "llm_model emb_model mcp_servers".split()
)


class SetupError(ValueError):
    """A configuration error that setup can present without a traceback."""


@contextmanager
def _setup_errors(message: str, errors=(TypeError, ValueError)):
    try:
        yield
    except errors as exc:
        raise SetupError(f"{message}: {exc}") from None


def validate_setup_name(value: Any, noun: str) -> str:
    if not isinstance(value, str):
        raise SetupError(f"{noun} must be a string.")
    if not _NAME.fullmatch(value := value.strip()):
        raise SetupError(
            f"{noun} must start with a letter and contain only letters, "
            "numbers, '-' or '_'."
        )
    return value


def validate_setup_url(value: Any, noun: str) -> str:
    if isinstance(value, str):
        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return value
    raise SetupError(f"{noun} must be a complete HTTP URL.")


def _setup_view(config: UrsaConfig, names: Collection[str]) -> UrsaConfig:
    providers = {
        name: config.inference_providers[name] for name in map(str, names)
    }
    config = config.model_copy(update={"inference_providers": providers})
    if overlap := sorted(providers.keys() & config.mcp_servers.keys()):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")
    return config


def load_setup_config(path: Path) -> tuple[UrsaConfig, bool]:
    with _setup_errors(
        "Could not load configuration", (OSError, ValueError, yaml.YAMLError)
    ):
        raw = (
            load_config_file(path, interpolate=False) if path.is_file() else {}
        )
        providers = raw.get("inference_providers", {})
        if isinstance(providers, Mapping):
            for provider in providers.values():
                if isinstance(provider, dict):
                    provider.pop(
                        "ssl_verify", None
                    )  # Setup always verifies TLS.
        config = UrsaConfig.model_validate(raw)
    return _setup_view(config, providers), "llm_model" in raw


def load_setup_defaults(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SetupError(f"Could not read setup defaults: {exc}") from None
    except json.JSONDecodeError as exc:
        raise SetupError(f"Setup defaults are not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise SetupError("Setup defaults must be a JSON object.")
    if unknown := sorted(value.keys() - _SETUP_DEFAULT_FIELDS):
        raise SetupError(f"Unknown setup defaults: {unknown}")
    return value


def reject_literal_secrets(
    value: Any,
    *,
    default_templates: bool = False,
    error_message: str = "Literal secrets cannot be saved; use an environment or keyring secret reference.",
) -> None:
    stack = [("", value)]
    while stack:
        name, item = stack.pop()
        if (
            default_templates
            and isinstance(item, dict)
            and "template" in item
            and item.get("env") is None
            and item.get("keyring") in (None, False)
        ):
            item["keyring"] = True
        if name and _SENSITIVE_NAME.search(name):
            reference = SecretTemplate.maybe_validate(item)
            if not (
                isinstance(reference, SecretReference)
                or isinstance(item, str)
                and _ENV_REFERENCE.fullmatch(item)
            ):
                raise SetupError(error_message)
        elif isinstance(item, Mapping):
            stack.extend((str(key), child) for key, child in item.items())
        elif isinstance(item, list):
            stack.extend(("", child) for child in item)


def _replace_config(config: UrsaConfig, **updates: Any) -> UrsaConfig:
    with _setup_errors("Setup configuration is invalid"):
        values = config.model_dump(
            mode="python",
            exclude_unset=True,
            context={"include_defaults": False},
        )
        candidate = UrsaConfig.model_validate(values | updates)
        names = updates.get("inference_providers", config.inference_providers)
    return _setup_view(candidate, names)


def apply_setup_defaults(
    config: UrsaConfig,
    defaults: Mapping[str, Any],
    *,
    llm_configured: bool = False,
) -> bool:
    defaults = deepcopy(defaults)
    servers = defaults.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise SetupError("mcp_servers must be a JSON object.")
    secret_error = "Installer defaults must not contain API keys."
    for server in servers.values():
        headers = server.get("headers") if isinstance(server, dict) else None
        if isinstance(headers, dict):
            reject_literal_secrets(
                headers, default_templates=True, error_message=secret_error
            )
    reject_literal_secrets(defaults, error_message=secret_error)

    raw_providers = defaults.get("inference_providers", [])
    if not isinstance(raw_providers, list):
        raise SetupError("inference_providers must be a JSON array.")
    providers = dict(config.inference_providers)
    seen: set[str] = set()
    for item in raw_providers:
        if not isinstance(item, dict):
            raise SetupError("Provider defaults must be JSON objects.")
        value = dict(item)
        name = validate_setup_name(
            value.pop("name", None), "Inference provider"
        )
        value["base_url"] = validate_setup_url(
            value.get("base_url"), "Inference provider base URL"
        )
        if name in seen:
            raise SetupError(f"Provider default '{name}' is duplicated.")
        seen.add(name)
        with _setup_errors(f"Inference provider default '{name}' is invalid"):
            provider = InferenceProviderConfig.model_validate(value)
        if name in config.inference_providers:
            continue
        if (
            provider.api_key is None
            and urlparse(provider.base_url).hostname not in _LOCAL_HOSTS
        ):
            provider = provider.model_copy(
                update={"api_key": SecretReference(keyring=True)}
            )
        providers[name] = provider

    updates: dict[str, Any] = {}
    if len(providers) != len(config.inference_providers):
        updates["inference_providers"] = providers
    for field, configured in (
        ("llm_model", llm_configured),
        ("emb_model", config.emb_model is not None),
    ):
        value = None if configured else defaults.get(field)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise SetupError(f"Installer default {field} must be an object.")
        provider_name = value.get("inference_provider")
        if not isinstance(provider_name, str) or provider_name not in providers:
            continue
        updates[field] = value
        llm_configured |= field == "llm_model"

    new_servers = {
        validate_setup_name(str(name), "MCP server"): server
        for name, server in servers.items()
        if name not in config.mcp_servers
    }
    if new_servers:
        updates["mcp_servers"] = config.mcp_servers | new_servers
    if updates:
        candidate = _replace_config(config, **updates)
        config.__dict__.update(
            (field, getattr(candidate, field)) for field in updates
        )
        config.__pydantic_fields_set__.update(updates)
    return llm_configured


def replace_inference_providers(
    config: UrsaConfig,
    providers: Mapping[str, InferenceProviderConfig],
    *,
    renamed: Mapping[str, str] | None = None,
) -> UrsaConfig:
    if not providers:
        raise SetupError("At least one inference provider is required.")
    replacements = {
        validate_setup_name(name, "Inference provider"): deepcopy(provider)
        for name, provider in providers.items()
    }
    if any(
        isinstance(provider.api_key, SecretStr)
        for provider in replacements.values()
    ):
        raise SetupError("API keys must use an ENV or keyring reference.")
    rename_map = dict(renamed or {})
    if unknown := rename_map.keys() - config.inference_providers.keys():
        raise SetupError(
            f"Cannot rename unknown provider '{next(iter(unknown))}'."
        )
    if missing := set(rename_map.values()) - replacements.keys():
        raise SetupError(
            f"Renamed provider '{next(iter(missing))}' is not in the replacement set."
        )
    fallback = next(iter(replacements))

    def retarget(model):
        if model is None:
            return None
        current = model.inference_provider
        target = rename_map.get(current, current) if current else None
        target = target if target in replacements else fallback
        return model.model_copy(
            update={"inference_provider": target} if current else {}, deep=True
        )

    return _replace_config(
        config,
        inference_providers=replacements,
        llm_model=retarget(config.llm_model),
        emb_model=retarget(config.emb_model),
    )


def replace_models(
    config: UrsaConfig,
    llm_model: ChatModelConfig,
    emb_model: EmbModelConfig | None,
) -> UrsaConfig:
    return _replace_config(
        config,
        llm_model=llm_model.editable_values(),
        emb_model=emb_model.editable_values() if emb_model else None,
    )


def parse_setup_model_yaml(
    text: str,
    model_type: type[ChatModelConfig] | type[EmbModelConfig],
    original: ChatModelConfig | EmbModelConfig,
    displayed_values: Mapping[str, Any],
    known_inference_providers: Collection[str],
) -> ChatModelConfig | EmbModelConfig | None:
    with _setup_errors("Invalid model YAML", (yaml.YAMLError,)):
        values = yaml.safe_load(text)
    if values is None and model_type is EmbModelConfig:
        return None
    if not isinstance(values, dict):
        raise SetupError("Model configuration must be a YAML mapping.")

    missing = object()
    editable_fields = original.editable_values().keys()
    replacement = {
        field: value
        for field, value in values.items()
        if field in editable_fields
        or value != displayed_values.get(field, missing)
    }
    if isinstance(original.api_key, SecretStr) and replacement.get(
        "api_key", missing
    ) == str(original.api_key):
        replacement["api_key"] = deepcopy(original.api_key)
    if (
        replacement.get("inference_provider") is not None
        and replacement.get("base_url") is None
    ):
        replacement.pop("base_url", None)
    with _setup_errors("Model configuration is invalid"):
        model = model_type.model_validate(replacement)
    if not model.model.strip():
        kind = "Language" if model_type is ChatModelConfig else "Embedding"
        raise SetupError(f"{kind} model must not be blank")
    if model.inference_provider not in {*known_inference_providers, None}:
        raise SetupError(
            f"Unknown inference_provider '{model.inference_provider}'"
        )
    return model


def replace_mcp_servers(
    config: UrsaConfig, servers: Mapping[str, ServerParameters]
) -> UrsaConfig:
    replacements = {
        validate_setup_name(name, "MCP server"): deepcopy(server)
        for name, server in servers.items()
    }
    return _replace_config(config, mcp_servers=replacements)


def prepare_provider_probe(
    config: UrsaConfig, name: str, staged_secret: SecretStr | None = None
) -> InferenceProviderConfig:
    if name not in config.inference_providers:
        raise SetupError(f"Unknown inference provider '{name}'.")
    if staged_secret is not None and not isinstance(staged_secret, SecretStr):
        raise SetupError("A staged provider secret must be a SecretStr.")
    values = config.inference_providers[name].model_dump(mode="python")
    with _setup_errors(f"Invalid inference provider '{name}'"):
        runtime = InferenceProviderConfig.model_validate(
            deep_interp_env(values)
        ).resolve(name)
    resolved = runtime.api_key if staged_secret is None else staged_secret
    if isinstance(resolved, SecretReference):
        with suppress(Exception):
            resolved = resolved.resolve(name)
    model_provider = getattr(runtime, "model_provider", None) or next(
        (
            model.model_provider
            for model in (config.llm_model, config.emb_model)
            if model and model.inference_provider == name
        ),
        None,
    )
    return runtime.model_copy(
        update={"api_key": resolved, "model_provider": model_provider}
    )


def format_setup_config_yaml(config: UrsaConfig) -> str:
    with _setup_errors(
        "Could not format configuration",
        (TypeError, ValueError, yaml.YAMLError),
    ):
        values = config.editable_values()
        UrsaConfig.model_validate(values)
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)


def save_setup_config(
    path: Path, config: UrsaConfig, pending: Mapping[str, str]
) -> Path:
    if not config.inference_providers or not config.llm_model.model:
        raise SetupError("A provider and language model are required.")
    UrsaConfig.model_validate(config.editable_values())
    store = get_keyring()
    previous: dict[str, str | None] = {}
    try:
        for username, secret in pending.items():
            previous[username] = store.get_password(KEYRING_SERVICE, username)
            store.set_password(KEYRING_SERVICE, username, secret)
        return save_config_file(path, config)
    except BaseException as exc:
        for username, password in reversed(previous.items()):
            try:
                if password is None:
                    store.delete_password(KEYRING_SERVICE, username)
                else:
                    store.set_password(KEYRING_SERVICE, username, password)
            except PasswordDeleteError:
                pass
            except Exception as rollback_error:
                exc.add_note(
                    f"Could not restore keyring entry '{username}': {rollback_error}"
                )
        raise
