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
from ursa.util.secrets import (
    KEYRING_SERVICE,
    SecretReference,
    SecretTemplate,
)

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
    """Normalize and validate a provider or MCP entry name."""
    if not isinstance(value, str):
        raise SetupError(f"{noun} must be a string.")
    value = value.strip()
    if not _NAME.fullmatch(value):
        raise SetupError(
            f"{noun} must start with a letter and contain only letters, "
            "numbers, '-' or '_'."
        )
    return value


def validate_setup_url(value: Any, noun: str) -> str:
    """Normalize and validate a complete HTTP(S) setup URL."""
    if not isinstance(value, str):
        raise SetupError(f"{noun} must be a complete HTTP URL.")
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SetupError(f"{noun} must be a complete HTTP URL.")
    return value


def _setup_view(
    config: UrsaConfig, provider_names: Collection[str]
) -> UrsaConfig:
    providers = {
        name: config.inference_providers[name]
        for name in map(str, provider_names)
    }
    config = config.model_copy(update={"inference_providers": providers})
    if overlap := sorted(set(providers) & set(config.mcp_servers)):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")
    return config


def load_setup_config(path: Path) -> tuple[UrsaConfig, bool]:
    """Load editable config and report whether its LLM was explicit."""
    with _setup_errors(
        "Could not load configuration", (OSError, ValueError, yaml.YAMLError)
    ):
        raw = (
            load_config_file(path, interpolate=False) if path.is_file() else {}
        )
        providers = raw.get("inference_providers", {})
        if isinstance(providers, Mapping):
            for value in providers.values():
                if isinstance(value, dict):
                    # Setup always enables TLS verification.
                    value.pop("ssl_verify", None)
        config = UrsaConfig.model_validate(raw)
    return _setup_view(config, providers), "llm_model" in raw


def load_setup_defaults(path: Path) -> dict[str, Any]:
    """Load a non-secret installer payload used to seed guided setup."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SetupError(f"Could not read setup defaults: {exc}") from None
    except json.JSONDecodeError as exc:
        raise SetupError(f"Setup defaults are not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise SetupError("Setup defaults must be a JSON object.")
    if unknown := sorted(set(value) - _SETUP_DEFAULT_FIELDS):
        raise SetupError(f"Unknown setup defaults: {unknown}")
    return value


def reject_literal_secrets(
    value: Any,
    *,
    default_templates: bool = False,
    error_message: str = (
        "Literal secrets cannot be saved; use an environment or keyring "
        "secret reference."
    ),
) -> None:
    """Reject secret-looking literals while allowing typed references."""
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
            if not (
                isinstance(SecretTemplate.maybe_validate(item), SecretReference)
                or isinstance(item, str)
                and _ENV_REFERENCE.fullmatch(item)
            ):
                raise SetupError(error_message)
            continue
        if isinstance(item, Mapping):
            stack.extend((str(key), child) for key, child in item.items())
        elif isinstance(item, list):
            stack.extend(("", child) for child in item)


def _replace_config(config: UrsaConfig, **updates: Any) -> UrsaConfig:
    with _setup_errors("Setup configuration is invalid"):
        candidate = UrsaConfig.model_validate(
            config.model_dump(
                mode="python",
                exclude_unset=True,
                context={"include_defaults": False},
            )
            | updates
        )
        provider_names = updates.get(
            "inference_providers", config.inference_providers
        )
    return _setup_view(candidate, provider_names)


def apply_setup_defaults(
    config: UrsaConfig,
    defaults: Mapping[str, Any],
    *,
    llm_configured: bool = False,
) -> bool:
    """Add site defaults without replacing entries in the user's config."""
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
    provider_values = dict(config.inference_providers)
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
        provider_values[name] = provider

    updates: dict[str, Any] = {}
    if len(provider_values) != len(config.inference_providers):
        updates["inference_providers"] = provider_values

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
        if (
            not isinstance(provider_name, str)
            or provider_name not in provider_values
        ):
            continue
        updates[field] = value
        if field == "llm_model":
            llm_configured = True

    new_servers = {
        validate_setup_name(str(name), "MCP server"): server
        for name, server in servers.items()
        if name not in config.mcp_servers
    }

    if new_servers:
        updates["mcp_servers"] = config.mcp_servers | new_servers
    if not updates:
        return llm_configured
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
    """Replace providers, retargeting renamed or removed references."""
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
    """Atomically replace setup's language and embedding models."""
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
    """Parse a complete model block, omitting unchanged display defaults."""
    with _setup_errors("Invalid model YAML", (yaml.YAMLError,)):
        values = yaml.safe_load(text)

    if values is None and model_type is EmbModelConfig:
        return None
    if not isinstance(values, dict):
        raise SetupError("Model configuration must be a YAML mapping.")

    editable_fields = set(original.editable_values())
    missing = object()
    replacement = {
        field: value
        for field, value in values.items()
        if field in editable_fields
        or value != displayed_values.get(field, missing)
    }

    # Preserve a SecretStr while its unchanged display mask remains.
    if isinstance(original.api_key, SecretStr) and replacement.get(
        "api_key", missing
    ) == str(original.api_key):
        replacement["api_key"] = deepcopy(original.api_key)

    # A named provider owns its endpoint, so discard the editor's null URL.
    if (
        replacement.get("inference_provider") is not None
        and replacement.get("base_url") is None
    ):
        replacement.pop("base_url", None)

    with _setup_errors("Model configuration is invalid"):
        config = model_type.model_validate(replacement)
    if not config.model.strip():
        raise SetupError(
            f"{'Language' if model_type is ChatModelConfig else 'Embedding'} "
            "model must not be blank"
        )
    provider = config.inference_provider
    if provider is not None and provider not in known_inference_providers:
        raise SetupError(f"Unknown inference_provider '{provider}'")
    return config


def replace_mcp_servers(
    config: UrsaConfig, servers: Mapping[str, ServerParameters]
) -> UrsaConfig:
    """Atomically replace MCP servers using the canonical MCP parser."""
    replacements = {
        validate_setup_name(name, "MCP server"): deepcopy(server)
        for name, server in servers.items()
    }
    return _replace_config(config, mcp_servers=replacements)


def prepare_provider_probe(
    config: UrsaConfig, name: str, staged_secret: SecretStr | None = None
) -> InferenceProviderConfig:
    """Resolve one provider and staged secret for a connectivity probe."""
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
    """Format the exact editable configuration shown on setup's review page."""
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
    """Persist pending credentials and config as one rollback-safe operation."""
    if not config.inference_providers or not config.llm_model.model:
        raise SetupError("A provider and language model are required.")
    UrsaConfig.model_validate(config.editable_values())
    credential_store = get_keyring()
    previous: dict[str, str | None] = {}
    try:
        for username, secret in pending.items():
            previous[username] = credential_store.get_password(
                KEYRING_SERVICE, username
            )
            credential_store.set_password(KEYRING_SERVICE, username, secret)
        return save_config_file(path, config)
    except BaseException as exc:
        for username, previous_password in reversed(previous.items()):
            try:
                if previous_password is None:
                    credential_store.delete_password(KEYRING_SERVICE, username)
                else:
                    credential_store.set_password(
                        KEYRING_SERVICE, username, previous_password
                    )
            except PasswordDeleteError:
                pass
            except Exception as rollback_error:
                exc.add_note(
                    f"Could not restore keyring entry '{username}': "
                    f"{rollback_error}"
                )
        raise
