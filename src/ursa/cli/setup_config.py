"""Configuration services for guided setup.

This module owns setup's configuration-file boundaries.  It intentionally has
no Textual dependencies so loading, applying installer defaults, and saving
configuration do not depend on the setup UI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from keyring.errors import PasswordDeleteError
from pydantic import SecretStr, ValidationError

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    ModelConfig,
    UrsaConfig,
    deep_interp_env,
    load_config_file,
    save_config_file,
)
from ursa.util import get_keyring
from ursa.util.mcp import ServerParameters, parse_server_parameters
from ursa.util.secrets import (
    KEYRING_SERVICE,
    SecretReference,
    SecretTemplate,
)

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_SENSITIVE_NAME = re.compile(r"auth|token|secret|password|api.key", re.I)
_ENV_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_SETUP_DEFAULT_FIELDS = {
    "python_version",
    "extras",
    "extra_packages",
    "inference_providers",
    "llm_model",
    "emb_model",
    "mcp_servers",
}


class SetupError(ValueError):
    """A configuration error that setup can present without a traceback."""


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


def load_setup_config(path: Path) -> tuple[UrsaConfig, bool]:
    """Load setup's editable config without interpolating placeholders.

    The regular CLI supplies a default OpenAI provider.  Guided setup instead
    exposes only providers that were explicitly present in the editable file.
    The returned boolean records whether the language model was also explicit.
    """
    try:
        raw = (
            deepcopy(load_config_file(path, interpolate=False))
            if path.is_file()
            else {}
        )
        providers = raw.get("inference_providers", {})
        if not isinstance(providers, dict):
            raise ValueError("Existing inference_providers must be a mapping.")
        for value in providers.values():
            if isinstance(value, dict):
                # Setup always enables TLS verification.
                value.pop("ssl_verify", None)
        config = UrsaConfig.model_validate(raw)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise SetupError(f"Could not load configuration: {exc}") from None

    explicit = {
        str(name): config.inference_providers[str(name)] for name in providers
    }
    config = config.model_copy(update={"inference_providers": explicit})
    if overlap := sorted(set(explicit) & set(config.mcp_servers)):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")
    return config, "llm_model" in raw


def load_setup_defaults(path: Path) -> dict[str, Any]:
    """Load a non-secret installer payload used to seed guided setup."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SetupError(f"Could not read setup defaults: {exc}") from None
    except json.JSONDecodeError as exc:
        raise SetupError(
            f"Setup defaults are not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from None
    if not isinstance(value, dict):
        raise SetupError("Setup defaults must be a JSON object.")
    if unknown := sorted(set(value) - _SETUP_DEFAULT_FIELDS):
        raise SetupError(f"Unknown setup defaults: {unknown}")
    return value


def reject_literal_secrets(
    value: Any,
    *,
    allow_implicit_keyring_template: bool = False,
    error_message: str = (
        "Literal secrets cannot be saved; use an environment or keyring "
        "secret reference."
    ),
) -> None:
    """Reject secret-looking literals while allowing typed references.

    Installer defaults may opt into a template-only shorthand because
    ``apply_setup_defaults`` adds its implicit keyring source before parsing.
    Interactive advanced configuration requires that source explicitly.
    """

    def visit(item: Any, name: str = "") -> None:
        if name and _SENSITIVE_NAME.search(name):
            reference = SecretTemplate.maybe_validate(item)
            if isinstance(reference, SecretReference):
                return
            if isinstance(item, str) and _ENV_REFERENCE.fullmatch(item):
                return
            if (
                allow_implicit_keyring_template
                and isinstance(item, Mapping)
                and set(item) == {"template"}
            ):
                try:
                    SecretTemplate.model_validate({**item, "keyring": True})
                except ValidationError:
                    pass
                else:
                    return
            raise SetupError(error_message)
        if isinstance(item, Mapping):
            for child_name, child in item.items():
                visit(child, str(child_name))
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)


def _model_default(
    field: str,
    model_type: type[ModelConfig],
    value: Mapping[str, Any],
) -> ModelConfig:
    try:
        return model_type.model_validate(value)
    except ValidationError as exc:
        raise SetupError(
            f"Installer default {field} is invalid: {exc}"
        ) from None


def apply_setup_defaults(
    config: UrsaConfig,
    defaults: Mapping[str, Any],
    *,
    llm_configured: bool = False,
) -> bool:
    """Add site defaults without replacing entries in the user's config."""
    reject_literal_secrets(
        defaults,
        allow_implicit_keyring_template=True,
        error_message="Installer defaults must not contain API keys.",
    )
    providers = defaults.get("inference_providers", [])
    if not isinstance(providers, list):
        raise SetupError("inference_providers must be a JSON array.")
    seen: set[str] = set()
    for item in providers:
        if not isinstance(item, dict):
            raise SetupError(
                "Inference provider defaults must be JSON objects."
            )
        value = deepcopy(item)
        try:
            name = validate_setup_name(value.pop("name"), "Inference provider")
            value["base_url"] = validate_setup_url(
                value["base_url"], "Inference provider base URL"
            )
        except KeyError as exc:
            raise SetupError(
                f"Inference provider default is missing '{exc.args[0]}'."
            ) from None
        if name in seen:
            raise SetupError(
                f"Inference provider default '{name}' is duplicated."
            )
        seen.add(name)
        try:
            provider = InferenceProviderConfig.model_validate(value)
        except ValidationError as exc:
            raise SetupError(
                f"Inference provider default '{name}' is invalid: {exc}"
            ) from None
        if isinstance(provider.api_key, SecretStr):
            raise SetupError("Installer defaults must not contain API keys.")
        if name in config.inference_providers:
            continue
        if name in config.mcp_servers:
            raise SetupError(
                f"The name '{name}' is used by both provider and MCP."
            )
        if provider.api_key is None and urlparse(
            provider.base_url
        ).hostname not in {"localhost", "127.0.0.1", "::1"}:
            provider.api_key = SecretReference(keyring=True)
        config.inference_providers[name] = provider
        config.__pydantic_fields_set__.add("inference_providers")

    model_fields: tuple[tuple[str, type[ModelConfig]], ...] = (
        ("llm_model", ChatModelConfig),
        ("emb_model", EmbModelConfig),
    )
    for field, model_type in model_fields:
        if (
            field == "llm_model"
            and llm_configured
            or field == "emb_model"
            and config.emb_model is not None
        ):
            continue
        value = defaults.get(field)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise SetupError(
                f"Installer default {field} must be a JSON object."
            )
        provider_name = value.get("inference_provider")
        if (
            not isinstance(provider_name, str)
            or provider_name not in config.inference_providers
        ):
            continue
        model = _model_default(field, model_type, value)
        if isinstance(model.api_key, SecretStr):
            raise SetupError("Installer defaults must not contain API keys.")
        setattr(config, field, model)
        config.__pydantic_fields_set__.add(field)
        if field == "llm_model":
            llm_configured = True

    servers = defaults.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise SetupError("mcp_servers must be a JSON object.")
    for raw_name, raw_server in servers.items():
        name = validate_setup_name(str(raw_name), "MCP server")
        if name in config.mcp_servers:
            continue
        if name in config.inference_providers:
            raise SetupError(
                f"The name '{name}' is used by both provider and MCP."
            )
        if not isinstance(raw_server, dict):
            raise SetupError(f"MCP server '{name}' must be a JSON object.")
        server = deepcopy(raw_server)
        headers = server.get("headers")
        if isinstance(headers, dict):
            for header, raw_value in headers.items():
                if not isinstance(raw_value, dict):
                    continue
                if "template" in raw_value:
                    headers[header] = {
                        **raw_value,
                        **(
                            {}
                            if raw_value.get("env") is not None
                            or raw_value.get("keyring") not in (None, False)
                            else {"keyring": True}
                        ),
                    }
        try:
            config.mcp_servers[name] = parse_server_parameters(server)
        except (TypeError, ValueError, ValidationError) as exc:
            raise SetupError(
                f"MCP server default '{name}' is invalid: {exc}"
            ) from None
        config.__pydantic_fields_set__.add("mcp_servers")
    return llm_configured


def _validate_setup_candidate(
    config: UrsaConfig, *, check_model_providers: bool = False
) -> UrsaConfig:
    """Validate both setup invariants and canonical persisted config."""
    if overlap := sorted(
        set(config.inference_providers) & set(config.mcp_servers)
    ):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")
    if check_model_providers:
        for field in ("llm_model", "emb_model"):
            model = getattr(config, field)
            if model is None or model.inference_provider is None:
                continue
            if model.inference_provider not in config.inference_providers:
                raise SetupError(
                    f"{field} references unknown inference_provider "
                    f"'{model.inference_provider}'"
                )
    try:
        # Validate the complete in-memory model without broadening its explicit
        # field state. Persistability (including literal-secret rejection) is
        # enforced at the save boundary, after staged setup credentials have
        # been replaced by references.
        UrsaConfig.model_validate(config.model_dump(mode="python"))
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(f"Setup configuration is invalid: {exc}") from None
    return config


def replace_inference_providers(
    config: UrsaConfig,
    providers: Mapping[str, InferenceProviderConfig],
    *,
    renamed: Mapping[str, str] | None = None,
) -> UrsaConfig:
    """Atomically replace providers and retarget model references.

    A model follows an explicitly renamed provider.  References to removed
    providers fall back to the first replacement provider, matching setup's
    ordered provider editor.
    """
    if not providers:
        raise SetupError("At least one inference provider is required.")
    replacements: dict[str, InferenceProviderConfig] = {}
    for raw_name, provider in providers.items():
        name = validate_setup_name(raw_name, "Inference provider")
        try:
            replacements[name] = InferenceProviderConfig.model_validate(
                deepcopy(provider)
            )
        except ValidationError as exc:
            raise SetupError(
                f"Inference provider '{name}' is invalid: {exc}"
            ) from None
        if isinstance(replacements[name].api_key, SecretStr):
            raise SetupError("API keys must use an ENV or keyring reference.")
    if overlap := sorted(set(replacements) & set(config.mcp_servers)):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")

    rename_map = dict(renamed or {})
    for old_name, new_name in rename_map.items():
        if old_name not in config.inference_providers:
            raise SetupError(f"Cannot rename unknown provider '{old_name}'.")
        if new_name not in replacements:
            raise SetupError(
                f"Renamed provider '{new_name}' is not in the replacement set."
            )

    fallback = next(iter(replacements))

    def retarget(
        model: ChatModelConfig | EmbModelConfig | None,
    ) -> ChatModelConfig | EmbModelConfig | None:
        if model is None:
            return None
        current = model.inference_provider
        if current is None:
            return model.model_copy(deep=True)
        target = rename_map.get(current, current)
        if target not in replacements:
            target = fallback
        return model.model_copy(
            update={"inference_provider": target}, deep=True
        )

    llm_model = retarget(config.llm_model)
    emb_model = retarget(config.emb_model)
    assert isinstance(llm_model, ChatModelConfig)
    assert emb_model is None or isinstance(emb_model, EmbModelConfig)
    candidate = config.model_copy(
        update={
            "inference_providers": replacements,
            "llm_model": llm_model,
            "emb_model": emb_model,
        },
        deep=True,
    )
    candidate.__pydantic_fields_set__ = config.model_fields_set | {
        "inference_providers"
    }
    return _validate_setup_candidate(candidate, check_model_providers=True)


def replace_models(
    config: UrsaConfig,
    llm_model: ChatModelConfig,
    emb_model: EmbModelConfig | None,
) -> UrsaConfig:
    """Atomically replace setup's language and embedding models."""
    try:
        chat = ChatModelConfig.model_validate(
            deepcopy(llm_model.editable_values())
        )
        embedding = (
            EmbModelConfig.model_validate(deepcopy(emb_model.editable_values()))
            if emb_model is not None
            else None
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(f"Model configuration is invalid: {exc}") from None
    candidate = config.model_copy(
        update={"llm_model": chat, "emb_model": embedding}, deep=True
    )
    candidate.__pydantic_fields_set__ = config.model_fields_set | {
        "llm_model",
        "emb_model",
    }
    return _validate_setup_candidate(candidate, check_model_providers=True)


def parse_setup_model_yaml(
    text: str,
    model_type: type[ChatModelConfig] | type[EmbModelConfig],
    original: ChatModelConfig | EmbModelConfig,
    displayed_values: Mapping[str, Any],
    known_inference_providers: Collection[str],
) -> ChatModelConfig | EmbModelConfig | None:
    """Parse a complete setup model block through its canonical config type.

    Setup displays defaulted fields so the advanced editor is a complete
    configuration form.  Unchanged presentation-only defaults remain unset in
    the parsed model, while every field actually present in the original
    editable block is replaced by the YAML document.  Consequently, deleting
    a field from the document deletes it from the resulting configuration
    instead of merging the stale value back from ``original``.
    """
    try:
        values = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SetupError(
            f"Model configuration is invalid YAML: {exc}"
        ) from None

    if values is None and model_type is EmbModelConfig:
        return None
    if not isinstance(values, dict):
        raise SetupError("Model configuration must be a YAML mapping.")

    replacement = deepcopy(values)
    editable_fields = set(original.editable_values())
    missing = object()
    for field, displayed in displayed_values.items():
        if (
            field not in editable_fields
            and replacement.get(field, missing) == displayed
        ):
            replacement.pop(field)

    # SecretStr renders as a mask.  Preserve the actual value only while that
    # unchanged mask remains in the replacement document; deleting api_key
    # still removes it like any other field.
    if isinstance(original.api_key, SecretStr) and replacement.get(
        "api_key", missing
    ) == str(original.api_key):
        replacement["api_key"] = deepcopy(original.api_key)

    # A named provider owns its endpoint.  An explicit null produced when the
    # structured control switches endpoint styles must not mask that provider.
    if (
        replacement.get("inference_provider") is not None
        and replacement.get("base_url") is None
    ):
        replacement.pop("base_url", None)

    try:
        config = model_type.model_validate(replacement)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(f"Model configuration is invalid: {exc}") from None
    if not config.model.strip():
        noun = "Language" if model_type is ChatModelConfig else "Embedding"
        raise SetupError(f"{noun} model must not be blank")
    if (
        config.inference_provider is not None
        and config.inference_provider not in known_inference_providers
    ):
        raise SetupError(
            f"Unknown inference_provider '{config.inference_provider}'"
        )
    return config


def replace_mcp_servers(
    config: UrsaConfig,
    servers: Mapping[str, ServerParameters],
) -> UrsaConfig:
    """Atomically replace MCP servers using the canonical MCP parser."""
    replacements: dict[str, ServerParameters] = {}
    for raw_name, server in servers.items():
        name = validate_setup_name(raw_name, "MCP server")
        if name in config.inference_providers:
            raise SetupError(
                f"The name '{name}' is used by both provider and MCP."
            )
        try:
            replacements[name] = parse_server_parameters(server)
        except (TypeError, ValueError, ValidationError) as exc:
            raise SetupError(f"MCP server '{name}' is invalid: {exc}") from None
    candidate = config.model_copy(
        update={"mcp_servers": replacements}, deep=True
    )
    candidate.__pydantic_fields_set__ = config.model_fields_set | {
        "mcp_servers"
    }
    return _validate_setup_candidate(candidate)


def prepare_provider_probe(
    config: UrsaConfig,
    name: str,
    staged_secret: SecretStr | None = None,
) -> InferenceProviderConfig:
    """Build one runtime provider for a connectivity probe.

    Environment placeholders are interpolated, secret references are resolved
    unless a staged value is supplied, and a model provider is borrowed from a
    configured model that uses this named inference provider.
    """
    if name not in config.inference_providers:
        raise SetupError(f"Unknown inference provider '{name}'.")
    if staged_secret is not None and not isinstance(staged_secret, SecretStr):
        raise SetupError("A staged provider secret must be a SecretStr.")

    values = config.inference_providers[name].model_dump(mode="python")
    try:
        runtime = InferenceProviderConfig.model_validate(
            deep_interp_env(values)
        ).resolve(name)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(
            f"Inference provider '{name}' is invalid: {exc}"
        ) from None

    reference = runtime.api_key
    if staged_secret is not None:
        resolved: SecretStr | None = staged_secret
    elif isinstance(reference, SecretReference):
        try:
            resolved = reference.resolve(name)
        except Exception:
            resolved = None
    else:
        resolved = reference

    runtime_values = runtime.model_dump(mode="python")
    runtime_values["api_key"] = resolved
    if not runtime_values.get("model_provider"):
        model = next(
            (
                candidate
                for candidate in (config.llm_model, config.emb_model)
                if candidate is not None
                and candidate.inference_provider == name
            ),
            None,
        )
        if model is not None and model.model_provider is not None:
            runtime_values["model_provider"] = model.model_provider
    return InferenceProviderConfig.model_validate(runtime_values)


def format_setup_config_yaml(config: UrsaConfig) -> str:
    """Format the exact editable configuration shown on setup's review page."""
    try:
        values = config.editable_values()
        UrsaConfig.model_validate(values)
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    except (TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise SetupError(f"Could not format configuration: {exc}") from None


def save_setup_config(
    path: Path,
    config: UrsaConfig,
    pending: Mapping[str, str],
) -> Path:
    """Persist pending credentials and config as one rollback-safe operation."""
    if not config.inference_providers or not config.llm_model.model:
        raise SetupError("A provider and language model are required.")
    UrsaConfig.model_validate(config.editable_values())
    credential_store = get_keyring()
    previous = {
        username: credential_store.get_password(KEYRING_SERVICE, username)
        for username in pending
    }
    written: list[str] = []
    try:
        for username, secret in pending.items():
            credential_store.set_password(KEYRING_SERVICE, username, secret)
            written.append(username)
        return save_config_file(path, config)
    except BaseException as exc:
        for username in reversed(written):
            try:
                previous_password = previous[username]
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
