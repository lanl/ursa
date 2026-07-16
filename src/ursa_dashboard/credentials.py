from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Literal, Protocol
from urllib.parse import urlsplit

from ursa.security import normalize_base_url, validate_group_name

CredentialKind = Literal["llm", "embedding"]
CredentialSource = Literal["environment", "stored", "llm", "none"]

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STORED_CREDENTIAL_VERSION = 1


class CredentialStoreError(RuntimeError):
    """Raised when the configured credential backend cannot be used."""


class CredentialConfigurationError(ValueError):
    """Raised when credential metadata is missing or unsafe."""


class CredentialStore(Protocol):
    def set_secret(self, credential_id: str, value: str) -> None: ...

    def get_secret(self, credential_id: str) -> str | None: ...

    def delete_secret(self, credential_id: str) -> None: ...


class KeyringCredentialStore:
    """Store dashboard credentials in the operating system credential store."""

    def __init__(self, *, service_name: str = "ursa-ai") -> None:
        self.service_name = service_name

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as e:  # pragma: no cover - installation failure
            raise CredentialStoreError(
                "Secure credential storage is unavailable. Install the "
                "dashboard credential-store dependency or use an environment "
                "variable."
            ) from e
        return keyring

    def set_secret(self, credential_id: str, value: str) -> None:
        try:
            self._keyring().set_password(
                self.service_name, credential_id, value
            )
        except Exception as e:
            raise CredentialStoreError(
                "The operating system credential store could not save the key."
            ) from e

    def get_secret(self, credential_id: str) -> str | None:
        try:
            return self._keyring().get_password(
                self.service_name, credential_id
            )
        except Exception as e:
            raise CredentialStoreError(
                "The operating system credential store could not read the key."
            ) from e

    def delete_secret(self, credential_id: str) -> None:
        keyring = self._keyring()
        try:
            if keyring.get_password(self.service_name, credential_id) is None:
                return
            keyring.delete_password(self.service_name, credential_id)
        except Exception as e:
            raise CredentialStoreError(
                "The operating system credential store could not remove the key."
            ) from e


class MemoryCredentialStore:
    """In-memory credential store for tests and embedded integrations."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def set_secret(self, credential_id: str, value: str) -> None:
        self._values[credential_id] = value

    def get_secret(self, credential_id: str) -> str | None:
        return self._values.get(credential_id)

    def delete_secret(self, credential_id: str) -> None:
        self._values.pop(credential_id, None)


def credential_id(group: str, kind: CredentialKind) -> str:
    """Return the only keyring identifier allowed for a dashboard slot."""
    return f"dashboard:{validate_group_name(group)}:{kind}"


def credential_target(config: Mapping[str, object]) -> str:
    """Bind a credential to an endpoint origin or model provider."""
    base_url = normalize_base_url(
        str(config.get("base_url")) if config.get("base_url") else None
    )
    if base_url:
        parsed = urlsplit(base_url)
        if parsed.scheme and parsed.netloc:
            host = (parsed.hostname or "").lower()
            port = f":{parsed.port}" if parsed.port else ""
            return f"origin:{parsed.scheme.lower()}://{host}{port}"
        return f"endpoint:{base_url}"

    model = str(config.get("model") or "openai").strip()
    provider = model.split(":", 1)[0] if ":" in model else "openai"
    return f"provider:{provider.lower()}"


def store_api_key(
    store: CredentialStore,
    *,
    credential_id: str,
    target: str,
    value: str,
) -> None:
    """Store the key and its trusted endpoint binding as one secure record."""
    record = {
        "version": _STORED_CREDENTIAL_VERSION,
        "target": target,
        "value": value,
    }
    store.set_secret(credential_id, json.dumps(record, separators=(",", ":")))


def read_api_key(
    store: CredentialStore, *, credential_id: str
) -> tuple[str, str] | None:
    """Read and validate a bound key from the secure credential store."""
    raw = store.get_secret(credential_id)
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise CredentialConfigurationError(
            "The saved credential record is invalid. Save the API key again."
        ) from e
    if (
        not isinstance(record, dict)
        or record.get("version") != _STORED_CREDENTIAL_VERSION
        or not isinstance(record.get("target"), str)
        or not isinstance(record.get("value"), str)
        or not record["value"]
    ):
        raise CredentialConfigurationError(
            "The saved credential record is invalid. Save the API key again."
        )
    return record["value"], record["target"]


def effective_credential_source(
    config: Mapping[str, object],
) -> CredentialSource:
    value = str(config.get("credential_source") or "").strip().lower()
    if not value:
        # Backward compatibility for settings created before credential_source.
        return "environment" if config.get("api_key_env") else "none"
    if value not in {"environment", "stored", "llm", "none"}:
        raise CredentialConfigurationError(
            f"Unknown credential source: {value}"
        )
    return value  # type: ignore[return-value]


def assert_no_raw_api_key(
    config: Mapping[str, object], *, context: str
) -> None:
    """Reject literal API keys in objects that will be persisted or returned."""
    for key, value in config.items():
        normalized = str(key).strip().lower()
        if normalized in {"api_key", "apikey", "api-key"}:
            raise CredentialConfigurationError(
                f"{context} must not contain a literal API key; use the "
                "credential endpoint or api_key_env instead."
            )
        if isinstance(value, Mapping):
            assert_no_raw_api_key(value, context=context)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping):
                    assert_no_raw_api_key(item, context=context)


def assert_no_credential_metadata(
    config: Mapping[str, object], *, context: str
) -> None:
    """Keep secure-store references under server control."""
    if {"credential_id", "credential_target"}.intersection(config):
        raise CredentialConfigurationError(
            f"{context} contains server-managed credential metadata."
        )


def resolve_api_key(
    config: Mapping[str, object],
    *,
    group: str,
    kind: CredentialKind,
    store: CredentialStore,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a key without adding it to persistent configuration."""
    assert_no_raw_api_key(config, context=kind)
    source = effective_credential_source(config)
    if source == "none":
        return None

    if source == "environment":
        env_name = str(config.get("api_key_env") or "").strip()
        if not env_name:
            return None
        if not _ENV_NAME_RE.fullmatch(env_name):
            raise CredentialConfigurationError(
                f"Invalid {kind} API-key environment variable name."
            )
        value = (environ or os.environ).get(env_name)
        if not value:
            raise CredentialConfigurationError(
                f"{kind.capitalize()} API-key environment variable "
                f"'{env_name}' is not set."
            )
        return value

    if source == "llm":
        if kind != "embedding":
            raise CredentialConfigurationError(
                "Only embedding settings may reuse the LLM credential."
            )
        stored = read_api_key(store, credential_id=credential_id(group, "llm"))
        if not stored:
            raise CredentialConfigurationError(
                "No stored LLM API key is available to reuse."
            )
        value, trusted_target = stored
        expected_target = credential_target(config)
        if trusted_target != expected_target:
            raise CredentialConfigurationError(
                "The stored LLM credential is not approved for the embedding "
                "endpoint. Save a separate embedding key or use the same endpoint."
            )
        return value

    expected_id = credential_id(group, kind)
    configured_id = str(config.get("credential_id") or "")
    if configured_id != expected_id:
        raise CredentialConfigurationError(
            f"The stored {kind} credential reference is invalid."
        )

    expected_target = credential_target(config)
    bound_target = str(config.get("credential_target") or "")
    if not bound_target or bound_target != expected_target:
        raise CredentialConfigurationError(
            f"The stored {kind} credential is not approved for the current "
            "model endpoint. Save the key again after reviewing the endpoint."
        )

    stored = read_api_key(store, credential_id=expected_id)
    if not stored:
        raise CredentialConfigurationError(
            f"No stored {kind} API key is available. Save one in Settings."
        )
    value, trusted_target = stored
    if trusted_target != expected_target:
        raise CredentialConfigurationError(
            f"The stored {kind} credential is not approved for the current "
            "model endpoint. Save the key again after reviewing the endpoint."
        )
    return value


def credential_status(
    config: Mapping[str, object],
    *,
    group: str,
    kind: CredentialKind,
    store: CredentialStore,
) -> dict[str, object]:
    """Return non-secret status suitable for a dashboard API response."""
    source = effective_credential_source(config)
    target = credential_target(config)
    configured = False
    usable = source == "none"
    needs_reentry = False

    if source == "environment":
        env_name = str(config.get("api_key_env") or "").strip()
        configured = bool(env_name and os.environ.get(env_name))
        usable = configured or not env_name
    elif source == "llm":
        stored = (
            read_api_key(store, credential_id=credential_id(group, "llm"))
            if kind == "embedding"
            else None
        )
        configured = stored is not None
        trusted_target = stored[1] if stored else None
        usable = configured and trusted_target == target
        needs_reentry = configured and not usable
    elif source == "stored":
        expected_id = credential_id(group, kind)
        metadata_matches = (
            config.get("credential_id") == expected_id
            and config.get("credential_target") == target
        )
        stored = read_api_key(store, credential_id=expected_id)
        configured = stored is not None
        trusted_target = stored[1] if stored else None
        binding_matches = trusted_target == target
        usable = configured and metadata_matches and binding_matches
        needs_reentry = configured and not usable

    return {
        "kind": kind,
        "source": source,
        "configured": configured,
        "usable": usable,
        "needs_reentry": needs_reentry,
        "target": target,
    }
