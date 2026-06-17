from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from langchain.chat_models import BaseChatModel

URSA_CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", "~/.cache/ursa")).expanduser()
GROUP_CONFIG_FILENAME = "group.yaml"


class GroupBaseURLPolicyError(ValueError):
    """Raised when a model endpoint violates a group's base URL policy."""


DEFAULT_GROUP_NAME = "default"


def validate_group_name(group: str | None) -> str:
    value = (group or DEFAULT_GROUP_NAME).strip()
    if not value:
        raise ValueError("Group name must not be empty")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")
    return value


def group_root_dir(group: str | None = DEFAULT_GROUP_NAME) -> Path:
    return URSA_CACHE_DIR.expanduser() / validate_group_name(group)


def group_agents_dir(group: str | None = DEFAULT_GROUP_NAME) -> Path:
    return group_root_dir(group) / "agents"


def group_rag_dir(group: str | None = DEFAULT_GROUP_NAME) -> Path:
    return group_root_dir(group) / "rag"


def group_dashboard_dir(group: str | None = DEFAULT_GROUP_NAME) -> Path:
    return group_root_dir(group) / "dashboard"


def group_config_file(group: str | None = DEFAULT_GROUP_NAME) -> Path:
    return group_root_dir(group) / GROUP_CONFIG_FILENAME


def normalize_base_url(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip()
    if not value:
        return None
    return value.rstrip("/")


def get_model_base_url(model: object) -> str | None:
    for attr in ("base_url", "api_base", "openai_api_base"):
        value = getattr(model, attr, None)
        normalized = normalize_base_url(value)
        if normalized:
            return normalized
    return None


def _load_group_allowed_base_urls(group: str) -> list[str] | None:
    if group == DEFAULT_GROUP_NAME:
        return None

    group_dir = group_root_dir(group)
    if not group_dir.exists() or not group_dir.is_dir():
        raise GroupBaseURLPolicyError(
            f"Group '{group}' does not exist. Please create it before use."
        )

    config_file = group_config_file(group)
    if not config_file.exists() or not config_file.is_file():
        raise GroupBaseURLPolicyError(
            f"Group '{group}' is missing required config file '{GROUP_CONFIG_FILENAME}'."
        )

    with open(config_file, "r", encoding="utf-8") as fid:
        data = yaml.safe_load(fid)

    if not isinstance(data, dict):
        raise GroupBaseURLPolicyError(
            f"Group config '{config_file}' must be a YAML mapping."
        )

    allowed = data.get("allowed_base_urls")
    if not isinstance(allowed, list) or not allowed:
        raise GroupBaseURLPolicyError(
            f"Group config '{config_file}' must define a non-empty 'allowed_base_urls' list."
        )

    normalized = []
    for url in allowed:
        if not isinstance(url, str):
            raise GroupBaseURLPolicyError(
                f"Group config '{config_file}' contains a non-string allowed URL."
            )
        norm = normalize_base_url(url)
        if not norm:
            raise GroupBaseURLPolicyError(
                f"Group config '{config_file}' contains an empty allowed URL entry."
            )
        normalized.append(norm)
    return normalized


def _same_origin(url_a: str, url_b: str) -> bool:
    a = urlsplit(url_a)
    b = urlsplit(url_b)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)


def is_base_url_allowed(base_url: str | None, group: str | None) -> bool:
    effective_group = validate_group_name(group)
    if effective_group == DEFAULT_GROUP_NAME:
        return True

    normalized = normalize_base_url(base_url)
    if not normalized:
        return False

    allowed = _load_group_allowed_base_urls(effective_group)
    assert allowed is not None
    return any(
        normalized == candidate or _same_origin(normalized, candidate)
        for candidate in allowed
    )


def enforce_group_base_url_policy(
    base_url: str | None, group: str | None
) -> None:
    effective_group = validate_group_name(group)
    if effective_group == DEFAULT_GROUP_NAME:
        return

    normalized = normalize_base_url(base_url)
    if is_base_url_allowed(normalized, effective_group):
        return

    allowed = _load_group_allowed_base_urls(effective_group) or []

    if not normalized:
        raise GroupBaseURLPolicyError(
            f"Group '{effective_group}' requires an explicit model base_url that matches its whitelist. "
            f"\nAllowed base URLs:\n {', '.join(allowed)}"
        )

    raise GroupBaseURLPolicyError(
        f"Base URL '{normalized}' is not allowed for group '{effective_group}'. "
        f"Allowed base URLs: {', '.join(allowed)}"
    )


def enforce_model_group_policy(
    model: BaseChatModel | object, group: str | None
) -> None:
    enforce_group_base_url_policy(get_model_base_url(model), group)
