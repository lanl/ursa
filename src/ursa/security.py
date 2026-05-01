from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import yaml
from langchain.chat_models import BaseChatModel

from ursa.cli.groups import AGENT_GROUPS_DIR, GROUP_CONFIG_FILENAME


class GroupBaseURLPolicyError(ValueError):
    """Raised when a model endpoint violates a group's base URL policy."""


DEFAULT_GROUP_NAME = "default"


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


def _group_dir(group: str) -> Path:
    return AGENT_GROUPS_DIR.expanduser() / group


def _load_group_allowed_base_urls(group: str) -> list[str] | None:
    if group == DEFAULT_GROUP_NAME:
        return None

    group_dir = _group_dir(group)
    if not group_dir.exists() or not group_dir.is_dir():
        raise GroupBaseURLPolicyError(
            f"Group '{group}' does not exist. Please create it before use."
        )

    config_file = group_dir / GROUP_CONFIG_FILENAME
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
    effective_group = group or DEFAULT_GROUP_NAME
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


def enforce_group_base_url_policy(base_url: str | None, group: str | None) -> None:
    effective_group = group or DEFAULT_GROUP_NAME
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


def enforce_model_group_policy(model: BaseChatModel | object, group: str | None) -> None:
    enforce_group_base_url_policy(get_model_base_url(model), group)
