from __future__ import annotations

from types import SimpleNamespace

import pytest

import ursa.security as security
from ursa.security import GroupBaseURLPolicyError


def _write_group_config(
    root, group: str, allowed_base_urls: list[str] | None
) -> None:
    group_dir = root / group
    group_dir.mkdir(parents=True)
    if allowed_base_urls is not None:
        lines = ["allowed_base_urls:"] + [
            f"  - {url}" for url in allowed_base_urls
        ]
        (group_dir / "group.yaml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


@pytest.fixture
def isolated_security_cache(monkeypatch, tmp_path):
    root = tmp_path / "ursa-cache"
    monkeypatch.setattr(security, "URSA_CACHE_DIR", root)
    return root


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "default"),
        ("", "default"),
        (" science ", "science"),
        ("science", "science"),
    ],
)
def test_validate_group_name_defaults_and_strips(raw, expected):
    assert security.validate_group_name(raw) == expected


@pytest.mark.parametrize(
    "group",
    [
        "../science",
        "science/agents",
        "science\\agents",
        ".",
        "..",
    ],
)
def test_validate_group_name_rejects_path_like_names(group):
    with pytest.raises(ValueError):
        security.validate_group_name(group)


def test_validate_group_name_rejects_whitespace_only_name():
    with pytest.raises(ValueError, match="must not be empty"):
        security.validate_group_name("   ")


def test_group_directory_helpers_use_shared_group_root(isolated_security_cache):
    assert (
        security.group_root_dir("science")
        == isolated_security_cache / "science"
    )
    assert (
        security.group_agents_dir("science")
        == isolated_security_cache / "science" / "agents"
    )
    assert (
        security.group_rag_dir("science")
        == isolated_security_cache / "science" / "rag"
    )
    assert (
        security.group_dashboard_dir("science")
        == isolated_security_cache / "science" / "dashboard"
    )
    assert (
        security.group_config_file("science")
        == isolated_security_cache / "science" / "group.yaml"
    )


@pytest.mark.parametrize("base_url", [None, "", "https://anything.example/v1"])
def test_default_group_does_not_require_whitelist_config(
    isolated_security_cache, base_url
):
    assert security.is_base_url_allowed(base_url, "default") is True
    security.enforce_group_base_url_policy(base_url, "default")
    assert not isolated_security_cache.exists()


def test_non_default_group_requires_existing_group_directory(
    isolated_security_cache,
):
    with pytest.raises(GroupBaseURLPolicyError, match="does not exist"):
        security.enforce_group_base_url_policy(
            "https://models.example.org/v1", "science"
        )


def test_non_default_group_requires_group_config_file(isolated_security_cache):
    (isolated_security_cache / "science").mkdir(parents=True)

    with pytest.raises(
        GroupBaseURLPolicyError, match="missing required config"
    ):
        security.enforce_group_base_url_policy(
            "https://models.example.org/v1", "science"
        )


@pytest.mark.parametrize(
    "yaml_text,match",
    [
        ("- https://models.example.org/v1\n", "must be a YAML mapping"),
        ("{}\n", "non-empty 'allowed_base_urls' list"),
        ("allowed_base_urls: []\n", "non-empty 'allowed_base_urls' list"),
        ("allowed_base_urls:\n  - 123\n", "non-string allowed URL"),
        ("allowed_base_urls:\n  - '   '\n", "empty allowed URL"),
    ],
)
def test_group_config_must_define_valid_allowed_base_urls(
    isolated_security_cache, yaml_text, match
):
    group_dir = isolated_security_cache / "science"
    group_dir.mkdir(parents=True)
    (group_dir / "group.yaml").write_text(yaml_text, encoding="utf-8")

    with pytest.raises(GroupBaseURLPolicyError, match=match):
        security.enforce_group_base_url_policy(
            "https://models.example.org/v1", "science"
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://models.example.org/v1",
        "https://models.example.org/v1/",
        " https://models.example.org/v1 ",
        # Whitelist policy permits any path on the same scheme + netloc.
        "https://models.example.org/v2/chat/completions",
    ],
)
def test_group_whitelist_allows_exact_url_and_same_origin(
    isolated_security_cache, base_url
):
    _write_group_config(
        isolated_security_cache,
        "science",
        ["https://models.example.org/v1/"],
    )

    assert security.is_base_url_allowed(base_url, "science") is True
    security.enforce_group_base_url_policy(base_url, "science")


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "",
        "   ",
        "https://unsafe.example.org/v1",
        "http://models.example.org/v1",
        "https://models.example.org.evil/v1",
        "https://user:pass@models.example.org/v1",
        "https://models.example.org:444/v1",
    ],
)
def test_group_whitelist_rejects_missing_or_different_origin(
    isolated_security_cache, base_url
):
    _write_group_config(
        isolated_security_cache,
        "science",
        ["https://models.example.org/v1"],
    )

    assert security.is_base_url_allowed(base_url, "science") is False
    with pytest.raises(GroupBaseURLPolicyError):
        security.enforce_group_base_url_policy(base_url, "science")


def test_policy_error_for_missing_base_url_lists_allowed_urls(
    isolated_security_cache,
):
    _write_group_config(
        isolated_security_cache,
        "science",
        ["https://models.example.org/v1"],
    )

    with pytest.raises(GroupBaseURLPolicyError) as excinfo:
        security.enforce_group_base_url_policy(None, "science")

    message = str(excinfo.value)
    assert "requires an explicit model base_url" in message
    assert "https://models.example.org/v1" in message


def test_policy_error_for_disallowed_base_url_names_group_and_allowed_urls(
    isolated_security_cache,
):
    _write_group_config(
        isolated_security_cache,
        "science",
        ["https://models.example.org/v1"],
    )

    with pytest.raises(GroupBaseURLPolicyError) as excinfo:
        security.enforce_group_base_url_policy(
            "https://unsafe.example.org/v1", "science"
        )

    message = str(excinfo.value)
    assert "https://unsafe.example.org/v1" in message
    assert "science" in message
    assert "https://models.example.org/v1" in message


@pytest.mark.parametrize(
    "attrs",
    [
        {"base_url": " https://base.example/v1/ "},
        {"api_base": "https://api-base.example/v1/"},
        {"openai_api_base": "https://openai-base.example/v1/"},
    ],
)
def test_get_model_base_url_extracts_and_normalizes_supported_attrs(attrs):
    model = SimpleNamespace(**attrs)

    assert security.get_model_base_url(model) == next(
        iter(attrs.values())
    ).strip().rstrip("/")


def test_get_model_base_url_uses_first_available_attr():
    model = SimpleNamespace(
        base_url="https://base.example/v1",
        api_base="https://api-base.example/v1",
        openai_api_base="https://openai-base.example/v1",
    )

    assert security.get_model_base_url(model) == "https://base.example/v1"


def test_enforce_model_group_policy_uses_model_base_url(
    isolated_security_cache,
):
    _write_group_config(
        isolated_security_cache,
        "science",
        ["https://models.example.org/v1"],
    )
    safe_model = SimpleNamespace(base_url="https://models.example.org/v1/")
    unsafe_model = SimpleNamespace(base_url="https://unsafe.example.org/v1")

    security.enforce_model_group_policy(safe_model, "science")
    with pytest.raises(GroupBaseURLPolicyError):
        security.enforce_model_group_policy(unsafe_model, "science")
