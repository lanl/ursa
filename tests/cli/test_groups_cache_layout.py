import pytest

import ursa.security as security
from ursa.cli import groups, main


def test_create_group_initializes_hierarchical_subdirectories(
    monkeypatch, tmp_path
):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(groups, "URSA_CACHE_DIR", cache_root)
    config_file = tmp_path / "group.yaml"
    config_file.write_text(
        "allowed_base_urls:\n  - https://example.com\n",
        encoding="utf-8",
    )

    groups.create_group("science", config_file)

    for group_name in ("default", "science"):
        group_dir = cache_root / group_name
        assert (group_dir / "agents").is_dir()
        assert (group_dir / "rag").is_dir()
        assert (group_dir / "dashboard").is_dir()
        assert (group_dir / "environments").is_dir()

    assert (cache_root / "science" / "group.yaml").exists()


@pytest.mark.parametrize("group", ["default", "science"])
@pytest.mark.parametrize("existing_policy", [False, True])
def test_update_group_cli_stores_and_enforces_whitelist(
    monkeypatch, tmp_path, group, existing_policy, capsys
):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(groups, "URSA_CACHE_DIR", cache_root)
    monkeypatch.setattr(security, "URSA_CACHE_DIR", cache_root)
    group_dir = cache_root / group
    if group != "default" or existing_policy:
        group_dir.mkdir(parents=True)
    if existing_policy:
        (group_dir / "group.yaml").write_text(
            "allowed_base_urls:\n  - https://old.example.com\n",
            encoding="utf-8",
        )
    config_file = tmp_path / "updated_allowed_urls.yaml"
    config_file.write_text(
        "allowed_base_urls:\n  - https://example.com\n",
        encoding="utf-8",
    )

    main(["update-group", group, str(config_file)])

    assert (group_dir / "group.yaml").read_bytes() == config_file.read_bytes()
    assert f"Updated group '{group}'" in capsys.readouterr().out
    security.enforce_group_base_url_policy("https://example.com/v1", group)
    with pytest.raises(security.GroupBaseURLPolicyError):
        security.enforce_group_base_url_policy("https://old.example.com", group)
    with pytest.raises(security.GroupBaseURLPolicyError):
        security.enforce_group_base_url_policy(None, group)
    if group == "default":
        for subdir in ("agents", "rag", "dashboard", "environments"):
            assert (group_dir / subdir).is_dir()


def test_update_missing_non_default_group_still_fails(monkeypatch, tmp_path):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(groups, "URSA_CACHE_DIR", cache_root)
    config_file = tmp_path / "allowed_urls.yaml"
    config_file.write_text(
        "allowed_base_urls:\n  - https://example.com\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="Group does not exist"):
        groups.update_group("science", config_file)
    assert not cache_root.exists()


@pytest.mark.parametrize("existing_policy", [False, True])
def test_invalid_update_does_not_create_or_replace_default_policy(
    monkeypatch, tmp_path, existing_policy
):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(groups, "URSA_CACHE_DIR", cache_root)
    policy = cache_root / "default" / "group.yaml"
    original = "allowed_base_urls:\n  - https://example.com\n"
    if existing_policy:
        policy.parent.mkdir(parents=True)
        policy.write_text(original, encoding="utf-8")
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("allowed_base_urls: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty 'allowed_base_urls' list"):
        groups.update_group("default", config_file)
    if existing_policy:
        assert policy.read_text(encoding="utf-8") == original
    else:
        assert not cache_root.exists()
