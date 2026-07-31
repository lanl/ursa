from ursa.cli import groups


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
