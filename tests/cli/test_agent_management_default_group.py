from ursa.cli import agent_management
from ursa.cli.agent_management import ensure_group_dir


def test_ensure_group_dir_creates_missing_default_group(monkeypatch, tmp_path):
    root = tmp_path / "ursa"
    monkeypatch.setattr(agent_management, "AGENT_GROUPS_DIR", root)

    group_dir = ensure_group_dir("default")

    assert group_dir == root / "default" / "agents"
    assert group_dir.is_dir()


def test_ensure_group_dir_still_requires_non_default_group(
    monkeypatch, tmp_path
):
    root = tmp_path / "ursa"
    monkeypatch.setattr(agent_management, "AGENT_GROUPS_DIR", root)

    try:
        ensure_group_dir("science")
    except FileNotFoundError as exc:
        assert "science" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError for missing non-default group"
        )
