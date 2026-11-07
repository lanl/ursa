import pytest

from ursa.cli import agent_management
from ursa.cli.agent_management import ensure_group_dir


def test_ensure_group_dir_creates_missing_default_group(monkeypatch, tmp_path):
    root = tmp_path / "ursa"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", root)

    group_dir = ensure_group_dir("default")

    assert group_dir == root / "default" / "agents"
    assert group_dir.is_dir()


def test_ensure_group_dir_still_requires_non_default_group(
    monkeypatch, tmp_path
):
    root = tmp_path / "ursa"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", root)

    with pytest.raises(FileNotFoundError) as exc:
        ensure_group_dir("science")
    assert "science" in str(exc.value)
