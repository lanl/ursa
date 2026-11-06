import pytest

from ursa.cli import agent_management
from ursa.cli.agent_management import delete_agent


@pytest.fixture
def agents_dir(monkeypatch, tmp_path):
    root = tmp_path / "ursa"
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", root)
    directory = root / "default" / "agents"
    directory.mkdir(parents=True)
    for name in (
        "ursa_tmp_0a1b",
        "ursa_tmp_9f8e",
        "tmp_user_agent",
        "hfbtho_bot",
    ):
        (directory / name / "db").mkdir(parents=True)
    return directory


def _names(directory):
    return sorted(path.name for path in directory.iterdir())


def test_wildcard_deletes_only_matching_agents(agents_dir, capsys):
    delete_agent("ursa_tmp_*")
    assert _names(agents_dir) == ["hfbtho_bot", "tmp_user_agent"]
    out = capsys.readouterr().out
    assert "Deleted 2 agent(s) matching 'ursa_tmp_*'" in out


def test_wildcard_is_case_sensitive_and_anchored(agents_dir):
    # "*bot" must not match in the middle, and case must match exactly.
    with pytest.raises(FileNotFoundError, match="No agents match"):
        delete_agent("*BOT")
    delete_agent("*bot")
    assert "hfbtho_bot" not in _names(agents_dir)
    assert "tmp_user_agent" in _names(agents_dir)


def test_wildcard_without_matches_raises(agents_dir):
    with pytest.raises(FileNotFoundError, match="No agents match"):
        delete_agent("nothing_*")
    assert len(_names(agents_dir)) == 4


@pytest.mark.parametrize("pattern", ["../*", "*/*", "a*?", "*[ab]", " /*"])
def test_wildcard_rejects_path_and_glob_syntax(agents_dir, pattern):
    with pytest.raises(ValueError, match="pattern may only contain"):
        delete_agent(pattern)
    assert len(_names(agents_dir)) == 4


def test_wildcard_never_follows_symlinks(agents_dir, tmp_path):
    outside = tmp_path / "precious"
    (outside / "keep.txt").parent.mkdir()
    (outside / "keep.txt").write_text("data")
    (agents_dir / "ursa_tmp_link").symlink_to(outside, target_is_directory=True)

    delete_agent("ursa_tmp_*")

    assert (outside / "keep.txt").read_text() == "data"
    assert (agents_dir / "ursa_tmp_link").is_symlink()


def test_exact_name_still_works(agents_dir):
    delete_agent("hfbtho_bot")
    assert "hfbtho_bot" not in _names(agents_dir)
