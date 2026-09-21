from pathlib import Path

import pytest

from ursa.skills import bundled
from ursa.skills.bundled import (
    BUNDLED_VERSION,
    CHECKSUM_KEY,
    SKILL_CREATION_NAME,
    VERSION_KEY,
    ensure_bundled_skills,
    write_bundled_skill,
)
from ursa.skills.discovery import discover_skills, split_frontmatter


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: root)
    return root


def skill_path(home):
    return home / ".agents" / "skills" / SKILL_CREATION_NAME / "SKILL.md"


def test_creates_skill_creation_on_first_launch(home, tmp_path):
    written = ensure_bundled_skills()

    assert written == [skill_path(home)]
    metadata, body = split_frontmatter(
        skill_path(home).read_text(encoding="utf-8")
    )
    assert metadata["name"] == SKILL_CREATION_NAME
    assert metadata["description"]
    assert metadata[VERSION_KEY] >= 1
    assert "SKILL.md" in body


def test_second_launch_writes_nothing(home):
    ensure_bundled_skills()

    assert ensure_bundled_skills() == []


def test_bundled_skill_is_discoverable(home, tmp_path):
    ensure_bundled_skills()

    catalog = discover_skills(tmp_path)

    assert catalog[SKILL_CREATION_NAME].scope == "user"
    assert "skill" in catalog[SKILL_CREATION_NAME].description.lower()


def ship_new_version(monkeypatch, home):
    monkeypatch.setattr(bundled, "BUNDLED_VERSION", BUNDLED_VERSION + 1)
    return write_bundled_skill(
        home / ".agents" / "skills",
        SKILL_CREATION_NAME,
        "New description",
        "New body",
    )


def test_locally_edited_skill_is_left_alone_on_version_bump(home, monkeypatch):
    ensure_bundled_skills()
    path = skill_path(home)
    original = path.read_text(encoding="utf-8")
    path.write_text(original + "\nMy own extra rule.\n", encoding="utf-8")
    edited = path.read_text(encoding="utf-8")

    assert ship_new_version(monkeypatch, home) is None
    assert path.read_text(encoding="utf-8") == edited


def test_unmodified_skill_is_rewritten_on_version_bump(home, monkeypatch):
    ensure_bundled_skills()
    path = skill_path(home)

    assert ship_new_version(monkeypatch, home) == path
    metadata, body = split_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata["description"] == "New description"
    assert body.strip() == "New body"
    assert metadata[VERSION_KEY] == BUNDLED_VERSION + 1


def test_same_version_is_not_rewritten(home):
    ensure_bundled_skills()
    path = skill_path(home)
    before = path.read_text(encoding="utf-8")

    assert (
        write_bundled_skill(
            home / ".agents" / "skills",
            SKILL_CREATION_NAME,
            "Ignored description",
            "Ignored body",
        )
        is None
    )
    assert path.read_text(encoding="utf-8") == before


def test_user_authored_skill_of_same_name_is_never_touched(home):
    path = skill_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: Mine\n---\n\nMy instructions.\n", encoding="utf-8"
    )

    assert ensure_bundled_skills() == []
    metadata, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata == {"description": "Mine"}
    assert VERSION_KEY not in metadata


def test_checksum_is_recorded_so_edits_can_be_detected(home):
    ensure_bundled_skills()

    metadata, _ = split_frontmatter(
        skill_path(home).read_text(encoding="utf-8")
    )

    assert len(metadata[CHECKSUM_KEY]) == 64


def test_unwritable_home_does_not_raise(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: blocked)

    assert ensure_bundled_skills() == []
