from pathlib import Path

import pytest

from ursa.skills.bundled import (
    SKILL_CREATION_NAME,
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
    return home / ".ursa" / "skills" / SKILL_CREATION_NAME / "SKILL.md"


def test_creates_skill_creation_on_first_launch(home, tmp_path):
    written = ensure_bundled_skills()

    assert written == [skill_path(home)]
    metadata, body = split_frontmatter(
        skill_path(home).read_text(encoding="utf-8")
    )
    assert metadata == {
        "name": SKILL_CREATION_NAME,
        "description": metadata["description"],
    }
    assert metadata["description"]
    assert "SKILL.md" in body


def test_second_launch_writes_nothing(home):
    ensure_bundled_skills()

    assert ensure_bundled_skills() == []


def test_bundled_skill_is_discoverable(home, tmp_path):
    ensure_bundled_skills()

    catalog = discover_skills(tmp_path)

    assert catalog[SKILL_CREATION_NAME].scope == "user"
    assert "skill" in catalog[SKILL_CREATION_NAME].description.lower()


def test_existing_file_is_never_overwritten(home):
    ensure_bundled_skills()
    path = skill_path(home)
    before = path.read_text(encoding="utf-8")

    assert (
        write_bundled_skill(
            home / ".ursa" / "skills",
            SKILL_CREATION_NAME,
            "Ignored description",
            "Ignored body",
        )
        is None
    )
    assert path.read_text(encoding="utf-8") == before


def test_local_edits_survive_a_later_launch(home):
    ensure_bundled_skills()
    path = skill_path(home)
    edited = path.read_text(encoding="utf-8") + "\nMy own extra rule.\n"
    path.write_text(edited, encoding="utf-8")

    assert ensure_bundled_skills() == []
    assert path.read_text(encoding="utf-8") == edited


def test_user_authored_skill_of_same_name_is_never_touched(home):
    path = skill_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: Mine\n---\n\nMy instructions.\n", encoding="utf-8"
    )

    assert ensure_bundled_skills() == []
    metadata, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata == {"description": "Mine"}


def test_unwritable_home_does_not_raise(monkeypatch, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: blocked)

    assert ensure_bundled_skills() == []
