from pathlib import Path

import pytest

from ursa.skills.discovery import (
    MAX_INSTRUCTION_CHARACTERS,
    discover_skills,
    is_valid_skill_name,
    project_skills_root,
    split_frontmatter,
    user_skills_root,
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: root)
    return root


def write_skill(root, name, description, body="Do the thing."):
    path = root / ".ursa" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    front = "---\n"
    if description is not None:
        front += f"description: {description}\n"
    front += "---\n"
    path.write_text(f"{front}\n{body}\n", encoding="utf-8")
    return path


def test_roots_use_home_and_given_project_dir(home, tmp_path):
    assert user_skills_root() == home / ".ursa" / "skills"
    assert project_skills_root(tmp_path) == tmp_path / ".ursa" / "skills"


def test_project_root_defaults_to_cwd(home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert project_skills_root() == tmp_path / ".ursa" / "skills"


def test_discovers_user_and_project_skills(home, tmp_path):
    write_skill(home, "user-only", "A user skill")
    write_skill(tmp_path, "project-only", "A project skill")

    catalog = discover_skills(tmp_path)

    assert {name: skill.scope for name, skill in catalog.items()} == {
        "project-only": "project",
        "user-only": "user",
    }
    assert catalog["user-only"].description == "A user skill"
    assert catalog["user-only"].instructions == "Do the thing."


def test_project_skill_shadows_user_skill_of_same_name(home, tmp_path):
    write_skill(home, "shared", "User version", body="User body")
    write_skill(tmp_path, "shared", "Project version", body="Project body")

    skill = discover_skills(tmp_path)["shared"]

    assert skill.scope == "project"
    assert skill.description == "Project version"
    assert skill.instructions == "Project body"


def test_missing_roots_yield_no_skills(home, tmp_path):
    assert discover_skills(tmp_path) == {}


def test_directory_without_skill_file_is_ignored(home, tmp_path):
    (tmp_path / ".ursa" / "skills" / "empty").mkdir(parents=True)
    write_skill(tmp_path, "real", "Real skill")

    assert list(discover_skills(tmp_path)) == ["real"]


def test_malformed_frontmatter_still_loads_with_placeholder(home, tmp_path):
    path = tmp_path / ".ursa" / "skills" / "broken" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\n: : not: yaml\n---\n\nBody text\n", encoding="utf-8")

    skill = discover_skills(tmp_path)["broken"]

    assert "no description provided" in skill.description


def test_missing_description_gets_placeholder(home, tmp_path):
    write_skill(tmp_path, "nodesc", None)

    assert (
        "no description provided"
        in discover_skills(tmp_path)["nodesc"].description
    )


def test_file_without_frontmatter_is_all_instructions(home, tmp_path):
    path = tmp_path / ".ursa" / "skills" / "bare" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("Just instructions.\n", encoding="utf-8")

    skill = discover_skills(tmp_path)["bare"]

    assert skill.instructions == "Just instructions."
    assert "no description provided" in skill.description


def test_description_is_collapsed_to_one_line(home, tmp_path):
    write_skill(tmp_path, "wrapped", "First part\n  second part")

    assert (
        discover_skills(tmp_path)["wrapped"].description
        == "First part second part"
    )


def test_oversized_instructions_are_truncated(home, tmp_path):
    write_skill(tmp_path, "big", "Big", body="x" * (2 * 10**5))

    instructions = discover_skills(tmp_path)["big"].instructions

    assert len(instructions) < 2 * 10**5
    assert instructions.startswith("x" * MAX_INSTRUCTION_CHARACTERS)
    assert "Truncated" in instructions


def test_non_utf8_bytes_do_not_break_discovery(home, tmp_path):
    path = tmp_path / ".ursa" / "skills" / "binary" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"---\ndescription: Latin\n---\n\ncaf\xe9\n")

    assert discover_skills(tmp_path)["binary"].description == "Latin"


def test_skill_exposes_its_directory_for_supporting_files(home, tmp_path):
    path = write_skill(tmp_path, "helper", "Has helpers")

    assert discover_skills(tmp_path)["helper"].directory == path.parent


@pytest.mark.parametrize(
    "name", ["write-python", "a", "skill.v2", "under_score", "x1"]
)
def test_valid_skill_names(name):
    assert is_valid_skill_name(name)


@pytest.mark.parametrize(
    "name", ["", ".", "..", "-leading", "has space", "a/b", "a\\b", ".hidden"]
)
def test_invalid_skill_names(name):
    assert not is_valid_skill_name(name)


def test_split_frontmatter_returns_metadata_and_body():
    metadata, body = split_frontmatter("---\nname: a\n---\n\nBody\n")

    assert metadata == {"name": "a"}
    assert body.strip() == "Body"


def test_split_frontmatter_ignores_non_mapping_metadata():
    metadata, body = split_frontmatter("---\n- one\n- two\n---\nBody\n")

    assert metadata == {}
    assert body.strip() == "Body"
