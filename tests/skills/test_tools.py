from pathlib import Path

import pytest

from ursa.skills.activation import annotate_skill_requests
from ursa.skills.tools import (
    SKILL_TOOL_NAME,
    build_skill_tool,
    load_skill_instructions,
)


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: root)
    return root


def write_skill(root, name, description, body="Do the thing."):
    path = root / ".ursa" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndescription: {description}\n---\n\n{body}\n", encoding="utf-8"
    )
    return path


def test_no_tool_when_no_skills_exist(home, tmp_path):
    assert build_skill_tool(tmp_path) is None


def test_tool_lists_every_skill_in_its_description(home, tmp_path):
    write_skill(home, "user-skill", "From the user root")
    write_skill(tmp_path, "write-python", "House style for Python")

    tool = build_skill_tool(tmp_path)

    assert tool.name == SKILL_TOOL_NAME
    assert list(tool.args) == ["name"]
    assert "- write-python: House style for Python" in tool.description
    assert "- user-skill: From the user root" in tool.description


def test_tool_returns_the_skill_instructions(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python", body="Run with `uv run`.")

    result = build_skill_tool(tmp_path).invoke({"name": "write-python"})

    assert "# Skill: write-python" in result
    assert "Run with `uv run`." in result
    assert "project skill" in result


def test_tool_reports_skills_added_after_it_was_built(home, tmp_path):
    write_skill(tmp_path, "first", "First")
    tool = build_skill_tool(tmp_path)

    write_skill(tmp_path, "second", "Second")

    assert "# Skill: second" in tool.invoke({"name": "second"})


def test_unknown_skill_returns_a_recoverable_message(home, tmp_path):
    write_skill(tmp_path, "known", "Known")

    result = build_skill_tool(tmp_path).invoke({"name": "nope"})

    assert "No skill named 'nope'" in result
    assert "known" in result


@pytest.mark.parametrize(
    "name", ["../../../etc/passwd", "/etc/passwd", "..", "a/b"]
)
def test_path_like_names_cannot_escape_the_skills_roots(home, tmp_path, name):
    write_skill(tmp_path, "known", "Known")

    result = load_skill_instructions(name, tmp_path)

    assert result.startswith("No skill named")


def test_leading_dollar_from_the_model_is_tolerated(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python")

    result = load_skill_instructions("$write-python", tmp_path)

    assert "# Skill: write-python" in result


def test_project_skill_shadows_user_skill_through_the_tool(home, tmp_path):
    write_skill(home, "shared", "User", body="User body")
    write_skill(tmp_path, "shared", "Project", body="Project body")

    result = load_skill_instructions("shared", tmp_path)

    assert "Project body" in result
    assert "User body" not in result


def test_dollar_reference_becomes_a_tool_directive(home, tmp_path):
    write_skill(tmp_path, "write-python", "Python")

    annotated = annotate_skill_requests(
        "Tidy this up $write-python", project_root=tmp_path
    )

    assert annotated.startswith("Tidy this up $write-python")
    assert "write-python" in annotated.splitlines()[-1]
    assert f"`{SKILL_TOOL_NAME}`" in annotated
