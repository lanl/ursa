from importlib.resources import files
from pathlib import Path

from tests.skills.test_discovery import write_skill
from ursa.skills.discovery import SkillScope, discover_skills


def test_bundled_skill_is_available_without_home_writes(tmp_path):
    home = Path.home()
    before = set(home.rglob("*"))
    skill = discover_skills(tmp_path)["skill-creation"]
    assert skill.scope is SkillScope.BUNDLED
    assert "SKILL.md" in skill.instructions
    assert "https://agentskills.io/specification" in skill.instructions
    assert files("ursa.skills").joinpath("skill-creation/SKILL.md").is_file()
    assert set(home.rglob("*")) == before


def test_user_and_project_can_shadow_bundled_skill(tmp_path):
    path = write_skill(
        Path.home() / ".ursa" / "skills",
        "skill-creation",
        body="User instructions",
    )
    before = path.read_text()
    assert (
        discover_skills(tmp_path)["skill-creation"].instructions
        == "User instructions"
    )
    write_skill(
        tmp_path / ".agents" / "skills",
        "skill-creation",
        body="Project instructions",
    )
    assert (
        discover_skills(tmp_path)["skill-creation"].instructions
        == "Project instructions"
    )
    assert path.read_text() == before


def test_unusable_home_does_not_hide_bundled_skills(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setattr(Path, "home", lambda: blocked)
    assert (
        discover_skills(tmp_path)["skill-creation"].scope is SkillScope.BUNDLED
    )
