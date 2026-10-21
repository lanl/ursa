from pathlib import Path

import pytest

from ursa.cli.config import UrsaConfig
from ursa.skills import SkillScope, discover_skills
from ursa.skills.runtime import explicit_skills


def test_precedence_and_workspace_isolation(tmp_path, make_skill):
    home, project, workspace = [
        tmp_path / name for name in ("home", "project", "workspace")
    ]
    make_skill(home)
    local = make_skill(project)
    make_skill(home, "home-only")
    catalog = discover_skills(cwd=project, home=home)
    assert [s.name for s in catalog.skills] == ["home-only", "lab-report"]
    assert catalog.get("lab-report").path == local / "SKILL.md"
    assert "shadowed" in catalog.diagnostics[0]
    assert (
        discover_skills(workspace=workspace, cwd=project, home=home).skills
        == ()
    )
    make_skill(workspace, "workspace-only")
    assert [
        s.name
        for s in discover_skills(
            workspace=workspace, cwd=project, home=home
        ).skills
    ] == ["workspace-only"]
    assert [
        s.name
        for s in discover_skills(
            workspace=project, cwd=project, home=home
        ).skills
    ] == ["lab-report"]


def test_refresh_symlinks_and_malformed_skills(tmp_path, make_skill):
    workspace = tmp_path / "workspace"
    scope = SkillScope(workspace=workspace)
    assert scope.discover().skills == ()
    outside = make_skill(tmp_path / "outside")
    root = workspace / ".ursa" / "skills"
    root.mkdir(parents=True)
    (root / "lab-report").symlink_to(outside, target_is_directory=True)
    bad = make_skill(workspace, "broken")
    (bad / "SKILL.md").write_text("---\nname: [bad yaml\n---\nbody")
    make_skill(root / "nested", "not-discovered")
    catalog = scope.discover()
    assert [s.name for s in catalog.skills] == ["lab-report"]
    assert catalog.skills[0].path == root / "lab-report" / "SKILL.md"
    assert len(catalog.diagnostics) == 1
    (outside / "SKILL.md").unlink()
    assert scope.discover().skills == ()


@pytest.mark.parametrize(
    "metadata",
    [
        "name: Bad\ndescription: test",
        "name: wrong\ndescription: test",
        "name: lab-report\ndescription: ''",
        "name: lab-report\ndescription: [bad]",
        "name: lab-report\ndescription: test\nmetadata: {version: 1}",
        "name: lab-report\ndescription: test\ncompatibility: ''",
        "name: lab-report\ndescription: test\nallowed-tools: [run_command]",
    ],
)
def test_invalid_metadata_is_reported(tmp_path, make_skill, metadata):
    folder = make_skill(tmp_path)
    (folder / "SKILL.md").write_text(f"---\n{metadata}\n---\nbody")
    catalog = discover_skills(workspace=tmp_path)
    assert not catalog.skills
    assert catalog.diagnostics


def test_standard_optional_metadata(tmp_path, make_skill):
    folder = make_skill(tmp_path)
    (folder / "SKILL.md").write_text(
        "---\nname: lab-report\ndescription: test\nlicense: MIT\ncompatibility: Python\n"
        "allowed-tools: read_file\nmetadata: {author: URSA}\n---\nbody"
    )
    skill = discover_skills(workspace=tmp_path).skills[0]
    assert skill.metadata["allowed-tools"] == "read_file"
    assert skill.as_dict()["metadata"]["metadata"] == {"author": "URSA"}


def test_explicit_references_ignore_code_and_escapes():
    text = "$lab-report $lab-report $other-skill `echo $ignored` \\$escaped\n```sh\necho $ignored-too\n```\n"
    assert explicit_skills(text) == ("lab-report", "other-skill")
    assert explicit_skills("${HOME} $HOME") == ()


@pytest.mark.parametrize("explicit", [False, True])
def test_resolution_preserves_workspace_provenance(explicit):
    config = UrsaConfig(**({"workspace": Path(".")} if explicit else {}))
    resolved = config.resolve().resolve()
    assert ("workspace" in resolved.model_fields_set) is explicit


def test_tmp_workspace_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = UrsaConfig(workspace="tmp").resolve()
    assert config.workspace.is_dir()
    assert "workspace" in config.model_fields_set
    assert config.resolve().workspace == config.workspace
    config._temp_workspace.cleanup()


def test_symlink_installation_keeps_logical_skill_directory(
    tmp_path, make_skill
):
    from ursa.skills.runtime import SkillRun

    external = tmp_path / "shared-bundle"
    external.mkdir()
    (external / "SKILL.md").write_text(
        "---\nname: lab-report\ndescription: report\n---\nSYMLINK_BODY"
    )
    root = tmp_path / "workspace" / ".ursa" / "skills"
    root.mkdir(parents=True)
    (root / "lab-report").symlink_to(external, target_is_directory=True)
    catalog = discover_skills(workspace=tmp_path / "workspace")
    run = SkillRun(catalog)
    run.activate("lab-report")
    assert "SYMLINK_BODY" in run.prompt()
