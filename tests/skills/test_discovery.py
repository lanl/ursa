import asyncio
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ursa.skills.discovery import (
    SkillCatalog,
    SkillScope,
    adiscover_skills,
    discover_skills,
    discover_skills_in,
    get_skill_catalog,
    is_valid_skill_name,
    load_skill,
    skill_roots,
    split_frontmatter,
)


def write_skill(
    root,
    name="example",
    description="Example",
    body="Instructions.",
    **metadata,
):
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump({
        "name": name,
        "description": description,
        **metadata,
    })
    path.write_text(f"---\n{front}---\n\n{body}\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["a", "x1", "1", "write-python", "a" * 64])
def test_valid_skill_names(name):
    assert is_valid_skill_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "-a",
        "a-",
        "a--b",
        "A",
        "a_b",
        "a.b",
        "a/b",
        "a\\b",
        "a b",
        "a" * 65,
    ],
)
def test_invalid_skill_names(name):
    assert not is_valid_skill_name(name)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"name": "example"},
        {"name": "wrong", "description": "ok"},
        {"name": 12, "description": "ok"},
        {"name": "example", "description": ""},
        {"name": "example", "description": " \n "},
        {"name": "example", "description": 123},
        {"name": "example", "description": "a" * 1025},
        {"name": "example", "description": "ok", "compatibility": "a" * 501},
        {"name": "example", "description": "ok", "metadata": {"n": 1}},
    ],
)
def test_invalid_metadata_is_skipped_with_diagnostic(
    tmp_path, metadata, caplog
):
    path = tmp_path / "example" / "SKILL.md"
    path.parent.mkdir()
    path.write_text(f"---\n{yaml.safe_dump(metadata)}---\nBody")
    assert discover_skills_in(tmp_path, SkillScope.PROJECT) == {}
    assert str(path) in caplog.text


@pytest.mark.parametrize(
    "text",
    [
        "Just instructions.",
        "---\n: : not: yaml\n---\nBody",
        "---\nname: example\n",
        "---\n- one\n- two\n---\nBody",
    ],
)
def test_malformed_files_are_skipped(tmp_path, text):
    path = tmp_path / "example" / "SKILL.md"
    path.parent.mkdir()
    path.write_text(text)
    assert load_skill(path.parent, SkillScope.PROJECT) is None


def test_description_is_preserved_and_body_is_lazy(tmp_path):
    description = "First line\n  indented second line"
    path = write_skill(tmp_path, description=description, body="original")
    skill = load_skill(path.parent, SkillScope.PROJECT)
    assert skill.description == description
    assert "instructions" not in skill.__dict__
    write_skill(
        tmp_path, description=description, body="edited before activation"
    )
    assert skill.instructions == "edited before activation"
    path.unlink()
    assert skill.instructions == "edited before activation"


def test_discovery_does_not_read_instruction_lines(tmp_path, monkeypatch):
    path = write_skill(tmp_path)

    class MetadataOnly:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.lines)

        lines = iter([
            "---\n",
            "name: example\n",
            "description: Example\n",
            "---\n",
        ])

    stream = MetadataOnly()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: stream)
    assert load_skill(path.parent, SkillScope.PROJECT).name == "example"


def test_long_description_and_instructions_are_not_truncated_on_discovery(
    tmp_path,
):
    path = write_skill(tmp_path, description="x" * 1024, body="x" * 200_000)
    skill = load_skill(path.parent, SkillScope.PROJECT)
    assert len(skill.description) == 1024
    assert len(skill.instructions) == 200_000


@pytest.mark.parametrize(
    "prefix,newline", [("", "\n"), ("\ufeff", "\n"), ("\n", "\n"), ("", "\r\n")]
)
def test_frontmatter_handles_bom_and_newlines(tmp_path, prefix, newline):
    text = (
        prefix
        + "---\nname: example\ndescription: Example\n---\nBody\n".replace(
            "\n", newline
        )
    )
    path = tmp_path / "example" / "SKILL.md"
    path.parent.mkdir()
    path.write_bytes(text.encode())
    metadata, body = split_frontmatter(text)
    assert metadata["name"] == "example"
    assert body.strip() == "Body"
    assert load_skill(path.parent, SkillScope.PROJECT).instructions == "Body"


def test_frontmatter_delimiters_must_be_at_the_start_and_on_their_own_lines():
    assert split_frontmatter("Intro\n---\nname: a\n---")[0] == {}
    metadata, body = split_frontmatter(
        "---\nname: a\ndescription: |\n  ---example\n---\nBody"
    )
    assert metadata["description"] == "---example\n"
    assert body == "Body"


@pytest.mark.parametrize("client", [".claude", ".codex", ".agents", ".ursa"])
def test_discovers_shared_and_client_roots(tmp_path, client):
    write_skill(Path.home() / client / "skills", "user-skill")
    write_skill(tmp_path / client / "skills", "project-skill")
    skills = discover_skills(tmp_path)
    assert skills["user-skill"].scope is SkillScope.USER
    assert skills["project-skill"].scope is SkillScope.PROJECT
    assert skills["project-skill"].path.is_absolute()


@pytest.mark.parametrize("xdg", [None, "custom", "relative-is-invalid"])
def test_xdg_config_root(tmp_path, monkeypatch, xdg):
    root = Path.home() / ".config"
    if xdg == "custom":
        root = tmp_path / "config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(root))
    elif xdg:
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
    write_skill(root / "ursa" / "skills")
    assert discover_skills(tmp_path)["example"].scope is SkillScope.USER


def test_precedence_and_shadow_diagnostics(tmp_path, monkeypatch, caplog):
    roots = skill_roots(tmp_path)[1:]
    for i, (root, _) in enumerate(roots):
        write_skill(root, description=f"Version {i}")
    skills = discover_skills(tmp_path)
    assert skills["example"].description == f"Version {len(roots) - 1}"
    assert skills["example"].scope is SkillScope.PROJECT
    assert "shadows" in caplog.text
    assert str(roots[-1][0]) in caplog.text


def test_discovery_uses_workspace_not_launch_directory(tmp_path, monkeypatch):
    launch, workspace = tmp_path / "launch", tmp_path / "workspace"
    write_skill(launch / ".ursa" / "skills", "launch-only")
    write_skill(workspace / ".ursa" / "skills", "workspace-only")
    monkeypatch.chdir(launch)
    assert "launch-only" not in discover_skills(workspace)
    assert "workspace-only" in discover_skills(workspace)
    assert "launch-only" in discover_skills()


def test_catalog_caches_discovery_and_isolates_workspaces(tmp_path):
    write_skill(tmp_path / ".ursa" / "skills")
    catalog = get_skill_catalog(tmp_path)
    assert catalog is get_skill_catalog(tmp_path / ".")
    with patch(
        "ursa.skills.discovery.discover_skills_in", wraps=discover_skills_in
    ) as scan:
        first = catalog.load()
        calls = scan.call_count
        assert first is catalog.load()
        assert scan.call_count == calls
    assert "example" not in get_skill_catalog(tmp_path / "other").load()
    with pytest.raises(TypeError):
        first["x"] = first["example"]


async def test_async_discovery_runs_off_the_event_loop(tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    threads = []

    def scan(root, scope):
        threads.append(threading.get_ident())
        return {}

    monkeypatch.setattr("ursa.skills.discovery.discover_skills_in", scan)
    assert await adiscover_skills(tmp_path) == {}
    catalog = SkillCatalog(((tmp_path, SkillScope.PROJECT),))
    await asyncio.gather(catalog.aload(), catalog.aload())
    assert threads and all(thread != loop_thread for thread in threads)


def test_home_can_also_be_the_workspace():
    write_skill(Path.home() / ".ursa" / "skills")
    assert discover_skills(Path.home())["example"].scope is SkillScope.PROJECT


def test_indented_yaml_delimiter_is_part_of_the_description(tmp_path):
    path = tmp_path / "example" / "SKILL.md"
    path.parent.mkdir()
    path.write_text(
        "---\nname: example\ndescription: |\n  First\n  ---\n  Second\n---\nBody"
    )
    skill = load_skill(path.parent, SkillScope.PROJECT)
    assert skill.description == "First\n---\nSecond\n"
    assert skill.instructions == "Body"
