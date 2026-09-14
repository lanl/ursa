from pathlib import Path

from ursa.util.skills import (
    discover_skills,
    load_skill_body,
    parse_frontmatter,
    render_loaded_skill,
    render_skill_catalog,
)


def _write_skill(
    root: Path, dirname: str, name: str, description: str, body: str
) -> Path:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_parse_frontmatter_splits_meta_and_body() -> None:
    text = "---\nname: foo\ndescription: does foo\n---\n\nBody line one.\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"name": "foo", "description": "does foo"}
    assert body.strip() == "Body line one."


def test_parse_frontmatter_without_frontmatter_returns_text() -> None:
    text = "# Just markdown\n\nNo frontmatter here.\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_unterminated_block_is_ignored() -> None:
    text = "---\nname: foo\nno closing fence\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_discover_skills_finds_skill(tmp_path: Path) -> None:
    _write_skill(
        tmp_path, "greeter", "greeter", "Greets the user", "Say hello."
    )
    skills = discover_skills(roots=[tmp_path])
    assert set(skills) == {"greeter"}
    assert skills["greeter"].description == "Greets the user"
    assert skills["greeter"].body == "Say hello."
    assert skills["greeter"].source == "global"


def test_discover_skills_project_shadows_global(tmp_path: Path) -> None:
    project = tmp_path / "project"
    glob_root = tmp_path / "global"
    _write_skill(project, "dup", "dup", "Project version", "project body")
    _write_skill(glob_root, "dup", "dup", "Global version", "global body")

    skills = discover_skills(roots=[project, glob_root])
    assert skills["dup"].description == "Project version"
    assert skills["dup"].body == "project body"


def test_discover_skills_skips_malformed(tmp_path: Path) -> None:
    # Missing description in frontmatter -> skipped.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: bad\n---\nbody\n", encoding="utf-8"
    )
    # Valid skill alongside it.
    _write_skill(tmp_path, "good", "good", "A good skill", "good body")

    skills = discover_skills(roots=[tmp_path])
    assert set(skills) == {"good"}


def test_discover_skills_falls_back_to_dir_name(tmp_path: Path) -> None:
    # No name in frontmatter -> falls back to the parent directory name.
    d = tmp_path / "fallback"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\ndescription: uses dir name\n---\nbody\n", encoding="utf-8"
    )
    skills = discover_skills(roots=[tmp_path])
    assert set(skills) == {"fallback"}


def test_render_skill_catalog_lists_names_and_descriptions(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha", "alpha", "Handles alpha", "alpha body")
    _write_skill(tmp_path, "beta", "beta", "Handles beta", "beta body")
    catalog = render_skill_catalog(discover_skills(roots=[tmp_path]))
    assert "alpha: Handles alpha" in catalog
    assert "beta: Handles beta" in catalog
    assert "load_skill" in catalog


def test_render_skill_catalog_empty_is_blank() -> None:
    assert render_skill_catalog({}) == ""


def test_load_skill_body_returns_body_or_none(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter", "greeter", "Greets", "Full instructions.")
    assert load_skill_body("greeter", roots=[tmp_path]) == "Full instructions."
    assert load_skill_body("missing", roots=[tmp_path]) is None


def test_render_loaded_skill_includes_directory_and_body(
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path,
        "runner",
        "runner",
        "Runs a script",
        "Run ./go.py to do the thing.",
    )
    skill = discover_skills(roots=[tmp_path])["runner"]
    rendered = render_loaded_skill(skill)

    # The skill directory is surfaced so relative paths (like ./go.py) resolve.
    assert str(tmp_path / "runner") in rendered
    assert "run_command" in rendered
    assert "Run ./go.py to do the thing." in rendered
