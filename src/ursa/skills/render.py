"""Markdown rendering of the skill catalog for the TUI's ``/skills`` page."""

from pathlib import Path

from ursa.skills.discovery import discover_skills

_INTRO = """
Skills are loaded from `.ursa/skills/` in the directory URSA was \
started in, and from `~/.ursa/skills/`. A project skill shadows a user \
skill with the same name.
""".strip()

_USAGE = """
Type `$` in the prompt to pick a skill, or write `$name` directly. You \
can also just describe the task: URSA loads a matching skill on its own.
""".strip()

_EMPTY = """
No skills found yet.

Ask URSA to create one and it will use the bundled `skill-creation` skill to \
write it for you.
""".strip()


def _cell(text: str) -> str:
    """Escape one markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def skills_markdown(project_root: Path | None = None) -> str:
    """Render the discovered skills as a markdown page."""
    catalog = discover_skills(project_root)
    lines = [_INTRO, ""]

    if not catalog:
        lines.append(_EMPTY)
        return "\n".join(lines)

    lines += [
        "| Skill | Scope | Description |",
        "|---|---|---|",
        *(
            f"| `${skill.name}` | {skill.scope} | {_cell(skill.description)} |"
            for skill in catalog.values()
        ),
        "",
        _USAGE,
        "",
        "### Locations",
        "",
        *(f"- `${skill.name}` — `{skill.path}`" for skill in catalog.values()),
    ]
    return "\n".join(lines)
