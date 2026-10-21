"""Markdown rendering of the skill catalog for the TUI's /skills page."""

from pathlib import Path

from ursa.skills.discovery import SkillCatalog, get_skill_catalog

_INTRO = """
Skills follow the [Agent Skills specification](https://agentskills.io/specification).
Project skills come from the configured workspace and override user and bundled skills.
""".strip()

_USAGE = """
Type `$` in the prompt to pick a skill, or describe a matching task.
Restart URSA after adding or editing skills to reload the catalog.
""".strip()


def skills_markdown(
    project_root: Path | None = None, *, catalog: SkillCatalog | None = None
) -> str:
    catalog = get_skill_catalog(project_root) if catalog is None else catalog
    skills = catalog.load()
    lines = [_INTRO, "", _USAGE, ""]
    if not skills:
        lines.append("No skills found.")
    for skill in skills.values():
        # Paragraphs wrap naturally even with a 1024-character description.
        lines += [
            f"### ${skill.name}",
            "",
            skill.description,
            "",
            f"Scope: {skill.scope.value}. Source: `{skill.path}`",
            "",
        ]
    lines += ["### Search locations (lowest to highest priority)", ""]
    lines += [f"- {scope.value}: `{root}`" for root, scope in catalog.roots]
    return "\n".join(lines)
