"""Discover and parse Agent Skills for progressive disclosure.

An Agent Skill is a directory containing a ``SKILL.md`` file with YAML
frontmatter (at least ``name`` and ``description``) followed by a markdown
body. Skills follow the Claude/Codex convention: only the lightweight
frontmatter (name + description) is always loaded into the agent's context,
while the full body is pulled in dynamically -- via the ``load_skill`` tool --
only when the current task actually calls for it.

Skills are discovered from two roots, project first so it shadows the global
copy on a name collision:

- ``./.ursa/skills/`` -- project-local skills (current working directory)
- ``~/.ursa/skills/`` -- user-global skills (home directory)

Each skill lives at ``<root>/<skill>/SKILL.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from ursa.util.parse import read_text_from_file

logger = logging.getLogger(__name__)

SKILL_DIR_NAME = ".ursa/skills"
SKILL_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """A discovered Agent Skill.

    Attributes:
        name: The skill name from frontmatter (used to load it by the tool).
        description: One-line description advertised in the always-in-context
            catalog.
        path: Path to the skill's ``SKILL.md`` file.
        source: Where the skill was found, ``"project"`` or ``"global"``.
        body: The full markdown body (frontmatter stripped).
    """

    name: str
    description: str
    path: Path
    source: str
    body: str


def default_skill_roots() -> list[Path]:
    """Return the skill roots to scan, project first then user-global."""
    return [
        Path.cwd() / SKILL_DIR_NAME,
        Path.home() / SKILL_DIR_NAME,
    ]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` YAML frontmatter block from a markdown document.

    Args:
        text: Raw file contents.

    Returns:
        A ``(metadata, body)`` tuple. ``metadata`` is an empty dict when no
        valid frontmatter block is present, in which case ``body`` is the
        original text unchanged.
    """
    stripped = text.lstrip("﻿")  # tolerate a leading BOM
    if not stripped.startswith("---"):
        return {}, text

    # Frontmatter is the block between the first two '---' fences.
    lines = stripped.splitlines(keepends=True)
    # lines[0] is the opening '---'. Find the closing fence.
    closing = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing = i
            break
    if closing is None:
        return {}, text

    front_text = "".join(lines[1:closing])
    body = "".join(lines[closing + 1 :])
    # Strip a single leading newline left after the closing fence.
    body = body[1:] if body.startswith("\n") else body

    try:
        meta = yaml.safe_load(front_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse skill frontmatter: %s", exc)
        return {}, text

    if not isinstance(meta, dict):
        return {}, text

    return meta, body


def _load_skill_from_file(path: Path, source: str) -> Skill | None:
    """Parse a single ``SKILL.md`` into a :class:`Skill`, or ``None`` if invalid."""
    try:
        text = read_text_from_file(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read skill file %s: %s", path, exc)
        return None

    meta, body = parse_frontmatter(text)
    name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()

    # Fall back to the directory name when frontmatter omits a name.
    if not name:
        name = path.parent.name.strip()

    if not name or not description:
        logger.warning(
            "Skipping skill %s: missing required 'name' or 'description' "
            "frontmatter.",
            path,
        )
        return None

    return Skill(
        name=name,
        description=description,
        path=path,
        source=source,
        body=body.strip(),
    )


def discover_skills(
    roots: list[Path] | None = None,
) -> dict[str, Skill]:
    """Discover all skills across the given roots.

    Args:
        roots: Directories to scan for ``<skill>/SKILL.md``. Defaults to
            :func:`default_skill_roots` (project then user-global).

    Returns:
        A mapping of skill name to :class:`Skill`. When the same name is found
        in more than one root, the first root wins (project shadows global).
    """
    if roots is None:
        roots = default_skill_roots()

    skills: dict[str, Skill] = {}
    for root in roots:
        source = "project" if root == Path.cwd() / SKILL_DIR_NAME else "global"
        if not root.is_dir():
            continue
        for skill_file in sorted(root.glob(f"*/{SKILL_FILENAME}")):
            skill = _load_skill_from_file(skill_file, source)
            if skill is None:
                continue
            # First-seen wins so earlier roots (project) shadow later ones.
            skills.setdefault(skill.name, skill)
    return skills


def render_skill_catalog(skills: dict[str, Skill]) -> str:
    """Render the always-in-context catalog block for a set of skills.

    Args:
        skills: Mapping of skill name to :class:`Skill`.

    Returns:
        A markdown block listing each skill's name and description, or an empty
        string when there are no skills.
    """
    if not skills:
        return ""

    lines = [
        "",
        "## Available Skills",
        "",
        "You have access to skills: reusable sets of instructions for specific "
        "tasks. Only their names and descriptions are shown below. When the "
        "current task matches a skill, call the `load_skill` tool with the "
        "skill's name ONCE to load its full instructions, then treat that "
        "content as authoritative and answer directly from it. Loaded "
        "instructions start with the skill's directory; if the skill bundles a "
        "script, resolve its relative path against that directory and run it "
        "with the `run_command` tool using the absolute path. Do not reload a "
        "skill you have already loaded this conversation, and do not hedge "
        "about or second-guess skill content. Do not load a skill unless the "
        "task calls for it.",
        "",
        "Available skills:",
    ]
    for skill in skills.values():
        lines.append(f"- {skill.name}: {skill.description}")
    lines.append("")
    return "\n".join(lines)


def render_loaded_skill(skill: Skill) -> str:
    """Render a skill's body with the context needed to run bundled scripts.

    Skills may ship scripts or data files alongside ``SKILL.md``. The body
    refers to them with paths relative to that directory, but tools such as
    ``run_command`` execute in the workspace, not the skill directory. Prefixing
    the body with the skill's absolute directory lets the model resolve those
    relative paths and invoke bundled scripts with an absolute path.

    Args:
        skill: The skill whose body to render.

    Returns:
        The skill body preceded by a short context block naming the skill's
        directory and how to run any bundled scripts.
    """
    skill_dir = skill.path.parent
    header = (
        f"[Skill: {skill.name}]\n"
        f"Skill directory: {skill_dir}\n"
        "Any file paths in the instructions below are relative to this "
        "directory. To run a bundled script or reference a bundled file, build "
        "an absolute path from the skill directory above (for example, run "
        f"`python {skill_dir / 'script.py'}` with the run_command tool). Do not "
        "assume bundled files exist in the workspace.\n\n"
        "---\n\n"
    )
    return header + skill.body


def load_skill_body(name: str, roots: list[Path] | None = None) -> str | None:
    """Return the full body of the named skill, or ``None`` if not found.

    Args:
        name: The skill name to load.
        roots: Optional discovery roots (see :func:`discover_skills`).

    Returns:
        The skill's markdown body, or ``None`` when no skill matches ``name``.
    """
    skills = discover_skills(roots)
    skill = skills.get(name.strip())
    return skill.body if skill is not None else None
