"""Discovery and parsing of on-disk agent skills.

A skill is a directory holding a ``SKILL.md`` file whose YAML frontmatter
carries at least a ``description``. The markdown body after the frontmatter is
the instruction text handed to the model.

Skills are discovered from two roots, in ascending order of precedence:

* ``~/.agents/skills`` - user skills, available in every workspace.
* ``<cwd>/.agents/skills`` - project skills, which shadow user skills by name.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

AGENTS_DIRNAME = ".agents"
SKILLS_SUBDIR = "skills"
SKILL_FILENAME = "SKILL.md"

PROJECT_SCOPE = "project"
USER_SCOPE = "user"

MAX_INSTRUCTION_CHARACTERS = 30000
"""Cap on instruction text, aligned with ``AgentContext.tool_character_limit``."""

_FRONTMATTER = re.compile(
    r"\A---[ \t]*\r?\n(?P<meta>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_VALID_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Skill:
    """One discovered skill."""

    name: str
    description: str
    instructions: str
    path: Path
    scope: str

    @property
    def directory(self) -> Path:
        """Directory holding ``SKILL.md`` and any supporting files."""
        return self.path.parent


def is_valid_skill_name(name: str) -> bool:
    """Return whether a name is a safe, simple skill identifier.

    Rejecting separators and ``..`` keeps a skill name from escaping its root
    when it is used to build a path or looked up from model-supplied input.
    """
    return bool(_VALID_NAME.fullmatch(name)) and name not in {".", ".."}


def user_skills_root() -> Path:
    """Return the user-wide skills root, ``~/.agents/skills``."""
    return Path.home() / AGENTS_DIRNAME / SKILLS_SUBDIR


def project_skills_root(project_root: Path | None = None) -> Path:
    """Return ``<project_root>/.agents/skills``, defaulting to the cwd.

    The process working directory is used rather than the configured workspace
    so that project skills are still found when the workspace is a throwaway
    temporary directory.
    """
    base = Path.cwd() if project_root is None else Path(project_root)
    return base / AGENTS_DIRNAME / SKILLS_SUBDIR


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading YAML frontmatter from the markdown body.

    Raises
    ------
    yaml.YAMLError
        If the frontmatter block is present but not parseable.
    """
    match = _FRONTMATTER.match(text.lstrip("﻿"))
    if match is None:
        return {}, text
    loaded = yaml.safe_load(match.group("meta"))
    metadata = loaded if isinstance(loaded, dict) else {}
    return metadata, text[match.end() :]


def load_skill(directory: Path, scope: str) -> Skill | None:
    """Load one skill directory, or return None when it is unusable."""
    name = directory.name
    if not is_valid_skill_name(name):
        logger.warning("Ignoring skill with unusable name: %s", directory)
        return None

    path = directory / SKILL_FILENAME
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read skill %s: %s", path, exc)
        return None

    try:
        metadata, body = split_frontmatter(text)
    except yaml.YAMLError as exc:
        logger.warning("Malformed frontmatter in %s: %s", path, exc)
        metadata, body = {}, text

    description = " ".join(str(metadata.get("description") or "").split())
    if not description:
        description = f"Skill '{name}' (no description provided)"

    instructions = body.strip()
    if len(instructions) > MAX_INSTRUCTION_CHARACTERS:
        instructions = (
            instructions[:MAX_INSTRUCTION_CHARACTERS]
            + f"\n\n[Truncated at {MAX_INSTRUCTION_CHARACTERS} characters."
            + f" Read {path} for the full text.]"
        )

    return Skill(
        name=name,
        description=description,
        instructions=instructions,
        path=path,
        scope=scope,
    )


def discover_skills_in(root: Path, scope: str) -> dict[str, Skill]:
    """Load every usable skill directly beneath one skills root."""
    found: dict[str, Skill] = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not (entry / SKILL_FILENAME).is_file():
            continue
        if (skill := load_skill(entry, scope)) is not None:
            found[skill.name] = skill
    return found


def discover_skills(project_root: Path | None = None) -> dict[str, Skill]:
    """Return every discovered skill by name, project skills winning."""
    skills = discover_skills_in(user_skills_root(), USER_SCOPE)
    skills.update(
        discover_skills_in(project_skills_root(project_root), PROJECT_SCOPE)
    )
    return dict(sorted(skills.items()))
