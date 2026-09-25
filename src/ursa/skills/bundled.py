"""Skills that URSA ships and materializes into the user skills root.

The bundled text lives here as a module constant rather than as package data so
that no ``[tool.setuptools.package-data]`` entry is required to ship it.

A bundled file is written once, when it is not already on disk, and is never
overwritten afterwards. Local edits to it and a user's own skill of the same
name are therefore both left alone.
"""

import logging
from pathlib import Path

import yaml

from ursa.skills.discovery import SKILL_FILENAME, user_skills_root

logger = logging.getLogger(__name__)

SKILL_CREATION_NAME = "skill-creation"
SKILL_CREATION_DESCRIPTION = (
    "Create, edit, or review an URSA skill. Use when the user asks to make a "
    "new skill, capture a repeatable workflow or house style as a skill, or "
    "wants to know how skills are structured."
)
SKILL_CREATION_BODY = """
# Authoring an URSA skill

A skill is a folder containing `SKILL.md`. URSA loads skills from two roots:

| Root | Scope | Use for |
|---|---|---|
| `.agents/skills/` in the working directory | project | Conventions for one repository or study |
| `~/.agents/skills/` | user | Habits that follow the user everywhere |

A project skill shadows a user skill of the same name.

## Layout

```
.agents/skills/write-python/
├── SKILL.md          # required
├── reference.md      # optional supporting material
└── scripts/check.sh  # optional helper scripts
```

## SKILL.md format

```markdown
---
name: write-python
description: House style for Python in this repo. Use when writing or editing Python files.
---

# Writing Python here

- Run code with `uv run`, never bare `python`.
- Keep lines under 80 characters.
```

Rules that matter:

- The folder name is the skill's identity. It must start with a letter or
  digit and contain only letters, digits, `.`, `_`, and `-`. The user types
  `$write-python` to invoke it, so prefer short kebab-case names.
- `description` is the only frontmatter field URSA requires, and it is the
  single most important line in the file: it is what URSA sees when deciding
  whether a skill applies to the conversation. Write it as
  "*what it does* + *when to use it*", naming the concrete triggers.
- Everything after the closing `---` is the instruction body. It is injected
  verbatim, so write imperative instructions addressed to the agent.

## Writing a good skill

1. **Be specific about triggers.** "Use when writing Python" beats "for
   Python work". Name file extensions, tool names, and task phrasings.
2. **Give procedures, not essays.** Numbered steps and checklists are
   followed reliably; prose is not.
3. **State the commands verbatim.** Include exact command lines to run.
4. **Keep it short.** A page or two. Move long material into a sibling file
   and reference it by path so it is read only when needed.
5. **Say what not to do.** Explicit prohibitions prevent common mistakes.

## Procedure for creating a skill

1. Decide the scope. Project-specific conventions go in
   `.agents/skills/`; personal habits go in `~/.agents/skills/`.
2. Pick a short kebab-case name and create
   `<root>/<name>/SKILL.md`.
3. Write the frontmatter `name` and `description` first, then the body.
4. Confirm it loads: the user can run `/skills` in the TUI to see it listed,
   and `$<name>` to invoke it.

Use `write_code` to create `SKILL.md` and `edit_code` to revise an existing
one. Read the current file with `read_file` before editing it.
"""


def render_bundled_skill(name: str, description: str, body: str) -> str:
    """Render a bundled skill file."""
    frontmatter = yaml.safe_dump(
        {"name": name, "description": description},
        sort_keys=False,
        allow_unicode=True,
        width=10**6,
    )
    return f"---\n{frontmatter}---\n\n{body.strip()}\n"


def write_bundled_skill(
    root: Path, name: str, description: str, body: str
) -> Path | None:
    """Write one bundled skill if it is not already on disk.

    Returns the written path, or None when the file already exists.
    """
    path = root / name / SKILL_FILENAME
    if path.exists():
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_bundled_skill(name, description, body), encoding="utf-8"
    )
    logger.debug("Created bundled skill %s", path)
    return path


def ensure_bundled_skills() -> list[Path]:
    """Materialize bundled skills under ``~/.agents/skills``.

    Called once per launch. Never raises: a read-only or unusable home
    directory must not stop URSA from starting.
    """
    try:
        root = user_skills_root()
        written = write_bundled_skill(
            root,
            SKILL_CREATION_NAME,
            SKILL_CREATION_DESCRIPTION,
            SKILL_CREATION_BODY,
        )
        return [written] if written is not None else []
    except Exception:
        logger.warning("Could not materialize bundled skills", exc_info=True)
        return []
