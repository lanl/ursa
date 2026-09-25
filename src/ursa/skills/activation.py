"""Explicit ``$skill-name`` activation from prompt text.

Explicit and automatic activation share one path: a ``$name`` reference is
turned into a directive to call the ``skill`` tool, rather than splicing the
skill body into the prompt. That keeps the transcript identical either way, and
avoids the mid-conversation ``SystemMessage`` that ``_sanitize_history`` in
``ursa.agents.base`` strips out.
"""

import re
from collections.abc import Mapping
from pathlib import Path

from ursa.skills.discovery import Skill, discover_skills
from ursa.skills.tools import SKILL_TOOL_NAME

# Requires a non-word character before the "$" so that shell-style "$VAR"
# inside a code span still matches, but "US$5" does not.
_REFERENCE = re.compile(r"(?<![\w$])\$([A-Za-z0-9][A-Za-z0-9._-]*)")


def requested_skill_names(
    prompt: str,
    skills: Mapping[str, Skill] | None = None,
    project_root: Path | None = None,
) -> list[str]:
    """Return the known skills referenced as ``$name``, in order of appearance.

    Unknown references are ignored so that ordinary uses of ``$`` in a prompt
    are left alone.
    """
    candidates = _REFERENCE.findall(prompt)
    if not candidates:
        return []
    catalog = discover_skills(project_root) if skills is None else skills
    if not catalog:
        return []

    names: list[str] = []
    for candidate in candidates:
        # Trailing punctuation is part of the sentence, not of the name.
        name = candidate.rstrip("._-")
        if name in catalog and name not in names:
            names.append(name)
    return names


def annotate_skill_requests(
    prompt: str,
    skills: Mapping[str, Skill] | None = None,
    project_root: Path | None = None,
) -> str:
    """Append a directive naming every skill the prompt asked for.

    The original ``$name`` text is left in place because it reads naturally.
    """
    names = requested_skill_names(prompt, skills, project_root)
    if not names:
        return prompt
    return (
        f"{prompt}\n\n"
        f"[The user explicitly requested these skills: {', '.join(names)}. "
        f"Call the `{SKILL_TOOL_NAME}` tool once for each of them, then follow "
        "the instructions it returns.]"
    )
