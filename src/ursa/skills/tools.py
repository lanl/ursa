"""The ``skill`` tool, which is how skills reach the model.

A single dispatcher tool is used rather than one tool per skill so that skills
added or edited mid-session take effect without rebuilding the agent's graph:
the tool re-reads the skills roots on every call. The tool *description* lists
the skills present when the agent was built, which is what lets the model
activate a skill on its own from the conversation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ursa.skills.discovery import Skill, discover_skills

SKILL_TOOL_NAME = "skill"

_GUIDANCE = (
    "Load a skill's instructions, then follow them for the rest of the task. "
    "A skill holds expert procedures and conventions for one kind of work. "
    "Call this as soon as the conversation matches a skill's description and "
    "before starting the work that skill covers, not after. Pass the skill "
    "name exactly as listed below."
)


class SkillInput(BaseModel):
    """Arguments accepted by the ``skill`` tool."""

    name: str = Field(
        description=(
            "Name of the skill to load, exactly as listed in this tool's "
            "description."
        )
    )


def skill_catalog_listing(skills: Mapping[str, Skill]) -> str:
    """Render the skill list embedded in the tool description."""
    return "\n".join(
        f"- {skill.name}: {skill.description}" for skill in skills.values()
    )


def render_skill_instructions(skill: Skill) -> str:
    """Render one skill as the text returned to the model."""
    return "\n".join([
        f"# Skill: {skill.name}",
        "",
        f"Source: {skill.path} ({skill.scope} skill)",
        f"Supporting files, if any, are in {skill.directory}",
        "",
        "Follow these instructions for the remainder of this task. They "
        "take precedence over your general habits, but not over an "
        "explicit instruction from the user.",
        "",
        "---",
        "",
        skill.instructions,
    ])


def load_skill_instructions(name: str, project_root: Path | None = None) -> str:
    """Resolve a skill name to its instructions, or to an error message.

    The name arrives from the model, so it is looked up in the catalog rather
    than used to build a path.
    """
    requested = name.strip().lstrip("$").strip()
    catalog = discover_skills(project_root)
    skill = catalog.get(requested)
    if skill is None:
        available = ", ".join(catalog) or "none"
        return (
            f"No skill named '{requested}'. Available skills: {available}. "
            "Do not guess; ask the user or continue without a skill."
        )
    return render_skill_instructions(skill)


def build_skill_tool(
    project_root: Path | None = None,
    skills: Mapping[str, Skill] | None = None,
) -> BaseTool | None:
    """Build the ``skill`` tool, or return None when no skills exist."""
    catalog = discover_skills(project_root) if skills is None else dict(skills)
    if not catalog:
        return None

    def load(name: str) -> str:
        return load_skill_instructions(name, project_root)

    return StructuredTool.from_function(
        func=load,
        name=SKILL_TOOL_NAME,
        description=(
            f"{_GUIDANCE}\n\nAvailable skills:\n"
            f"{skill_catalog_listing(catalog)}"
        ),
        args_schema=SkillInput,
    )
