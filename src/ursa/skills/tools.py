"""The skill tool exposes a stable catalog and loads instructions on demand."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

from ursa.agents.base import AgentContext
from ursa.skills.discovery import Skill, get_skill_catalog

SKILL_TOOL_NAME = "skill"

_GUIDANCE = (
    "Load a skill when the current work matches its description, before "
    "starting that work. Apply only the instructions relevant to the current "
    "request, and reassess applicability when the task changes. Explicit user "
    "instructions take precedence. Resolve relative file references against "
    "the directory containing the source file. Pass a name from the catalog."
)


def skill_catalog_listing(skills: Mapping[str, Skill]) -> str:
    return "\n".join(
        f"- {skill.name}: {skill.description}" for skill in skills.values()
    )


def render_skill_instructions(skill: Skill, character_limit: int) -> str:
    frontmatter = yaml.safe_dump(
        {"name": skill.name, "source": str(skill.path)}, sort_keys=False
    )
    text = f"---\n{frontmatter}---\n\n{skill.instructions}"
    limit = max(0, character_limit)
    if len(text) > limit:
        notice = (
            f"\n\n[Truncated. Read {skill.path} for the full instructions.]"
        )
        if len(notice) >= limit:
            return "[Truncated]"[:limit]
        text = text[: limit - len(notice)] + notice
    return text


def load_skill_instructions(
    name: str,
    project_root: Path | None = None,
    *,
    skills: Mapping[str, Skill] | None = None,
    character_limit: int = AgentContext.tool_character_limit,
) -> str:
    requested = name.strip().lstrip("$").strip()
    catalog = (
        get_skill_catalog(project_root).load() if skills is None else skills
    )
    skill = catalog.get(requested)
    if skill is None:
        available = ", ".join(catalog) or "none"
        return f"No skill named '{requested}'. Available skills: {available}."
    try:
        return render_skill_instructions(skill, character_limit)
    except (OSError, yaml.YAMLError) as exc:
        return f"Could not load skill '{requested}' from {skill.path}: {exc}"


def build_skill_tool(
    project_root: Path | None = None,
    skills: Mapping[str, Skill] | None = None,
) -> BaseTool | None:
    catalog = (
        get_skill_catalog(project_root).load()
        if skills is None
        else dict(skills)
    )
    if not catalog:
        return None

    def load(name: str, runtime: ToolRuntime[AgentContext] = None) -> str:
        limit = (
            runtime.context.tool_character_limit
            if runtime is not None
            else AgentContext.tool_character_limit
        )
        return load_skill_instructions(
            name, skills=catalog, character_limit=limit
        )

    async def aload(
        name: str, runtime: ToolRuntime[AgentContext] = None
    ) -> str:
        return await asyncio.to_thread(load, name, runtime)

    tool = StructuredTool.from_function(
        func=load,
        coroutine=aload,
        name=SKILL_TOOL_NAME,
        description=f"{_GUIDANCE}\n\nAvailable skills:\n{skill_catalog_listing(catalog)}",
    )
    tool.args_schema = create_model(
        "SkillInput",
        __base__=tool.args_schema,
        name=(
            Literal[tuple(catalog)],
            Field(description="Name of the skill to load."),
        ),
    )
    return tool
