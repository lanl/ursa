from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents
from ursa.util.rendering import file_artifact
from ursa.util.skills import discover_skills, load_skill_body


@tool
def load_skill(name: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Load the full instructions for an available skill by name.

    Skills are reusable, task-specific instruction sets. The agent's system
    prompt advertises each available skill's name and description; call this
    tool to pull a skill's full instructions into context ONLY when the current
    task calls for it.

    Args:
        name: The name of the skill to load (as listed in the available skills).

    Returns:
        The full markdown instructions for the skill, or a message listing the
        available skills when no skill matches the requested name.
    """
    events = ToolEvents.from_runtime("load_skill", runtime)
    skills = discover_skills()
    skill = skills.get(name.strip())

    if skill is None:
        events.emit(
            "Skill not found",
            stage="load_skill",
            phase="error",
            requested=name,
        )
        available = ", ".join(sorted(skills)) or "none"
        return f"Skill '{name}' not found. Available skills: {available}."

    # Guard against reloading the same skill within a conversation. The model
    # is instructed to load a skill once, but nudge-y prompting can make it call
    # again; a repeat load just wastes context, so short-circuit it.
    store = runtime.store
    thread_id = runtime.config.get("metadata", {}).get("thread_id")
    store_key = f"{thread_id}:{skill.name}"
    if store is not None and store.get(("skills", "loaded"), store_key):
        events.emit(
            "Skill already loaded",
            stage="load_skill",
            phase="end",
            name=skill.name,
        )
        return (
            f"Skill '{skill.name}' was already loaded earlier in this "
            "conversation. Answer directly from its instructions; do not "
            "reload it."
        )

    body = load_skill_body(name) or skill.body
    if store is not None:
        store.put(
            ("skills", "loaded"),
            store_key,
            {
                "name": skill.name,
                "tool_call_id": runtime.tool_call_id,
                "thread_id": thread_id,
            },
        )
    events.emit(
        "Skill loaded",
        stage="load_skill",
        phase="end",
        name=skill.name,
        source=skill.source,
        path=str(skill.path),
        artifact=file_artifact(skill.path, title=f"Skill: {skill.name}"),
    )
    return body
