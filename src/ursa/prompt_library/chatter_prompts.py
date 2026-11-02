import json
import os
from pathlib import Path

from ursa.skills import create_skill_catalog
from ursa.skills.prompt import skills_system_prompt_template

chatter_system_prompt_template = """
You are the chat interface to URSA, a flexible agentic workflow for accelerating scientific tasks.
Do not speculate about the capabilities of URSA beyond the information given to you.
The documentation for URSA is available at https://lanl.github.io/ursa/.

{extra}
"""


def get_chatter_system_prompt():
    local_skills_dir = Path.cwd() / ".ursa" / "skills"

    skills_dirs = [local_skills_dir]
    if paths := os.environ.get("URSA_SKILLS_PATH", "").strip():
        path_list = (
            paths.strip().split(";")
            if os.name == "nt"  # For windows
            else paths.strip().split(":")
        )
        skills_dirs.extend(path_list)

    skill_catalog = create_skill_catalog(
        directories=[Path(d) for d in skills_dirs if Path(d).is_dir()]
    )

    if skill_catalog:
        skills_system_prompt = skills_system_prompt_template.format(
            catalog=json.dumps(skill_catalog, indent=2, sort_keys=True)
        )
    else:
        skills_system_prompt = ""

    return chatter_system_prompt_template.format(
        extra=skills_system_prompt
    ).strip()
