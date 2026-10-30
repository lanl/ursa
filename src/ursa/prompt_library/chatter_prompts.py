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


def resolved(path: str) -> Path:
    return Path(path).expanduser().resolve()


def get_chatter_system_prompt():
    # Set URSA_SKILLS_DIR env var to colon-separated dirs to discover skills.
    skills_directories = os.environ.get("URSA_SKILLS_DIRS")
    skill_catalog = create_skill_catalog(
        directories=[]
        if skills_directories is None
        else [
            resolved(d)
            for d in skills_directories.strip().split(":")
            if resolved(d).is_dir()
        ]
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
