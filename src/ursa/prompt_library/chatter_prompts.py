import json

from ursa.skills import create_skill_catalog, skill_directories
from ursa.skills.prompt import skills_system_prompt_template

chatter_system_prompt_template = """
You are the chat interface to URSA, a flexible agentic workflow for accelerating scientific tasks.
Do not speculate about the capabilities of URSA beyond the information given to you.
The documentation for URSA is available at https://lanl.github.io/ursa/.

{extra}
"""


def get_chatter_system_prompt():
    # Discover skills and build the catalog from the same resolved roots that
    # read_file authorizes.
    skill_catalog = create_skill_catalog(
        directories=[root for root in skill_directories() if root.is_dir()]
    )

    # Build the skills catalog prompt
    if skill_catalog:
        skills_system_prompt = skills_system_prompt_template.format(
            catalog=json.dumps(skill_catalog, indent=2, sort_keys=True)
        )
    else:
        skills_system_prompt = ""

    return chatter_system_prompt_template.format(
        extra=skills_system_prompt
    ).strip()
