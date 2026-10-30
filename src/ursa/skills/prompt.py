# From: https://agentskills.io/client-implementation/adding-skills-support#behavioral-instructions
skills_system_prompt_template = """
The following skills provide specialized instructions for specific tasks.
When a task matches a skill's description, use your read_file tool to load
the SKILL.md at the listed location before proceeding.
When a skill references relative paths, resolve them against the skill's
directory (the parent of SKILL.md) and use absolute paths in tool calls.

{catalog}
""".strip()
