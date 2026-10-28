---
name: skill-creation
description: Create, edit, or review an URSA skill. Use when the user asks to make a new skill, capture a repeatable workflow or house style as a skill, or wants to know how skills are structured.
---

# Authoring an URSA skill

A skill is a folder containing `SKILL.md`. URSA supports the [Agent Skills specification](https://agentskills.io/specification).
Prefer these shared locations:

| Root | Scope | Use for |
|---|---|---|
| `.agents/skills/` in the configured workspace | project | Conventions for one repository or study |
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

- `name` and `description` are required frontmatter fields.
- `name` must match the folder name, be 1–64 lowercase letters or digits
  separated by single hyphens, and have no leading or trailing hyphen.
- `description` must be a non-empty string of at most 1024 characters.
  Write it as "what it does + when to use it", naming concrete triggers.
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
4. Restart URSA to reload the catalog. The user can run `/skills` in the TUI to see it listed,
   and `$<name>` to invoke it.

Use `write_code` to create `SKILL.md` and `edit_code` to revise an existing
one. Read the current file with `read_file` before editing it.
