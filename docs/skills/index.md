# Skills

URSA supports the [Agent Skills specification](https://agentskills.io/specification).
A skill is a folder containing a `SKILL.md` file: instructions that teach URSA
how you want a particular kind of work done. Skills let you capture a house
style, a repeatable procedure, or hard-won conventions once and have the
[chat agent](../agents/chat.md) apply them whenever they are relevant.

## Where skills come from

URSA searches these locations in increasing priority order. Later locations
shadow earlier ones with the same skill name, and URSA logs each shadowed path.

| Root | Scope |
|---|---|
| `skills/` inside the URSA package | bundled |
| `~/.claude/skills/`, `~/.codex/skills/`, `~/.agents/skills/`, `~/.ursa/skills/` (in this order) | user |
| `$XDG_CONFIG_HOME/ursa/skills/` (defaults to `~/.config/ursa/skills/`) | user |
| `<workspace>/.claude/skills/`, `<workspace>/.codex/skills/`, `<workspace>/.agents/skills/`, `<workspace>/.ursa/skills/` (in this order) | project |

The project root is the configured `--workspace`, which defaults to the current
directory. URSA scans immediate skill folders under these roots; it does not
walk parent directories outside the selected workspace. See the
[discovery conventions](https://agentskills.io/client-implementation/adding-skills-support#where-to-scan)
for cross-client sharing through `.agents/skills/`.

## Layout

Use the standard [directory structure](https://agentskills.io/specification#directory-structure):

```
.ursa/skills/write-python/
├── SKILL.md          # required
├── reference.md      # optional supporting material
└── scripts/check.sh  # optional helper scripts
```

`SKILL.md` is YAML frontmatter followed by a markdown body:

```markdown
---
name: write-python
description: House style for Python in this repo. Use when writing or editing Python files.
---

# Writing Python here

- Run code with `uv run`, never bare `python`.
- Keep lines under 80 characters.
```

Both `name` and `description` are required. The name must match its folder,
contain 1–64 lowercase letters or digits separated by single hyphens, and have
no leading or trailing hyphen. The description must be a non-empty string of
at most 1024 characters. Write it as *what it does* plus *when to use it*;
multiline descriptions are preserved. URSA skips invalid skills and logs why.

Metadata is validated once per catalog. Instruction bodies are loaded and cached
when activated, with tool responses bounded by `AgentContext.tool_character_limit`.

Supporting files are not loaded automatically. URSA tells the model where the
skill's folder is, so instructions can point at a sibling file by name and it
will be read only when needed.

## Using a skill

Skills activate two ways, and both deliver the same instructions.

**Automatically.** The chat agent has a `skill` tool whose description lists
every discovered skill and its description. When your request matches one, the
agent loads it before starting the work.

**Explicitly.** Write `$` followed by the skill name anywhere in a prompt:

```text
Tidy up analysis.py $write-python
```

In the TUI, typing `$` opens a fuzzy picker over the discovered skills and
inserts the name you choose. A `$` that does not match a known skill name is
left alone, so `$PATH` and `US$5` pass through untouched.

Run `/skills` in the TUI for the full catalog, each skill's scope, and the file
it was loaded from.

## Creating a skill

URSA ships `skills/skill-creation/SKILL.md` as package data, discovered through
`importlib.resources` at the lowest priority. To use it, ask:

```text
Create a skill that captures how we run simulations in this repo $skill-creation
```

Or write the file yourself. The agent and TUI share a cached workspace catalog;
restart URSA after adding or editing skills. Discovery runs off the TUI event loop.

To customize `skill-creation`, place a skill with the same name in a user or
project root. URSA never copies bundled files into your home directory.

## Disabling skills

`ChatAgent(llm=llm, use_skills=False)` omits the `skill` tool and leaves `$`
references in prompts untouched.

## Trust

Skills are local files that you or your project placed on disk, and their
instructions reach the model verbatim — the same trust level as your own
configuration. Review a skill before copying it in from elsewhere. Skill
instructions do not bypass URSA's other safeguards; `run_command` still applies
its safety checks.
