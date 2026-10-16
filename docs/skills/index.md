# Skills

A skill is a folder containing a `SKILL.md` file: instructions that teach URSA
how you want a particular kind of work done. Skills let you capture a house
style, a repeatable procedure, or hard-won conventions once and have the
[chat agent](../agents/chat.md) apply them whenever they are relevant.

## Where skills come from

URSA discovers skills from two roots:

| Root | Scope | Use for |
|---|---|---|
| `.ursa/skills/` in the directory URSA was started in | project | Conventions for one repository or study |
| `~/.ursa/skills/` | user | Habits that follow you everywhere |

A project skill shadows a user skill of the same name, so a repository can pin
its own version of a skill you also keep globally.

The project root is the directory URSA was launched from, not the `--workspace`
directory. A workspace is often a scratch directory, while skills belong with
the project you are working on.

## Layout

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

The folder name is the skill's identity: it must start with a letter or digit
and contain only letters, digits, `.`, `_`, and `-`. `description` is the only
field URSA requires, and it is the line that decides whether a skill is applied,
so write it as *what it does* plus *when to use it*. Everything after the
closing `---` is passed to the model verbatim.

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
inserts the name you choose; typing the name by hand works the same way. A `$`
that does not match a known skill name is left alone, so `$PATH` and `US$5` pass
through untouched.

Run `/skills` in the TUI for the full catalog, each skill's scope, and the file
it was loaded from.

## Creating a skill

URSA ships a `skill-creation` skill and writes it to
`~/.ursa/skills/skill-creation/SKILL.md` on first launch, so the fastest route
is to ask:

```text
Create a skill that captures how we run simulations in this repo $skill-creation
```

Or write the file yourself; discovery re-reads both roots on each use, so a new
skill is available immediately without restarting URSA.

URSA only writes that file when it is not already there, so any edit you make to
it is permanent, and a hand-written `skill-creation` of your own is never
touched.

## Disabling skills

`ChatAgent(llm=llm, use_skills=False)` omits the `skill` tool and leaves `$`
references in prompts untouched.

## Trust

Skills are local files that you or your project placed on disk, and their
instructions reach the model verbatim — the same trust level as your own
configuration. Review a skill before copying it in from elsewhere. Skill
instructions do not bypass URSA's other safeguards; `run_command` still applies
its safety checks.
