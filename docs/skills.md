# Skills

Skills give URSA reusable instructions for particular tasks. URSA follows the
[Agent Skills specification](https://agentskills.io/specification): each skill
is a directory containing `SKILL.md`, with YAML metadata and Markdown instructions.

## Create a skill

For example, create `.ursa/skills/lab-report/SKILL.md`:

```markdown
---
name: lab-report
description: Write a laboratory report from experimental results. Use when organizing methods, observations, uncertainty, and conclusions.
---

When writing a report:
1. Separate measured results from interpretations.
2. Describe methods and units.
3. Report uncertainty and limitations.
4. End with conclusions supported by the observations.

Read references/report-template.md for the report structure.
```

Put the supporting template in
`.ursa/skills/lab-report/references/report-template.md`. Optional `scripts/`,
`references/`, and `assets/` directories keep supporting resources with the skill.
References are relative to the skill directory, not URSA's working directory.
Skill scripts run using the agent's existing tools and permission checks.

Names must match their directory, contain 1–64 lowercase ASCII letters, digits,
or single hyphens, and cannot begin or end with a hyphen. Descriptions must be
nonempty strings of at most 1024 characters. Standard optional metadata fields
are retained. The experimental `allowed-tools` field does not grant permissions.

## Where URSA looks

| Selection | Skill directories |
|---|---|
| No explicit workspace | `~/.ursa/skills` and `<launch-directory>/.ursa/skills` |
| Explicit workspace | Only `<workspace>/.ursa/skills` |

Project skills override home skills with the same name. Explicit workspace
selection includes `--workspace`, configuration files/environment overrides,
Python's `workspace=` constructor argument, and dashboard workspace choices.
Even `--workspace .` isolates discovery from home skills. Temporary workspaces
also use only their own `.ursa/skills`; an empty catalog never falls back.

URSA scans immediate child skill directories, including symlinks. It does not
search ancestor directories or other clients' skill locations. Invalid or
unreadable skills are skipped with diagnostics. Listing and each new request
refresh the catalog, so newly installed skills do not require a restart.

## List and activate skills

In the TUI, enter `/skills` or choose **skills** from the `/` command menu.
In the dashboard, submit `/skills` in the session composer. Both show available
names, descriptions, paths, and diagnostics without starting an agent run.
The dashboard reads files on the machine running its server.

Request a skill explicitly:

```text
$lab-report Write a report from these measurements.
```

Multiple references are supported. Repeated references load once. Escaped
references (`\$lab-report`) and references inside Markdown code are literal.
An unknown explicit name produces an error with guidance to list skills.

Agents also see the catalog and are instructed to call `load_skill` when a
description matches the task or conversation. This contextual choice is made
by the model. Full instructions are added only when a skill is loaded, and
supporting text files can be loaded with `read_skill_resource`.

Loaded instructions and resources remain available throughout the current
request, including context compaction. They are not saved as permanent
conversation instructions. Later requests reconsider relevance and load skills
again as needed. Skill activation appears in the usual agent event output.

## Python

```python
from pathlib import Path
from ursa.agents import ChatAgent
from ursa.skills import discover_skills

# Inspect skills without constructing an LLM or agent.
catalog = discover_skills(workspace=Path("./experiment"))
for skill in catalog.skills:
    print(skill.name, skill.description, skill.path)

# Supply your configured LangChain chat model as llm.
agent = ChatAgent(llm=llm, workspace=Path("./experiment"))
print(agent.list_skills().markdown())
result = agent.invoke("$lab-report Summarize the results in measurements.csv")
print(agent.format_result(result))
```

Omit `workspace` to combine home and launch-directory skills. `invoke`,
`ainvoke`, and `stream` support activation; `/skills` is an interface command,
so scripts use the listing API instead.

## Supported agents

Skills are supported by ChatAgent, ExecutionAgent, DSIAgent, DeepReviewAgent,
and their subclasses, including GitAgent and GitGoAgent. Non-tool agents such
as BasicChatAgent, PromptingAgent, PlanningAgent, and other specialized agents
do not activate skills. Workflows can use skills in their supported component
agents; selecting a workflow does not add skill support to every component.

Custom conversational tool agents can use `self.skill_messages(messages)` on
outbound model messages. This supplies current skill context without adding it
to checkpointed conversation messages. Skill tools are registered through
`AgentWithTools`.
