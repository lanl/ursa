# Environments: agents working together

A single URSA agent is useful when one assistant can own the whole task. Some work,
though, benefits from more structure: one agent should coordinate, another should
analyze data, another should critique assumptions, and another might synthesize the
final answer for a human reader. URSA environments are built for that kind of work.

An environment composes agents, and sometimes other environments, behind the same
simple interface used by individual workflows:

```python
result = environment.invoke("Solve this problem...")
```

or, inside async code:

```python
result = await environment.ainvoke("Solve this problem...")
```

The important difference is what happens behind that call. Instead of sending the
prompt to one agent, an environment gives the task to a structured group and guides
how the members interact.

## The two built-in environment patterns

URSA currently includes two multi-agent environment patterns.

<div class="grid cards" markdown>

-   :material-account-group: **Agent Teams**

    ---

    A hierarchical team led by a PI, or principal investigator. The PI is the
    user-facing coordinator. Team members are exposed to the PI as delegation
    tools, so the PI can assign focused work, inspect returned results, ask for
    follow-up, and produce a single coherent answer.

    Use a team when you want one agent to manage a small group of specialists.

    [:octicons-arrow-right-24: Agent Teams](agent-teams.md)

-   :material-forum: **Agent Symposia**

    ---

    A peer-review style environment. Multiple members, which can be individual
    agents or nested teams, work independently on the same problem. They then
    review the submitted writeups, revise their own work, and an organizer
    synthesizes the final report.

    Use a symposium when you want independent approaches, critique, comparison,
    and revision before the final answer.

    [:octicons-arrow-right-24: Agent Symposia](agent-symposia.md)

</div>

## Visualize environment runs

Environment runs can be recorded and replayed with `run_with_visualization(...)`
and `arun_with_visualization(...)`. Recorded runs include the environment graph,
major phases, delegation events, tool activity, errors, and final results.

Use visualization when you want to debug or audit how a team or symposium reached
its answer.

[:octicons-arrow-right-24: Visualizing Environment Runs](visualization.md)

## Why use an environment instead of one larger prompt?

Environments make the collaboration pattern explicit. That matters because complex
work often fails for social reasons as much as technical ones: an assistant jumps
to a conclusion too early, ignores a specialized check, forgets to document a
result, or never asks a critic to look for failure modes.

With an environment you can encode the shape of the work:

- who coordinates,
- who contributes specialist evidence,
- where shared files live,
- whether members work sequentially or independently,
- when critique happens,
- and who owns the final synthesis.

This is not a guarantee that every answer is correct. It is a way to give agents a
better workflow: delegation for teams, independent comparison and review for
symposia, and persistent state so named members can build continuity over time.

## A minimal Python example

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentTeamEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")

team = AgentTeamEnvironment.from_yaml(
    "examples/environments/agent_team.yaml",
    llm=llm,
)

result = team.invoke(
    "Analyze this research question and return a concise recommendation."
)

print(result["messages"][-1].content if "messages" in result else result)
```

The same pattern works for a symposium:

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")

symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml",
    llm=llm,
)

result = symposium.invoke("Solve this complex problem and compare alternatives.")
print(result["final"])
```

!!! note "No separate environment CLI yet"
    Environments are currently used from Python. They load from YAML and expose
    `invoke(...)` and `ainvoke(...)`, but the implementation inspected here does
    not add a dedicated `ursa environments ...` command.

## YAML as the collaboration plan

A YAML environment file describes the people in the room. For example, an agent
team file names the team, the PI, the member agents, their roles, and optional
member-specific guidance. A symposium file names an organizer, the participants,
and how many review/revision rounds to run.

Each member can use a short built-in class name such as `ChatAgent`,
`ExecutionAgent`, or `AgentTeamEnvironment`, or a fully qualified Python class path
for custom agents. Members can also receive their own model configuration, which
lets you vary model providers or model sizes across the group.

## Persistence and workspace behavior

Environments separate collaboration configuration from agent checkpoint state:

- saved team configs default to
  `~/.cache/ursa/<group>/environments/agent_teams/<name>/team.yaml`,
- saved symposium configs default to
  `~/.cache/ursa/<group>/environments/agent_symposia/<name>/symposium.yaml`,
- environment workspaces default to
  `~/.cache/ursa/<group>/environments/workspaces/<name>` when no workspace is set,
- persistent member agents use the normal URSA named-agent cache under the same
  group.

For agent teams, members and the PI share the team workspace so they can
collaborate through files. For symposia, each member is built as an agent or nested
environment with its configured workspace behavior.

## Safety reminder

Environments can contain agents that use web access, write files, or run commands.
Treat the whole environment as powerful as its most capable member. Use dedicated
workspaces, review generated code and commands, and apply the same sandboxing
practices you would use for an `ExecutionAgent`.

Next, read about [Agent Teams](agent-teams.md) or [Agent Symposia](agent-symposia.md).
