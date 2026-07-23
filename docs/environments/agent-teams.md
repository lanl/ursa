# Agent Teams

An Agent Team is URSA's hierarchical collaboration pattern. It is modeled after a
small research group: a PI understands the user's goal, decides who should do what,
delegates focused work to specialists, reviews their responses, and writes the
final answer.

This pattern is useful when you want one coherent output but do not want one agent
to do every job alone. For example, a team might include:

- a PI that manages scope and writes the final recommendation,
- a research specialist that gathers background evidence,
- a data analyst that inspects files or writes code,
- and a critic that looks for missing assumptions or weak claims.

The user talks to the PI. The PI sees the team roster and receives one delegation
tool per member. When the PI calls a delegation tool, URSA sends the member a
self-contained task prompt that includes the member's role, any member-specific
guidance, the relevant context, and the delegated task. The member returns a
writeup, and the PI integrates it into the final response.

## When a team is the right shape

Choose an Agent Team when the work has one owner but several specialties.

Good fits include:

- preparing a report that needs both research and analysis,
- assigning implementation to an execution-capable member while a PI manages the
  final narrative,
- asking one specialist to gather evidence and another to test a calculation,
- building reusable named specialists that share a workspace and accumulate state.

A team is less suitable when you want several fully independent solutions before
comparison. For that, use an [Agent Symposium][agent-symposia].

## What happens during a team run

A team run follows a simple story:

1. You invoke the team with a task.
2. URSA wraps that task in a PI prompt that describes the team, the PI's
   responsibility, and the available members.
3. The PI decides whether to delegate.
4. Each delegated member receives a focused prompt and completes the work with its
   own tools and configuration.
5. The PI reviews the returned work and produces the final answer.

If delegation tracing is enabled, the environment prints a compact trace showing
PI-to-member assignments and member-to-PI returns. This is meant to make the team
interaction visible while fuller event logging evolves.

## Minimal team YAML

```yaml
name: example_team
group: default
workspace: team_workspace
description: >
  A small research team dedicated to data analysis and evidence synthesis.

pi:
  name: pi
  role: Principal investigator and user-facing coordinator
  agent: ExecutionAgent
  config:
    use_web: true
  prompt: >
    Plan before delegating, ask team members for focused contributions, and
    synthesize a concise answer with limitations and reproducibility notes.

members:
  - name: literature_specialist
    role: Finds and summarizes relevant background evidence
    agent: ChatAgent
    config:
      use_web: true
    prompt: >
      Be detailed, methodical, and evidence-based. Note sources and uncertainty.

  - name: analyst
    role: Performs calculations, data checks, and reproducible analysis
    agent: ExecutionAgent
    config:
      use_web: false
    prompt: >
      Prefer reproducible scripts and clearly report assumptions, outputs, and files.
```

The repository also includes an example at
`examples/environments/agent_team.yaml`.

## Run a team from Python

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentTeamEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")

team = AgentTeamEnvironment.from_yaml(
    "examples/environments/agent_team.yaml",
    llm=llm,
)

result = team.invoke(
    "Review the evidence, run any useful checks, and give me a recommendation."
)

print(result)
```

Use `await team.ainvoke(...)` when calling from code that already has a running
async event loop. Calling `team.invoke(...)` from inside an existing event loop
raises a clear error so the environment does not hide async execution issues.

## Programmatic construction

YAML is usually the clearest way to describe a team, but you can also construct a
team directly:

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentTeamEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")

team = AgentTeamEnvironment(
    llm=llm,
    name="analysis_team",
    group="default",
    pi={
        "name": "pi",
        "role": "Team lead and final synthesizer",
        "agent": "ExecutionAgent",
    },
    members=[
        {
            "name": "analyst",
            "role": "Runs calculations and writes reproducible notes",
            "agent": "ExecutionAgent",
        },
        {
            "name": "critic",
            "role": "Checks assumptions and identifies weak evidence",
            "agent": "ChatAgent",
        },
    ],
    workspace="./analysis-team-workspace",
)
```

## Choosing the PI

[`ExecutionAgent`][ursa.agents.execution_agent.ExecutionAgent] is the preferred
PI implementation because it accepts the team's
member-delegation tools during construction. Other tool-capable agents may work if
they expose an `add_tool(...)` method. If the configured PI cannot accept
delegation tools, team construction raises a `TypeError`.

If no PI is configured, the default PI agent class is `ExecutionAgent`.

## Member names, persistence, and shared work

Team members use their configured `name` as their persistent URSA agent name when
member persistence is enabled. In other words, a member named `analyst` uses the
normal named-agent persistence for `analyst` in the configured group. This makes it
possible to reuse an existing named specialist in a team YAML file.

The PI and all team members receive the same team workspace. That shared workspace
is what lets one member create a file and another member, or the PI, inspect it
later.

By default, if you do not provide `workspace`, URSA creates one under:

```text
~/.cache/ursa/<group>/environments/workspaces/<team-name>
```

## Member-specific models

A team member can inherit the default `llm` passed to the environment, or provide a
`model:` block to use a different model endpoint:

```yaml
members:
  - name: fast_researcher
    role: Drafts quick background summaries
    agent: ChatAgent
    model:
      model: openai:gpt-4o-mini
      api_key_env: OPENAI_API_KEY

  - name: local_checker
    role: Checks reasoning with a local model
    agent: ChatAgent
    model:
      model: ollama:llama3.1
      base_url: http://localhost:11434
```

This is optional for teams, but it can be a good way to match model cost,
latency, and capability to each member's role.

## Saving a team configuration

You can persist a team config with `save_team_config(...)`:

```python
from ursa.environments import AgentTeamConfig, save_team_config

config = AgentTeamConfig(name="science_team", group="default")
path = save_team_config(config)
print(path)
```

When no path is supplied, URSA writes to:

```text
~/.cache/ursa/<group>/environments/agent_teams/<name>/team.yaml
```

## Practical guidance

- Give members narrow roles. "Data analyst" is more useful than "helpful agent."
- Put expectations in each member's `prompt`, especially evidence standards,
  file-writing expectations, and known constraints.
- Use an execution-capable PI only when the coordinator needs tools as well as
  delegation.
- Use a dedicated workspace for teams that can write or execute code.
- Remember that the PI decides whether to delegate. If delegation is essential,
  say so clearly in the PI prompt and the user task.
