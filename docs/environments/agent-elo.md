# Agent Elo

Agent Elo is URSA's evolutionary competition environment. Multiple agents work
independently on the same task, compete in pairs, and receive Elo ratings based on
the quality of their work. Selected losers leave the active population, while
successful survivors produce descendants that inherit their research state.

The goal is to develop solutions over several generations. A descendant starts
with its parent's files and persistent state, then works independently to address
limitations or improve the inherited solution. Surviving founders also continue
improving their existing work.

## When Elo is the right shape

Use Agent Elo when you want to:

- explore several approaches to a task independently;
- compare solutions using explicit evidence and quality criteria;
- preserve successful work while replacing weaker competitors;
- develop and validate scientific or computational methods over repeated rounds;
- inspect how solutions and agent lineages evolve over time.

Use an [Agent Team](agent-teams.md) when a PI should delegate specialist subtasks
and synthesize an answer. Use an [Agent Symposium](agent-symposia.md) when you want
peer review, revision, and an organizer's final report. Elo returns competition
results and standings; it does not have an organizer that synthesizes one final
answer.

## What happens during a generation

Each generation follows this sequence:

1. **Independent work.** Every active member receives the same task, its role,
   member-specific guidance, and instructions appropriate to its lineage. Members
   run asynchronously in their own workspaces.
2. **Pairing.** The environment randomly pairs the active members. Each member
   participates in exactly one match.
3. **Judging.** Completed and timed-out competitors are compared by an LLM judge,
   which chooses candidate A, candidate B, or a draw using the task’s evaluation
   criteria. Ordinary execution failures are handled before calling the judge.
4. **Rating updates.** Match results update both competitors' Elo ratings.
5. **Selection and reproduction.** Up to `deaths_per_round` decisive losers are
   selected for elimination. The highest-rated survivors each produce one child
   to replace an eliminated member. All replacement children are prepared before
   the losers are removed.
6. **Saving.** The environment advances its generation counter and writes
   `environment_state.json` for restart.

A population must contain at least two members and have an even size. Names must
be unique. The population size stays constant after each successful generation,
although its members may change. Children first compete in the next generation.

## Minimal Elo YAML

Save the following as `elo.yaml`:

```yaml
name: numerical_integration_elo
group: default
workspace: ./elo_workspace

generations: 2
initial_rating: 1500
k_factor: 32
deaths_per_round: 1
seed: 12345

# Put evaluation criteria in the task supplied to invoke().

members:
  - name: researcher_1
    role: Develops and validates numerical integration methods
    agent: ExecutionAgent
    config:
      use_web: false

  - name: researcher_2
    role: Explores alternative methods and checks numerical accuracy
    agent: ExecutionAgent
    config:
      use_web: false
```

The repository also includes `examples/environments/agent_elo.yaml` and
`examples/environments/run_elo.py`. That runner asks agents to integrate
`exp(-x^2)` from zero to one, compare against a reference computed with `math.erf`,
and save executable code and validation results.

## Run Elo from Python

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentEloEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")
env = AgentEloEnvironment.from_yaml("elo.yaml", llm=llm)

task = """
Numerically integrate exp(-x^2) from 0 to 1 using Python.
Create and execute code in your workspace. Compare the result with a reference
computed using math.erf, report absolute error, and perform a convergence check.
Save the code and useful results. Explain remaining limitations.

Evaluation criteria, in priority order:
1. Correctness and numerical accuracy.
2. Evidence from executed validation and convergence checks.
3. Reproducibility.
4. Meaningful improvement over existing work, when applicable.
"""

try:
    result = env.invoke({"task": task})
    for row in result["standings"]:
        print(row["name"], row["rating"], row["parent"])
    print("Restart state:", result["environment_state"])
finally:
    for member in env.members.values():
        close = getattr(member, "close", None)
        if callable(close):
            close()
```

One invocation runs the configured number of `generations` on the supplied task.
Invoking the same environment again runs that many additional generations using
its current population and ratings.

Inside an existing async event loop, use `await env.ainvoke(...)`. Calling
`env.invoke(...)` from that context raises an error.

## Programmatic construction

You can also create an environment without a YAML file:

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentEloEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")
env = AgentEloEnvironment(
    llm=llm,
    name="method_comparison",
    workspace="./method-comparison-workspace",
    generations=3,
    deaths_per_round=1,
    seed=42,
    members=[
        {
            "name": "adaptive_method",
            "role": "Develops an adaptive numerical method",
            "agent": "ExecutionAgent",
        },
        {
            "name": "reference_method",
            "role": "Develops a simple, well-validated reference method",
            "agent": "ExecutionAgent",
        },
    ],
)
```

## Competition settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `initial_rating` | `1500` | Rating assigned to founding members. |
| `k_factor` | `32` | Positive scale controlling rating changes per match. |
| `deaths_per_round` | `1` | Maximum number of decisive losers replaced per generation; zero disables replacement. |
| `generations` | `1` | Number of additional generations run by each invocation; must be at least one. |
| `seed` | `null` | Optional seed for pairing and selection tie-breaking. |
| `member_timeout_seconds` | `null` | Positive execution budget for the independent-work phase; no environment-imposed deadline when omitted. |
| `judge_prompt` | `null` | Optional additional judging guidance; put evaluation criteria in the task. |
| `restart_from_json` | `null` | Path to a saved `environment_state.json`. |

Elo uses the standard expected-score calculation. A win scores `1`, a draw `0.5`,
and a loss `0`. The rating change is `k_factor * (score - expected_score)`.
For two equally rated agents with a K-factor of 32, a decisive match adds 16 to
the winner and subtracts 16 from the loser.

Only decisive losers are eligible for elimination. If there are more eligible
losers than available replacement slots, the lowest-rated losers are selected
using their updated ratings. Draws do not cause elimination, although a draw
between unequally rated agents still changes their ratings.

Parents are selected from the highest-rated survivors. Equal ratings in selection
are resolved using the environment's random generator. A fixed seed makes those
random choices reproducible given the same population, ordering, and match
outcomes; it does not make LLM responses deterministic.

## Judging and incomplete runs

Specify evaluation criteria in the task so both competitors and judge receive
the same definition of success. The built-in judge prompt provides comparison
and inspection instructions, without a task-specific quality rubric. If the task
omits criteria, the judge evaluates how well each candidate fulfills it.
`judge_prompt` remains available for optional additional guidance.

The member prompt asks founders to establish an independent approach, surviving
founders to favor refinement and validation, and descendants to explore a
substantive improvement or alternative. A descendant retains that guidance in
later rounds, even if it has itself produced children. The task determines what
counts as a useful improvement.

Submissions use the following order:

1. The member's final response, when available.
2. Its saved progress report when there is no usable final response.
3. A request for brief workspace inspection when neither is available.

Members with file-writing tools are asked to maintain a concise report at
`_elo_progress/generation_<number>.md` inside their own workspace. The report
should describe completed work, verified results, limitations, and supporting
file paths, distinguishing new work from inherited results. There is no fixed
word limit. Members should update it after meaningful milestones, using a
temporary file and rename to avoid partially written reports. The report is
optional for agents without file-writing tools.

The environment reads the current generation's report when handling a timeout,
or when a completed run has an empty final response. It passes that captured text
directly to the judge; it does not make an extra competitor LLM call or scan the
workspace for a submission. An earlier generation's report is not substituted.

The judge evaluates supplied submissions first, with optional file verification.
When no submission is available, it is instructed to inspect obvious relevant
artifacts briefly, avoid extensive searches or lengthy calculations, and note
insufficient evidence if useful results cannot be found. It must not modify
candidate files. These efficiency instructions are not a hard judging timeout.
The judge uses the environment-level `llm`, even when members have their own models.

If the agentic judge fails to initialize, execute, or return a valid decision,
URSA tries a direct LLM judgment. That fallback uses the supplied text without
workspace tools. If it also fails, the match is recorded as a draw with an
explanation.

A timeout alone is not a loss: timed-out members can win, draw, survive, and
reproduce based on the judgment. Ordinary execution exceptions retain their
separate handling:

| Candidate A | Candidate B | Outcome |
| --- | --- | --- |
| Completed or timed out | Completed or timed out | Ask the judge. |
| Completed | Failed | A wins automatically. |
| Failed | Completed | B wins automatically. |
| Timed out | Failed | Draw. |
| Failed | Timed out | Draw. |
| Failed | Failed | Draw. |

When `member_timeout_seconds` is set, all members receive the same UTC deadline
for that generation's independent work. This budget does not cover judging or
reproduction. The environment cancels member execution at the deadline, but an
already-running blocking subprocess may continue until it exits or reaches its
own timeout. Neither report capture nor optional workspace inspection guarantees
a filesystem snapshot taken precisely at the deadline.

## Workspaces, persistence, and lineage

Each member receives a separate directory under the environment workspace:

```text
elo_workspace/
  researcher_1/
  researcher_2/
  researcher_1_g1_1/
  environment_state.json
```

The child name above is illustrative. Descendant names include the parent name,
the child's lineage generation, and a counter; existing names and directories are
checked to avoid collisions.

If no top-level `workspace` is supplied, the environment uses:

```text
~/.cache/ursa/<group>/environments/workspaces/<environment-name>/
```

Persistent members use identities of the form
`<environment-name>_<member-name>`, stored under:

```text
~/.cache/ursa/<group>/agents/<environment-name>_<member-name>/
```

Elo manages these paths and identities. Member-level `config.workspace` and
`config.agent_name` are rejected, including during restart. Set the environment's
top-level workspace and use each member's `name` field instead.

With `persist_members=True`, the default, children inherit copies of the parent's
workspace, checkpoint database, and LangGraph store when present. They also
inherit the parent's updated Elo rating and member configuration, including its
role, prompt, and model settings. Parent and child subsequently use separate
files and databases.

Lineage generation and environment generation are different: a founder remains at
lineage generation zero even after surviving several environment generations.
A child's lineage generation is its parent's plus one.

Eliminated members leave the active roster, but their workspaces and persistent
directories remain on disk for inspection. With `persist_members=False`, workspace
copying and rating inheritance still occur, but persistent checkpoint inheritance
is disabled and restart from JSON is unavailable.

## Member-specific models

Members inherit the default `llm` unless they specify a `model` block. For example,
these entries can be used in the YAML `members` list:

```yaml
members:
  - name: researcher_1
    role: Develops and validates numerical methods
    agent: ExecutionAgent
    model:
      model: openai:gpt-4o-mini

  - name: researcher_2
    role: Investigates an alternative numerical approach
    agent: ExecutionAgent
    model:
      model: ollama:llama3.1
      base_url: http://localhost:11434
```

Model diversity can provide different approaches, but ratings measure performance
under this task and judge. They are not a general benchmark of model capability.

## Saving configuration and restarting a run

Saving configuration and saving evolutionary state serve different purposes.
Use `save_elo_config(...)` to save the environment definition:

```python
from ursa.environments import load_elo_config, save_elo_config

config = load_elo_config("elo.yaml")
path = save_elo_config(config)
print(path)
```

Without an explicit destination, the configuration is written to:

```text
~/.cache/ursa/<group>/environments/agent_elo/<name>/elo.yaml
```

After each completed generation, Elo separately writes `environment_state.json`
in the environment workspace. It contains the active member configurations,
ratings, lineage, completed-generation count, and random-generator state.

To restart, use the same environment name, group, and workspace, and supply the
snapshot path. For example, save this as `elo-restart.yaml`:

```yaml
name: numerical_integration_elo
group: default
workspace: ./elo_workspace
restart_from_json: ./elo_workspace/environment_state.json
generations: 2
k_factor: 32
deaths_per_round: 1
```

Load it with `AgentEloEnvironment.from_yaml("elo-restart.yaml", llm=llm)` and invoke
it with the task again. This runs two additional generations. The task is not
stored in the restart snapshot.

The snapshot replaces the YAML member list, including per-member model settings.
Members without an explicit saved model use the default LLM supplied on restart.
Resolved model settings are saved without requiring the original named-provider
registry; credential references still require the corresponding credentials to be
available. Current environment settings such as `generations`, `k_factor`,
`deaths_per_round`, timeout, and judge instructions come from the restart
configuration, so retain any custom values you want to continue using.

The JSON file is metadata, not a self-contained backup. Keep the member workspaces
and persistent databases alongside it in their expected locations. Restart
requires member persistence and existing member directories.

If child preparation fails, Elo cleans up prepared children and keeps the original
population active. Ratings and evolutionary random state are restored so the
failed generation can be retried. This does not undo research or file changes
already made by the competing agents.

## Understanding the results

The returned mapping includes:

- `task`: the normalized task;
- `starting_generation`, `completed_generations`, and `ending_generation`:
  progress for this invocation;
- `generations`: detailed reports for the completed generations;
- `standings`: final active members, ratings, ranks, lineage generations, and
  parent names;
- `population_size`: the final active population size;
- `environment_state`: the restart snapshot path.

Each generation report includes member `outputs`, `member_runs`, `pairs`,
`matches` with scores and reasoning, `ratings_before`, `standings_after_matches`,
`eliminated`, `reproducing_parents`, `children`, and the resulting `standings`.
It also lists failed and timed-out members. Each `member_runs` entry includes
`progress_report` when captured; `outputs` remains reserved for final responses.
Timed-out members retain their `timed_out` status regardless of the judgment.

Inspect outputs and match reasoning alongside ratings. Final standings include
new children that have inherited ratings but have not yet competed themselves.
The generation's `outputs` belong to the agents that actually worked that round.

## Monitoring runs

The dashboard's **Environment Runs** page can validate Elo YAML, launch a run,
and display its progress. For Python-driven recording and replay, see
[Visualizing Environment Runs](visualization.md). Elo emits generation, member,
match, elimination, reproduction, and topology events.

## Practical guidance

- Start with two members and a small number of generations to evaluate whether
  the task and judge criteria produce useful improvements.
- Give agents a task with room for substantive development, such as stronger
  numerical validation, a better method, or a tested implementation.
- Ask for concise final outputs that identify evidence, generated files, and
  remaining weaknesses so the judge can compare the work efficiently.
- Account for every active member running each generation, plus one judgment per
  completed pair and the cost of copying persistent state for children.
- Preserve the full workspace and persistence directories if you intend to
  restart or inspect extinct lineages.
- Treat high ratings as comparative evidence within the run, and check the
  underlying results before relying on them.
