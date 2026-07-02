# Visualizing environment runs

URSA environments can record a structured replay of an `AgentTeamEnvironment` or
`AgentSymposiumEnvironment` run. The replay captures the environment topology,
major phases, delegation events, tool activity, errors, and final results so you
can inspect how the group worked after the run completes or while it is still
running.

Use this feature when you want to answer questions such as:

- Which agent or environment member handled a piece of work?
- When did the PI delegate to a specialist?
- Which symposium phase produced the final answer?
- Which tool calls happened during a nested environment run?
- What failed, and at what point in the collaboration?

The visualization helpers are available from `ursa.environments`:

```python
from ursa.environments import arun_with_visualization, run_with_visualization
```

## Run a team with visualization

Use `run_with_visualization(...)` anywhere you would normally call
`environment.invoke(...)` synchronously.

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentTeamEnvironment, run_with_visualization

llm = init_chat_model(model="openai:gpt-5.4-mini")

team = AgentTeamEnvironment.from_yaml(
    "examples/environments/agent_team.yaml",
    llm=llm,
)

result = run_with_visualization(
    team,
    {"task": "Analyze this research question and recommend a next step."},
)

print(result)
```

`run_with_visualization(...)` creates an environment run recorder, invokes the
environment with the recorder attached as a LangChain callback, and writes a
replayable event stream to disk.

## Run a symposium with visualization

The same helper works for `AgentSymposiumEnvironment`.

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment, run_with_visualization

llm = init_chat_model(model="openai:gpt-5.4-mini")

symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml",
    llm=llm,
)

result = run_with_visualization(
    symposium,
    {"task": "Compare two solution strategies and recommend one."},
)

print(result["final"])
```

Symposium replays include the initial independent work, review rounds, revision
rounds, synthesis step, and final symposium completion event.

## Async usage

Use `arun_with_visualization(...)` when running an environment from async code.
It mirrors `environment.ainvoke(...)`.

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment, arun_with_visualization

llm = init_chat_model(model="openai:gpt-5.4-mini")

symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml",
    llm=llm,
)

result = await arun_with_visualization(
    symposium,
    {"task": "Evaluate competing hypotheses and produce a final recommendation."},
)

print(result["final"])
```

## Choose a run ID

By default, URSA generates a unique run ID. You can provide one when you want a
stable name for a run:

```python
result = run_with_visualization(
    team,
    {"task": "Summarize the experiment plan."},
    run_id="experiment-plan-review",
)
```

If you reuse a run ID, the recorder writes into the same run directory. Prefer a
new run ID for each replay unless you intentionally want to update or replace a
specific run record.

## Pass LangChain runnable config

Both helpers accept an optional `config` argument. URSA merges this config with
the visualization recorder config so you can keep passing callbacks, tags, or
metadata that your application already uses.

```python
result = run_with_visualization(
    team,
    {"task": "Review these results."},
    config={
        "tags": ["paper-review"],
        "metadata": {"project": "demo"},
    },
)
```

The visualization callback is added to the run while your supplied config remains
available to the underlying LangChain calls.

## Limit recorded payload size

Environment event payloads can include prompts, task text, tool inputs, tool
outputs, errors, and final results. Large strings are truncated before they are
written to the event file. The default per-string limit is 30,000 characters.

You can lower or raise that limit with `max_payload_chars`:

```python
result = run_with_visualization(
    team,
    {"task": "Inspect these files and report findings."},
    max_payload_chars=10_000,
)
```

Use a smaller value when you expect very large tool outputs or want smaller replay
files.

## Where runs are stored

Recorded environment runs are stored under the active URSA group cache:

```text
~/.cache/ursa/<group>/environment_runs/<run_id>/
```

Each run directory contains:

```text
manifest.json
events.jsonl
artifacts/
logs/
```

`manifest.json` stores run metadata such as the run ID, group, environment name,
environment type, status, timestamps, and task preview.

`events.jsonl` stores one normalized event per line. Each event has a sequence
number, timestamp, event type, source, target, message, payload, tags, and
metadata.

The `artifacts/` and `logs/` directories are reserved for run-related files and
log output.

## View runs in the dashboard

Start the URSA dashboard as usual, then open the **Environment Runs** page from
the dashboard toolbar. You can also navigate directly to:

```text
/ui/environment-runs
```

The environment run list shows recorded runs for the dashboard group. Opening a
run displays a replay page with:

- an environment graph,
- a work timeline,
- the current or selected activity,
- task and final-result panels,
- workspace/path information,
- and a link to the raw event JSON API.

The run detail view can update live while a run is still active.

!!! note "Graph rendering"
    The replay page uses Cytoscape.js for the graph view. If Cytoscape.js cannot
    load, the dashboard falls back to a simpler graph display.

## Raw run APIs

The dashboard also exposes authenticated JSON endpoints for recorded runs:

```text
GET /environment-runs
GET /environment-runs/{run_id}
GET /environment-runs/{run_id}/events
GET /environment-runs/{run_id}/stream
```

Use the events endpoint with `after_seq` and `limit` to page through the event
stream:

```text
/environment-runs/example-run/events?after_seq=0&limit=1000
```

The stream endpoint uses server-sent events for live updates:

```text
/environment-runs/example-run/stream?after_seq=42
```

## What is recorded

Agent team runs record events such as:

- `team_started`
- `topology_declared`
- `delegation_started`
- `delegation_completed`
- `delegation_failed`
- `team_failed`
- `team_completed`

Agent symposium runs record events such as:

- `symposium_started`
- `topology_declared`
- `initial_work_started`
- `initial_work_completed`
- `review_round_started`
- `review_round_completed`
- `revision_round_started`
- `revision_round_completed`
- `synthesis_started`
- `synthesis_completed`
- `symposium_failed`
- `symposium_completed`

Tool events can also be associated with the environment member that invoked them
when the runtime metadata includes that ownership information.

## Privacy and cleanup

Environment replays are useful debugging artifacts, but they may contain
sensitive information. Depending on what the environment did, recorded events may
include:

- user task text,
- prompts sent to agents,
- tool inputs and outputs,
- error messages,
- file or workspace paths,
- intermediate writeups,
- and final answers.

Review recorded runs before sharing them. Delete old run directories from
`~/.cache/ursa/<group>/environment_runs/` when you no longer need them.
