# Evolve numerical methods with Agent Elo

Run two independent agents on a small numerical-integration problem, compare
their evidence, and let successful agents produce descendants. This walkthrough
uses two generations to show how a child inherits a solution and then develops
it further.

The task is to integrate `exp(-x^2)` from zero to one. Agents must execute Python,
compare their result with a reference computed using `math.erf`, and save their
implementation and validation results. The task itself specifies the evaluation
criteria seen by both competitors and the judge.

For competition rules, persistence details, and configuration options, see the
[Agent Elo guide](../../../docs/environments/agent-elo.md).

## 1. Set up the example

Clone URSA and start from the repository root. You will need
[uv](https://docs.astral.sh/uv/), Python 3.11 or newer, and credentials for the
model used in [run_elo.py](run_elo.py). The supplied runner uses OpenAI's `gpt-5`
for both competitors and the judge.

=== "macOS/Linux"

    ```bash
    cd examples/environments/agent_elo
    export OPENAI_API_KEY="your-api-key"
    uv sync
    ```

=== "Windows PowerShell"

    ```powershell
    Set-Location examples\environments\agent_elo
    $env:OPENAI_API_KEY = "your-api-key"
    uv sync
    ```

The local [pyproject.toml](pyproject.toml) installs URSA from this checkout in
editable mode. Model and credential configuration is described in
[Models and inference providers](../../../docs/configuration/models.md).

## 2. Inspect the population

Open [agent_elo.yaml](agent_elo.yaml). It defines two `ExecutionAgent` members,
two generations per invocation, and at most one replacement per generation.
The seed controls pairing and selection tie-breaking, not the model responses.

```yaml
--8<-- "examples/environments/agent_elo/agent_elo.yaml"
```

Keep member names unique and the population even. For this first run, leave the
two-member population and generation count unchanged.

## 3. Run the numerical task

From this example directory, run:

```bash
uv run python run_elo.py
```

The runner loads the adjacent YAML, supplies the task, prints each generation's
matches and standings, and reports the restart snapshot path. Its `TASK` string
defines correctness, executed validation, reproducibility, and improvement as
the evaluation criteria. Change those criteria when adapting the example to a
different problem.

```python
--8<-- "examples/environments/agent_elo/run_elo.py"
```

The runner creates `workspace_agent_elo/` under this directory. Each competitor
works in a separate subdirectory. After a decisive first match, the selected
survivor produces a child; in the second generation, that child starts from its
parent's saved work. A draw produces no replacement.

## 4. Inspect the results

Look for the following in the printed output:

- The winner and reasoning for each match, or a draw.
- Names of eliminated agents and their replacements.
- Updated ratings and the parent of each descendant.
- The path to `environment_state.json`.

The reference integral is approximately `0.746824132812427`. Inspect the agents'
saved scripts and convergence checks to understand how closely their methods
approach it. Filenames, ratings, winners, and numerical accuracy depend on the
generated work; this example does not prescribe an exact transcript.

Compare the first and second generation's artifacts. Did the survivor strengthen
its validation? Did a descendant implement a useful improvement? A high inherited
rating alone does not demonstrate new work, and children born at the end of the
last generation have not yet competed.

## 5. Continue or adapt the experiment

To continue from a completed run, set `restart_from_json` in `agent_elo.yaml` to
`./workspace_agent_elo/environment_state.json` and rerun the script. Keep the
environment name, group, workspace, and member persistence directories intact.
The runner supplies the task again, and `generations: 2` requests two additional
generations. See [restart behavior](../../../docs/environments/agent-elo.md#saving-configuration-and-restarting-a-run)
for which settings come from the snapshot.

For a new experiment, use a new environment name in the YAML and a new workspace
path in the runner. You can then vary the agents' roles, add an even number of
competitors, or change the task and evaluation criteria.

Optionally set `member_timeout_seconds` in the YAML. Timed-out agents remain
eligible for judging through saved progress reports or brief workspace
inspection. Start without a timeout so you can observe the complete workflow.

## Cleanup

The script closes its active members when it exits. To discard this example's
data, remove `workspace_agent_elo/` and the named agent directories beginning
with `numerical_integration_elo_` under
`~/.cache/ursa/default/agents/`, including any descendants. Keep both locations
if you intend to restart; the JSON snapshot alone is not a full backup.

The example's `.venv/` can also be removed when you no longer need its local
dependencies. Avoid removing unrelated agents from the shared URSA cache.
