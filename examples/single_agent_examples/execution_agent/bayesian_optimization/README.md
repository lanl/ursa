# Continue a Bayesian optimization run from a checkpoint

Use this example to give an `ExecutionAgent` a scientific programming task,
then return to the same thread with a follow-up visualization request. The
agent must implement and run a Bayesian optimization of the six-hump camel
function; the second script reopens its checkpoint and asks for convergence and
input-importance plots.

This is an agent-generated workflow, not a fixed optimization implementation.
Inspect the code, numerical results, and plots that the model produces before
relying on them.

## What the files demonstrate

- `bayesian_optimization.py` starts the OpenAI-backed run, using
  `workspace_BO/` and thread ID `BO_test`.
- `bayesian_optimization_continue.py` reuses that workspace and thread ID so it
  can continue from the first run's checkpoint.
- `bayesian_optimization_ollama.py` is an independent local-model variant. It
  does not participate in the two-step checkpoint walkthrough.

Read the [ExecutionAgent guide](../../../../docs/agents/execution.md) for its
code-writing and command-execution behavior, and review
[checkpointing and sharing](../../../../docs/persistence/checkpoints-and-sharing.md)
for the persistence concepts used here.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for the checkpoint walkthrough
- A directory whose generated `workspace_BO/` contents you are comfortable
  reviewing and removing

Run every command from this `bayesian_optimization` directory.

## 1. Install the example environment

=== "macOS/Linux"

    ```bash
    uv sync
    export OPENAI_API_KEY="..."
    ```

=== "Windows PowerShell"

    ```powershell
    uv sync
    $env:OPENAI_API_KEY = "..."
    ```

The scripts use `openai:gpt-5.4-mini`. To select another endpoint, update the
model initialization in both checkpoint scripts and follow the
[configuration guide](../../../../docs/configuration/index.md). Keep both
scripts on the same model configuration when comparing the initial and
continued runs.

## 2. Start the optimization

```bash
uv run bayesian_optimization.py
```

The execution agent receives the optimization objective, writes its chosen
implementation under `workspace_BO/`, runs it, and reports its result. URSA also
records state for thread `BO_test` in that workspace and prints a timing
summary.

Before continuing:

1. Read the generated implementation.
2. Confirm that it evaluates the standard six-hump camel function on an
   appropriate bounded domain.
3. Check that the reported best point and value are supported by saved
   evaluations rather than prose alone.
4. Review any commands and dependency installations performed by the agent.

Because an LLM chooses the implementation, exact filenames and optimization
libraries can differ between runs.

## 3. Continue the checkpointed thread

Run the continuation only after the first command completes successfully:

```bash
uv run bayesian_optimization_continue.py
```

The continuation script points to the same `workspace_BO/`, creates a
checkpointer from that workspace, and invokes `ExecutionAgent` with the same
`BO_test` thread ID. Its prompt asks the agent to use the existing evaluation
history to create:

- a convergence plot with the running minimum; and
- a second plot highlighting important function inputs.

Confirm that the plots use results from the first run. If they silently create
a new optimization history, inspect the checkpoint files and first-run output
before trying again.

## Optional: run the Ollama variant

The Ollama script is a separate choose-and-run example; it does not resume the
OpenAI checkpoint. Install and start [Ollama](https://ollama.com/), then pull the
model named by the script:

=== "macOS/Linux"

    ```bash
    ollama pull gpt-oss:20b
    uv run bayesian_optimization_ollama.py
    ```

=== "Windows PowerShell"

    ```powershell
    ollama pull gpt-oss:20b
    uv run bayesian_optimization_ollama.py
    ```

Set `set_workspace = True` in the script if you want this independent run to
write under `workspace_BO/`. Do not assume it can continue the OpenAI thread:
the Ollama variant does not configure the same checkpointer or thread ID.

## Adapt the Python workflow

Edit the `problem` string to change the scientific task. If you want a new
checkpoint lineage, change both `workspace` and `thread_id` consistently in the
initial and continuation scripts. Reusing only one of them can attach the
follow-up to the wrong state or make the expected state unavailable.

See the [Python getting-started guide](../../../../docs/getting-started/python-scripts.md)
for model initialization and direct agent invocation patterns.

## Troubleshooting and cleanup

- If authentication fails, verify the active key or provider with the
  [configuration guide](../../../../docs/configuration/index.md).
- If continuation cannot find prior state, confirm that the first run completed
  and that neither script's `workspace` or `thread_id` was changed alone.
- If Ollama cannot find its model, run `ollama list` and make the script's model
  string match the locally installed tag.
- To start the OpenAI walkthrough from scratch, move `workspace_BO/` somewhere
  safe or remove it after confirming that you no longer need its generated code,
  results, or checkpoints.

These scripts make real LLM calls and allow generated code execution. Runtime,
cost, dependencies, and artifacts vary by model response.
