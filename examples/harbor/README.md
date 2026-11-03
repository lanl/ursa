# Run URSA with Harbor

This example connects URSA's `ExecutionAgent` to
[Harbor](https://www.harborframework.com/docs/core-concepts) and includes
GPAW, Parthenon, and PyTorch installation tasks.

## Run a task

From this directory, install Python 3.12+, [uv](https://docs.astral.sh/uv/),
and Docker for Harbor's default environment. Set a provider credential, then:

```bash
export OPENAI_API_KEY=...
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config="$PWD/ursa.yaml" \
  --model openai/openai:gpt-5.4-nano
```

A successful task reports reward `1`; inspect results with `uv run harbor view
jobs`. Use `--path benchmark/tasks` to run all included tasks.

To continue a completed trial locally, run:

```bash
uv run harbor trial handoff jobs/<job>/<trial>
```

Handoff imports the trial checkpoint into its recorded, existing URSA group and
opens the TUI; the task container's files are not copied into the workspace.

Validate tasks against URSA's Singularity constraints, or warm the SIF cache,
without running an agent:

```bash
uv run -m ursa.integrations.harbor validate benchmark/tasks
uv run -m ursa.integrations.harbor prebuild benchmark/tasks
```

See [advanced usage](ADVANCED.md) for the Singularity environment, cache
options, and non-Docker prerequisites.

## Add a task

Start with Harbor's generator, then follow its task tutorial and reference for
the generated instruction, environment, task, and verifier files:

```bash
uv run harbor task init --tasks-dir benchmark/tasks my-org/my-task
```

The official [task tutorial](https://www.harborframework.com/docs/tasks/task-tutorial),
[format reference](https://www.harborframework.com/docs/tasks), and
[publishing guide](https://www.harborframework.com/docs/tasks/publishing) cover
authoring and sharing tasks.

## Configuration

Harbor models use `<inference-provider>/<model-provider>:<model-name>`, such
as `ollama/ollama:gemma4:latest`. The inference provider must exist in
`ursa.yaml`; its `model_provider` allows omitting the model-provider prefix.
`--model` replaces the configured model, and Harbor
`[[environment.mcp_servers]]` entries are attached automatically. See the
[MCP task tutorial](https://www.harborframework.com/docs/tutorials/mcp-server-task).
