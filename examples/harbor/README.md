# Run URSA with Harbor

This example connects URSA's `ExecutionAgent` to
[Harbor](https://www.harborframework.com/docs/core-concepts). The included
benchmark asks the agent to install GPAW, Parthenon, and PyTorch.

## Run one task

Install Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and Docker. Set
the credential for your inference provider, then run from this directory:

```bash
export OPENAI_API_KEY=...
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config_file="$PWD/ursa.yaml" \
  --model openai/gpt-5.4-nano
```

A successful task reports reward `1`. Inspect the result with:

```bash
uv run harbor view jobs
```

To run all three tasks, repeat the command with `--path benchmark/tasks`.

## Add a task

Start with Harbor's generator:

```bash
uv run harbor task init --tasks-dir benchmark/tasks my-org/my-task
```

Then work through the generated files one at a time:

1. Write the request in `instruction.md`.
2. Define the base environment and tools available to the agent in
   `environment/Dockerfile`.
3. Describe the task in `task.toml`.
4. Make `tests/test.sh` write a numeric reward to
   `/logs/verifier/reward.txt`.
5. Run the task by passing its directory to `--path`.

The official [task tutorial](https://www.harborframework.com/docs/tasks/task-tutorial)
walks through these files. Use the
[task-format reference](https://www.harborframework.com/docs/tasks) for details
and the [publishing guide](https://www.harborframework.com/docs/tasks/publishing)
when the task is ready to share.

## Configuration

Harbor models use `<inference-provider>/<model-name>`. The provider must exist
in `ursa.yaml`; the adapter replaces its configured model with `--model`.
Harbor `[[environment.mcp_servers]]` entries are also attached automatically;
see the [MCP task tutorial](https://www.harborframework.com/docs/tutorials/mcp-server-task).

See [advanced usage](ADVANCED.md) for extra Python packages, Singularity,
SLURM, and cleanup.
