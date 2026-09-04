# Connect URSA to MIST through Nomad

In this tutorial, we'll connect URSA to a
[MIST surrogate model](https://huggingface.co/mist-models) that predicts quantum
mechanical properties. We'll use [Nomad](https://github.com/lanl/nomad), an MCP
server for scientific foundation models (SciFMs), to expose the MIST model as an
MCP tool. URSA can then query the model to predict properties such as the
HOMO–LUMO gap of caffeine.

MIST is a family of molecular foundation models for property prediction. Browse
the [MIST models on Hugging Face](https://huggingface.co/mist-models) before you
begin to see the available weights and model cards. This tutorial connects one
MIST model, but Nomad can serve additional MIST models or
[a different SciFM](https://lanl.github.io/nomad/guides/model-builder.html).

## What you will do

1. Install this example's URSA environment.
2. Start Nomad's prebuilt demo container.
3. Configure URSA to connect to the local Nomad MCP endpoint.
4. Run a guided prompt and inspect `mist_caffeine_report.json`.

The first container launch downloads model weights. The model used here is
small enough to run without a GPU, although a GPU makes inference faster.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Docker Desktop or Docker Engine
- An API key for the inference provider URSA will use
- Optional: an NVIDIA GPU available to Docker

Run every command from the [`examples/nomad_mist`](.) directory.

## Install the example environment

```bash
uv sync
uv run ursa --help
```

This example uses the URSA checkout two directories above it. `uv run` ensures
that the command comes from the example's environment.

If this is your first URSA session, follow the
[configuration overview](../../docs/configuration/index.md) to configure your
LLM credentials.

## Start the Nomad container

The following command starts the Nomad demo container in a terminal. Wait for
it to start, then leave it running and open a second terminal.

=== "macOS/Linux"

    === "With GPU"

        ```bash
        docker run --rm \
          --gpus all \
          --publish 38217:38217 \
          --volume "nomad-cache:/var/cache/nomad" \
          ghcr.io/lanl/nomad:v0.2.0 \
          serve \
          --transport=streamable-http \
          --host=0.0.0.0 \
          --port=38217 \
          /nomad/container/demo/nomad.yml
        ```

    === "Without GPU"

        ```bash
        docker run --rm \
          --publish 38217:38217 \
          --volume "nomad-cache:/var/cache/nomad" \
          ghcr.io/lanl/nomad:v0.2.0 \
          serve \
          --transport=streamable-http \
          --host=0.0.0.0 \
          --port=38217 \
          /nomad/container/demo/nomad.yml
        ```

=== "Windows PowerShell"

    === "With GPU"

        ```powershell
        docker run --rm `
          --gpus all `
          --publish 38217:38217 `
          --volume "nomad-cache:/var/cache/nomad" `
          ghcr.io/lanl/nomad:v0.2.0 `
          serve `
          --transport=streamable-http `
          --host=0.0.0.0 `
          --port=38217 `
          /nomad/container/demo/nomad.yml
        ```

    === "Without GPU"

        ```powershell
        docker run --rm `
          --publish 38217:38217 `
          --volume "nomad-cache:/var/cache/nomad" `
          ghcr.io/lanl/nomad:v0.2.0 `
          serve `
          --transport=streamable-http `
          --host=0.0.0.0 `
          --port=38217 `
          /nomad/container/demo/nomad.yml
        ```

Wait until Nomad reports that the server is listening on port `38217`.

!!! tip "Port already in use"
    If port `38217` is already in use, choose another five-digit port. Update
    both values in `--publish`, the value passed to `--port`, and the URL in
    `ursa.yaml` below.

## Connect URSA to Nomad

Create a file named `ursa.yaml` with the following content:

```yaml
mcp_servers:
  nomad:
    transport: streamable-http
    url: http://localhost:38217/mcp
  # Other existing MCP servers here
```

!!! tip "Configuration files"
    To connect to this MCP server by default, add the same block to your user
    configuration instead. See the
    [URSA configuration guide](../../docs/configuration/index.md) for details.

In the second terminal, confirm that URSA loaded the configuration:

```bash
uv run ursa --config ursa.yaml --print-config
```

Look for `mcp_servers.nomad` in the output, then start the TUI:

```bash
uv run ursa --config ursa.yaml
```

## Run the MIST workflow

Once URSA starts, confirm that the Nomad MCP server is attached by entering
`/agents` at the prompt. Under the `#execute` tab, verify that some tools show
`(MCP: nomad)` next to their names. This label indicates that the tools came
from the `nomad` MCP server.

Press `Esc` to close the Agents modal, then enter the following prompt:

```text
#execute Use the connected Nomad tools to compute the chemical properties of
caffeine. Save all computed properties to `mist_caffeine_report.json`.
```

URSA should resolve caffeine to a canonical SMILES string, call the MIST model,
and create `mist_caffeine_report.json` in the selected workspace. Review the
report and tool activity rather than treating the generated predictions as
experimentally validated data.

You can also ask URSA to tell you more about how the MIST model was trained:

```text
#execute Use the connected Nomad tools to explain how the MIST 26.9M QM9 model
was trained.
```

URSA will query Nomad for the
[MIST QM9 model card](https://huggingface.co/mist-models/mist-26.9M-kkgx0omx-qm9)
to answer your question.

## Use the dashboard instead

You can run the same workflow in the browser interface. Follow the
[dashboard guide](../../docs/getting-started/dashboard.md) to install and launch
the dashboard, configure the same Nomad MCP endpoint, and select a workspace.
Choose the execution agent and submit the same prompt, then inspect the activity
timeline and generated report in the artifacts panel.

## Troubleshooting

- If no Nomad tools appear, recheck the resolved configuration and confirm that
  the container is still listening on port `38217`. Also verify that the URL in
  `ursa.yaml` uses the same port.
- If Docker rejects `--gpus all`, remove that line and run on the CPU.
- If the first model call is slow, watch the Nomad terminal; it may still be
  downloading weights into its cache.

Stop Nomad with **Ctrl+C**. Stop passing `--config ursa.yaml` when you no longer
want URSA to connect to the local server.

The endpoint uses unauthenticated local HTTP for this exercise. Before exposing
Nomad on a network, apply appropriate authentication and transport security.
See Nomad's [getting-started guide](https://lanl.github.io/nomad/guides/getting-started.html)
and [deployment guide](https://lanl.github.io/nomad/deployments/guide.html) for
production considerations.
