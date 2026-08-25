# Connect URSA to MIST through Nomad

Use this walkthrough to ask URSA for a molecular-property prediction from a
MIST scientific foundation model. [Nomad](https://github.com/lanl/nomad) serves
the model as an MCP tool; URSA discovers the tool, finds caffeine in PubChem,
and writes a short report from the model output.

MIST is a family of molecular foundation models for property prediction. Browse
the [MIST models on Hugging Face](https://huggingface.co/mist-models) before you
begin if you want to see the available weights and model cards.

## What you will do

1. Install this example's URSA environment.
2. Start Nomad's pre-built demo container.
3. Add the local Nomad MCP endpoint to your URSA user configuration.
4. Run a guided prompt and inspect `mist_caffeine_report.txt`.

The first container launch downloads model weights. The models used here are
small enough to run without a GPU, although a GPU makes inference faster.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Docker Desktop or Docker Engine
- An API key for the language model that will direct URSA's tool calls
- Optional: an NVIDIA GPU available to Docker

Run every command from this `examples/nomad_mist` directory.

## 1. Install the example environment

```bash
uv sync
uv run ursa --help
```

This example uses the URSA checkout two directories above it. `uv run` ensures
that the command comes from the example's environment.

If this is your first URSA session, follow the
[configuration overview](../../docs/configuration/index.md) to configure your
LLM credentials. Standard OpenAI access only requires `OPENAI_API_KEY`.

## 2. Pull the Nomad image

=== "macOS/Linux"

    ```bash
    docker pull ghcr.io/lanl/nomad:latest
    mkdir -p cache
    ```

=== "Windows PowerShell"

    ```powershell
    docker pull ghcr.io/lanl/nomad:latest
    New-Item -ItemType Directory -Force cache | Out-Null
    ```

The `cache` directory retains downloaded model weights between container runs.

## 3. Start Nomad

Choose the command for your platform and leave this terminal running.

=== "macOS/Linux"

    ```bash
    # CPU-only: delete the "--gpus all" line.
    docker run --rm \
      --gpus all \
      --publish 38217:38217 \
      --volume "$PWD/cache:/var/cache/nomad" \
      ghcr.io/lanl/nomad:latest \
      serve \
        --transport=streamable-http \
        --host=0.0.0.0 \
        --port=38217 \
        /nomad/container/demo/nomad-smoke.yml
    ```

=== "Windows PowerShell"

    ```powershell
    # CPU-only: delete the "--gpus all" line.
    docker run --rm `
      --gpus all `
      --publish 38217:38217 `
      --volume "${PWD}/cache:/var/cache/nomad" `
      ghcr.io/lanl/nomad:latest `
      serve `
        --transport=streamable-http `
        --host=0.0.0.0 `
        --port=38217 `
        /nomad/container/demo/nomad-smoke.yml
    ```

Wait until Nomad reports that the server is listening on port `38217`.

## 4. Connect URSA to Nomad

Open your persistent URSA user configuration and merge in this block without
replacing your model settings:

```yaml
mcp_servers:
  nomad:
    transport: streamable-http
    url: http://localhost:38217/mcp
```

The [configuration-file guide](../../docs/configuration/files-and-env.md) lists
the user-config location for macOS, Linux, and Windows. The
[MCP configuration guide](../../docs/configuration/mcp.md) explains transports,
headers, timeouts, and environment expansion.

In a second terminal, confirm that URSA loaded the user setting:

```bash
uv run ursa --print-config=user,resolved
```

Look for `mcp_servers.nomad` in the output, then start the TUI:

```bash
uv run ursa
```

## 5. Run the MIST workflow

Paste this prompt into the TUI:

```text
#execute Use the connected Nomad tools to find caffeine in PubChem, inspect the
model card for mist_models---mist_26p9M_kkgx0omx_qm9, and run that MIST model
on caffeine's canonical SMILES string. Save mist_caffeine_report.txt with the
SMILES input, every predicted property, units and descriptions when available,
and a brief explanation of what this demonstrates about connecting scientific
foundation models to URSA.
```

URSA should first resolve caffeine to a canonical SMILES string, inspect the
served model card, call the MIST model, and create
`mist_caffeine_report.txt` in the selected workspace. Review the report and the
tool activity rather than treating the generated prediction as experimentally
validated data.

The requested model has a
[MIST QM9 model card](https://huggingface.co/mist-models/mist-26.9M-kkgx0omx-qm9).

## Use the dashboard instead

The same user-level MCP configuration is available to the browser interface.
Follow the [dashboard guide](../../docs/getting-started/dashboard.md) for the
dashboard installation, credential settings, workspace selection, and launch
command. Choose the execution agent and submit the same prompt, then inspect the
activity timeline and generated report in the artifacts panel.

## Troubleshooting

- Ask `#execute List the connected Nomad tools` to confirm MCP discovery before
  running the full prompt.
- If no Nomad tools appear, recheck the resolved configuration and confirm that
  the container is still listening on port `38217`.
- If Docker rejects `--gpus all`, remove that line and run on the CPU.
- If the first model call is slow, watch the Nomad terminal; it may still be
  downloading weights into `cache`.

Stop Nomad with **Ctrl+C**. Remove the `nomad` block from your user config when
you no longer want URSA to connect to the local server.

The endpoint uses unauthenticated local HTTP for this exercise. Before exposing
Nomad on a network, apply appropriate authentication and transport security.
See Nomad's [getting-started guide](https://lanl.github.io/nomad/guides/getting-started.html)
and [deployment guide](https://lanl.github.io/nomad/deployments/guide.html) for
production considerations.
