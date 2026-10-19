# Getting started

This guide takes you from installation to a first URSA conversation, then shows
the terminal interface and browser dashboard. OpenAI models work without a config
file; configuration is only needed when you want to change a default or use a
different endpoint.

## Install URSA

URSA requires Python 3.11 or newer. On macOS and Linux, the guided installer
installs `uv` if needed, installs URSA and the dashboard with Python 3.13, then
walks through provider, model, and optional MCP configuration:

```bash
curl -LsSf https://lanl.github.io/ursa/latest/install.sh | sh
```

The development installer at
`https://lanl.github.io/ursa/main/install.sh` installs URSA from GitHub main.

Credentials are validated before setup continues and are stored in the
operating-system keyring. The resulting configuration is written to
`~/.config/ursa/config.yaml`. Re-run the wizard with `ursa self setup`.

For manual installation, the `uv` tool installation is recommended:

=== "uv tool (recommended)"

    Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if
    needed, then install URSA and the dashboard in an isolated tool environment:

    ```bash
    uv tool install --python 3.13 'ursa-ai[dashboard]'
    ```

    Update URSA with: `ursa self update` or `uv tool upgrade ursa-ai`

=== "venv + pip"

    Use this option when you already manage Python virtual environments:

    === "macOS/Linux"

        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install --upgrade pip
        python -m pip install 'ursa-ai[dashboard]'
        ```

    === "Windows PowerShell"

        ```powershell
        py -3 -m venv .venv
        .\.venv\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        python -m pip install 'ursa-ai[dashboard]'
        ```

=== "Conda + pip"

    ```bash
    conda create -y -n ursa-env python=3.13
    conda activate ursa-env
    python -m pip install 'ursa-ai[dashboard]'
    ```

Verify both applications:

```bash
ursa --help
ursa-dashboard --help
```

## Optional dependencies

Install optional features using extras on the `ursa-ai` package. Multiple extras
can be combined, for example, `python -m pip install 'ursa-ai[dashboard,lammps]'`
in your activated environment, or
`uv tool install --python 3.13 'ursa-ai[dashboard,lammps]'` for a new tool installation.

| Extra | Purpose | Additional requirements |
| --- | --- | --- |
| `dashboard` | Browser dashboard (FastAPI and Uvicorn) | A web browser. |
| `fm` | Expose PyTorch models as MCP tools | Model weights and any model-specific dependencies; compatible accelerator drivers if using a GPU. CPU execution is also supported. |
| `lammps` | Potential discovery and LAMMPS workflow support | Separately installed LAMMPS and MPI executables; network access for remote potential discovery. See [LAMMPS dependencies](../agents/lammps.md#dependencies). |
| `office_readers` | Read Word and PowerPoint documents | No additional external executable for these readers. |
| `opt` | Google's [OR (Operations Research)-Tools](https://developers.google.com/optimization) optimization support | Other solvers used by the Optimization Agent may require separate installation and licenses. |
| `otel` | Export OpenTelemetry data over OTLP/HTTP | A configured, reachable OTLP collector or compatible service. |
| `dsi` | Data Science Infrastructure integration | Data sources and backend-specific dependencies required by the DSI workflow. |
| `image` | Image handling with Pillow | A model with image support when sending images to an LLM. |
| `mp` | Materials Project API access | A Materials Project API key and network access. |
| `metric_plots` | Plot and analyze collected metrics such as execution times, LLM token usage, and tokens per second | No additional external executable. |

Extras install Python dependencies; they do not install external simulation codes,
MPI, or model weights. The LAMMPS Agent defaults to `lmp_mpi` and `mpirun`;
configure its `lammps_cmd` and `mpirun_cmd` arguments if your executables have
different names or paths. Your LAMMPS build must include the packages required by
the selected potential and simulation.

## Start with the built-in OpenAI configuration

Set your OpenAI API key and launch URSA:

=== "macOS/Linux"

    ```bash
    export OPENAI_API_KEY="..."
    ursa
    ```

=== "Windows PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "..."
    ursa
    ```

That is a complete working setup. The built-in `openai` inference provider
already supplies the model and OpenAI base URL, so an OpenAI-only config file is
unnecessary.

!!! warning "Choose workspaces deliberately"
    The execution agent can write files and run commands. Start URSA in a
    disposable exercise directory, or pass `--workspace` with a directory you
    are comfortable modifying.

## Optional: customize your user configuration

Use a user config for defaults that should follow you across projects. Do not
copy the same `config.yaml` into every project.

| Platform | User configuration path |
| --- | --- |
| macOS | `~/Library/Application Support/ursa/config.yaml` |
| Linux | `~/.config/ursa/config.yaml` |
| Windows | `%APPDATA%/ursa/config.yaml` |

For example, this changes only the embedding model and leaves the built-in
OpenAI chat configuration intact:

```yaml
emb_model:
  model: openai:text-embedding-3-large
```

Inspect the merged user configuration with:

```bash
ursa --print-config=user,resolved
```

See [Configuration][configuration] for other providers and precedence rules.

## Learn the TUI

Run `ursa`. The welcome panel confirms the active model, workspace, and agent.

- Enter ordinary text to chat.
- Type `#` to open the agent picker. `#plan` creates a plan; `#execute` can use
  tools, run commands, and create workspace artifacts.
- Type `/` to browse application commands. `/keymap` shows every shortcut.
- Type `@` to find and insert a workspace file into a prompt.

Try these in order:

```text
Explain the difference between the chat, planning, and execution agents.
```

```text
#plan Plan a small parameter sweep and describe the outputs we should retain.
```

```text
#execute Create hello_ursa.txt containing a one-sentence description of this workspace.
```

Review proposed tool actions before approving them. Use a named agent when you
want its state to persist between launches:

```bash
ursa --name tutorial
```

The [TUI guide][getting-started-tui] covers commands, web-tool opt-in, and named
agents in more detail.

## Use the dashboard

Launch the browser interface:

```bash
ursa-dashboard
```

It opens `http://127.0.0.1:8080`. Then:

1. Open **Settings → LLM** and confirm the endpoint and credential source.
2. Create a session and select a folder or a temporary workspace.
3. Choose an agent, enter a prompt, and follow the live activity timeline.
4. Inspect generated files in the workspace/artifacts panel.
5. Use **Environment runs** when you want to launch a team or symposium from
   YAML instead of a single-agent session.

The dashboard and TUI use the same URSA concepts, but browser credentials are
managed in **Settings** and each dashboard session has an explicit workspace.
See the [dashboard guide][getting-started-web-dashboard] for credential storage,
remote-access safety, and environment runs.

## Run an example

Continue with the [examples gallery][examples]. The environment walkthrough is
a good first exercise; the Nomad/MIST example shows how URSA can call a served
scientific model through MCP.
