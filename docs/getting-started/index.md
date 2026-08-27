# Getting started

This guide takes you from installation to a first URSA conversation, then shows
the terminal interface and browser dashboard. OpenAI models work without a config
file; configuration is only needed when you want to change a default or use a
different endpoint.

## 1. Install URSA

URSA requires Python 3.11 or newer. The `uv` tool installation is recommended:

=== "uv tool (recommended)"

    Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if
    needed, then install URSA and the dashboard in an isolated tool environment:

    ```bash
    uv tool install 'ursa[dashboard]'
    ```

    Upgrade it later with:

    ```bash
    uv tool upgrade ursa
    ```

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
    conda create -y -n ursa-env python=3.12
    conda activate ursa-env
    python -m pip install 'ursa-ai[dashboard]'
    ```

Verify both applications:

```bash
ursa --help
ursa-dashboard --help
```

## 2. Start with the built-in OpenAI configuration

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

## 3. Optional: customize your user configuration

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

## 4. Learn the TUI

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

## 5. Use the dashboard

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

## 6. Run an example

Continue with the [examples gallery][examples]. The environment walkthrough is
a good first exercise; the Nomad/MIST example shows how URSA can call a served
scientific model through MCP.
