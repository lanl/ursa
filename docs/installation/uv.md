# Install with uv

[`uv`](https://docs.astral.sh/uv/) is the recommended way to install URSA. It is fast, works well with isolated environments, and makes it easy to reproduce an installation.

## Install uv

If you do not already have `uv`, install it from Astral's official installer:

=== "macOS/Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows PowerShell"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

See the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for additional methods.

## Create an environment and install URSA

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install ursa-ai
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install with dashboard support

```bash
uv pip install "ursa-ai[dashboard]"
```

Then verify:

```bash
ursa --help
ursa-dashboard --help
```

## Project-style installation

If you are creating a new project around URSA:

```bash
uv init -p 3.12 my-ursa-project
cd my-ursa-project
uv add ursa-ai
```

With dashboard support:

```bash
uv add "ursa-ai[dashboard]"
```

## Optional: install as a uv tool

For command-line-only use, you can install URSA as a `uv` tool:

```bash
uv tool install ursa-ai
```

If you need the dashboard command from the tool installation:

```bash
uv tool install "ursa-ai[dashboard]"
```

For Python scripting, prefer a project or virtual environment installation so your scripts and URSA share the same environment.

## Next step

Continue with [Getting Started - CLI](../getting-started/cli.md).
