# Install with pip

Use `pip` if you already manage Python environments with `venv`, Conda, or another environment manager.

## Create a clean virtual environment

=== "macOS/Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```

=== "Windows PowerShell"

    ```powershell
    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```

## Install URSA

```bash
python -m pip install ursa-ai
```

## Install with dashboard support

```bash
python -m pip install "ursa-ai[dashboard]"
```

Then verify:

```bash
ursa --help
ursa-dashboard --help
```

## Conda environment with pip

If you prefer Conda for Python environment management:

```bash
conda create -y -n ursa-env python=3.12
conda activate ursa-env
python -m pip install ursa-ai
```

With dashboard support:

```bash
python -m pip install "ursa-ai[dashboard]"
```

## Next step

Continue with [Getting Started - CLI][getting-started-cli].
