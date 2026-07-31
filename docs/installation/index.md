# Installation

URSA is published on PyPI as [`ursa-ai`](https://pypi.org/project/ursa-ai/). The installed command-line entry points are:

```text
ursa
ursa-dashboard
```

We recommend installing URSA with [`uv`](https://docs.astral.sh/uv/) because it creates reproducible Python environments quickly and handles dependency resolution well. Clear `pip` instructions are also provided because many users already manage Python environments with `venv`, Conda, or system tooling.

## Python version

URSA requires Python 3.11 or newer. For most users, Python 3.12 or newer is a good default.

## Choose an installation path

- [Install with uv][install-with-uv] — recommended for most new URSA projects.
- [Install with pip][install-with-pip] — useful if you already use `venv`,
  Conda, or another Python environment manager.

## Optional extras

URSA includes optional dependency groups for features that are not required by the core package. The most commonly used extra is the web dashboard:

```bash
uv pip install "ursa-ai[dashboard]"
# or
python -m pip install "ursa-ai[dashboard]"
```

The dashboard extra installs the web-server dependencies needed by `ursa-dashboard`.

Other optional extras exist for specialized workflows, such as LAMMPS, DSI, office-document readers, OpenTelemetry, Materials Project, image support, and optimization tooling. Install them only when needed.

## Verify your installation

After installing, run:

```bash
ursa --help
```

If you installed the dashboard extra, also check:

```bash
ursa-dashboard --help
```

## Next steps

After installation, create a reusable model configuration file and run the
[CLI getting-started guide][getting-started-cli].
