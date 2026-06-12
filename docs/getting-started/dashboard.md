# Getting Started - Web Dashboard

The URSA web dashboard provides a browser-based interface for running URSA workflows.

!!! note "Screenshots forthcoming"
    A complete dashboard walkthrough should include screenshots. This page currently covers installation and launch commands; a fuller visual guide can be added later.

## Install dashboard dependencies

Install URSA with the dashboard extra:

=== "uv"

    ```bash
    uv pip install "ursa-ai[dashboard]"
    ```

=== "pip"

    ```bash
    python -m pip install "ursa-ai[dashboard]"
    ```

## Launch the dashboard

```bash
ursa-dashboard
```

By default this serves on `127.0.0.1:8080`.

You can set the host, port, group, and initial config file:

```bash
ursa-dashboard \
  --host 127.0.0.1 \
  --port 8080 \
  --group default \
  --config config.yaml
```

The config file initializes the dashboard LLM endpoint settings.

## Where next?

- [Configuration](../configuration/index.md)
- [Persistence](../persistence/index.md)
- [Sandboxing and information control](../best-practices/sandboxing.md)
