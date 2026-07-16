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

## Configure API credentials

Open **Settings -> LLM** and choose an API-key source:

- **Secure system storage** stores the key in macOS Keychain, Windows
  Credential Manager, or the available Linux keyring service. The key field is
  always blank when Settings opens; the dashboard reports only whether a usable
  key is configured.
- **Environment variable** retains the existing headless and automation
  workflow. Enter the variable name, not its value.
- **No API key** is appropriate for endpoints that do not require one.

Embedding credentials are configured independently under
**Settings -> Embedding/RAG**, or can explicitly reuse the saved LLM key when
both configurations resolve to the same provider or endpoint origin. Saved keys
are bound to the configured provider or endpoint origin. After changing the
endpoint host, save the key again to approve its use with that host.

The raw key is never written to dashboard settings, sessions, run records, or
worker configuration files. In remote dashboard mode, credential changes must
be served over HTTPS.

!!! note "Headless Linux"
    Secure system storage requires an available desktop keyring service.
    Headless deployments should continue to use environment variables or a
    deployment-managed secret provider.

## Where next?

- [Configuration](../configuration/index.md)
- [Persistence](../persistence/index.md)
- [Sandboxing and information control](../best-practices/sandboxing.md)
