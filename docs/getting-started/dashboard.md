# Getting Started - Web Dashboard

The URSA web dashboard provides a browser-based interface for running URSA workflows.

Install URSA first with `uv tool install --python 3.13 'ursa-ai[dashboard]'` as described in the
[getting started guide][getting-started].

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

The optional config file initializes the dashboard model settings and makes its
named `inference_providers` available in the LLM and Embedding/RAG settings.
Select a provider to use its configured endpoint and load the models advertised
by that endpoint. Model fields remain editable when an endpoint does not support
model discovery. The config file is not needed for the built-in OpenAI provider.

Dashboard model, MCP, and RAG-tool settings use this precedence, from highest
to lowest: an explicit `--config` file, standard URSA environment overrides,
the user config, dashboard `settings.json`, and the system config. The explicit
file is applied at runtime and is not copied into `settings.json`. This means a
value declared in the user config remains authoritative over the corresponding
dashboard default.

## Enable web tools

Web, arXiv, and OSTI search tools are opt in, matching `ursa --use-web` in the
CLI and TUI:

```bash
ursa-dashboard --use-web
```

Equivalently, set `URSA_DASHBOARD_USE_WEB=1` in the environment.

This sets the default for agents that support web access (chat, execution,
planning/execution workflow, prompt refinement, and deep review). Individual
runs can still override it through the agent's advanced parameters, so you can
launch with `--use-web` and disable it for a specific run.

## First session

The dashboard offers a guided walkthrough on its first visit. You can close it
at any point and reopen it with **Take a guided tour** on the welcome page.
It covers model setup, example prompts, behaviors, persistent agents, workspace
selection, logs and artifacts, and multi-agent environments. The prime-spacing
example prepares an editable prompt with Chat selected; it runs only when
you press **Send**. The workspace picker then asks for the folder where URSA
will start the session's work. You can change it later with **Set workspace**
in the Artifacts panel.

First-time connection setup is split into three short steps: the Base URL,
the API key source, and the language model. It explains each choice without
showing the full configuration editor. Choose **Enter a key · store securely**
to paste a provider key if you have not set an environment variable. Advanced
options and embedding models remain available in **Default config** later.

1. Open **Model and Agent Settings → Default config** to connect your models,
   or **LLM** to inspect the effective dashboard settings.
2. Create a session.
3. Select a folder you are comfortable modifying, or choose **Temporary
   workspace** for disposable work.
4. Choose an agent and submit a prompt.
5. Follow the activity timeline and inspect generated files in the artifacts
   panel.

New sessions snapshot the effective LLM, embedding, MCP, and RAG-tool settings.
Open **Session settings** from a session card to adjust the same controls for
that session without changing the dashboard defaults. Securely saved session
keys use an isolated credential-store entry, so replacing one does not replace
the global key or another session's key. Theme and agent-management actions
remain dashboard-wide.

## Edit your default configuration

**Model and Agent Settings → Default config** edits the normal URSA user config,
shared with the CLI and TUI. It loads the highest-priority existing user config
(including `~/.config/ursa/config.yaml`), or offers a starter configuration when
no user file exists. The selected path is displayed in the editor.

Add providers using a name, Base URL, API type, and credential source. Keys
entered here are saved in the operating system keyring under the standard URSA
service; the YAML contains only a `keyring` reference. Environment-variable
references and unauthenticated endpoints are also supported. Select default
language and optional embedding models, use **Find available models** for
suggestions, and edit advanced model options if needed.

If an older config contains a literal API key, saving through the editor moves
that key into secure storage as well. The original backup retains the original
file contents and is created with owner-only permissions.

**Test language model** sends a small real generation request;
**Test embeddings** checks an embedding request. These are explicit actions,
may incur provider charges, and do not save the draft. A **Passed** or **Failed**
result appears beside each test; editing the draft marks previous results as
needing a new test. The walkthrough's
**Test, save & continue** checks the language model before saving.

**Update** validates the draft, saves the user config, and reloads defaults and
the provider catalog immediately. An explicit launch config or environment
override still takes precedence. Existing sessions retain their snapshots.
External file edits still require a dashboard restart. The Default config pane
is shared across sessions; the other model/tool panes in Session settings edit
that session instead.

Unrelated config options and existing provider options are preserved. YAML is
rewritten, so comments and formatting may change; an exact backup is saved
beside the original as `config.backup-*.yaml`. A concurrent external edit is
reported instead of silently overwritten. Replacing a provider key creates a
new keyring reference so old sessions and config backups retain access to the
previous credential; old entries are not automatically deleted.

## Configure API credentials

Open **Model and Agent Settings → LLM** and choose an API-key source:

- **Secure system storage** stores the key in macOS Keychain, Windows
  Credential Manager, or the available Linux keyring service. The key field is
  always blank when Settings opens; the dashboard reports only whether a usable
  key is configured.
- **Environment variable** retains the existing headless and automation
  workflow. Enter the variable name, not its value.
- **No API key** is appropriate for endpoints that do not require one.
- **URSA config key** uses the standard keyring reference from the selected
  provider. Edit these keys in **Default config**.

Embedding credentials are configured independently under
**Model and Agent Settings → Embedding/RAG**, or can explicitly reuse the saved LLM key when
both configurations resolve to the same provider or endpoint origin. Saved keys
are bound to the configured provider or endpoint origin. After changing the
endpoint host, save the key again to approve its use with that host.

The raw key is never written to dashboard settings, sessions, run records, or
worker configuration files. In remote dashboard mode, credential changes must
be served over HTTPS.

## Choose a session workspace

Every new dashboard session requires an explicit workspace choice. Select a
folder you can find and reuse, or choose **Temporary workspace** for disposable
work. Temporary mode behaves like `ursa --workspace tmp`: the dashboard creates
the workspace in the operating system's temporary directory and removes it when
the session is deleted or the dashboard stops.

The dashboard no longer creates a hidden default session workspace under
`~/.cache/ursa`. Older sessions without an explicit workspace remain available,
but the dashboard prompts for a folder or temporary workspace before their next
run. When an older session still has its former UUID-named cached workspace, the
folder field is prefilled with that path so existing work remains easy to
recover. A user-selected folder is never deleted when its dashboard session is
deleted.

## Launch an agent team or symposium

Open **Environment runs** from the dashboard sidebar. The page provides **New
team** and **New symposium** actions that let you edit a starter YAML definition,
optionally choose a unique Run ID, enter the task prompt, validate the
configuration, and launch it directly from the browser. Reuse the same
environment name with a new Run ID for follow-on work that should retain the
team or symposium workspace while creating a separate replay.

The run is queued in the background and appears on the page immediately. Open it
to follow the environment graph and work timeline live, inspect the final result,
or cancel a dashboard-launched run. The dashboard uses the LLM, credential, and
timeout settings configured under **Model and Agent Settings**. Member-specific model blocks may
refer to API-key environment variable names, but literal API keys are rejected
and never stored in YAML or run metadata.

!!! note "Headless Linux"
    Secure system storage requires an available desktop keyring service.
    Headless deployments should continue to use environment variables or a
    deployment-managed secret provider.

## Where next?

- [Configuration](../configuration/index.md)
- [Persistence](../persistence/index.md)
- [Sandboxing and information control][sandboxing-and-information-control]
