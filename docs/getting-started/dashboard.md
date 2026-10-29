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

The optional config file initializes the dashboard LLM endpoint settings. It is
not needed for the built-in OpenAI provider.

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

1. Open **Settings → LLM** and confirm the endpoint and credential source.
2. Create a session.
3. Select a folder you are comfortable modifying, or choose **Temporary
   workspace** for disposable work.
4. Choose an agent and submit a prompt.
5. Follow the activity timeline and inspect generated files in the artifacts
   panel.

## Summarization settings

Under **Settings → LLM**, set **Summarization fallback (tokens)** to the threshold
used when the model's context limit is unknown (default: 50,000). When the limit
is known, agents summarize at 70% of it. The setting also appears in session
settings; changes apply to new runs, and explicit agent fallbacks take precedence.
The dashboard's `--config` option imports `tokens_before_summarize` from CLI
configuration files when provided.

After saving model settings, use **Check saved model** to display the effective
threshold. This checks metadata without generating a response. For an unloaded
Ollama model, the allocated context may be unavailable; configure `num_ctx` or
check again once the model is running.

## Log previews and full captured output

The Run logs panel keeps a recent preview rather than rendering entire archives:
up to 128 Ki characters per stream, within the configured line limit. It loads
bounded tails from the last five completed runs in a session, and bounds initial
live-event replay. Select any session run in **Full logs** to download its captured
**stdout** or **stderr** without loading the full file into the browser's log panel.
Environment run replay pages also provide full stdout/stderr download links.
Timeline redraws are batched, ordinary events preserve graph layout, and timeline
cards and large event-content previews are limited while playback remains available.

Authenticated API downloads are available at:

- `/runs/{run_id}/logs/stdout` and `/runs/{run_id}/logs/stderr`
- `/environment-runs/{run_id}/logs/stdout` and
  `/environment-runs/{run_id}/logs/stderr`

Adding `?tail_bytes=131072` requests a bounded preview (maximum: 262144 bytes).
Without that parameter, the server sends the captured file as an attachment.
Downloads use the dashboard's existing authentication and path-jail checks.

To free disk space, select a finished run under **Full logs** and click **Delete
stored logs**. Environment run pages offer the same action next to their log
downloads. A confirmation warns that deletion is permanent; download anything
needed for provenance first. Only the selected run's captured `stdout.log` and
`stderr.log` are removed. Structured event history (which can also contain log
excerpts), results, artifacts, and checkpoints are retained. Active runs cannot
have their stored logs deleted. **Clear preview** only clears the visible buffer
and does not remove files from disk.

Authenticated `DELETE /runs/{run_id}/logs` and
`DELETE /environment-runs/{run_id}/logs` provide the same per-run operation.
Repeating a deletion is safe; missing captures are treated as already deleted.

New dashboard runs retain the full captured worker stdout/stderr on disk by
default, independently of browser-preview and event-log limits. Programmatic
`RunConfig` and `EnvironmentRunManagerConfig` still accept finite
`stdout_cap_bytes` / `stderr_cap_bytes` limits when disk budgets require them;
those explicitly capped captures are not complete archives. Output already lost
under earlier capture limits cannot be recovered. Tool-level output truncation
is unchanged; these downloads contain the worker process streams, not an
additional archive of every tool's subprocess output.

Full capture can use significant disk space. Ordinary agent runs remain subject
to dashboard retention (by default seven days, 100 runs per agent, and a 2 GiB
completed-run disk budget). That cleanup runs after completion, not as an active
run's disk quota. Environment-run directories do not currently have automatic
retention; archive or remove them explicitly as needed. Save provenance downloads
before deleting runs or retention removes them.

This is a bounded-preview optimization, not a fully virtualized environment replay
or indexed event store. Environment event data still grows in browser memory during
long live sessions, and historical/resume API scans can still grow with the archive.

## Configure API credentials

Open **Settings -> LLM** and choose an API-key source:

- **Keyring** stores the key using URSA's [keyring
  support](../configuration/secrets.md). The key field is always blank when
  Settings opens; the dashboard reports only whether a usable key is configured.
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
timeout settings configured under **Settings**. Member-specific model blocks may
refer to API-key environment variable names, but literal API keys are rejected
and never stored in YAML or run metadata.

!!! note "Headless Linux"
    When no native keyring is available, URSA uses its file-backed keyring.
    Environment variables and deployment-managed secret providers remain
    available when preferred.

## Where next?

- [Configuration](../configuration/index.md)
- [Persistence](../persistence/index.md)
- [Sandboxing and information control][sandboxing-and-information-control]
