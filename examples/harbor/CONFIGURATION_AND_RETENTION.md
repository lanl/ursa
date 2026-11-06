# Reusable run configuration and artifact retention

Install `ursa-ai[harbor]` with Python 3.12 or newer. The Harbor extra supports
Harbor 0.25; the lockfile selects 0.25.0.

## Configure one invocation

`ursa.integrations.harbor_config` exports `AgentConfig`, `ModelConfig`,
`load_agent_config`, `load_provider_config`, and `resolve_run`. It contains no
campaign expansion, output naming, repetition, or scheduler policy.

```yaml
# agent.yaml
harbor_agent: codex
kwargs: {}
setup_timeout_multiplier: 1.2
```

```yaml
# provider.yaml
base_url: https://inference.example/v1
api_key: {env: LAB_API_KEY}
model_provider: openai
```

```python
import os
import subprocess
from pathlib import Path

from ursa.integrations.harbor_config import (
    ModelConfig, load_agent_config, load_provider_config, resolve_run,
)

resolved = resolve_run(
    Path("tasks/example"),
    load_agent_config(Path("agent.yaml")),
    ModelConfig(provider="lab", name="model-name"),
    load_provider_config(Path("provider.yaml"), "lab"),
    reasoning="high",
)
# These are inspectable without resolving the credential.
print(resolved.agent.model_dump())
subprocess.run(
    ["harbor", "run", *resolved.cli_args(), "--jobs-dir", "jobs"],
    env={**os.environ, **resolved.environment()},
    check=True,
)
```

Resolution does not launch a process or read the selected credential. The
environment is materialized separately at launch. The provider uses URSA's
existing `api_key` secret/reference schema, including keyring references.
`api_key_env` in the agent config chooses the host variable handed to Harbor's
`--api-key-name` (default `URSA_HARBOR_API_KEY`). Harbor owns projection to the
harness. Provider credential values do not enter CLI arguments or generated
native configuration. Use secret references for any additional native secrets
and avoid literal credentials in `env` or arbitrary agent kwargs.

Endpoints use `--base-url`; there is no `base_url_env` setting. A `base_url=`
argument to `resolve_run` overrides the provider URL, for example for a local
timing relay. `api_format` selects Harbor's `--api-format`. OpenCode can use
`openai_chat_completions` with a custom endpoint without an URL environment
variable or custom provider overlay.

Optional agent fields:

- `provider_kwargs` maps provider aliases to shallow overrides of `kwargs`.
- `native_config` accepts an inline mapping or a YAML/JSON/TOML file, relative
  to the agent file. Harbor receives a mapping and owns native serialization.
  A harness must declare native-config support.
- `reasoning_kwarg` selects a harness-specific option such as OpenCode's
  `variant`. Otherwise Harbor's `reasoning_effort` capability is required.
  URSA receives reasoning through `llm_model.reasoning_effort`.
- `model_template` supports `{provider}`, `{model_provider}`, and `{model}`.
  The default is `{model_provider}/{model}` for native harnesses and
  `{provider}/{model}` for URSA. Use `{model}` for a bare model selector.
- `env` contains Harbor environment settings/references.

The resolver checks Harbor options and capabilities, resolves only the selected
provider into URSA's native layer, and preserves caller inputs. Model names are
not restricted to filesystem slugs. Runtime availability and API connectivity
are still checked by Harbor at execution time.

## Retention plugin

```yaml
# retention.yaml
retention:
  version: 1
  collections: [artifacts]
  rules:
    - match: ["**/*.temporary"]
      action: delete
    - match: ["summary.json"]
      action: keep
  default_action: archive
  archive:
    format: tar.xz
    path: retained-artifacts.tar.xz
    remove_originals: true
    compression_level: 6
```

```bash
harbor run --path tasks/example --agent oracle \
  --plugin ursa.integrations.harbor_retention:RetentionPlugin \
  --plugin-kwarg policy=retention.yaml
```

The plugin runs at job end after all trial/verifier consumers have finished.
Completed failed trials are eligible too. Cancelled jobs that never reach job
finalization are left alone. The adapter recognizes single-step and multi-step
trial `artifacts` directories. Optionally select `verifier` to include verifier
payloads. Agent logs, job/trial control files and result records are outside its
scope. Collection `manifest.json`, verifier `reward.txt`/`reward.json`, and
`test-stdout.txt`/`test-stderr.txt` are protected even from catch-all rules.

Patterns match paths relative to each collection, using gitignore-style glob
syntax, anchored at the collection root. `**/*.dat` includes root and nested
files. The first matching rule wins; unmatched files default to `keep`.
Negative/comment patterns are rejected. Archive members preserve the full
job-relative trial/step path, avoiding name collisions between trials.

The archive is written to a temporary file, closed, checked for exact inventory
and content hashes, then published without overwriting an existing file. Only
then are archived originals and delete-selected files removed. Keep enough
free space for sources plus the temporary archive. Compression level is 0–9;
CPU/disk limits come from the surrounding job. A lock serializes retention
writers; callers must ensure no other process is still writing the payloads.
POSIX directory fsync is used for publication and journal durability.

`retention.json` records the plan, file identities, input bytes by action,
member/archive hashes, deleted paths, link warnings, compression size, elapsed
time, and status. Reapplying the same completed policy verifies it without
rewriting the manifest. Interrupted compression can restart; interrupted cleanup
resumes against the verified archive. Changed inputs, different policies, and
unrelated existing archives are rejected. One policy/archive per job is supported.

Harbor logs plugin errors without failing the job. Check
`retention-status.json` for `status: complete` if retention is required.
The plugin never changes verifier scores. Its duration is recorded separately
from trial execution. A retention exception remains distinguishable from the
scientific outcome.

## Preview, resume, and restore

```bash
ursa-artifacts plan --run-dir jobs/example --policy retention.yaml
ursa-artifacts apply --run-dir jobs/example --policy retention.yaml
ursa-artifacts restore --manifest jobs/example/retention.json \
  --destination restored-example
```

Preview reads inventories and hashes archive-selected files without writing.
Apply requires finished job and trial result metadata and also supports resuming
an interrupted policy. It updates the same status sidecar used by the plugin.
Nothing scans or rewrites historical jobs automatically.

Restore verifies the archive and extracts into a fresh directory, rejecting
path traversal and escaping symlinks. Symlinks are archived as links without
following their targets; warnings identify targets absent from the archive.
An archive containing an escaping link is intentionally rejected by restore.
Deleted files are reported as unrecoverable. Kept loose files are not copied
by restore; retrieve them from the original job.

The Harbor-independent `ursa.artifacts.retention` library exposes
`plan_retention(root, policy, roots=..., protected=...)`, `apply_retention(plan)`,
`retain_artifacts(...)`, and `restore_artifacts(manifest, destination)`. Other
benchmark adapters can supply explicit artifact roots and protected paths.
