# Run URSA with Harbor

This example connects URSA's `ExecutionAgent` to
[Harbor](https://www.harborframework.com/docs/core-concepts) and includes
GPAW, Parthenon, and PyTorch installation tasks.

## Run a task

From this directory, install Python 3.12+, [uv](https://docs.astral.sh/uv/),
and Docker for Harbor's default environment. Set a provider credential, then:

```bash
export OPENAI_API_KEY=...
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config="$PWD/ursa.yaml" \
  --model openai/openai:gpt-5.4-nano
```

A successful task reports reward `1`; inspect results with `uv run harbor view
jobs`. Use `--path benchmark/tasks` to run all included tasks.

To continue a completed trial locally, run:

```bash
uv run harbor trial handoff jobs/<job>/<trial>
```

Handoff imports the trial checkpoint into its recorded, existing URSA group and
opens the TUI; the task container's files are not copied into the workspace.

### Add a task

Start with Harbor's generator, then follow its task tutorial and reference for
the generated instruction, environment, task, and verifier files:

```bash
uv run harbor task init --tasks-dir benchmark/tasks my-org/my-task
```

The official [task tutorial](https://www.harborframework.com/docs/tasks/task-tutorial),
[format reference](https://www.harborframework.com/docs/tasks), and
[publishing guide](https://www.harborframework.com/docs/tasks/publishing) cover
authoring and sharing tasks.

### Configure URSA

Harbor models use `<inference-provider>/<model-provider>:<model-name>`, such
as `ollama/ollama:gemma4:latest`. The inference provider must exist in
`ursa.yaml`; its `model_provider` allows omitting the model-provider prefix.
`--model` replaces the configured model, and Harbor
`[[environment.mcp_servers]]` entries are attached automatically. See the
[MCP task tutorial](https://www.harborframework.com/docs/tutorials/mcp-server-task).

### Advanced installation

The adapter installs Python and URSA with `uv` inside each task container.
`ursa_install_spec` must be a [PEP 508 requirement](https://peps.python.org/pep-0508/)
for `ursa-ai`. It defaults to the host checkout or installed version.

| Source | `ursa_install_spec` example |
| --- | --- |
| Package version | `ursa-ai==0.17.3` |
| Git revision | `ursa-ai @ git+https://github.com/lanl/ursa.git@main` |
| Local checkout | `ursa-ai @ file:///absolute/path/to/ursa` |
| Local wheel | `ursa-ai @ file:///absolute/path/to/ursa_ai-0.17.3-py3-none-any.whl` |
| Local sdist | `ursa-ai @ file:///absolute/path/to/ursa_ai-0.17.3.tar.gz` |

Quote the whole kwarg, including the package name and `@` for URL references;
bare paths and bare Git URLs are not requirements. Use `Path(...).resolve().as_uri()`
to create a file URL with spaces or platform-specific paths encoded correctly.
Local sources are uploaded before installation; checkouts exclude Git-ignored
and common secret files. An implicit host checkout is built into a wheel first.

```bash
uv run harbor run \
  --path benchmark/tasks/install-pytorch \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --agent-kwarg config="$PWD/ursa.yaml" \
  --model openai/openai:gpt-5.4-nano \
  --agent-kwarg 'ursa_install_spec=ursa-ai @ file:///absolute/path/to/ursa' \
  --agent-kwarg ursa_extras=image \
  --agent-kwarg 'extra_packages=["numpy", "scipy"]'
```

Requirement extras and environment markers are preserved; the `harbor` extra
is always added. `ursa_extras` adds extras, and `extra_packages` installs
additional requirements alongside URSA.

The agent's native `config` is the complete URSA configuration before Harbor's
model and MCP layer are applied.
Secret and keyring references resolve on the host; only generated environment
references reach the runner, never host config or keyring files. Trial output
includes `agent/ursa.log`, lifecycle records in `agent/ursa.jsonl` (including
chat token usage), and `agent/ursa_result.out`.

## Singularity environment

Use this environment for direct runs:

```bash
--env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment
```

It builds `environment/Dockerfile` with Buildah, Podman, or Docker and converts
it to a SIF; no Apptainer definition is needed. Compute nodes need `apptainer`
or `singularity` plus one builder (Docker also needs a running daemon and
socket access). SIFs cache by build context in `$XDG_CACHE_HOME/ursa/harbor/sif`
or `~/.cache/ursa/harbor/sif`; concurrent builds share a lock. Set
`URSA_HARBOR_SIF_CACHE`, pass the environment constructor's
`singularity_image_cache_dir`, or use `prebuild --image-cache-dir` to choose a
shared cache.

`singularity_fakeroot` defaults to `None`: fakeroot is enabled with Apptainer
and disabled with Singularity. Passing
`--environment-kwarg singularity_fakeroot=true` or
`--environment-kwarg singularity_fakeroot=false` overrides this choice.
Apptainer is preferred when both runtimes are installed. A failed build or task
command that reports requiring root under Singularity warns that switching to
Apptainer may help.

With Apptainer fakeroot, writable overlays are created by
`apptainer overlay create --fakeroot --sparse` so their ownership matches the
container's user namespace. Overlay size follows `[environment].storage_mb`
(default 1024 MiB); Apptainer requires at least 64 MiB. Singularity and runs
without fakeroot retain the host-created ext3 overlay.

Validate tasks against URSA's Singularity constraints, or warm the SIF cache,
without running an agent:

```bash
uv run -m ursa.integrations.harbor validate benchmark/tasks
uv run -m ursa.integrations.harbor prebuild benchmark/tasks
```

Prebuild includes separate verifier environments and every Compose service:
Dockerfile builds are built, and `image:` references are fetched as local SIFs.
Runs reuse these files from the same cache without contacting the registry.
Use `prebuild --force` to rebuild Dockerfiles and refresh fetched images.

### Agent install layers

Prebuild can cache the complete installation, including system packages and
installed binaries, in a read-only overlay:

```bash
export URSA_HARBOR_AGENT_CACHE="/shared/$USER/ursa-agent-layers"
uv run -m ursa.integrations.harbor prebuild benchmark/tasks \
  --agent goose --agent claude-code --agent codex
```

Each installer runs in a fresh environment based on the task's main SIF, without
trial mounts, model credentials, inference, or verification. Only a successfully
installed, stopped overlay is published. Concurrent prebuilds share a lock.
Build overlays have at least 4 GiB of sparse capacity for temporary installer
files; this does not change the trial's writable storage setting.
Layers are keyed by the base image contents, agent/version, Harbor/runtime
version, fakeroot setting, host user/home, and working directory. Use the same
per-user cache on the prebuild and compute nodes.

Select a layer when starting the environment and use URSA's installed-agent
adapter so Harbor skips installation:

```bash
harbor run --path benchmark/tasks/example \
  --agent ursa.integrations.harbor.installed_agents:Goose \
  --model provider/model \
  --env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment \
  --environment-kwarg singularity_agent=goose
```

| Prebuild / `singularity_agent` name | Harbor agent import path |
| --- | --- |
| `ursa` | `ursa.integrations.harbor:UrsaHarborAgent` |
| `goose` | `ursa.integrations.harbor.installed_agents:Goose` |
| `claude-code` | `ursa.integrations.harbor.installed_agents:ClaudeCode` |
| `codex` | `ursa.integrations.harbor.installed_agents:Codex` |

Each trial mounts a private sparse copy of the cached installation read-only
beneath its own writable overlay; its writable Harbor directories are seeded
from that combined image. Filesystems that support reflinks share the copy's
unchanged blocks. The private copy also isolates filesystem mount metadata.
Trial changes never update the cached layer. An absent or mismatched layer fails
with instructions to prebuild it. The adapters retain normal Harbor installation
when no layer is selected. The original Harbor `goose` agent unconditionally
reinstalls, so use the adapter to benefit from the cached installation.
Separate verifier environments retain their own images without the agent layer.

For URSA, prebuild with `--agent ursa`, then add
`--environment-kwarg singularity_agent=ursa` to your usual URSA Harbor run.
The existing `UrsaHarborAgent` handles reuse directly. Its default install source
is the current checkout or installed package. Each trial still prepares its own
native configuration, model settings, and secret references; these are not cached.

To customize the URSA installation, pass `--agent-kwarg` to prebuild (with one
`--agent`). Supported options are `ursa_install_spec` (a PEP 508 requirement),
`ursa_extras`, and `extra_packages`. Match those options in both the runtime
agent kwargs and the environment's `singularity_agent_kwargs` JSON object:

```bash
uv run -m ursa.integrations.harbor prebuild benchmark/tasks --agent ursa \
  --agent-kwarg 'ursa_install_spec=ursa-ai==0.17.3'

harbor run --path benchmark/tasks/example \
  --agent ursa.integrations.harbor:UrsaHarborAgent \
  --model provider/model \
  --agent-kwarg 'ursa_install_spec=ursa-ai==0.17.3' \
  --env ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment \
  --environment-kwarg singularity_agent=ursa \
  --environment-kwarg 'singularity_agent_kwargs={"ursa_install_spec":"ursa-ai==0.17.3"}'
```

URSA layers also track the package requirement, extras, additional packages,
Python/uv versions, and local source contents. Source changes require a new
prebuild. Use `--force` to refresh an unpinned remote requirement.

Use `--agent NAME@VERSION` during prebuild and
`--environment-kwarg singularity_agent=NAME@VERSION` with
`--agent-kwarg version=VERSION` during the run. Version strings are forwarded to
Harbor's installer unchanged. For URSA, select versions through
`ursa_install_spec` instead. `--agent-install-timeout-sec` adjusts the
1800-second limit per installer. Prebuild honors the task's setup network policy;
run it where installation is allowed, then reuse the layer for offline setup.
Match any runtime fakeroot override with prebuild's `--fakeroot` or
`--no-fakeroot`.

The default cache is `$XDG_CACHE_HOME/ursa/harbor/agents` or
`~/.cache/ursa/harbor/agents`. Override it with `URSA_HARBOR_AGENT_CACHE`,
prebuild's `--agent-cache-dir`, or the environment's
`singularity_agent_cache_dir`. `prebuild --force` rebuilds both images and the
selected install layers. Omit `--agent` to build only images. Cached layers
persist independently of artifact retention; clear them only when no trials or
prebuilds are running.

### Compose and networking

`environment/docker-compose.yaml` starts sidecars through
[`singularity-compose`](https://singularityhub.github.io/singularity-compose/).
The `main` service may use an image or a Dockerfile build; verifier overlays
are resolved relative to their overlay file. The adapter supports images,
Dockerfile builds, list `depends_on`, commands, bind mounts, TCP ports,
replicas, environment, `env_file`, and per-service artifacts. Unsupported
Compose features fail before launch; consult Harbor's
[multi-container guide](https://docs.harborframework.com/core-concepts/tasks/multi-container)
and the [`singularity-compose` specification](https://github.com/singularityhub/singularity-compose/blob/master/docs/spec/spec-2.0.md).
Apptainer and SingularityCE work without a system-wide `singularity` alias;
the host runtime needs fakeroot CNI networking.

The adapter supports Harbor `public` and `no-network` policies. The
`[environment]` policy is the setup baseline; omitted agent and verifier phase
policies inherit it:

```toml
[environment]
network_mode = "public"

[agent]
network_mode = "no-network"

[verifier]
network_mode = "no-network"
```

Before each phase it restarts the same SIF with the requested policy, preserving
its writable overlay and Harbor binds. `public` enables URSA web tools and
`no-network` disables them, while Harbor remains the actual network boundary.
Keep the overlay enabled: `singularity_overlay=false` disables phase changes.
`no-network` is unavailable for Compose projects with sidecars, and network
allowlists are unsupported because SingularityCE 3.6.2 cannot enforce them
securely; see Harbor's [network-policy reference](https://docs.harborframework.com/tasks/network-policies#phases)
and Singularity's [networking guide](https://docs.sylabs.io/guides/3.6/user-guide/networking.html).

Set `[environment].workdir` when a Dockerfile has a variable or inherited
non-root `WORKDIR`, or when Compose uses an image for its `main` service; the
adapter rejects unresolved workdirs before launch.

## Configure other agents

For other Harbor agents, `ursa.integrations.harbor.config` exports `AgentConfig`, `ModelConfig`,
`load_agent_config`, `load_provider_config`, and `resolve_run`. It resolves one invocation;
callers control scheduling and output locations.

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

from ursa.integrations.harbor.config import (
    ModelConfig, load_agent_config, load_provider_config, resolve_run,
)

resolved = resolve_run(
    Path("benchmark/tasks/install-pytorch"),
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

## Artifact retention

```yaml
# retention.yaml
retention:
  version: 1
  collections: [trial]
  default_action: archive
  archive:
    format: tar.xz
    path: retained-artifacts.tar.xz
    remove_originals: true
    compression_level: 6
```

```bash
uv run harbor run --path benchmark/tasks/install-pytorch --agent oracle \
  --plugin ursa.integrations.harbor.retention:RetentionPlugin \
  --plugin-kwarg policy=retention.yaml
```

The plugin runs after job finalization and all trial/verifier consumers finish,
including failed trials. Cancelled, unfinalized jobs are untouched. It selects
all files in each trial directory (the parent of `artifacts`) when
`collections: [trial]` is selected. This includes agent logs and checkpoints,
configuration, lock files, results, artifact manifests, verifier logs and rewards,
and all multi-step outputs. Job-level files and retention metadata stay outside
the archive. Select `trial` on its own.

The default scope remains `collections: [artifacts]`; `[artifacts, verifier]`
also includes verifier payloads. These narrower scopes protect collection
manifests, rewards, and verifier test stdout/stderr, and leave other trial files
untouched.

Optional rules use gitignore-style globs relative to each selected root: the
trial directory for `[trial]`, or each payload directory for the narrower scopes.
For example, `agent/**` matches agent logs in trial scope. The first
match wins; unmatched files default to `keep`. `**/*.dat` includes root and nested
files; negative/comment patterns are rejected. Archives preserve full
job-relative trial/step paths to avoid collisions.

Archives are written temporarily, verified against the inventory and content
hashes, then published without overwriting existing files. Only then are selected
originals deleted. Allow space for both sources and the archive; compression
levels range from 0–9. A lock serializes retention writers, but other payload
writers must have stopped. POSIX directory fsync protects publication and journals.
With whole-trial retention and `remove_originals: true`, restore the trial before
using Harbor commands that need its result, configuration, or logs.

`retention.json` records the plan, file identities, sizes, hashes, deletions,
link warnings, duration, and status. Reapplying a completed policy verifies it;
interrupted compression restarts and interrupted cleanup resumes. Changed inputs,
including new trials, are rejected; whole-trial cleanup can resume even after
trial results have been removed. Different policies
and unrelated archives are rejected. Each job supports one policy/archive.
Harbor logs plugin errors without failing the job or changing
scores: require `status: complete` in `retention-status.json` to confirm retention.

### Preview, resume, and restore

```bash
uv run -m ursa.integrations.harbor artifacts plan \
  --run-dir jobs/example --policy retention.yaml
uv run -m ursa.integrations.harbor artifacts apply \
  --run-dir jobs/example --policy retention.yaml
uv run -m ursa.integrations.harbor artifacts restore \
  --manifest jobs/example/retention.json --destination restored-example
```

Preview inventories files and hashes archive-selected content without writing.
Apply requires finished job/trial results, resumes interrupted cleanup, and updates
`retention-status.json`. Historical jobs are only processed when explicitly selected.

Restore verifies the archive and extracts into a fresh directory, rejecting path
traversal and escaping symlinks. Links are archived without following targets;
warnings identify missing targets. Deleted files are unrecoverable, and kept
loose files must be retrieved from the original job.

For other adapters, `ursa.artifacts.retention` exposes
`plan_retention(root, policy, roots=..., protected=...)`, `apply_retention(plan)`,
`retain_artifacts(...)`, and `restore_artifacts(manifest, destination)`.

Remove `jobs/` when results are no longer needed. Remove SIF cache entries
only when no Harbor jobs use them.
