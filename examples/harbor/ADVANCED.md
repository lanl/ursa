# Advanced Harbor usage

## Install and configure URSA

The adapter installs Python and URSA with `uv` inside each task container. Add
extras or packages with agent kwargs; `ursa_install_spec` accepts a package
requirement, Git URL, local project, wheel, or sdist. Local projects are staged
without Git-ignored and common secret files.

```bash
--agent-kwarg ursa_extras=image \
--agent-kwarg 'extra_packages=["numpy","scipy"]' \
--agent-kwarg ursa_install_spec=/path/to/checkout
```

By default, only `config_file` is merged before Harbor's model and MCP layer.
Set `--agent-kwarg config_only=false` to include system and user config first.
Secret and keyring references resolve on the host; only generated environment
references reach the runner, never host config or keyring files. Trial output
includes `agent/ursa.log`, lifecycle records in `agent/ursa.jsonl` (including
chat token usage), and `agent/ursa_result.out`.

## Singularity

Use this environment for direct runs:

```bash
--env ursa.integrations.harbor_singularity:DockerfileSingularityEnvironment
```

It builds `environment/Dockerfile` with Buildah, Podman, or Docker and converts
it to a SIF; no Apptainer definition is needed. Compute nodes need `apptainer`
or `singularity` plus one builder (Docker also needs a running daemon and
socket access). SIFs cache by build context in `$XDG_CACHE_HOME/ursa/harbor/sif`
or `~/.cache/ursa/harbor/sif`; concurrent builds share a lock. Set
`URSA_HARBOR_SIF_CACHE`, pass the environment constructor's
`singularity_image_cache_dir`, or use `prebuild --image-cache-dir` to choose a
shared cache.

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
non-root `WORKDIR`; the adapter rejects unresolved workdirs before launch.

## Clean up

Remove `jobs/` when results are no longer needed, and remove SIF cache entries
only when no Harbor jobs use them.
