# Advanced Harbor usage

## Customize the URSA installation

The adapter installs Python and URSA with `uv` inside each task container.
Add URSA extras or other packages with agent kwargs:

```bash
--agent-kwarg ursa_extras=image \
--agent-kwarg 'extra_packages=["numpy","scipy"]'
```

Set `ursa_install_spec` to a package requirement, Git URL, local project
directory, or local wheel/sdist archive. Local inputs are staged into the task
container; project directories exclude Git-ignored files and common secret
files:

```bash
--agent-kwarg ursa_install_spec=/path/to/checkout
```

## Choose the URSA config stack

By default the adapter merges only the file passed as `config_file`, then
applies Harbor's model and MCP settings last. Set
`--agent-kwarg config_only=false` to include URSA's system and user config
layers first. The supplied file is still merged below Harbor's settings.

Secret references are resolved on the host, including keyring references. The
adapter passes generated environment references only to the URSA runner; it
does not copy host config or keyring files into the task container.

Each trial captures the URSA runner's stdout, stderr, and progress logging in
`agent/ursa.log`. Metrics remain in `agent/ursa-metrics.json`.

## Singularity and SLURM

The custom environment builds `environment/Dockerfile` with Buildah, Podman,
or Docker, converts it to a cached SIF, and requires no Apptainer definition
file. Compute nodes need `apptainer` or `singularity`, plus one of `buildah`,
`podman`, or `docker`. When using Docker, its daemon must be running and the
invoking process must have socket access.

SIFs are cached by build context under `$XDG_CACHE_HOME/ursa/harbor/sif`, or
`~/.cache/ursa/harbor/sif` when `XDG_CACHE_HOME` is unset. Concurrent trials
with the same build context share a lock and build only once. Set
`URSA_HARBOR_SIF_CACHE` or pass `singularity_image_cache_dir` as an environment
kwarg to use a different shared cache.

```bash
export OPENAI_API_KEY=...
export URSA_HARBOR_SIF_CACHE=/shared/cache/harbor-sif
export URSA_HARBOR_JOBS_DIR=/shared/results/ursa-harbor
bash submit_slurm.sh
```

For a direct run, add:

```bash
--env ursa.integrations.harbor_singularity:DockerfileSingularityEnvironment
```

### Multi-container tasks

Add Harbor's standard `environment/docker-compose.yaml` to start sidecars with
[`singularity-compose`](https://singularityhub.github.io/singularity-compose/).
The reserved `main` service defaults to `environment/Dockerfile`, but it may
use an OCI `image` or a `build` with a different context, Dockerfile, build
arguments, or target. This also works in verifier overlays such as
`tests/docker-compose.yaml`, where paths are relative to the overlay file.
Sidecars support the same `image` and `build` forms:

```yaml
services:
  main:
    build:
      context: ..
      dockerfile: tests/Dockerfile
    depends_on:
      - api

  api:
    build:
      context: ./api
    command: [python, server.py]
    expose:
      - "8000"
```

The adapter translates the subset that `singularity-compose` can represent:
images, Dockerfile builds (`context`, `dockerfile`, `args`, and `target`), list
form `depends_on`, commands, bind mounts, TCP `host:container` ports,
`deploy.replicas`, `environment`, and `env_file`. Sidecars are reachable by
their Harbor service name. Harbor's per-service artifact collection is also
supported.

Unsupported Compose fields fail before launch. In particular,
`singularity-compose` cannot represent healthchecks or conditional
`depends_on`, named volumes, custom Compose networks, entrypoints, resource
limits, or privilege/capability settings. See Harbor's
[multi-container task guide](https://docs.harborframework.com/core-concepts/tasks/multi-container)
and the
[`singularity-compose` 2.0 specification](https://github.com/singularityhub/singularity-compose/blob/master/docs/spec/spec-2.0.md)
before porting a Docker Compose task.

Multi-container tasks work with either SingularityCE or Apptainer. Although
`singularity-compose` 0.1.19 invokes a command named `singularity` internally,
the adapter maps that command to the selected runtime inside its private
project directory. An Apptainer installation therefore does not need a
system-wide `singularity` alias. The `ursa-ai[harbor]` extra installs
`singularity-compose`; the host runtime must provide fakeroot CNI networking.

The adapter supports Harbor's static `public` and `no-network` modes on
SingularityCE 3.6.2. Set the baseline policy in `task.toml`:

```toml
[environment]
network_mode = "no-network"
```

The effective Harbor policy is also projected into URSA's `use_web` setting
for each agent phase: `public` and `allowlist` enable web tools, while
`no-network` disables them. Harbor still enforces the actual network boundary.

`no-network` starts the task as a named Singularity instance with an isolated
`none` network. It is not available to multi-container tasks because
`singularity-compose` cannot preserve communication between sidecars while
isolating the project from external networks. Network allowlists and `[agent]`
or `[verifier]` policies that differ from the environment baseline are rejected
because SingularityCE 3.6.2 cannot enforce them securely. See Harbor's
[network-policy reference](https://docs.harborframework.com/core-concepts/tasks/network-policies#network-modes)
and Singularity's
[networking guide](https://docs.sylabs.io/guides/3.6/user-guide/networking.html).

Set `[environment].workdir` in `task.toml` when a Dockerfile computes
`WORKDIR` from an environment variable or inherits a non-root workdir from its
base image. The task setting takes precedence over image metadata; the
Singularity adapter rejects variable workdirs it cannot resolve before launch.

## Clean up

Remove `jobs/` when local results are no longer needed. Remove the SIF cache
only when no Harbor jobs use it.
`submit_slurm.sh` prints the temporary task-manifest path, which can be removed
after the array finishes.
