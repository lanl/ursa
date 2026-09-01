# Advanced Harbor usage

## Customize the URSA installation

The adapter installs Python and URSA with `uv` inside each task container.
Add URSA extras or other packages with agent kwargs:

```bash
--agent-kwarg ursa_extras=image \
--agent-kwarg extra_packages=numpy,scipy
```

Use `ursa_source_dir=/path/to/checkout` while developing the integration, or
set `ursa_install_spec` to another package/version specification.

## Singularity and SLURM

The custom environment builds `environment/Dockerfile` with Buildah, Podman,
or Docker, converts it to a cached SIF, and requires no Apptainer definition
file. Compute nodes need `apptainer` or `singularity`, plus one of `buildah`,
`podman`, or `docker`.

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

Set `[environment].workdir` in `task.toml` when a Dockerfile computes
`WORKDIR` from an environment variable or inherits a non-root workdir from its
base image. The task setting takes precedence over image metadata; the
Singularity adapter rejects variable workdirs it cannot resolve before launch.

## Clean up

Remove `jobs/` when local results are no longer needed. Remove the directory
named by `$URSA_HARBOR_SIF_CACHE` only when no array jobs use it.
`submit_slurm.sh` prints the temporary task-manifest path, which can be removed
after the array finishes.
