# Sandboxing

URSA now sandboxes commands executed through `run_command` by default on Linux
and macOS. Commands can write inside the configured workspace and temporary
storage; other filesystem locations are read-only unless explicitly allowed.
File reads and network access remain enabled by default. These restrictions
apply to `run_command`, not every URSA tool.

## Start with sandboxing enabled

- **Linux:** Install [Bubblewrap](https://github.com/containers/bubblewrap) and
  ensure `bwrap` is on your `PATH`. If you use [`pixi`][1], run
  `pixi global install bubblewrap`. Your system must allow user namespaces.
- **macOS:** Ensure `sandbox-exec` is available on your `PATH`.

To enable sandboxing on Linux and macOS, leave `URSA_SBX` unset or set it to `1`.

## Customize access

Set environment variables before launching URSA. The following examples use a
POSIX shell on Linux or macOS.

Block network access for sandboxed commands:

```sh
URSA_ACCESS_NETWORK=0 ursa
```

Allow writes to additional existing directories, such as a package cache or
shared output folder:

```sh
URSA_EXTRA_WRITABLE_PATHS="$HOME/.cache/uv:$HOME/ursa-output" ursa
```

Separate paths with `:` on Linux and macOS. Paths are resolved and `~` is
expanded; nonexistent paths are ignored. Create directories before launching.

## Safety checks and troubleshooting

Model-based command safety checks remain enabled independently of the sandbox.
To skip those checks while keeping sandboxing enabled:

```sh
URSA_SAFETY_LEVEL=none URSA_SBX=1 ursa
```

`URSA_SAFETY_LEVEL=yolo` disables both safety checks and sandboxing.

If the backend is missing or cannot start, command execution raises an error
instead of silently running without a sandbox. Check `PATH` and, on Linux,
whether user namespaces are allowed; some HPC systems disable them. Windows
sandboxing is not implemented.

To explicitly run commands without sandboxing, including on unsupported systems:

```sh
URSA_SBX=0 ursa
```

This removes sandbox filesystem and network restrictions; model-based safety
checks still follow `URSA_SAFETY_LEVEL`.

## Container image

Containers can provide additional isolation. The following examples start
`ursa` in a container:

### Docker

```shell
# Pull the image
docker pull ghcr.io/lanl/ursa

# Launch ursa in container
docker run \
    -e "OPENAI_API_KEY"=$OPENAI_API_KEY \
    --mount type=bind,src=$PWD,dst=/mnt/workspace \
    ghcr.io/lanl/ursa \
    ursa --name my-session
```

### Charliecloud

[Charliecloud](https://charliecloud.io/)
is a rootless alternative to docker that is sometimes preferred on HPC. The
following commands replicate the behaviors above for docker.

```shell
# Pull the image
ch-image pull ghcr.io/lanl/ursa ursa

# Convert image to sqfs, for use on another system
ch-convert ursa ursa.sqfs

# Launch ursa in container
ch-run -W ursa \
    --unset-env="*" \
    --set-env \
    --set-env="OPENAI_API_KEY"=$OPENAI_API_KEY \
    --bind ${PWD}/scripts:/mnt/workspace \
    --cd /mnt/workspace \
    -- ursa --name my-session
```

### Apptainer

Apptainer is another rootless alternative to docker.

```shell
# Pull the image
apptainer pull /path/to/ursa.sif docker://ghcr.io/lanl/ursa

# Launch rusa in container
apptainer exec \
  --cleanenv \
  --containall \
  --no-mount cwd \
  --no-home \
  --home /root \
  --env OPENAI_API_KEY=$OPENAI_API_KEY \
  --env UV_NO_CACHE=1 \
  --env UV_SBX=0 \
  --bind $PWD:/mnt/workspace \
  --pwd /mnt/workspace \
  --writable-tmpfs \
  /path/to/ursa.sif \
  bash
```

[1]: https://pixi.prefix.dev
