# Sandboxing

See the [Command sandbox quickstart](sandbox-quickstart.md) for setup,
access controls, safety settings, and troubleshooting.

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
