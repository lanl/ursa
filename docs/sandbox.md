# Sandboxing
The [Execution Agent][executionagent-documentation] is allowed to run system
commands and write/run code. Being able to execute arbitrary system commands or write
and execute code has the potential to cause problems like:

- Damage code or data on the computer
- Damage the computer
- Transmit your local data

The Web Search Agent scrapes data from urls, so has the potential to attempt to pull information from questionable sources.

Some suggestions for sandboxing the agent:

- Creating a specific environment such that limits URSA's access to only what you want. Examples:

    - Creating/using a virtual machine that is sandboxed from the rest of your machine
    - Creating a new account on your machine specifically for URSA

- Creating a network blacklist/whitelist to ensure that network commands and webscraping are contained to safe sources

You have a duty for ensuring that you use URSA responsibly.

## Container image

To enable limited sandboxing insofar as containerization does this, you can run
the following commands:

### Docker

```shell
# Pull the image
docker pull ghcr.io/lanl/ursa

# Run script from host system
mkdir -p scripts
echo "import ursa; print('Hello from ursa')" > scripts/my_script.py
docker run -e "OPENAI_API_KEY"=$OPENAI_API_KEY \
    --mount type=bind,src=$PWD/scripts,dst=/mnt/workspace \
    ghcr.io/lanl/ursa \
    bash -c "uv run /mnt/workspace/my_script.py"
```

### Charliecloud

[Charliecloud](https://charliecloud.io/)
is a rootless alternative to docker
that is sometimes preferred on HPC. The following commands replicate the
behaviors above for docker.

```shell
# Pull the image
ch-image pull ghcr.io/lanl/ursa ursa

# Convert image to sqfs, for use on another system
ch-convert ursa ursa.sqfs

# Run script from host system (if wanted, replace ursa with /path/to/ursa.sqfs)
mkdir -p scripts
echo "import ursa; print('Hello from ursa')" > scripts/my_script.py
ch-run -W ursa \
    --unset-env="*" \
    --set-env \
    --set-env="OPENAI_API_KEY"=$OPENAI_API_KEY \
    --bind ${PWD}/scripts:/mnt/workspace \
    --cd /mnt/workspace \
    -- bash -c \
    "uv run --no-sync my_script.py"
```
