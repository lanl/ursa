#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
repository_root=$(CDPATH= cd -P "$script_dir/.." && pwd)
repository_commit=${URSA_COMMIT:-$(git -C "$repository_root" rev-parse HEAD)}

case ${1:-} in
    -h|--help)
        cat <<'EOF'
Run an URSA installer in a clean Ubuntu container.

Usage: scripts/run_installer.sh [INSTALL_SCRIPT] [INSTALL_ARGUMENT ...]

INSTALL_SCRIPT defaults to the install.sh in this repository. Remaining
arguments are passed to the installer inside the container.
EOF
        exit 0
        ;;
esac

installer=${1:-$repository_root/install.sh}
if [ "$#" -gt 0 ]; then
    shift
fi

case $installer in
    /*) ;;
    *) installer=$(CDPATH= cd -P "$(dirname "$installer")" 2>/dev/null && pwd)/$(basename "$installer") \
        || {
            printf 'Invalid installer path: %s\n' "$installer" >&2
            exit 1
        } ;;
esac

if [ ! -f "$installer" ]; then
    printf 'Installer not found: %s\n' "$installer" >&2
    exit 1
fi
if [ ! -r "$installer" ]; then
    printf 'Installer is not readable: %s\n' "$installer" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is required to run the installer harness.\n' >&2
    exit 1
fi

image=ursa-installer-test:ubuntu-24.04

docker build --tag "$image" - <<'DOCKERFILE'
FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash ursa

USER ursa
WORKDIR /home/ursa

ENV HOME=/home/ursa \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/home/ursa/.local/bin:/home/ursa/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TERM=xterm-256color

ENTRYPOINT ["/bin/sh", "/opt/ursa/install.sh"]
DOCKERFILE

docker run --rm --interactive --tty \
    --env "URSA_COMMIT=$repository_commit" \
    --env URSA_INSTALLER_SOURCE=/opt/ursa/source-install.sh \
    --env URSA_LOCAL_INSTALLER=/opt/ursa/install_local.sh \
    --mount "type=bind,src=$installer,dst=/opt/ursa/install.sh,readonly" \
    --mount "type=bind,src=$repository_root/install.sh,dst=/opt/ursa/source-install.sh,readonly" \
    --mount "type=bind,src=$repository_root/scripts/install_local.sh,dst=/opt/ursa/install_local.sh,readonly" \
    "$image" "$@"
