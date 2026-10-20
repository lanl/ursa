#!/bin/sh
set -eu
repository_root=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)

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
installer=$repository_root/install.sh
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    installer=$1
    shift
fi
case $installer in /*) ;; *) installer=$PWD/$installer ;; esac
[ -f "$installer" ] || { printf 'Installer not found: %s\n' "$installer" >&2; exit 1; }
config=
previous=
for argument do
    [ "$previous" != --config ] || config=$argument
    previous=$argument
done
if [ -n "$config" ]; then
    case $config in /*) config_source=$config; config_target=$config ;;
        *) config_source=$PWD/$config; config_target=/home/ursa/$config ;;
    esac
    [ -f "$config_source" ] || { printf 'Config not found: %s\n' "$config" >&2; exit 1; }
fi
image=ursa-installer-test:ubuntu-24.04

docker build --tag "$image" - <<'DOCKERFILE'
FROM ubuntu:24.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates curl git \
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

set -- "$image" "$@"
[ -z "$config" ] || set -- --mount \
    "type=bind,src=$config_source,dst=$config_target,readonly" "$@"
docker run --rm --interactive --tty \
    --mount "type=bind,src=$installer,dst=/opt/ursa/install.sh,readonly" \
    "$@"
