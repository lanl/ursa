#!/bin/sh
set -eu
root=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
case ${1:-} in
-h|--help)
    echo "Usage: scripts/run_installer.sh [INSTALL_SCRIPT] [INSTALL_ARGUMENT ...]"
    exit 0
    ;;
esac
installer=$root/install.sh
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then installer=$1; shift; fi
case $installer in /*) ;; *) installer=$PWD/$installer ;; esac
[ -f "$installer" ] || { printf 'Installer not found: %s\n' "$installer" >&2; exit 1; }

config=
previous=
for argument do
    [ "$previous" != --config ] || config=$argument
    previous=$argument
done
if [ -n "$config" ]; then
    case $config in
    /*) source=$config; target=$config ;;
    *) source=$PWD/$config; target=/home/ursa/$config ;;
    esac
    [ -f "$source" ] || { printf 'Config not found: %s\n' "$config" >&2; exit 1; }
fi

docker build -t ursa-installer-test:ubuntu-24.04 - <<'DOCKERFILE'
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl git && rm -rf /var/lib/apt/lists/* && useradd --create-home --shell /bin/bash ursa
USER ursa
WORKDIR /home/ursa
ENV HOME=/home/ursa LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/ursa/.local/bin:/home/ursa/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin TERM=xterm-256color
ENTRYPOINT ["/bin/sh", "/opt/ursa/install.sh"]
DOCKERFILE

set -- ursa-installer-test:ubuntu-24.04 "$@"
[ -z "$config" ] || set -- --mount "type=bind,src=$source,dst=$target,readonly" "$@"
docker run --rm -it --mount "type=bind,src=$installer,dst=/opt/ursa/install.sh,readonly" "$@"
