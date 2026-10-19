#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
URSA_COMMIT=${URSA_COMMIT:-$(git -C "$script_dir/.." rev-parse HEAD)}
URSA_INSTALLER_SOURCE=${URSA_INSTALLER_SOURCE:-$script_dir/../install.sh}

installer=$(mktemp "${TMPDIR:-/tmp}/ursa-install.XXXXXX")
trap 'rm -f "$installer"' EXIT HUP INT TERM

sed "s|git+https://github.com/lanl/ursa.git@main|git+https://github.com/lanl/ursa.git@$URSA_COMMIT|g" \
    "$URSA_INSTALLER_SOURCE" >"$installer"
sh "$installer" "$@"
