#!/bin/sh

set -eu

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'URSA installer: %s\n' "$*" >&2
    exit 1
}

download() {
    download_url=$1
    download_target=$2
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$download_url" -o "$download_target"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$download_url" -O "$download_target"
    else
        fail "curl or wget is required to download uv"
    fi
}

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    for uv_candidate in \
        "${UV_INSTALL_DIR:-}/uv" \
        "${XDG_BIN_HOME:-}/uv" \
        "${HOME:-}/.local/bin/uv" \
        "${HOME:-}/.cargo/bin/uv"
    do
        if [ -n "$uv_candidate" ] && [ -x "$uv_candidate" ]; then
            printf '%s\n' "$uv_candidate"
            return
        fi
    done
    return 1
}

uv_executable=$(find_uv || true)
if [ -z "$uv_executable" ]; then
    say "Installing uv…"
    installer_tmp=$(mktemp "${TMPDIR:-/tmp}/ursa-uv-installer.XXXXXX") \
        || fail "could not create a temporary file"
    trap 'rm -f "$installer_tmp"' EXIT HUP INT TERM
    download "https://astral.sh/uv/install.sh" "$installer_tmp"
    sh "$installer_tmp"
    rm -f "$installer_tmp"
    trap - EXIT HUP INT TERM
    uv_executable=$(find_uv || true)
    [ -n "$uv_executable" ] \
        || fail "uv installed, but its executable could not be found"
else
    say "Using uv at $uv_executable"
fi

say "Installing URSA and the dashboard…"
"$uv_executable" tool install \
    --force --upgrade --compile-bytecode 'ursa-ai[dashboard]'

tool_bin=$("$uv_executable" tool dir --bin)
ursa_executable=$tool_bin/ursa
if [ ! -x "$ursa_executable" ]; then
    ursa_executable=$(command -v ursa || true)
fi
[ -n "$ursa_executable" ] && [ -x "$ursa_executable" ] \
    || fail "URSA installed, but its executable could not be found"

say "Starting guided setup…"
if [ -t 0 ]; then
    "$ursa_executable" self setup
elif ( : </dev/tty ) 2>/dev/null; then
    "$ursa_executable" self setup </dev/tty
else
    say "No interactive terminal was found. Finish setup by running:"
    say "  $ursa_executable self setup"
fi

say "URSA is installed. Run 'ursa' or 'ursa-dashboard' to get started."
