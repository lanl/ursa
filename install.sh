#!/bin/sh

set -eu

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'URSA installer: %s\n' "$*" >&2
    exit 1
}

if [ "$#" -gt 0 ]; then
    case $1 in
    -h|--help)
        cat <<'EOF'
Install URSA with uv and start the guided setup.

Usage: install.sh
EOF
        exit 0
        ;;
    *) fail "unknown option: $1" ;;
    esac
fi

installer_work=$(mktemp -d "${TMPDIR:-/tmp}/ursa-installer.XXXXXX") \
    || fail "could not create a temporary directory"
cleanup() {
    rm -rf "$installer_work"
}
trap cleanup 0
trap 'exit 1' HUP INT TERM

# The documentation build pins this requirement for a release.
URSA_DASHBOARD_REQUIREMENT='ursa-ai[dashboard] @ git+https://github.com/lanl/ursa.git@main'
case ${URSA_COMMIT:-} in
    '') ;;
    *[!0123456789abcdefABCDEF]*) fail "URSA_COMMIT must be a Git commit hash" ;;
    *)
        URSA_DASHBOARD_REQUIREMENT="ursa-ai[dashboard] @ git+https://github.com/lanl/ursa.git@$URSA_COMMIT"
        ;;
esac

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return
    fi
    for uv_candidate in \
        "${UV_UNMANAGED_INSTALL:+$UV_UNMANAGED_INSTALL/uv}" \
        "${UV_INSTALL_DIR:+$UV_INSTALL_DIR/uv}" \
        "${XDG_BIN_HOME:+$XDG_BIN_HOME/uv}" \
        "${XDG_DATA_HOME:+$XDG_DATA_HOME/../bin/uv}" \
        "${HOME:+$HOME/.local/bin/uv}" \
        "${HOME:+$HOME/.cargo/bin/uv}"
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
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh -o "$installer_work/uv-install.sh"
    elif command -v wget >/dev/null 2>&1; then
        wget -q https://astral.sh/uv/install.sh -O "$installer_work/uv-install.sh"
    else
        fail "curl or wget is required to download uv"
    fi
    sh "$installer_work/uv-install.sh"
    uv_executable=$(find_uv || true)
    [ -n "$uv_executable" ] \
        || fail "uv installed, but its executable could not be found"
else
    say "Using uv at $uv_executable"
fi

say "Installing URSA with Python 3.13…"
set -- tool install --python 3.13 --force --upgrade --compile-bytecode \
    "$URSA_DASHBOARD_REQUIREMENT"
"$uv_executable" "$@"

tool_bin=$("$uv_executable" tool dir --bin)
tool_bin=$(CDPATH= cd -P "$tool_bin" 2>/dev/null && pwd) \
    || fail "uv returned an invalid tool bin directory: $tool_bin"
ursa_executable=$tool_bin/ursa
[ -x "$ursa_executable" ] \
    || fail "URSA installed, but its executable was not found at $ursa_executable"
uv_dir=$(dirname "$uv_executable")
uv_dir=$(CDPATH= cd -P "$uv_dir" 2>/dev/null && pwd) \
    || fail "uv has an invalid executable path: $uv_executable"

current_ursa=$(command -v ursa || true)
ursa_command=ursa
if [ -n "$current_ursa" ] && [ "$current_ursa" != "$ursa_executable" ]; then
    say "Warning: $current_ursa currently takes precedence over $ursa_executable."
    ursa_command=$ursa_executable
fi
if [ "$current_ursa" != "$ursa_executable" ]; then
    say "Adding uv's tool directory to PATH…"
    "$uv_executable" tool update-shell \
        || say "Could not update this shell automatically; add $tool_bin to PATH."
    say "Follow uv's instructions above to update this shell. Or restart your shell."
fi
PATH="$tool_bin:$uv_dir${PATH:+:$PATH}"
export PATH

"$ursa_executable" self setup --help >/dev/null 2>&1 \
    || fail "installed URSA does not support guided setup. This installer may target an unreleased URSA version; use https://lanl.github.io/ursa/latest/install.sh or install the matching release."

say "Starting guided setup…"
if [ -t 0 ]; then
    "$ursa_executable" self setup
elif ( : </dev/tty ) 2>/dev/null; then
    "$ursa_executable" self setup </dev/tty
else
    say "No interactive terminal was found. Finish setup by running:"
    say "  $ursa_command self setup"
fi

say "URSA is installed. Run '$ursa_command' to get started."
