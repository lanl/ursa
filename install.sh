#!/bin/sh

set -eu

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'URSA installer: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Install URSA with uv and start the guided setup.

Usage: install.sh [--config FILE | --config-json JSON | --config-stdin]

  --config FILE       Read a setup defaults JSON payload from FILE.
  --config-json JSON  Read the payload directly from the argument.
  --config-stdin      Read the payload from stdin. Download this script first;
                      stdin is already occupied when using curl ... | sh.
EOF
}

config_source=
config_value=
while [ "$#" -gt 0 ]; do
    case $1 in
    --config)
        [ "$#" -ge 2 ] || fail "--config requires a file path"
        [ -z "$config_source" ] || fail "setup defaults may only be specified once"
        config_source=file
        config_value=$2
        shift 2
        ;;
    --config-json)
        [ "$#" -ge 2 ] || fail "--config-json requires a JSON payload"
        [ -z "$config_source" ] || fail "setup defaults may only be specified once"
        config_source=argument
        config_value=$2
        shift 2
        ;;
    --config-stdin)
        [ -z "$config_source" ] || fail "setup defaults may only be specified once"
        config_source=stdin
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *) fail "unknown option: $1" ;;
    esac
done

installer_work=$(mktemp -d "${TMPDIR:-/tmp}/ursa-installer.XXXXXX") \
    || fail "could not create a temporary directory"
cleanup() {
    rm -rf "$installer_work"
}
trap cleanup 0
trap 'exit 1' HUP INT TERM

payload_file=
if [ -n "$config_source" ]; then
    payload_file=$installer_work/defaults.json
    case $config_source in
    file)
        [ -f "$config_value" ] \
            || fail "setup defaults not found: $config_value"
        cp "$config_value" "$payload_file"
        ;;
    argument)
        printf '%s\n' "$config_value" >"$payload_file"
        ;;
    stdin)
        cat >"$payload_file"
        ;;
    esac
fi

setup_input=none
if [ -t 0 ]; then
    setup_input=stdin
elif ( : </dev/tty ) 2>/dev/null; then
    setup_input=tty
fi
if [ -n "$payload_file" ] && [ "$setup_input" = none ]; then
    fail "setup defaults require an interactive terminal"
fi

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
    ursa-ai
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

if ! "$ursa_executable" self setup --help >/dev/null 2>&1; then
    ursa_version=$("$ursa_executable" --version 2>/dev/null || true)
    [ -n "$ursa_version" ] || ursa_version="version unavailable"
    fail "installed URSA ($ursa_version) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
fi

say "Starting guided setup…"
set -- self setup
if [ -n "$payload_file" ]; then
    set -- "$@" --defaults-json "$payload_file"
fi
if [ "$setup_input" = stdin ]; then
    "$ursa_executable" "$@"
elif [ "$setup_input" = tty ]; then
    "$ursa_executable" "$@" </dev/tty
else
    say "No interactive terminal was found. Finish setup by running:"
    say "  $ursa_command self setup"
fi

say "URSA is installed. Run '$ursa_command' to get started."
