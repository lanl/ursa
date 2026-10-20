#!/bin/sh
set -eu
say() { printf '%s\n' "$*"; }
fail() { printf 'URSA installer: %s\n' "$*" >&2; exit 1; }

config_file=
selected_python=3.13
ursa_requirement=ursa-ai
source_option=
while [ "$#" -gt 0 ]; do
    case $1 in
    --config|--version|--branch|--python)
        option=$1
        shift
        [ "$#" -gt 0 ] && [ -n "$1" ] || fail "$option requires a value"
        case $option in
        --config) config_file=$1 ;;
        --version|--branch)
            [ -z "$source_option" ] || [ "$source_option" = "$option" ] \
                || fail "--branch and --version cannot be used together"
            source_option=$option
            [ "$option" = --version ] && ursa_requirement="ursa-ai==$1" \
                || ursa_requirement="ursa-ai @ git+https://github.com/lanl/ursa.git@$1"
            ;;
        --python) selected_python=$1 ;;
        esac
        shift
        ;;
    -h|--help)
        cat <<'EOF'
Install URSA with uv and start the guided setup.

Usage: install.sh [--config FILE] [--branch BRANCH | --version VERSION] [--python VERSION]

  --config FILE              Read setup defaults from a JSON file.
  --branch BRANCH            Install from a branch in the LANL URSA repository.
  --version VERSION          Install a PyPI version.
  --python VERSION           Install with this Python version (default: 3.13).
EOF
        exit 0
        ;;
    *) fail "unknown option: $1" ;;
    esac
done
[ -z "$config_file" ] || [ -f "$config_file" ] \
    || fail "setup defaults not found: $config_file"

setup_input=
if [ -t 0 ]; then
    setup_input=stdin
elif ( : </dev/tty ) 2>/dev/null; then
    setup_input=tty
fi
[ -z "$config_file" ] || [ -n "$setup_input" ] \
    || fail "setup defaults require an interactive terminal"

find_uv() {
    command -v uv 2>/dev/null && return
    for candidate in \
        "${UV_UNMANAGED_INSTALL:+$UV_UNMANAGED_INSTALL/uv}" \
        "${UV_INSTALL_DIR:+$UV_INSTALL_DIR/uv}" \
        "${XDG_BIN_HOME:+$XDG_BIN_HOME/uv}" \
        "${XDG_DATA_HOME:+$XDG_DATA_HOME/../bin/uv}" \
        "${HOME:+$HOME/.local/bin/uv}" \
        "${HOME:+$HOME/.cargo/bin/uv}"
    do
        [ -n "$candidate" ] && [ -x "$candidate" ] || continue
        printf '%s\n' "$candidate"
        return 0
    done
    return 1
}

uv_executable=$(find_uv || true)
if [ -z "$uv_executable" ]; then
    say "Installing uv…"
    if command -v curl >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        fail "curl or wget is required to download uv"
    fi
    uv_executable=$(find_uv || true)
    [ -n "$uv_executable" ] \
        || fail "uv installed, but its executable could not be found"
else
    say "Using uv at $uv_executable"
fi

say "Installing URSA with Python $selected_python…"
"$uv_executable" tool install --python "$selected_python" --force --upgrade --compile-bytecode \
    "$ursa_requirement"
tool_bin=$("$uv_executable" tool dir --bin)
ursa_executable="$tool_bin/ursa"
[ -x "$ursa_executable" ] || fail \
    "URSA installed, but its executable was not found at $ursa_executable"
current_ursa=$(command -v ursa || true)
ursa_command=ursa
[ -z "$current_ursa" ] || [ "$current_ursa" = "$ursa_executable" ] || {
    say "Warning: $current_ursa currently takes precedence over $ursa_executable."
    ursa_command=$ursa_executable
}
if [ "$current_ursa" != "$ursa_executable" ]; then
    say "Adding uv's tool directory to PATH…"
    "$uv_executable" tool update-shell \
        || say "Could not update this shell automatically; add $tool_bin to PATH."
    say "Follow uv's instructions above to update this shell. Or restart your shell."
fi
PATH="$tool_bin:$(dirname "$uv_executable")${PATH:+:$PATH}"
export PATH

if ! "$ursa_executable" self setup --help >/dev/null 2>&1; then
    ursa_version=$("$ursa_executable" --version 2>/dev/null || true)
    [ -n "$ursa_version" ] || ursa_version="version unavailable"
    fail "installed URSA ($ursa_version) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
fi

say "Starting guided setup…"
set -- self setup
[ -z "$config_file" ] || set -- "$@" --defaults-json "$config_file"
if [ "$setup_input" = stdin ]; then
    "$ursa_executable" "$@"
elif [ "$setup_input" = tty ]; then
    "$ursa_executable" "$@" </dev/tty
else
    say "No interactive terminal was found. Finish setup by running:"
    say "  $ursa_command self setup"
fi

say "URSA is installed. Run '$ursa_command' to get started."
