#!/bin/sh
set -eu
say() { printf '%s\n' "$*"; }
fail() { printf 'URSA installer: %s\n' "$*" >&2; exit 1; }

config=
python=3.13
source=
requirement=ursa-ai
while [ "$#" -gt 0 ]; do
    option=$1
    case $option in
    --config) config=$2 ;;
    --python) python=$2 ;;
    --version|--branch)
        [ -z "$source" ] || [ "$source" = "$option" ] || fail "--branch and --version cannot be used together"
        source=$option
        [ "$option" = --version ] && requirement="ursa-ai==$2" || requirement="ursa-ai @ git+https://github.com/lanl/ursa.git@$2"
        ;;
    -h|--help)
        say "Usage: install.sh [--config FILE] [--branch BRANCH | --version VERSION] [--python VERSION]"
        exit 0 ;;
    *) fail "unknown option: $option" ;;
    esac
    shift 2
done

[ -z "$config" ] || [ -f "$config" ] || fail "setup defaults not found: $config"
input=
if [ -t 0 ]; then input=stdin
elif ( : </dev/tty ) 2>/dev/null; then input=tty
fi
[ -z "$config" ] || [ -n "$input" ] || fail "setup defaults require an interactive terminal"

uv=$(command -v uv || true)
if [ -z "$uv" ]; then
    say "Installing uv…"
    command -v curl >/dev/null || fail "curl is required to download uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    directory=${UV_UNMANAGED_INSTALL:-${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}}
    uv=$directory/uv
    [ -x "$uv" ] || fail "uv installed, but its executable could not be found"
else
    say "Using uv at $uv"
fi

say "Installing URSA with Python $python…"
"$uv" tool install --python "$python" --force --upgrade --compile-bytecode "$requirement"
tool_bin=$("$uv" tool dir --bin)
ursa="$tool_bin/ursa"
"$uv" tool update-shell || say "Could not update this shell automatically; add $tool_bin to PATH."
PATH="$tool_bin:$(dirname "$uv")${PATH:+:$PATH}"; export PATH

if ! "$ursa" self setup --help >/dev/null 2>&1; then
    fail "installed URSA ($("$ursa" --version)) does not support guided setup. See https://lanl.github.io/ursa/latest/getting-started."
fi

say "Starting guided setup…"
set -- self setup
[ -z "$config" ] || set -- "$@" --defaults-json "$config"
case $input in
stdin) "$ursa" "$@" ;;
tty) "$ursa" "$@" </dev/tty ;;
*) say "No interactive terminal was found. Finish setup by running:"; say "  ursa self setup" ;;
esac
say "URSA is installed. Run 'ursa' to get started."
