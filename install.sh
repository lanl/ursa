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

  --config FILE       Read an installer JSON payload from FILE.
  --config-json JSON  Read the payload directly from the argument.
  --config-stdin      Read the payload from stdin. Download this script first;
                      stdin is already occupied when using curl ... | sh.
EOF
}

config_source=""
config_value=""
while [ "$#" -gt 0 ]; do
    case $1 in
        --config)
            [ "$#" -ge 2 ] || fail "--config requires a file path"
            config_source=file
            config_value=$2
            shift 2
            ;;
        --config-json)
            [ "$#" -ge 2 ] || fail "--config-json requires a JSON payload"
            config_source=argument
            config_value=$2
            shift 2
            ;;
        --config-stdin)
            config_source=stdin
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

installer_work=$(mktemp -d "${TMPDIR:-/tmp}/ursa-installer.XXXXXX") \
    || fail "could not create a temporary directory"
cleanup() {
    rm -f \
        "$installer_work/payload.json" \
        "$installer_work/ursa-paths" \
        "$installer_work/uv-install.sh"
    rmdir "$installer_work" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

payload_file=""
if [ -n "$config_source" ]; then
    payload_file=$installer_work/payload.json
    case $config_source in
        file)
            [ -f "$config_value" ] \
                || fail "installer configuration not found: $config_value"
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

find_on_path() {
    command_name=$1
    remaining_path=${PATH:-}:
    seen_paths=''
    while [ -n "$remaining_path" ]; do
        path_dir=${remaining_path%%:*}
        remaining_path=${remaining_path#*:}
        [ -n "$path_dir" ] || path_dir=.
        canonical_dir=$(CDPATH= cd -P "$path_dir" 2>/dev/null && pwd) \
            || continue
        candidate=$canonical_dir/$command_name
        [ -x "$candidate" ] || continue
        case "
$seen_paths
" in
            *"
$candidate
"*) continue ;;
        esac
        printf '%s\n' "$candidate"
        seen_paths="${seen_paths}${candidate}
"
    done
}

uv_executable=$(find_uv || true)
if [ -z "$uv_executable" ]; then
    say "Installing uv…"
    download "https://astral.sh/uv/install.sh" \
        "$installer_work/uv-install.sh"
    sh "$installer_work/uv-install.sh"
    uv_executable=$(find_uv || true)
    [ -n "$uv_executable" ] \
        || fail "uv installed, but its executable could not be found"
else
    say "Using uv at $uv_executable"
fi

say "Installing URSA with Python 3.13…"
set -- tool install --python 3.13 \
    --force --upgrade --compile-bytecode ursa-ai
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

say "Applying selected extras and additional packages…"
if [ -n "$payload_file" ]; then
    set -- self modify --installer-config "$payload_file"
else
    set -- self modify --clean --extra dashboard
fi
PATH="$uv_dir${PATH:+:$PATH}" "$ursa_executable" "$@"
[ -x "$ursa_executable" ] \
    || fail "URSA modification completed, but $ursa_executable is missing"

path_list=$installer_work/ursa-paths
find_on_path ursa >"$path_list"
installed_on_path=false
while IFS= read -r path_ursa; do
    [ -n "$path_ursa" ] || continue
    if [ "$path_ursa" = "$ursa_executable" ]; then
        installed_on_path=true
    fi
done <"$path_list"

if [ "$installed_on_path" != true ]; then
    say "Adding uv's tool directory to PATH…"
    "$uv_executable" tool update-shell \
        || fail "URSA was installed at $ursa_executable, but $tool_bin could not be added to PATH"
    PATH="$tool_bin${PATH:+:$PATH}"
    export PATH
    find_on_path ursa >"$path_list"
fi

installed_on_path=false
other_ursa=false
while IFS= read -r path_ursa; do
    [ -n "$path_ursa" ] || continue
    if [ "$path_ursa" = "$ursa_executable" ]; then
        installed_on_path=true
    else
        other_ursa=true
    fi
done <"$path_list"
[ "$installed_on_path" = true ] \
    || fail "URSA was installed, but its executable is still unavailable on PATH"

if [ "$(command -v ursa || true)" != "$ursa_executable" ]; then
    other_ursa=true
fi

if [ "$other_ursa" = true ]; then
    say ""
    say "Warning: other URSA executables are also on PATH:"
    while IFS= read -r path_ursa; do
        [ -n "$path_ursa" ] || continue
        version=$("$path_ursa" --version 2>/dev/null | sed -n '1p' || true)
        [ -n "$version" ] || version="version unknown"
        if [ "$path_ursa" = "$ursa_executable" ]; then
            label="installed now"
        else
            label="other installation"
        fi
        say "  - $path_ursa ($label; $version)"
    done <"$path_list"
    resolved_ursa=$(command -v ursa || true)
    say "The current shell resolves 'ursa' to: $resolved_ursa"
    say ""
fi

set -- self setup
if [ -n "$payload_file" ]; then
    set -- "$@" --defaults-json "$payload_file"
fi

say "Starting guided setup…"
if [ -t 0 ]; then
    "$ursa_executable" "$@"
elif ( : </dev/tty ) 2>/dev/null; then
    "$ursa_executable" "$@" </dev/tty
else
    say "No interactive terminal was found. Finish setup by running:"
    say "  ursa self setup"
fi

say "URSA is installed. Run 'ursa' or 'ursa-dashboard' to get started."
