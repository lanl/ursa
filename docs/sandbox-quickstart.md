# Command sandbox quickstart

URSA now sandboxes commands executed through `run_command` by default on Linux
and macOS. Commands can write inside the configured workspace and temporary
storage; other filesystem locations are read-only unless explicitly allowed.
File reads and network access remain enabled by default. These restrictions
apply to `run_command`, not every URSA tool.

## Start with sandboxing enabled

- **Linux:** Install [Bubblewrap](https://github.com/containers/bubblewrap) and
  ensure `bwrap` is on your `PATH`. If you use [`pixi`][1], run
  `pixi global install bubblewrap`. Your system must allow user namespaces.
- **macOS:** Ensure `sandbox-exec` is available on your `PATH`.

Launch URSA with sandboxing on macOS and Linux doesn't requires either setting
the environment variable `URSA_SBX=1` or not setting `URSA_SBX`.

## Customize access

Set environment variables before launching URSA. The following examples use a
POSIX shell on Linux or macOS.

Block network access for sandboxed commands:

```sh
URSA_ACCESS_NETWORK=0 ursa
```

Allow writes to additional existing directories, such as a package cache or
shared output folder:

```sh
URSA_EXTRA_WRITABLE_PATHS="$HOME/.cache/uv:$HOME/ursa-output" ursa
```

Separate paths with `:` on Linux and macOS. Paths are resolved and `~` is
expanded; nonexistent paths are ignored. Create directories before launching.

## Safety checks and troubleshooting

Model-based command safety checks remain enabled independently of the sandbox.
To skip those checks while keeping sandboxing enabled:

```sh
URSA_SAFETY_LEVEL=none URSA_SBX=1 ursa
```

`URSA_SAFETY_LEVEL=yolo` disables both safety checks and sandboxing.

If the backend is missing or cannot start, command execution raises an error
instead of silently running without a sandbox. Check `PATH` and, on Linux,
whether user namespaces are allowed; some HPC systems disable them. Windows
sandboxing is not implemented.

To explicitly run commands without sandboxing, including on unsupported systems:

```sh
URSA_SBX=0 ursa
```

This removes sandbox filesystem and network restrictions; model-based safety
checks still follow `URSA_SAFETY_LEVEL`.

[1]: https://pixi.prefix.dev
