# Terminal tools

URSA's terminal tools run commands in managed shell sessions. A short command
normally returns its output directly. Interactive, long-running, or large-output
commands return `Terminal ID: <id>`, where `<id>` is eight characters; use the
ID with the other terminal tools.

!!! warning
    Terminal sessions execute commands with the permissions of the URSA
    process. Treat command text and environment values as sensitive input, and
    use a restricted workspace when running untrusted commands.

The terminal tool set is opt-in. `TERM_TOOLS` is not part of the default Chat
or Execution agent tool set because later calls such as `term_send_line` can
submit arbitrary shell input without a complete command to assess. Add the
tools explicitly only when an agent should control interactive processes.
Use `get_supported_term_tools()` to omit screen operations that the selected
fallback backend cannot implement:

```python
from ursa.agents import ExecutionAgent
from ursa.tools import get_supported_term_tools

agent = ExecutionAgent(llm, extra_tools=get_supported_term_tools())
```

`TERM_TOOLS` remains available as the complete, unfiltered tool catalog.

The initial terminal launch uses the same fail-closed safety assessment as
`run_command`. The assessment includes the command, resolved shell arguments,
working directory, and every environment override key and value. Do not put
secrets in `env`: those values are sent to the configured safety-review model
because environment variables can change what a shell executes. A launch that
cannot be assessed as safe is rejected before a session starts. Shell arguments
must not contain command-execution flags such as `-c`, `--command`,
`-Command`, or `-EncodedCommand`; pass the command through `cmd`. The initial
check does not cover input sent later through the `term_send_*` tools.

## Backends

`GhosttyTerm` is the preferred backend on supported macOS and Linux systems. It
uses [`pyghostty`](https://github.com/AnswerDotAI/pyghostty) to maintain a real
terminal screen, including scrollback, cursor position, and resize behavior.
It is installed automatically with URSA on the supported operating-system and
CPU combinations where upstream publishes a wheel:

```bash
uv sync
```

- macOS 13 or newer on x86-64 or Apple silicon;
- glibc-based Linux on x86-64 or AArch64.

Python dependency markers cannot distinguish glibc Linux from musl Linux or
express a minimum macOS release. Consequently, automatic dependency selection
can still fail during installation on musl-based Linux or macOS older than 13;
those environments must omit `pyghostty` when constructing an installation in
order to use the runtime `ProcessTerm` fallback.

`ProcessTerm` is the fallback when `pyghostty` is unavailable, including on
Windows and unsupported architectures. It captures combined standard output
and standard error in a private temporary file. It supports command execution,
input, status, waiting, and line-based reads, but it does not emulate a
terminal screen; cursor, size, and resize operations are therefore unavailable.

Set `URSA_TERM_BACKEND=process` before starting URSA to force the portable
fallback, even when Ghostty is installed. This is useful for testing the exact
backend that Windows and unsupported platforms use. `URSA_TERM_BACKEND=ghostty`
forces Ghostty and reports a configuration error if it is unavailable;
`URSA_TERM_BACKEND=auto` (the default) prefers Ghostty and falls back to
ProcessTerm. `/status` identifies forced backend selection.

On Unix, sessions use Bash by default. On Windows, URSA prefers Git Bash when it
is installed and otherwise uses PowerShell. Pass `shell` to `term` to override
the default.

## Start a command

```text
term(
    cmd: str | list[str],
    env: dict[str, str] | None = None,
    session: bool = False,
    shell: list[str] | None = None,
)
```

Set `session=true` for an interactive program and `term` immediately returns
`Terminal ID: <id>`. With the default `session=false`, output is returned as
`Terminal contents:\n<contents>`
only when the command finishes before `URSA_TERM_TIMEOUT` and remains within
both output limits. If the output reaches either limit, the session stays
registered and `term` returns `Terminal ID: <id>` so output can be inspected
incrementally. In other words, direct output must be strictly below both byte
and line thresholds.

`cmd` may be shell text or a list of arguments. `env` is merged into the child
process environment; it does not replace the complete environment.

## Interact with a session

| Tool | Purpose |
| --- | --- |
| `term_send_bytes(term_id, data)` | Send `bytes`, or a JSON-compatible list of integer byte values from 0 through 255. |
| `term_send_text(term_id, text)` | Send UTF-8 text without a newline. |
| `term_send_line(term_id, line)` | Send UTF-8 text followed by a newline. |
| `term_send_key(term_id, key, modifiers=None)` | Send a printable or named key with optional modifiers. |
| `term_read(term_id, offset=0, lines=None)` | Read terminal text, or select lines back from the end. |
| `term_is_alive(term_id)` | Return `{"is_alive": true}` while running, or report `exit_code` after exit. |
| `term_wait_for(term_id, pattern, timeout=None)` | Search output emitted after the call begins, newest-first, and return the newest matching line and stream offset. |
| `term_wait_screen(term_id, condition="stable", bounding_box=None, include_styling=true, timeout=None)` | Wait for a Ghostty screen to remain unchanged for one second or ten frames (minimum two frames), or to change. |
| `term_click(term_id, row, col, button="left", modifiers=None)` | Click a mouse button at a Ghostty screen cell. |
| `term_mouse_down(term_id, row, col, button="left", modifiers=None)` | Press and hold a mouse button at a Ghostty screen cell. |
| `term_mouse_up(term_id, row, col, button="left", modifiers=None)` | Release a mouse button at a Ghostty screen cell. |
| `term_hover(term_id, row, col, modifiers=None)` | Move the pointer to a Ghostty screen cell. |
| `term_scroll(term_id, row, col, delta_y, delta_x=0, modifiers=None)` | Send vertical or horizontal wheel events at a Ghostty screen cell. |
| `term_resize(term_id, rows, cols)` | Resize a Ghostty-backed screen. |
| `term_cursor(term_id)` | Return the Ghostty-backed cursor as `(row, column)`. |
| `term_size(term_id)` | Return the Ghostty-backed size as `(rows, columns)`. |
| `term_screenshot(term_id)` | Return a styled PNG image of a Ghostty-backed screen. |

With Ghostty, `term_read(id)` returns the visible screen. Supplying a nonzero
`offset` or a `lines` value switches to the complete terminal contents,
including scrollback, and selects from the end. `ProcessTerm` has no screen
model, so an unbounded read returns its complete captured output. For a
tail-like read, `offset` skips trailing lines and `lines` limits the preceding
selection. For example, `term_read(id, offset=10, lines=20)` returns the 20
lines immediately before the final 10 lines.

`term_send_key` supports printable characters, Enter, Tab, Escape, Backspace,
Delete, the arrow keys, Home, End, Page Up, Page Down, Insert, and F1 through
F12. Modifier names are case-insensitive: `ctrl`/`control`, `alt`/`option`,
`shift`, and `super`/`cmd`/`meta`. Modified navigation and function keys use
xterm modifier parameters. Super-modified printable characters use Kitty's
CSI-u keyboard encoding, so applications that do not understand that protocol
may not recognize them.

`term_wait_for` uses Python regular-expression syntax and returns `Pattern not
found` when its deadline expires. Terminal wait tools default to five times
`URSA_TERM_TIMEOUT`; a requested timeout cannot exceed ten times that value.

`term_wait_screen` compares text and styling by default. Set
`include_styling=false` to ignore color and other visual-style changes. Its
optional bounding box is `(top, left, bottom, right)` in zero-based terminal
cells, with exclusive bottom and right edges. It is available only with the
Ghostty backend.

Mouse coordinates are zero-based. Click, press, and release accept left,
middle, or right buttons and the same modifiers as `term_send_key`. Positive
scroll deltas move down or right; negative deltas move up or left. A held
button can be dragged with `term_mouse_down`, one or more `term_hover` calls,
then `term_mouse_up`. Mouse tools are available only with Ghostty and emit
input only when the terminal application has enabled mouse tracking. New
Ghostty sessions default to 120 columns by 40 rows.

## Limits

The defaults can be changed before starting URSA:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `URSA_TERM_BACKEND` | `auto` | Backend selection: `auto`, `ghostty`, or `process`. |
| `URSA_TERM_TIMEOUT` | `10` | Seconds allowed for a direct short-command result. |
| `URSA_TERM_MAX_BYTES` | `20000` | UTF-8 output size at which `term` returns a session ID. |
| `URSA_TERM_MAX_LINES` | `200` | Output line count at which `term` returns a session ID. |

The values are read when the terminal package is imported. All sessions receive
an ID internally, but a successfully completed short command returns its
captured output prefixed by `Terminal contents:`.

## Exporting the Textual view as PNG

Textual's compositor can export the currently running app as SVG. URSA can
rasterize that exact view to PNG through PyMuPDF, which is already a core
dependency:

```python
from ursa.cli.tui.image_export import textual_app_to_png

png = textual_app_to_png(app, scale=2)
```

This is an application screenshot helper, not a `term_read` mode or a headless
terminal-rendering API. `textual_app_to_png` asks a running Textual app for its
current composed SVG screen, then rasterizes that SVG. It returns PNG bytes and
does not write to the filesystem. It preserves the composed Textual appearance;
theme blending and modal opacity may make the resulting pixels differ from raw
Ghostty RGB values.

Rasterization rejects SVG input above 16 MiB, scale factors above 8, and output
above 64 million pixels before allocating the output pixmap.

The app must already be running, as required by Textual's screenshot API. The
helper captures the whole screen exactly as composed, including overlays. To
capture the terminal browser, open `/terms`, select the wanted tab, and call the
helper while that modal is active. It does not temporarily push a screen,
select a terminal, or crop the result to the terminal widget. Use
`textual_screenshot_to_png(svg)` only when an SVG screenshot has already been
captured and needs rasterization.

## Terminal screenshots

On the Ghostty backend, `term_screenshot(term_id)` returns a PNG image of the
terminal's current screen, including colors and text styling. It uses the same
styled snapshot and Rich/Textual rendering path as the live terminal view.
Newly started screen sessions are sampled briefly for initial output and a
stable frame, preventing an immediate screenshot from capturing the empty PTY
state before the child process has rendered its first prompt.
The Process fallback has no emulated screen, so it does not advertise this
tool; use `term_read` for Process terminal output.
