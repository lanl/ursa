# Terminal tools

Terminal tools run managed shell sessions. Short completed commands return
`Terminal contents:`; interactive, slow, or large-output commands return an
eight-character `Terminal ID: <id>` for use with the other tools.

!!! warning
    Sessions run with URSA's permissions. Use a restricted workspace for
    untrusted work. The initial launch is safety-reviewed, including `cmd`,
    shell, workspace, and `env`; do not put secrets in `env`. Later session
    input is not reassessed.

Tools are opt-in because later calls can submit arbitrary shell input:

```python
from ursa.agents import ExecutionAgent
from ursa.tools import get_supported_term_tools

agent = ExecutionAgent(llm, extra_tools=get_supported_term_tools())
```

`TERM_TOOLS` is the complete catalog. `get_supported_term_tools()` omits
screen-only tools when the selected backend cannot implement them. Shell
arguments cannot include command flags (`-c`, `--command`, `-Command`, or
`-EncodedCommand`); pass the command as `cmd`.

## Backends and limits

`GhosttyTerm` is the default on supported macOS and Linux installations through
the [`pyghostty`](https://github.com/AnswerDotAI/pyghostty) dependency. It
emulates a screen,
scrollback, cursor, resizing, styling, paste, and mouse input. `ProcessTerm`
is the portable Windows/unsupported-platform fallback: it supports command
execution, input, status, waiting, and line reads, but has no screen. Set
`URSA_TERM_BACKEND` to `auto` (default), `ghostty`, or `process` before import.

The dependency covers glibc-based x86-64/AArch64 Linux and macOS 13+ on Intel
or Apple silicon. Other platforms use ProcessTerm. Unix defaults to Bash;
Windows prefers Git Bash, then PowerShell. Pass `shell` to override either
choice.

| Limit | Default |
| --- | ---: |
| Ghostty scrollback | 2,000 lines |
| ProcessTerm capture | 100 MiB combined output |
| Live or starting sessions | 16 |
| `URSA_TERM_TIMEOUT` | 10 seconds |
| `URSA_TERM_MAX_BYTES` / `URSA_TERM_MAX_LINES` | 20,000 / 200 |

The three `URSA_TERM_*` values are read when the terminal package imports.
Graceful runtime and interpreter shutdown both clean up registered sessions.

## Tool reference

`term(cmd, env=None, session=False, shell=None)` starts a command. `cmd` can
be shell text or an argument list; `env` augments the child environment. With
`session=True`, it returns an ID immediately. Otherwise it returns direct
output only when completion occurs before the timeout and output stays strictly
below both output limits; the retained session otherwise returns an ID.

| Tool | Purpose |
| --- | --- |
| `term_send_bytes`, `term_send_text`, `term_send_line` | Send raw bytes, UTF-8 text, or text plus newline. |
| `term_paste_text` | Send one negotiated Ghostty bracketed paste; rejects NUL/Escape/newlines and does not submit. |
| `term_send_key` | Send printable, navigation, editing, or F1–F12 keys with ctrl/control, alt/option, shift, and super/cmd/meta modifiers. |
| `term_read` | Read visible Ghostty screen, or use `offset`/`lines` for scrollback/tail selection; ProcessTerm reads its capture. |
| `term_is_alive`, `term_resources`, `term_close` | Report process state, sample the shell tree's CPU/memory/process/thread use, or terminate and discard a session. |
| `term_wait_for` | Search output emitted after the call begins and return the newest matching line/offset, or `Pattern not found`. |
| `term_wait_screen` | Wait for Ghostty screen `stable` or `changed`; accepts an end-exclusive `(top, left, bottom, right)` box and optional styling comparison. |
| `term_click`, `term_mouse_down`, `term_mouse_up`, `term_hover`, `term_scroll` | Send Ghostty mouse input at zero-based cells; left/middle/right buttons and key modifiers are supported. |
| `term_resize`, `term_cursor`, `term_size`, `term_screenshot` | Resize or inspect the Ghostty screen, or return its styled PNG. |

Ghostty's modified navigation/function keys use xterm parameters; super-
modified printable keys use Kitty CSI-u. Mouse input takes effect only when the
application enables tracking. New Ghostty sessions start at 120×40 cells.
`term_wait_for` accepts Python regex and defaults to five times the base timeout
(requests cannot exceed ten times it). `term_wait_screen` needs Ghostty and
defaults to comparing text and styling until one second or ten frames settle.

## Textual PNG export

`textual_app_to_png(app, scale=1)` captures the running Textual compositor as
SVG and rasterizes it through PyMuPDF; `textual_screenshot_to_png(svg)` converts
an existing SVG. Both return bytes without writing files. Exports include
overlays (open `/terms` and select its tab first), limit SVG input to 16 MiB,
scale to 8, and output to 64 million pixels. `term_screenshot` instead captures
one Ghostty screen and waits briefly for initial rendering; use `term_read` for
the ProcessTerm fallback.
