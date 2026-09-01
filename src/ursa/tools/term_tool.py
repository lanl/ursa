"""LangChain tools for persistent asynchronous terminal sessions."""

from __future__ import annotations

import asyncio
import base64
import re
import shlex
from pathlib import Path
from typing import Annotated, Literal

from langchain.tools import ToolRuntime
from langchain_core.messages.content import (
    ImageContentBlock,
    TextContentBlock,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import AfterValidator, Field, ValidationError

from ursa.agents.base import AgentContext
from ursa.tools.run_command_tool import assess_command_safety
from ursa.tools.terminal import (
    TERM_MAX_BYTES,
    TERM_MAX_LINES,
    TERM_TIMEOUT,
    TermManager,
    settled_screen_snapshot,
    term_manager,
    terminal_snapshot_to_png,
)

_SHELL_COMMAND_FLAGS = {
    "-c",
    "--command",
    "-command",
    "-encodedcommand",
}

_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "super": "super",
    "cmd": "super",
    "meta": "super",
}
_MODIFIER_BITS = {"shift": 1, "alt": 2, "ctrl": 4, "super": 8}
_SIMPLE_KEYS = {
    "enter": b"\r",
    "tab": b"\t",
    "escape": b"\x1b",
    "esc": b"\x1b",
    "backspace": b"\x7f",
}
_CSI_FINAL_KEYS = {
    "up": "A",
    "arrowup": "A",
    "down": "B",
    "arrowdown": "B",
    "right": "C",
    "arrowright": "C",
    "left": "D",
    "arrowleft": "D",
    "home": "H",
    "end": "F",
}
_CSI_TILDE_KEYS = {
    "insert": 2,
    "delete": 3,
    "pageup": 5,
    "pagedown": 6,
    "f5": 15,
    "f6": 17,
    "f7": 18,
    "f8": 19,
    "f9": 20,
    "f10": 21,
    "f11": 23,
    "f12": 24,
}
_FUNCTION_FINAL_KEYS = {"f1": "P", "f2": "Q", "f3": "R", "f4": "S"}
_CONTROL_CHARACTERS = {
    " ": 0,
    "@": 0,
    "[": 27,
    "\\": 28,
    "]": 29,
    "^": 30,
    "_": 31,
    "?": 127,
}
_ESC = bytes((27,))


NonEmptyStr = Annotated[str, Field(min_length=1)]
TermId = Annotated[str, Field(pattern=r"^[A-Za-z0-9]{8}$")]
NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
ByteValue = Annotated[int, Field(ge=0, le=255, strict=True)]
WaitTimeout = Annotated[float, Field(ge=0, le=TERM_TIMEOUT * 10)]
ScrollDelta = Annotated[int, Field(ge=-100, le=100, strict=True)]
MouseButton = Literal["left", "middle", "right"]


def _validate_paste_text(text: str) -> str:
    if any(character in text for character in ("\x00", chr(27), "\r", "\n")):
        raise ValueError(
            "paste text must not contain NUL, Escape, or line-break characters"  # pragma: no mutate
        )
    return text


PasteText = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_paste_text),
]


def _canonical_modifiers(
    modifiers: set[str] | list[str] | None,
) -> frozenset[str]:
    """Validate modifier names and return their canonical forms."""
    canonical: set[str] = set()
    for modifier in modifiers or ():
        if not isinstance(modifier, str):
            raise ValueError("modifiers must be strings")
        normalized = _MODIFIER_ALIASES.get(modifier.casefold())
        if normalized is None:
            raise ValueError(f"unknown modifier: {modifier!r}")
        if normalized in canonical:
            raise ValueError(f"duplicate modifier: {modifier!r}")
        canonical.add(normalized)
    return frozenset(canonical)


def _modifier_parameter(modifiers: frozenset[str]) -> int:
    """Return the xterm/Kitty modifier parameter for *modifiers*."""
    return 1 + sum(_MODIFIER_BITS[modifier] for modifier in modifiers)


def encode_term_key(
    key: str,
    modifiers: set[str] | list[str] | None = None,
) -> bytes:
    """Encode a printable or named key into terminal input bytes."""
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    mods = _canonical_modifiers(modifiers)
    name = key.casefold().replace("_", "").replace("-", "")

    if len(key) == 1 and key.isprintable():
        character = key
        if "shift" in mods and key.isalpha():
            shifted = key.upper()
            if len(shifted) == 1:
                character = shifted
        if "super" in mods:
            parameter = _modifier_parameter(mods)
            return f"\x1b[{ord(character)};{parameter}u".encode()
        if "ctrl" in mods:
            if character.isascii() and character.isalpha():
                control_code = ord(character) & 0x1F
                payload = bytes((control_code,))
            elif character in _CONTROL_CHARACTERS:
                payload = bytes((_CONTROL_CHARACTERS[character],))
            else:
                raise ValueError(
                    f"Ctrl+{key} has no representable ASCII control code"
                )
        else:
            payload = character.encode()
        return _ESC + payload if "alt" in mods else payload

    if name in _SIMPLE_KEYS:
        if mods == {"shift"} and name == "tab":
            return b"\x1b[Z"  # pragma: no mutate
        unsupported = mods - {"alt"}
        if unsupported:
            raise ValueError(
                f"modifiers are not supported for named key {key!r}"
            )
        payload = _SIMPLE_KEYS[name]
        return _ESC + payload if "alt" in mods else payload

    parameter = _modifier_parameter(mods)
    suffix = "" if not mods else f";{parameter}"
    if name in _CSI_FINAL_KEYS:
        prefix = "\x1b[" if not mods else "\x1b[1"  # pragma: no mutate
        return f"{prefix}{suffix}{_CSI_FINAL_KEYS[name]}".encode()
    if name in _CSI_TILDE_KEYS:
        return f"\x1b[{_CSI_TILDE_KEYS[name]}{suffix}~".encode()
    if name in _FUNCTION_FINAL_KEYS:
        if not mods:
            return f"\x1bO{_FUNCTION_FINAL_KEYS[name]}".encode()
        return f"\x1b[1{suffix}{_FUNCTION_FINAL_KEYS[name]}".encode()
    raise ValueError(f"unknown key: {key!r}")


def _launch_safety_text(
    cmd: str | list[str],
    env: dict[str, str] | None,
    shell: list[str] | None,
    cwd: Path,
) -> str:
    """Return an unambiguous representation of execution-affecting inputs."""
    if isinstance(cmd, str):
        command = f"text {cmd!r}"
    else:
        command = f"argv {shlex.join(cmd)!r}"
    shell_argv = TermManager.default_shell() if shell is None else shell
    shell_text = shlex.join(shell_argv)
    environment = (
        "none"
        if not env
        else "\n".join(
            f"  {key!r}={value!r}" for key, value in sorted(env.items())
        )
    )
    return (
        f"COMMAND:\n  {command}\n"
        f"SHELL ARGV:\n  {shell_text!r}\n"
        f"WORKING DIRECTORY:\n  {str(cwd)!r}\n"
        f"ENVIRONMENT OVERRIDES:\n{environment}"
    )


def _validate_shell(shell: list[str] | None) -> list[str] | None:
    """Reject shell arguments that compete with backend command handling."""
    if shell is None:
        return None
    if not shell:
        raise ValueError("shell must contain at least one argument")
    conflicting = [
        argument
        for argument in shell[1:]
        if argument.casefold() in _SHELL_COMMAND_FLAGS
    ]
    if conflicting:
        flags = ", ".join(repr(flag) for flag in conflicting)
        raise ValueError(
            "shell must not contain command-execution flags; pass the command "
            f"through cmd instead (found {flags})"
        )
    return shell


def _validate_key(key: str) -> str:
    """Validate a key independently of its modifiers for tool input parsing."""
    if len(key) == 1 and key.isprintable():
        return key
    name = key.casefold().replace("_", "").replace("-", "")
    if name not in {
        *_SIMPLE_KEYS,
        *_CSI_FINAL_KEYS,
        *_CSI_TILDE_KEYS,
        *_FUNCTION_FINAL_KEYS,
    }:
        raise ValueError(f"unknown key: {key!r}")
    return key


def _validate_modifiers(
    modifiers: set[str] | list[str] | None,
) -> set[str] | list[str] | None:
    _canonical_modifiers(modifiers)
    return modifiers


def _validate_pattern(pattern: str) -> str:
    """Reject malformed regular expressions at the tool schema boundary."""
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"invalid regular expression: {error}") from error
    return pattern


ShellArgv = Annotated[
    Annotated[list[NonEmptyStr], Field(min_length=1)] | None,
    AfterValidator(_validate_shell),
]
CommandArgv = Annotated[list[NonEmptyStr], Field(min_length=1)]
TermKey = Annotated[NonEmptyStr, AfterValidator(_validate_key)]
TermModifiers = Annotated[
    set[str] | list[str] | None,
    AfterValidator(_validate_modifiers),
]
RegexPattern = Annotated[NonEmptyStr, AfterValidator(_validate_pattern)]


def _validate_bounding_box(
    box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    """Validate a zero-based, end-exclusive screen rectangle."""
    if box is None:
        return None
    top, left, bottom, right = box
    if bottom <= top or right <= left:
        raise ValueError("bounding box must have positive height and width")
    return box


ScreenBoundingBox = Annotated[
    tuple[
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
        NonNegativeInt,
    ]
    | None,
    AfterValidator(_validate_bounding_box),
]


def _unknown_terminal(term_id: str) -> ToolException:
    """Translate a stale terminal ID into a retryable tool failure."""
    return ToolException(
        f"Unknown terminal ID {term_id!r}. Use an ID returned by the term tool."
    )


@tool
async def term(
    cmd: NonEmptyStr | CommandArgv,
    runtime: ToolRuntime[AgentContext],
    env: dict[str, str] | None = None,
    session: bool = False,
    shell: ShellArgv = None,
) -> str:
    """Run a command, returning its output or a persistent terminal ID.

    Set ``session`` for interactive or long-running work. Short commands return
    their output directly when they finish within the configured time and size
    limits; otherwise they remain available through the other terminal tools.

    Args:
        cmd: Command text, or an argument list to execute.
        env: Environment variables to add to the shell environment.
        session: Return immediately with a persistent terminal ID.
        shell: Optional shell executable and arguments.
    """
    # Keep direct coroutine callers safe too; normal tool calls have already
    # performed this check through ``ShellArgv``'s Pydantic validator.
    _validate_shell(shell)
    workspace = Path(runtime.context.workspace)
    launch_text = _launch_safety_text(cmd, env, shell, workspace)
    safety_result = await assess_command_safety(launch_text, runtime)
    if not safety_result.is_safe:
        return (
            "[UNSAFE] That terminal launch was deemed unsafe and "
            f"cannot be run.\nFor reason: {safety_result.reason}"
        )

    terminal = await term_manager.create(
        cmd,
        env=env,
        shell=shell,
        cwd=workspace,
    )
    try:
        if session:
            # Deliver cancellation requested during creation before disclosing
            # the session ID, so ownership cannot be lost between those steps.
            await asyncio.sleep(0)
            return f"Terminal ID: {terminal.term_id}"

        try:
            await asyncio.wait_for(
                term_manager.wait(terminal.term_id), timeout=TERM_TIMEOUT
            )
        except TimeoutError:
            return f"Terminal ID: {terminal.term_id}"

        contents = await term_manager.contents(terminal.term_id)
        if (
            len(contents.encode("utf-8")) >= TERM_MAX_BYTES
            or len(contents.splitlines()) >= TERM_MAX_LINES
        ):
            return f"Terminal ID: {terminal.term_id}"

        await term_manager.remove(terminal.term_id, terminate=True)
        return f"Terminal contents:\n{contents}"
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(
            term_manager.remove(terminal.term_id, terminate=True)
        )
        try:
            await asyncio.shield(cleanup)
        except BaseException:
            # Cleanup failure must not replace the caller's cancellation.
            pass
        raise


@tool
async def term_send_bytes(
    term_id: TermId, data: bytes | list[ByteValue]
) -> str:
    """Send raw bytes to a terminal session."""
    if isinstance(data, list):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in data
        ):
            raise ValueError("byte values must be integers from 0 through 255")
        payload = bytes(data)
    else:
        payload = data
    try:
        await term_manager.send_bytes(term_id, payload)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    return f"Sent {len(payload)} bytes to terminal {term_id}"


@tool
async def term_send_text(term_id: TermId, text: str) -> str:
    """Send text to a terminal session without appending a newline."""
    try:
        await term_manager.send_text(term_id, text)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    return f"Sent text to terminal {term_id}"


@tool
async def term_paste_text(term_id: TermId, text: PasteText) -> str:
    """Paste literal text without triggering per-character terminal bindings."""
    try:
        await term_manager.paste_text(term_id, text)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    except NotImplementedError as error:
        raise ToolException(str(error)) from error
    return f"Pasted text to terminal {term_id}"


@tool
async def term_send_line(term_id: TermId, line: str) -> str:
    """Send text followed by a newline to a terminal session."""
    try:
        await term_manager.send_line(term_id, line)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    return f"Sent line to terminal {term_id}"


@tool
async def term_send_key(
    term_id: TermId,
    key: TermKey,
    modifiers: TermModifiers = None,
) -> str:
    """Send a printable or named key with optional keyboard modifiers."""
    try:
        payload = encode_term_key(key, modifiers)
    except ValueError as error:
        # Cross-field constraints (for example Ctrl+Enter) cannot be expressed
        # on either annotation alone. Still expose them as retryable Pydantic
        # input errors rather than leaking an implementation ValueError.
        raise ValidationError.from_exception_data(
            "term_send_key",
            [
                {
                    "type": "value_error",
                    "loc": ("key", "modifiers"),
                    "input": {"key": key, "modifiers": modifiers},
                    "ctx": {"error": error},
                }
            ],
        ) from error
    try:
        await term_manager.send_bytes(term_id, payload)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    return f"Sent key {key!r} to terminal {term_id}"


@tool
async def term_read(
    term_id: TermId,
    offset: NonNegativeInt = 0,
    lines: PositiveInt | None = None,
) -> str:
    """Read terminal text, optionally selecting lines back from the end."""
    try:
        return await term_manager.read(term_id, offset=offset, lines=lines)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error


@tool
async def term_is_alive(term_id: TermId) -> dict[str, bool | int]:
    """Report whether a terminal is alive, or return its process exit code."""
    try:
        return await term_manager.is_alive(term_id)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error


@tool
async def term_wait_for(
    term_id: TermId,
    pattern: RegexPattern,
    timeout: WaitTimeout | None = None,
) -> str:
    """Wait for new output matching a regex, returning the newest match."""
    try:
        return await term_manager.wait_for(term_id, pattern, timeout)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error


@tool
async def term_wait_screen(
    term_id: TermId,
    condition: Literal["stable", "change"] = "stable",
    bounding_box: ScreenBoundingBox = None,
    include_styling: bool = True,
    timeout: WaitTimeout | None = None,
) -> str:
    """Wait for a terminal screen to stabilize or change.

    Stability means the selected screen region remains unchanged for one
    second or ten frames, whichever comes first, with at least two frames.
    ``bounding_box`` is ``(top, left, bottom, right)`` using
    zero-based, end-exclusive coordinates. Styling participates in comparison
    by default; set ``include_styling`` false to compare only displayed text.
    """
    try:
        return await term_manager.wait_screen(
            term_id,
            condition=condition,
            bounding_box=bounding_box,
            include_styling=include_styling,
            timeout=timeout,
        )
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    except (ValueError, NotImplementedError) as error:
        raise ToolException(str(error)) from error


async def _send_mouse_events(
    term_id: str,
    row: int,
    col: int,
    events: tuple[tuple[str, str | None], ...],
    modifiers: set[str] | list[str] | None,
) -> None:
    canonical = _canonical_modifiers(modifiers)
    try:
        await term_manager.mouse_input(
            term_id, row, col, events, modifiers=canonical
        )
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    except (ValueError, NotImplementedError) as error:
        raise ToolException(str(error)) from error


@tool
async def term_click(
    term_id: TermId,
    row: NonNegativeInt,
    col: NonNegativeInt,
    button: MouseButton = "left",
    modifiers: TermModifiers = None,
) -> str:
    """Click a mouse button at a zero-based terminal cell."""
    await _send_mouse_events(
        term_id,
        row,
        col,
        (("press", button), ("release", button)),
        modifiers,
    )
    return f"Clicked {button} at ({row}, {col}) in terminal {term_id}"


@tool
async def term_mouse_down(
    term_id: TermId,
    row: NonNegativeInt,
    col: NonNegativeInt,
    button: MouseButton = "left",
    modifiers: TermModifiers = None,
) -> str:
    """Press and hold a mouse button at a zero-based terminal cell."""
    await _send_mouse_events(term_id, row, col, (("press", button),), modifiers)
    return f"Pressed {button} at ({row}, {col}) in terminal {term_id}"


@tool
async def term_mouse_up(
    term_id: TermId,
    row: NonNegativeInt,
    col: NonNegativeInt,
    button: MouseButton = "left",
    modifiers: TermModifiers = None,
) -> str:
    """Release a mouse button at a zero-based terminal cell."""
    await _send_mouse_events(
        term_id, row, col, (("release", button),), modifiers
    )
    return f"Released {button} at ({row}, {col}) in terminal {term_id}"


@tool
async def term_hover(
    term_id: TermId,
    row: NonNegativeInt,
    col: NonNegativeInt,
    modifiers: TermModifiers = None,
) -> str:
    """Move the pointer to a zero-based terminal cell."""
    await _send_mouse_events(term_id, row, col, (("motion", None),), modifiers)
    return f"Moved pointer to ({row}, {col}) in terminal {term_id}"


@tool
async def term_scroll(
    term_id: TermId,
    row: NonNegativeInt,
    col: NonNegativeInt,
    delta_y: ScrollDelta,
    delta_x: ScrollDelta = 0,
    modifiers: TermModifiers = None,
) -> str:
    """Scroll at a terminal cell; positive deltas move down and right."""
    if delta_y == 0 and delta_x == 0:
        error = ValueError("at least one scroll delta must be nonzero")
        raise ValidationError.from_exception_data(
            "term_scroll",
            [
                {
                    "type": "value_error",
                    "loc": ("delta_y", "delta_x"),
                    "input": {"delta_y": delta_y, "delta_x": delta_x},
                    "ctx": {"error": error},
                }
            ],
        )
    events: list[tuple[str, str | None]] = []
    if delta_y:
        button = "wheel_down" if delta_y > 0 else "wheel_up"
        events.extend(("press", button) for _ in range(abs(delta_y)))
    if delta_x:
        button = "wheel_right" if delta_x > 0 else "wheel_left"
        events.extend(("press", button) for _ in range(abs(delta_x)))
    await _send_mouse_events(term_id, row, col, tuple(events), modifiers)
    return (
        f"Scrolled ({delta_y}, {delta_x}) at ({row}, {col}) "
        f"in terminal {term_id}"
    )


@tool
async def term_resize(
    term_id: TermId, rows: PositiveInt, cols: PositiveInt
) -> str:
    """Resize a terminal screen to the requested rows and columns."""
    try:
        await term_manager.resize(term_id, rows, cols)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    return f"Resized terminal {term_id} to {rows} rows by {cols} columns"


@tool
async def term_cursor(term_id: TermId) -> tuple[int, int]:
    """Return a terminal's cursor position as ``(row, column)``."""
    try:
        return await term_manager.cursor(term_id)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error


@tool
async def term_size(term_id: TermId) -> tuple[int, int]:
    """Return a terminal's screen size as ``(rows, columns)``."""
    try:
        return await term_manager.size(term_id)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error


@tool
async def term_screenshot(
    term_id: TermId,
) -> list[TextContentBlock | ImageContentBlock]:
    """Capture a styled PNG of a screen-backed terminal."""
    try:
        snapshot = await settled_screen_snapshot(term_manager, term_id)
    except KeyError as error:
        raise _unknown_terminal(term_id) from error
    if not snapshot.screen or snapshot.rows is None or snapshot.cols is None:
        raise ToolException(
            "Terminal screenshots require the Ghostty backend. Use term_read "
            "for a Process terminal."
        )

    png = terminal_snapshot_to_png(snapshot)
    return [
        create_text_block(text="Screenshot attached."),
        create_image_block(
            base64=base64.b64encode(png).decode("ascii"),
            mime_type="image/png",
        ),
    ]


_BASE_TERM_TOOLS = [
    term,
    term_send_bytes,
    term_send_text,
    term_send_line,
    term_send_key,
    term_read,
    term_is_alive,
    term_wait_for,
]

_SCREEN_TERM_TOOLS = [
    term_paste_text,
    term_wait_screen,
    term_resize,
    term_cursor,
    term_size,
    term_screenshot,
    term_click,
    term_mouse_down,
    term_mouse_up,
    term_hover,
    term_scroll,
]

TERM_TOOLS = [
    *_BASE_TERM_TOOLS,
    *_SCREEN_TERM_TOOLS,
]


def get_supported_term_tools() -> list[BaseTool]:
    """Return terminal tools supported by the manager's selected backend."""
    tools = list(_BASE_TERM_TOOLS)
    if term_manager.supports_screen():
        tools.extend(_SCREEN_TERM_TOOLS)
    return tools
