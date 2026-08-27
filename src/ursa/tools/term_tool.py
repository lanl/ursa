"""LangChain tools for persistent asynchronous terminal sessions."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from ursa.agents.base import AgentContext
from ursa.tools.run_command_tool import assess_command_safety
from ursa.tools.terminal import (
    TERM_MAX_BYTES,
    TERM_MAX_LINES,
    TERM_TIMEOUT,
    TermManager,
    term_manager,
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


def _validate_shell(shell: list[str] | None) -> None:
    """Reject shell arguments that compete with backend command handling."""
    if shell is None:
        return
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


@tool
async def term(
    cmd: str | list[str],
    runtime: ToolRuntime[AgentContext],
    env: dict[str, str] | None = None,
    session: bool = False,
    shell: list[str] | None = None,
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
async def term_send_bytes(term_id: str, data: bytes | list[int]) -> str:
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
    await term_manager.send_bytes(term_id, payload)
    return f"Sent {len(payload)} bytes to terminal {term_id}"


@tool
async def term_send_text(term_id: str, text: str) -> str:
    """Send text to a terminal session without appending a newline."""
    await term_manager.send_text(term_id, text)
    return f"Sent text to terminal {term_id}"


@tool
async def term_send_line(term_id: str, line: str) -> str:
    """Send text followed by a newline to a terminal session."""
    await term_manager.send_line(term_id, line)
    return f"Sent line to terminal {term_id}"


@tool
async def term_send_keycode(term_id: str, keycode: int) -> str:
    """Send one byte-valued keycode (0 through 255) to a terminal session."""
    await term_manager.send_keycode(term_id, keycode)
    return f"Sent keycode {keycode} to terminal {term_id}"


@tool
async def term_send_key(
    term_id: str,
    key: str,
    modifiers: set[str] | list[str] | None = None,
) -> str:
    """Send a printable or named key with optional keyboard modifiers."""
    payload = encode_term_key(key, modifiers)
    await term_manager.send_bytes(term_id, payload)
    return f"Sent key {key!r} to terminal {term_id}"


@tool
async def term_read(
    term_id: str,
    offset: int = 0,
    lines: int | None = None,
) -> str:
    """Read terminal text, optionally selecting lines back from the end."""
    return await term_manager.read(term_id, offset=offset, lines=lines)


@tool
async def term_is_alive(term_id: str) -> dict[str, bool | int]:
    """Report whether a terminal is alive, or return its process exit code."""
    return await term_manager.is_alive(term_id)


@tool
async def term_wait_for(
    term_id: str,
    pattern: str,
    timeout: float | None = None,
) -> str:
    """Wait until a regex appears in terminal output and return its line."""
    return await term_manager.wait_for(term_id, pattern, timeout)


@tool
async def term_resize(term_id: str, rows: int, cols: int) -> str:
    """Resize a terminal screen to the requested rows and columns."""
    await term_manager.resize(term_id, rows, cols)
    return f"Resized terminal {term_id} to {rows} rows by {cols} columns"


@tool
async def term_cursor(term_id: str) -> tuple[int, int]:
    """Return a terminal's cursor position as ``(row, column)``."""
    return await term_manager.cursor(term_id)


@tool
async def term_size(term_id: str) -> tuple[int, int]:
    """Return a terminal's screen size as ``(rows, columns)``."""
    return await term_manager.size(term_id)


_BASE_TERM_TOOLS = [
    term,
    term_send_bytes,
    term_send_text,
    term_send_line,
    term_send_keycode,
    term_send_key,
    term_read,
    term_is_alive,
    term_wait_for,
]

_SCREEN_TERM_TOOLS = [
    term_resize,
    term_cursor,
    term_size,
]

TERM_TOOLS = [*_BASE_TERM_TOOLS, *_SCREEN_TERM_TOOLS]


def get_supported_term_tools() -> list[BaseTool]:
    """Return terminal tools supported by the manager's selected backend."""
    tools = list(_BASE_TERM_TOOLS)
    if term_manager.supports_screen():
        tools.extend(_SCREEN_TERM_TOOLS)
    return tools
