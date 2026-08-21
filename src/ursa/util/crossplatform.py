import os
import shlex
import shutil
import subprocess
import sys

SSH_ENV_VARS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
KITTY_KEYBOARD_ENV_VARS = (
    "ALACRITTY_WINDOW_ID",
    "KITTY_WINDOW_ID",
    "WEZTERM_PANE",
    "WT_SESSION",
)
KITTY_KEYBOARD_TERM_PROGRAMS = frozenset({
    "alacritty",
    "ghostty",
    "iterm.app",
    "rio",
    "warpterminal",
    "wezterm",
})
KITTY_KEYBOARD_TERM_PREFIXES = ("foot", "xterm-ghostty", "xterm-kitty")


def expects_kitty_keyboard() -> bool:
    """Infer expected Kitty keyboard support without touching the terminal.

    Terminfo has no standardized capability for the Kitty keyboard protocol,
    so this uses identifiers exported by terminal implementations known to
    support it. Unknown terminals and terminal multiplexers fail closed.

    Implementations: https://sw.kovidgoyal.net/kitty/keyboard-protocol/
    """
    if os.environ.get("TMUX") or os.environ.get("ZELLIJ"):
        return False
    if any(os.environ.get(name) for name in KITTY_KEYBOARD_ENV_VARS):
        return True
    term_program = os.environ.get("TERM_PROGRAM", "").casefold()
    if term_program in KITTY_KEYBOARD_TERM_PROGRAMS:
        return True
    term = os.environ.get("TERM", "").casefold()
    return term.startswith(KITTY_KEYBOARD_TERM_PREFIXES)


def platform_clipboard() -> list[str] | None:
    """Return the preferred clipboard command for the current platform."""
    override = os.environ.get("URSA_CLIPBOARD")
    if override:
        return shlex.split(override, posix=(os.name != "nt"))
    if any(os.environ.get(name) for name in SSH_ENV_VARS):
        return None
    if sys.platform == "darwin":
        return ["pbcopy"] if shutil.which("pbcopy") else None
    if sys.platform.startswith("win"):
        return ["clip"] if shutil.which("clip") else None
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy"]
    if os.environ.get("DISPLAY"):
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard"]
        if shutil.which("xsel"):
            return ["xsel", "--clipboard", "--input"]
    return None


def copy_to_clipboard(text: str) -> bool:
    """Copy text with the platform clipboard command, if one is available."""
    command = platform_clipboard()
    if command is None:
        return False
    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            timeout=2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True
