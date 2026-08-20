import os
import shlex
import shutil
import subprocess
import sys

SSH_ENV_VARS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")


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
