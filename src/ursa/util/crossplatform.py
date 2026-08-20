"""Cross-platform locations used by URSA."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def system_config_path() -> Path:
    """Return the platform-specific system-wide URSA configuration path."""
    if sys_config := os.environ.get("URSA_SYSTEM_CONFIG"):
        return Path(sys_config)
    if sys.platform == "win32":
        root = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
        return root / "ursa" / "config.yaml"
    if sys.platform == "darwin":
        return Path("/Library/Application Support/ursa/config.yaml")
    return Path("/etc/ursa/config.yaml")


def platform_user_config_path() -> Path:
    """Return the native platform-specific per-user configuration path."""
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
        return root / "ursa" / "config.yaml"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/ursa/config.yaml"
    return Path.home() / ".config/ursa/config.yaml"


def user_config_paths() -> list[Path]:
    """Return user config paths from lowest to highest precedence.

    The native platform location is followed by the portable ``~/.config``
    location and then an explicit ``XDG_CONFIG_HOME`` location. Duplicate
    paths are omitted while preserving that precedence.
    """
    candidates = [
        platform_user_config_path(),
        Path.home() / ".config/ursa/config.yaml",
    ]
    if xdg_home := os.environ.get("XDG_CONFIG_HOME"):
        candidates.append(Path(xdg_home).expanduser() / "ursa" / "config.yaml")

    paths: list[Path] = []
    for candidate in candidates:
        if candidate not in paths:
            paths.append(candidate)
    return paths
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
