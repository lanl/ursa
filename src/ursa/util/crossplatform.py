"""Cross-platform locations used by URSA."""

import os
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
