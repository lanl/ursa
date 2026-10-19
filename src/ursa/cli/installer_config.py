"""Shared loading for installer-provided JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INSTALLER_CONFIG_FIELDS = frozenset({
    "python_version",
    "extras",
    "extra_packages",
    "inference_providers",
    "mcp_servers",
})


class InstallerConfigError(ValueError):
    """An installer configuration could not be loaded."""


def load_installer_json(path: Path) -> dict[str, Any]:
    """Load one JSON object without interpreting its consumer-specific fields."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InstallerConfigError(f"could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise InstallerConfigError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from None
    if not isinstance(payload, dict):
        raise InstallerConfigError("the payload must be a JSON object")
    unknown = sorted(set(payload) - INSTALLER_CONFIG_FIELDS)
    if unknown:
        fields = ", ".join(unknown)
        raise InstallerConfigError(
            f"unknown field{'s' if len(unknown) != 1 else ''}: {fields}. "
            "Is this configuration for an unreleased URSA version?"
        )
    return payload
