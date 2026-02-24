from __future__ import annotations

import os
from pathlib import Path


class WorkspaceJailError(Exception):
    pass


def workspace_root_from_env(
    default: str | Path = Path.home() / ".cache/ursa_dashboard_workspace",
) -> Path:
    root = os.environ.get("URSA_DASHBOARD_WORKSPACE_ROOT")
    if root:
        p = Path(root)
        if not p.is_absolute():
            raise WorkspaceJailError(
                "URSA_DASHBOARD_WORKSPACE_ROOT must be an absolute path"
            )
        return p
    return Path(default).resolve()


def safe_join(root: Path, rel_path: str) -> Path:
    """Join a user-provided relative path to root, preventing traversal/symlink escapes."""
    if rel_path.startswith("/") or rel_path.startswith("\\"):
        raise WorkspaceJailError("absolute paths not allowed")
    if "\x00" in rel_path:
        raise WorkspaceJailError("NUL byte not allowed")

    # Normalize separators
    rel_path = rel_path.replace("\\", "/")
    parts = [p for p in rel_path.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise WorkspaceJailError("path traversal not allowed")

    candidate = root.joinpath(*parts)
    root_real = root.resolve()
    cand_real = candidate.resolve()
    try:
        cand_real.relative_to(root_real)
    except Exception:
        raise WorkspaceJailError("path escapes workspace")
    return cand_real
