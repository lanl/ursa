from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.parse import (
    SPECIAL_TEXT_FILENAMES,
    TEXT_EXTENSIONS,
    read_text_from_file,
)

DEFAULT_READ_LIMIT = 4000


def _resolve_workspace_path(path: str, workspace: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Path {resolved} is outside the workspace {workspace.resolve()}"
        ) from exc
    return resolved


def _read_file_chunk(path: Path, *, offset: int, limit: int) -> tuple[str, int]:
    safe_offset = max(0, int(offset))
    safe_limit = max(0, int(limit))
    is_text_file = (
        path.suffix.lower() in TEXT_EXTENSIONS
        or path.name.lower() in SPECIAL_TEXT_FILENAMES
    )

    if is_text_file:
        with open(path, encoding="utf-8") as handle:
            handle.seek(safe_offset)
            text = handle.read(safe_limit)
            next_offset = handle.tell()
        return text, next_offset

    full_text = read_text_from_file(path)
    if isinstance(full_text, tuple):
        full_text = "\n".join(str(part) for part in full_text)
    text = str(full_text or "")
    chunk = text[safe_offset : safe_offset + safe_limit]
    return chunk, safe_offset + len(chunk)


@tool("read_file")
def read_file(
    path: str,
    runtime: ToolRuntime[AgentContext],
    offset: int = 0,
    limit: int = DEFAULT_READ_LIMIT,
) -> dict[str, Any]:
    """
    Read a chunk from a workspace file.

    If the path is a directory, return a directory listing instead of failing.
    """
    workspace = Path(runtime.context.workspace).resolve()
    try:
        resolved = _resolve_workspace_path(path, workspace)
    except Exception as exc:
        return {
            "path": str(path),
            "type": "error",
            "error": str(exc),
        }

    if not resolved.exists():
        return {
            "path": str(resolved),
            "type": "missing",
            "message": "Path does not exist. Inspect the parent directory or try another file.",
        }

    if resolved.is_dir():
        return {
            "path": str(resolved),
            "type": "directory",
            "entries": sorted(child.name for child in resolved.iterdir()),
            "message": (
                "Path is a directory. Use read_file on a specific file path to read contents."
            ),
        }

    try:
        text, next_offset = _read_file_chunk(
            resolved,
            offset=offset,
            limit=limit,
        )
    except Exception as exc:
        return {
            "path": str(resolved),
            "type": "error",
            "error": f"Failed to read file: {exc}",
        }

    return {
        "path": str(resolved),
        "type": "file",
        "text": text,
        "next_offset": next_offset,
    }
