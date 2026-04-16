from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

TOOL_OUTPUT_DIR = "tool_outputs"


def _safe_tool_name(tool_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(tool_name or "").strip())
    slug = slug.strip("._-")
    return slug or "tool"


def serialize_tool_output(result: Any) -> tuple[str, str]:
    if isinstance(result, str):
        return result, ".txt"
    return json.dumps(result, ensure_ascii=False, indent=2, default=str), ".json"


def _artifact_path(
    *,
    workspace: Path,
    tool_name: str,
    suffix: str,
    label: str | None = None,
) -> tuple[Path, str]:
    artifact_dir = workspace / TOOL_OUTPUT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_tool_name(tool_name)
    middle = f"-{label}" if label else ""
    artifact_name = f"{stem}{middle}-{uuid.uuid4().hex[:8]}{suffix}"
    artifact_path = artifact_dir / artifact_name
    return artifact_path, str(artifact_path.relative_to(workspace))


def _preview_text(
    text: str,
    *,
    limit: int,
    saved_to: str,
    label: str | None = None,
) -> str:
    note_label = f"full {label} saved to" if label else "full output saved to"
    max_chars = max(0, int(limit))
    base_note = f"[truncated {{omitted}} chars; {note_label} {saved_to}]"
    if max_chars <= 0:
        return base_note.format(omitted=len(text))

    preview = text[:max_chars]
    while True:
        omitted = max(0, len(text) - len(preview))
        if omitted == 0:
            return preview
        note = "\n..." + base_note.format(omitted=omitted)
        remaining = max_chars - len(note)
        if remaining < 0:
            compact_note = base_note.format(omitted=len(text))
            return compact_note[:max_chars]
        candidate = text[:remaining]
        if len(candidate) == len(preview):
            return f"{candidate}{note}"
        preview = candidate


def spill_tool_output(
    *,
    result: Any,
    tool_name: str,
    workspace: Path,
    limit: int,
) -> Any:
    serialized, suffix = serialize_tool_output(result)
    if len(serialized) <= limit:
        return result

    artifact_path, saved_to = _artifact_path(
        workspace=workspace,
        tool_name=tool_name,
        suffix=suffix,
    )
    artifact_path.write_text(serialized, encoding="utf-8")

    preview = _preview_text(serialized, limit=int(limit), saved_to=saved_to)
    return {
        "truncated": True,
        "tool_name": tool_name,
        "saved_to": saved_to,
        "original_chars": len(serialized),
        "preview": preview,
    }


def _allocate_stream_budgets(
    stdout: str,
    stderr: str,
    *,
    total_limit: int,
) -> tuple[int, int]:
    label_overhead = len("STDOUT:\n") + len("\nSTDERR:\n")
    budget = max(0, int(total_limit) - label_overhead)
    if len(stdout) + len(stderr) <= budget:
        return len(stdout), len(stderr)
    total_len = max(1, len(stdout) + len(stderr))
    stdout_budget = int(budget * (len(stdout) / total_len))
    stderr_budget = budget - stdout_budget
    return stdout_budget, stderr_budget


def spill_command_streams(
    *,
    tool_name: str,
    workspace: Path,
    stdout: str,
    stderr: str,
    total_limit: int,
) -> dict[str, Any]:
    stdout = stdout or ""
    stderr = stderr or ""
    stdout_budget, stderr_budget = _allocate_stream_budgets(
        stdout,
        stderr,
        total_limit=total_limit,
    )
    payload: dict[str, Any] = {"stdout": stdout, "stderr": stderr}

    if len(stdout) > stdout_budget:
        artifact_path, saved_to = _artifact_path(
            workspace=workspace,
            tool_name=tool_name,
            label="stdout",
            suffix=".txt",
        )
        artifact_path.write_text(stdout, encoding="utf-8")
        payload["stdout_saved_to"] = saved_to
        payload["stdout"] = _preview_text(
            stdout,
            limit=stdout_budget,
            saved_to=saved_to,
            label="stdout",
        )

    if len(stderr) > stderr_budget:
        artifact_path, saved_to = _artifact_path(
            workspace=workspace,
            tool_name=tool_name,
            label="stderr",
            suffix=".txt",
        )
        artifact_path.write_text(stderr, encoding="utf-8")
        payload["stderr_saved_to"] = saved_to
        payload["stderr"] = _preview_text(
            stderr,
            limit=stderr_budget,
            saved_to=saved_to,
            label="stderr",
        )

    return payload
