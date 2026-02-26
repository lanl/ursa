from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .storage import append_jsonl, read_json, utc_now, write_json
from .ulid import new_ulid


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    meta_path: Path
    messages_path: Path
    workspace_dir: Path


_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _validate_session_id(session_id: str) -> str:
    sid = str(session_id)
    if not _ULID_RE.match(sid):
        raise ValueError("Invalid session_id")
    return sid


def session_paths(workspace_root: Path, session_id: str) -> SessionPaths:
    session_id = _validate_session_id(session_id)
    base = workspace_root / "sessions" / session_id
    return SessionPaths(
        session_dir=base,
        meta_path=base / "session.json",
        messages_path=base / "messages.jsonl",
        workspace_dir=base / "workspace",
    )


def create_session(
    workspace_root: Path, *, agent_id: str, title: str | None = None
) -> dict[str, Any]:
    session_id = new_ulid()
    paths = session_paths(workspace_root, session_id)
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)

    now = utc_now()
    rec: dict[str, Any] = {
        "session_id": session_id,
        "agent_id": agent_id,
        "title": title or f"{agent_id} session",
        "created_at": now,
        "updated_at": now,
        "active_run_id": None,
        "last_run_id": None,
    }
    write_json(paths.meta_path, rec)
    return rec


def read_session(workspace_root: Path, session_id: str) -> dict[str, Any]:
    paths = session_paths(workspace_root, session_id)
    return read_json(paths.meta_path)


def update_session(
    workspace_root: Path, session_id: str, patch: dict[str, Any]
) -> dict[str, Any]:
    paths = session_paths(workspace_root, session_id)
    rec = read_json(paths.meta_path)
    rec.update(patch)
    rec["updated_at"] = utc_now()
    write_json(paths.meta_path, rec)
    return rec


def list_sessions(
    workspace_root: Path, *, limit: int = 50, agent_id: str | None = None
) -> list[dict[str, Any]]:
    root = workspace_root / "sessions"
    if not root.exists():
        return []
    recs: list[dict[str, Any]] = []
    for p in root.glob("*/session.json"):
        try:
            rec = read_json(p)
        except Exception:
            continue
        if agent_id and rec.get("agent_id") != agent_id:
            continue
        recs.append(rec)
    recs.sort(
        key=lambda r: r.get("updated_at") or r.get("created_at") or "",
        reverse=True,
    )
    return recs[:limit]


def delete_session(workspace_root: Path, session_id: str) -> None:
    """Delete the session directory (messages + per-session workspace).

    Note: does not delete global run records; runs are stored separately.
    """

    paths = session_paths(workspace_root, session_id)
    if not paths.session_dir.exists():
        raise FileNotFoundError(paths.session_dir)
    # Safety: only delete under workspace_root/sessions
    sessions_root = (workspace_root / "sessions").resolve()
    sess_real = paths.session_dir.resolve()
    try:
        sess_real.relative_to(sessions_root)
    except Exception:
        raise ValueError("Refusing to delete outside sessions root")

    shutil.rmtree(sess_real)


def append_message(
    workspace_root: Path,
    *,
    session_id: str,
    role: str,
    text: str,
    run_id: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    paths = session_paths(workspace_root, session_id)
    msg = {
        "message_id": message_id or new_ulid(),
        "ts": utc_now(),
        "role": role,
        "text": text,
        "run_id": run_id,
    }
    append_jsonl(paths.messages_path, msg)
    update_session(workspace_root, session_id, {})
    return msg


def read_messages(
    workspace_root: Path, session_id: str, *, limit: int = 200
) -> list[dict[str, Any]]:
    paths = session_paths(workspace_root, session_id)
    if not paths.messages_path.exists():
        return []
    msgs: list[dict[str, Any]] = []
    with paths.messages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except Exception:
                continue
    return msgs[-limit:]


def build_prompt_from_messages(
    messages: Iterable[dict[str, Any]],
    *,
    new_user_text: str,
    max_chars: int = 12000,
    include_assistant: bool = True,
) -> str:
    """Generic prompt builder to approximate a conversational session.

    This is intentionally simple and model-agnostic. It formats the last part of
    the transcript into a single prompt string.
    """

    # Collect lines, then trim from the front to max_chars.
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role") or "")
        txt = str(m.get("text") or "")
        if role == "assistant" and not include_assistant:
            continue
        if role not in {"user", "assistant", "system"}:
            continue
        prefix = (
            "User"
            if role == "user"
            else ("Assistant" if role == "assistant" else "System")
        )
        lines.append(f"{prefix}: {txt}")

    lines.append(f"User: {new_user_text}")
    rendered = "\n\n".join(lines)

    if len(rendered) <= max_chars:
        return rendered

    # Trim from front.
    rendered = rendered[-max_chars:]
    # Avoid starting mid-character; try to cut to next blank line.
    cut = rendered.find("\n\n")
    if 0 <= cut < 2000:
        rendered = rendered[cut + 2 :]
    return rendered
