from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from .ulid import new_ulid

EventType = Literal[
    "state_change",
    "log",
    "final_output",
    "progress",
    "tool",
    "artifact_created",
    "artifact_updated",
    "artifact_deleted",
    "node",
    "error",
]

EventLevel = Literal["debug", "info", "warn", "error"]


class Event(TypedDict):
    event_id: str
    ts: str
    run_id: str
    agent_id: str
    seq: int
    type: EventType
    payload: dict[str, Any]
    level: EventLevel
    tags: list[str]


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_event(
    *,
    run_id: str,
    agent_id: str,
    seq: int,
    type: EventType,
    payload: dict[str, Any],
    level: EventLevel = "info",
    tags: list[str] | None = None,
) -> Event:
    return {
        "event_id": new_ulid(),
        "ts": utc_now_rfc3339(),
        "run_id": run_id,
        "agent_id": agent_id,
        "seq": seq,
        "type": type,
        "payload": payload,
        "level": level,
        "tags": tags or [],
    }
