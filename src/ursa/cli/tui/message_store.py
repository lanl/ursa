"""Disk-backed message history and lazy coordination-tool details.

Only bounded previews are returned for lists. Full bodies are fetched explicitly
when a card is expanded. SQLite work runs off the Textual event loop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
COORDINATION_TOOLS = frozenset({
    "send_message",
    "wait_for",
    "spawn_agent",
    "stop_agent",
})


def preview_text(content: str, limit: int = 240) -> str:
    text = " ".join(content[: limit * 4].split())
    return text[:limit] + ("…" if len(content) > limit else "")


@dataclass(frozen=True)
class HierarchyMessage:
    """Metadata and a bounded preview, never a retained full message body."""

    id: int
    sender: str
    recipients: tuple[str, ...]
    content: str
    timestamp: datetime


class MessageStore:
    """Use the agent's checkpoint database without sharing its connection."""

    def __init__(self, path: Path, thread_id: str) -> None:
        self.path = path
        self.thread_id = thread_id
        self._ready = False
        self._setup_lock = threading.Lock()

    def _run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            with self._setup_lock:
                if not self._ready:
                    db.execute("PRAGMA journal_mode=WAL")
                    db.executescript("""
                        CREATE TABLE IF NOT EXISTS hierarchy_messages (
                            id INTEGER PRIMARY KEY,
                            thread_id TEXT NOT NULL,
                            sender TEXT NOT NULL,
                            recipients TEXT NOT NULL,
                            content TEXT NOT NULL,
                            preview TEXT NOT NULL,
                            timestamp TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS hierarchy_message_thread
                            ON hierarchy_messages(thread_id, id);
                        CREATE TABLE IF NOT EXISTS coordination_actions (
                            id INTEGER PRIMARY KEY,
                            turn_id TEXT NOT NULL,
                            run_id TEXT NOT NULL,
                            tool TEXT NOT NULL,
                            target TEXT NOT NULL,
                            input_preview TEXT NOT NULL,
                            input TEXT NOT NULL,
                            output TEXT NOT NULL DEFAULT '',
                            preview TEXT NOT NULL DEFAULT '',
                            completed INTEGER NOT NULL DEFAULT 0,
                            failed INTEGER NOT NULL DEFAULT 0,
                            UNIQUE(turn_id, run_id)
                        );
                        CREATE INDEX IF NOT EXISTS coordination_action_order
                            ON coordination_actions(turn_id, id);
                        CREATE INDEX IF NOT EXISTS coordination_action_state
                            ON coordination_actions(turn_id, completed, failed);
                    """)
                    self._ready = True
            with db:
                return operation(db)

    async def append(
        self, sender: str, recipients: tuple[str, ...], content: str
    ) -> int:
        timestamp = datetime.now().astimezone().isoformat()
        return await asyncio.to_thread(
            self._run,
            lambda db: db.execute(
                "INSERT INTO hierarchy_messages "
                "(thread_id, sender, recipients, content, preview, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.thread_id,
                    sender,
                    json.dumps(recipients),
                    content,
                    preview_text(content),
                    timestamp,
                ),
            ).lastrowid,
        )

    async def messages(
        self,
        *,
        before: int | None = None,
        after: int | None = None,
        limit: int = 50,
    ) -> list[HierarchyMessage]:
        """Read one page of previews, with indexed cursor pagination."""

        def read(db: sqlite3.Connection) -> list[HierarchyMessage]:
            predicate = "thread_id = ?"
            args: list[Any] = [self.thread_id]
            if before is not None:
                predicate += " AND id < ?"
                args.append(before)
            if after is not None:
                predicate += " AND id > ?"
                args.append(after)
            args.append(limit)
            order = "ASC" if after is not None else "DESC"
            rows = db.execute(
                "SELECT id, sender, recipients, preview, timestamp "
                f"FROM hierarchy_messages WHERE {predicate} "
                f"ORDER BY id {order} LIMIT ?",
                args,
            ).fetchall()
            if after is None:
                rows.reverse()
            return [
                HierarchyMessage(
                    id=row["id"],
                    sender=row["sender"],
                    recipients=tuple(json.loads(row["recipients"])),
                    content=row["preview"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                )
                for row in rows
            ]

        return await asyncio.to_thread(self._run, read)

    async def message_body(self, message_id: int) -> str:
        return await asyncio.to_thread(
            self._run,
            lambda db: db.execute(
                "SELECT content FROM hierarchy_messages "
                "WHERE thread_id = ? AND id = ?",
                (self.thread_id, message_id),
            ).fetchone()[0],
        )

    async def record_action(
        self,
        turn_id: str,
        run_id: str,
        tool: str,
        tool_input: Mapping[str, Any],
        *,
        output: str = "",
        completed: bool = False,
        failed: bool = False,
    ) -> None:
        target = tool_input.get("name" if tool == "spawn_agent" else "target")
        if isinstance(target, list):
            target = ", ".join(str(item) for item in target)
        if tool == "spawn_agent" and tool_input.get("agent_type"):
            target = f"{target} ({tool_input['agent_type']})"

        def write(db: sqlite3.Connection) -> None:
            db.execute(
                "INSERT INTO coordination_actions "
                "(turn_id, run_id, tool, target, input_preview, input, output, preview, completed, failed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(turn_id, run_id) DO UPDATE SET "
                "output=excluded.output, preview=excluded.preview, "
                "completed=excluded.completed, failed=excluded.failed",
                (
                    turn_id,
                    run_id,
                    tool,
                    preview_text(str(target or "agents")),
                    json.dumps(
                        {
                            key: value
                            for key, value in tool_input.items()
                            if key
                            in {
                                "name",
                                "target",
                                "agent_type",
                                "timeout_seconds",
                                "wait_all",
                            }
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    json.dumps(tool_input, ensure_ascii=False, default=str),
                    output,
                    preview_text(output),
                    completed,
                    failed,
                ),
            )

        await asyncio.to_thread(self._run, write)

    async def action_summary(
        self, turn_id: str
    ) -> tuple[dict[str, int], list[dict]]:
        def read(db: sqlite3.Connection) -> tuple[dict[str, int], list[dict]]:
            counts = dict(
                db.execute(
                    "SELECT COUNT(*) AS total, COALESCE(SUM(completed=0), 0) AS running, "
                    "COALESCE(SUM(failed), 0) AS failed "
                    "FROM coordination_actions WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
            )
            rows = db.execute(
                "SELECT tool, target, preview, completed, failed "
                "FROM coordination_actions WHERE turn_id = ? "
                "ORDER BY id DESC LIMIT 2",
                (turn_id,),
            ).fetchall()
            return counts, [dict(row) for row in reversed(rows)]

        return await asyncio.to_thread(self._run, read)

    async def actions(
        self, turn_id: str, *, offset: int = 0, limit: int = 20
    ) -> list[dict]:
        """One page of action previews without message bodies or results."""
        return await asyncio.to_thread(
            self._run,
            lambda db: [
                dict(row)
                for row in db.execute(
                    "SELECT id, tool, input_preview, preview, completed, failed "
                    "FROM coordination_actions WHERE turn_id = ? "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (turn_id, limit, offset),
                ).fetchall()
            ],
        )

    async def action_body(
        self, turn_id: str, action_id: int
    ) -> tuple[dict[str, Any], str]:
        """Read full input/result only while an individual tool card is open."""

        def read(db: sqlite3.Connection) -> tuple[dict[str, Any], str]:
            row = db.execute(
                "SELECT input, output FROM coordination_actions WHERE turn_id = ? AND id = ?",
                (turn_id, action_id),
            ).fetchone()
            return json.loads(row["input"]), row["output"]

        return await asyncio.to_thread(self._run, read)
