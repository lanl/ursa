"""Durable card journal and indexed, bounded conversation reads.

The journal is append-only. Heads are a transactional projection, so opening a
viewport never replays events, scans hidden revisions, or uses SQL OFFSET.
Card content (including resource references) belongs exclusively in JSON.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class CardRecord:
    id: int
    position: int
    turn_id: int
    key: str
    card: dict[str, Any]
    view_state: dict[str, Any] = field(default_factory=dict)


class CardStore:
    def __init__(self, path: Path | str, session: str = "ursa") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            path, timeout=30, check_same_thread=False
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.session = session
        self.lock = RLock()
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS event_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                record_type TEXT NOT NULL,
                card_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_card_heads (
                session TEXT NOT NULL,
                position INTEGER NOT NULL,
                card_id INTEGER NOT NULL REFERENCES event_cards(id),
                turn_id INTEGER NOT NULL,
                card_key TEXT NOT NULL,
                PRIMARY KEY (session, position),
                UNIQUE (session, turn_id, card_key)
            );
            CREATE TABLE IF NOT EXISTS event_card_links (
                session TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                card_key TEXT NOT NULL,
                PRIMARY KEY (session, turn_id, alias)
            );
            CREATE INDEX IF NOT EXISTS event_cards_members
                ON event_cards(session, position, id);
            CREATE INDEX IF NOT EXISTS event_card_keys
                ON event_card_heads(session, card_key, position);
            CREATE TABLE IF NOT EXISTS event_card_state (
                session TEXT PRIMARY KEY,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_card_view_state (
                session TEXT NOT NULL,
                position INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (session, position)
            );
        """)

    @contextmanager
    def transaction(self):
        with self.lock, self.connection:
            yield

    def append(
        self,
        turn_id: int,
        key: str,
        card: dict[str, Any],
        *,
        record_type: str = "card",
        visible: bool = True,
    ) -> int:
        """Append a serialized card and atomically advance its head.

        Call inside transaction() when publishing several related records.
        Invisible member records are retained for bundle inspection; a commit
        carries a self-contained renderable snapshot and member IDs in JSON.
        """
        old = (
            self.connection.execute(
                "SELECT position FROM event_card_heads "
                "WHERE session=? AND turn_id=? AND card_key=?",
                (self.session, turn_id, key),
            ).fetchone()
            if visible
            else None
        )
        position = old[0] if old else 0
        cursor = self.connection.execute(
            "INSERT INTO event_cards "
            "(session, turn_id, position, record_type, card_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                self.session,
                turn_id,
                position,
                record_type,
                json.dumps(card, ensure_ascii=False, default=str),
            ),
        )
        card_id = cursor.lastrowid
        assert card_id is not None
        position = position or card_id
        self.connection.execute(
            "UPDATE event_cards SET position=? WHERE id=?",
            (position, card_id),
        )
        if visible:
            self.connection.execute(
                "INSERT INTO event_card_heads VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session, position) DO UPDATE SET "
                "card_id=excluded.card_id",
                (self.session, position, card_id, turn_id, key),
            )
        return card_id

    def link(self, turn_id: int, alias: str, key: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO event_card_links VALUES (?, ?, ?, ?)",
            (self.session, turn_id, alias, key),
        )

    def start_turn(self, card: dict[str, Any], key: str = "user") -> int:
        """Use the first entry's journal ID as a globally unique turn ID."""
        with self.transaction():
            turn_id = self.append(0, key, card)
            self.connection.execute(
                "UPDATE event_cards SET turn_id=? WHERE id=?",
                (turn_id, turn_id),
            )
            self.connection.execute(
                "UPDATE event_card_heads SET turn_id=? "
                "WHERE session=? AND position=?",
                (turn_id, self.session, turn_id),
            )
        return turn_id

    def find(self, turn_id: int, key: str) -> CardRecord | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT e.id, h.position, h.turn_id, h.card_key, e.card_json "
                "FROM event_card_heads h JOIN event_cards e ON e.id=h.card_id "
                "WHERE h.session=? AND h.turn_id=? AND h.card_key=?",
                (self.session, turn_id, key),
            ).fetchone()
            return self._record(row) if row else None

    def linked(self, turn_id: int, alias: str) -> CardRecord | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT card_key FROM event_card_links "
                "WHERE session=? AND turn_id=? AND alias=?",
                (self.session, turn_id, alias),
            ).fetchone()
            return self.find(turn_id, row[0]) if row else None

    @staticmethod
    def _record(row) -> CardRecord:
        return CardRecord(*row[:4], json.loads(row[4]))

    def window(
        self,
        limit: int,
        *,
        before: int | None = None,
        after: int | None = None,
        floor: int = 0,
        exclude_turn_ids: Collection[int] = (),
    ) -> list[CardRecord]:
        if limit < 1:
            return []
        if before is not None and after is not None:
            raise ValueError("Specify only one cursor")
        where = "h.session=? AND h.position>?"
        # Give SQLite one lower bound. Two parameterized bounds on the same
        # column can make it seek to floor and filter all rows up to after.
        args: list[Any] = [self.session, max(floor, after or 0)]
        excluded = tuple(exclude_turn_ids)
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            where += f" AND h.turn_id NOT IN ({placeholders})"
            args.extend(excluded)
        if before is not None:
            where += " AND h.position<?"
            args.append(before)
        order = "ASC" if after is not None else "DESC"
        with self.lock:
            rows = self.connection.execute(
                "SELECT e.id, h.position, h.turn_id, h.card_key, e.card_json, "
                "v.state_json "
                "FROM event_card_heads h JOIN event_cards e ON e.id=h.card_id "
                "LEFT JOIN event_card_view_state v "
                "ON v.session=h.session AND v.position=h.position "
                f"WHERE {where} ORDER BY h.position {order} LIMIT ?",
                (*args, limit),
            ).fetchall()
        if after is None:
            rows.reverse()
        return [
            CardRecord(*row[:4], json.loads(row[4]), json.loads(row[5] or "{}"))
            for row in rows
        ]

    def state(self) -> dict[str, Any]:
        with self.lock:
            row = self.connection.execute(
                "SELECT state_json FROM event_card_state WHERE session=?",
                (self.session,),
            ).fetchone()
            return json.loads(row[0]) if row else {}

    def prompts(self, limit: int = 100) -> list[str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT e.card_json FROM event_card_heads h "
                "JOIN event_cards e ON e.id=h.card_id "
                "WHERE h.session=? AND h.card_key='user' "
                "ORDER BY h.position DESC LIMIT ?",
                (self.session, limit),
            ).fetchall()
        return [json.loads(row[0])["content"] for row in reversed(rows)]

    def save_view_state(self, position: int, **state: Any) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT OR REPLACE INTO event_card_view_state VALUES (?, ?, ?)",
                (self.session, position, json.dumps(state)),
            )

    def save_state(self, **changes: Any) -> None:
        with self.transaction():
            state = self.state()
            state.update(changes)
            self.connection.execute(
                "INSERT OR REPLACE INTO event_card_state VALUES (?, ?)",
                (self.session, json.dumps(state)),
            )

    def close(self) -> None:
        with self.lock:
            self.connection.close()
