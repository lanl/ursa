"""Retention helpers for URSA-managed LangGraph SQLite checkpoints."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

CHECKPOINT_PRUNE_THRESHOLD_BYTES = 25 * 1024 * 1024
_KEEP_TABLE = "_ursa_checkpoint_retention_keep"


@dataclass(frozen=True)
class CheckpointPruneResult:
    """Summary of a checkpoint-retention attempt."""

    pruned: bool
    checkpoints_deleted: int = 0
    writes_deleted: int = 0
    size_before: int = 0
    size_after: int = 0
    compaction_error: str | None = None


def is_terminal_checkpoint(snapshot: Any) -> bool:
    """Return whether a LangGraph state snapshot is safe to prune behind."""
    if snapshot is None or tuple(getattr(snapshot, "next", ()) or ()):
        return False

    for task in tuple(getattr(snapshot, "tasks", ()) or ()):
        if getattr(task, "error", None) is not None:
            return False
        if tuple(getattr(task, "interrupts", ()) or ()):
            return False
    return True


def _database_path_from_rows(rows: list[tuple[Any, ...]]) -> Path | None:
    for _, name, filename in rows:
        if name == "main" and filename:
            return Path(filename)
    return None


def _database_family_size(path: Path | None) -> int:
    if path is None:
        return 0
    return sum(
        candidate.stat().st_size
        for candidate in (
            path,
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if candidate.exists()
    )


def _has_delta_channel_history(
    conn: sqlite3.Connection, thread_id: str
) -> bool:
    return (
        conn.execute(
            """
            SELECT 1
            FROM checkpoints
            WHERE thread_id = ?
              AND json_type(
                    CAST(metadata AS TEXT),
                    '$.counters_since_delta_snapshot'
                  ) IS NOT NULL
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
        is not None
    )


def _prune_rows(conn: sqlite3.Connection, thread_id: str) -> tuple[int, int]:
    before_checkpoints = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    before_writes = conn.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]

    conn.execute(
        f"""
        CREATE TEMP TABLE IF NOT EXISTS {_KEEP_TABLE} (
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            PRIMARY KEY (checkpoint_ns, checkpoint_id)
        )
        """
    )
    conn.execute(f"DELETE FROM {_KEEP_TABLE}")
    conn.execute(
        f"""
        INSERT INTO {_KEEP_TABLE} (checkpoint_ns, checkpoint_id)
        SELECT checkpoint_ns, MAX(checkpoint_id)
        FROM checkpoints
        WHERE thread_id = ?
        GROUP BY checkpoint_ns
        """,
        (thread_id,),
    )
    conn.execute(
        f"""
        DELETE FROM writes
        WHERE thread_id = ?
          AND NOT EXISTS (
              SELECT 1
              FROM {_KEEP_TABLE} AS keep
              WHERE keep.checkpoint_ns = writes.checkpoint_ns
                AND keep.checkpoint_id = writes.checkpoint_id
          )
        """,
        (thread_id,),
    )
    conn.execute(
        f"""
        DELETE FROM checkpoints
        WHERE thread_id = ?
          AND NOT EXISTS (
              SELECT 1
              FROM {_KEEP_TABLE} AS keep
              WHERE keep.checkpoint_ns = checkpoints.checkpoint_ns
                AND keep.checkpoint_id = checkpoints.checkpoint_id
          )
        """,
        (thread_id,),
    )
    conn.execute(
        """
        UPDATE checkpoints
        SET parent_checkpoint_id = NULL
        WHERE thread_id = ?
        """,
        (thread_id,),
    )
    conn.execute(f"DROP TABLE {_KEEP_TABLE}")

    after_checkpoints = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    after_writes = conn.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    return (
        before_checkpoints - after_checkpoints,
        before_writes - after_writes,
    )


def prune_sqlite_checkpoints(
    saver: SqliteSaver,
    thread_id: str,
    *,
    threshold_bytes: int = CHECKPOINT_PRUNE_THRESHOLD_BYTES,
) -> CheckpointPruneResult:
    """Keep only the latest checkpoint per namespace and compact SQLite."""
    saver.setup()
    with saver.lock:
        path = _database_path_from_rows(
            list(saver.conn.execute("PRAGMA database_list"))
        )
        size_before = _database_family_size(path)
        if size_before <= threshold_bytes:
            return CheckpointPruneResult(False, size_before=size_before)
        if _has_delta_channel_history(saver.conn, str(thread_id)):
            logger.warning(
                "Skipping checkpoint pruning for thread %s because it uses "
                "DeltaChannel history",
                thread_id,
            )
            return CheckpointPruneResult(False, size_before=size_before)

        saver.conn.execute("BEGIN IMMEDIATE")
        try:
            checkpoints_deleted, writes_deleted = _prune_rows(
                saver.conn, str(thread_id)
            )
            saver.conn.commit()
        except BaseException:
            saver.conn.rollback()
            raise

        compaction_error: str | None = None
        try:
            cursor = saver.conn.execute("VACUUM")
            cursor.close()
            cursor = saver.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            cursor.fetchall()
            cursor.close()
        except sqlite3.Error as exc:
            # The row pruning is already committed and its free pages remain
            # reusable even if physical compaction cannot run right now.
            compaction_error = str(exc)
            logger.warning(
                "Checkpoint rows were pruned, but SQLite compaction failed: %s",
                exc,
            )

        return CheckpointPruneResult(
            True,
            checkpoints_deleted=checkpoints_deleted,
            writes_deleted=writes_deleted,
            size_before=size_before,
            size_after=_database_family_size(path),
            compaction_error=compaction_error,
        )


async def _ahas_delta_channel_history(
    saver: AsyncSqliteSaver, thread_id: str
) -> bool:
    cursor = await saver.conn.execute(
        """
        SELECT 1
        FROM checkpoints
        WHERE thread_id = ?
          AND json_type(
                CAST(metadata AS TEXT),
                '$.counters_since_delta_snapshot'
              ) IS NOT NULL
        LIMIT 1
        """,
        (thread_id,),
    )
    try:
        return await cursor.fetchone() is not None
    finally:
        await cursor.close()


async def _acount(saver: AsyncSqliteSaver, table: str, thread_id: str) -> int:
    cursor = await saver.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (thread_id,)
    )
    try:
        row = await cursor.fetchone()
        return int(row[0])
    finally:
        await cursor.close()


async def _aprune_rows(
    saver: AsyncSqliteSaver, thread_id: str
) -> tuple[int, int]:
    before_checkpoints = await _acount(saver, "checkpoints", thread_id)
    before_writes = await _acount(saver, "writes", thread_id)

    await saver.conn.execute(
        f"""
        CREATE TEMP TABLE IF NOT EXISTS {_KEEP_TABLE} (
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            PRIMARY KEY (checkpoint_ns, checkpoint_id)
        )
        """
    )
    await saver.conn.execute(f"DELETE FROM {_KEEP_TABLE}")
    await saver.conn.execute(
        f"""
        INSERT INTO {_KEEP_TABLE} (checkpoint_ns, checkpoint_id)
        SELECT checkpoint_ns, MAX(checkpoint_id)
        FROM checkpoints
        WHERE thread_id = ?
        GROUP BY checkpoint_ns
        """,
        (thread_id,),
    )
    await saver.conn.execute(
        f"""
        DELETE FROM writes
        WHERE thread_id = ?
          AND NOT EXISTS (
              SELECT 1
              FROM {_KEEP_TABLE} AS keep
              WHERE keep.checkpoint_ns = writes.checkpoint_ns
                AND keep.checkpoint_id = writes.checkpoint_id
          )
        """,
        (thread_id,),
    )
    await saver.conn.execute(
        f"""
        DELETE FROM checkpoints
        WHERE thread_id = ?
          AND NOT EXISTS (
              SELECT 1
              FROM {_KEEP_TABLE} AS keep
              WHERE keep.checkpoint_ns = checkpoints.checkpoint_ns
                AND keep.checkpoint_id = checkpoints.checkpoint_id
          )
        """,
        (thread_id,),
    )
    await saver.conn.execute(
        """
        UPDATE checkpoints
        SET parent_checkpoint_id = NULL
        WHERE thread_id = ?
        """,
        (thread_id,),
    )
    await saver.conn.execute(f"DROP TABLE {_KEEP_TABLE}")

    after_checkpoints = await _acount(saver, "checkpoints", thread_id)
    after_writes = await _acount(saver, "writes", thread_id)
    return (
        before_checkpoints - after_checkpoints,
        before_writes - after_writes,
    )


async def aprune_sqlite_checkpoints(
    saver: AsyncSqliteSaver,
    thread_id: str,
    *,
    threshold_bytes: int = CHECKPOINT_PRUNE_THRESHOLD_BYTES,
) -> CheckpointPruneResult:
    """Async equivalent of :func:`prune_sqlite_checkpoints`."""
    # setup() takes the same non-reentrant lock, so it must run first.
    await saver.setup()
    async with saver.lock:
        cursor = await saver.conn.execute("PRAGMA database_list")
        try:
            path = _database_path_from_rows(await cursor.fetchall())
        finally:
            await cursor.close()
        size_before = _database_family_size(path)
        if size_before <= threshold_bytes:
            return CheckpointPruneResult(False, size_before=size_before)
        if await _ahas_delta_channel_history(saver, str(thread_id)):
            logger.warning(
                "Skipping checkpoint pruning for thread %s because it uses "
                "DeltaChannel history",
                thread_id,
            )
            return CheckpointPruneResult(False, size_before=size_before)

        await saver.conn.execute("BEGIN IMMEDIATE")
        try:
            checkpoints_deleted, writes_deleted = await _aprune_rows(
                saver, str(thread_id)
            )
            await saver.conn.commit()
        except BaseException:
            await saver.conn.rollback()
            raise

        compaction_error: str | None = None
        try:
            cursor = await saver.conn.execute("VACUUM")
            await cursor.close()
            cursor = await saver.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            await cursor.fetchall()
            await cursor.close()
        except sqlite3.Error as exc:
            compaction_error = str(exc)
            logger.warning(
                "Checkpoint rows were pruned, but SQLite compaction failed: %s",
                exc,
            )

        return CheckpointPruneResult(
            True,
            checkpoints_deleted=checkpoints_deleted,
            writes_deleted=writes_deleted,
            size_before=size_before,
            size_after=_database_family_size(path),
            compaction_error=compaction_error,
        )
