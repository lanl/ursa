import operator
import sqlite3
from types import SimpleNamespace
from typing import Annotated, TypedDict

import aiosqlite
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from ursa.util.checkpoint_retention import (
    aprune_sqlite_checkpoints,
    is_terminal_checkpoint,
    prune_sqlite_checkpoints,
)


class RetentionState(TypedDict):
    values: Annotated[list[str], operator.add]


def _graph(checkpointer):
    builder = StateGraph(RetentionState)
    builder.add_node("first", lambda _: {"values": ["first"]})
    builder.add_node("second", lambda _: {"values": ["second"]})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=checkpointer)


def _count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_terminal_checkpoint_requires_no_pending_or_failed_tasks():
    assert is_terminal_checkpoint(SimpleNamespace(next=(), tasks=()))
    assert not is_terminal_checkpoint(
        SimpleNamespace(next=("resume",), tasks=())
    )
    assert not is_terminal_checkpoint(
        SimpleNamespace(
            next=(),
            tasks=(SimpleNamespace(error="failed", interrupts=()),),
        )
    )
    assert not is_terminal_checkpoint(
        SimpleNamespace(
            next=(),
            tasks=(SimpleNamespace(error=None, interrupts=("review",)),),
        )
    )


def test_sync_pruning_retains_latest_checkpoint_and_state(tmp_path):
    db_path = tmp_path / "sync.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = _graph(saver)
    config = {"configurable": {"thread_id": "ursa"}}

    graph.invoke({"values": ["input-1"]}, config)
    graph.invoke({"values": ["input-2"]}, config)
    expected = graph.get_state(config).values
    assert _count(conn, "checkpoints") > 1

    result = prune_sqlite_checkpoints(saver, "ursa", threshold_bytes=0)

    assert result.pruned
    assert result.checkpoints_deleted > 0
    assert _count(conn, "checkpoints") == 1
    assert (
        conn.execute("SELECT parent_checkpoint_id FROM checkpoints").fetchone()[
            0
        ]
        is None
    )
    assert (
        conn.execute(
            """
        SELECT COUNT(*)
        FROM writes AS w
        LEFT JOIN checkpoints AS c
          ON c.thread_id = w.thread_id
         AND c.checkpoint_ns = w.checkpoint_ns
         AND c.checkpoint_id = w.checkpoint_id
        WHERE c.checkpoint_id IS NULL
        """
        ).fetchone()[0]
        == 0
    )
    assert graph.get_state(config).values == expected

    resumed = graph.invoke({"values": ["input-3"]}, config)
    assert resumed["values"][-3:] == ["input-3", "first", "second"]
    conn.close()


def test_pruning_skips_database_at_or_below_threshold(tmp_path):
    db_path = tmp_path / "small.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = _graph(saver)
    config = {"configurable": {"thread_id": "ursa"}}
    graph.invoke({"values": ["input"]}, config)
    before = _count(conn, "checkpoints")

    result = prune_sqlite_checkpoints(
        saver, "ursa", threshold_bytes=1024 * 1024 * 1024
    )

    assert not result.pruned
    assert _count(conn, "checkpoints") == before
    conn.close()


def test_pruning_skips_delta_channel_history(tmp_path):
    db_path = tmp_path / "delta.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = _graph(saver)
    config = {"configurable": {"thread_id": "ursa"}}
    graph.invoke({"values": ["input"]}, config)
    before = _count(conn, "checkpoints")
    conn.execute(
        "UPDATE checkpoints SET metadata = ?",
        (b'{"counters_since_delta_snapshot": {"values": 1}}',),
    )
    conn.commit()

    result = prune_sqlite_checkpoints(saver, "ursa", threshold_bytes=0)

    assert not result.pruned
    assert _count(conn, "checkpoints") == before
    conn.close()


@pytest.mark.asyncio
async def test_async_pruning_retains_latest_checkpoint_and_state(tmp_path):
    db_path = tmp_path / "async.db"
    conn = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(conn)
    graph = _graph(saver)
    config = {"configurable": {"thread_id": "ursa"}}

    await graph.ainvoke({"values": ["input-1"]}, config)
    await graph.ainvoke({"values": ["input-2"]}, config)
    expected = (await graph.aget_state(config)).values

    result = await aprune_sqlite_checkpoints(saver, "ursa", threshold_bytes=0)

    assert result.pruned
    cursor = await conn.execute("SELECT COUNT(*) FROM checkpoints")
    assert (await cursor.fetchone())[0] == 1
    await cursor.close()
    assert (await graph.aget_state(config)).values == expected

    resumed = await graph.ainvoke({"values": ["input-3"]}, config)
    assert resumed["values"][-3:] == ["input-3", "first", "second"]
    await conn.close()
