import sqlite3
from contextlib import closing

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.cli import agent_management
from ursa.cli.agent_management import import_agent


def _write_checkpoint_db(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        SqliteSaver(conn).setup()
        # Keep the original rollback-journal fixture used by the import tests.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(
            """
            INSERT INTO checkpoints (
                thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                type, checkpoint, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-thread",
                "",
                "checkpoint-1",
                None,
                "msgpack",
                b"checkpoint",
                b"metadata",
            ),
        )
        conn.execute(
            """
            INSERT INTO writes (
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                channel, type, value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "another-thread",
                "",
                "checkpoint-1",
                "task-1",
                0,
                "messages",
                "msgpack",
                b"value",
            ),
        )


@pytest.fixture
def group_dir(monkeypatch, tmp_path):
    root = tmp_path / "ursa"
    group_root = root / "science"
    group_root.mkdir(parents=True)
    monkeypatch.setattr(agent_management, "URSA_CACHE_DIR", root)
    return group_root / "agents"


def test_import_agent_db_creates_agent_checkpoint_and_rewrites_thread_ids(
    group_dir, tmp_path
):
    source_db = tmp_path / "execute.db"
    _write_checkpoint_db(source_db)

    import_agent(source_db, group_name="science", agent_name="converted_agent")

    imported_db = group_dir / "converted_agent" / "db" / "checkpointer.db"
    assert imported_db.is_file()

    with closing(sqlite3.connect(imported_db)) as conn:
        checkpoint_threads = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
        write_threads = conn.execute(
            "SELECT DISTINCT thread_id FROM writes"
        ).fetchall()

    assert checkpoint_threads == [("ursa",)]
    assert write_threads == [("ursa",)]

    with closing(sqlite3.connect(source_db)) as source_conn:
        original_threads = source_conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
    assert original_threads == [("old-thread",)]


def test_import_agent_db_requires_name(group_dir, tmp_path):
    source_db = tmp_path / "execute.db"
    _write_checkpoint_db(source_db)

    with pytest.raises(ValueError, match="requires --name"):
        import_agent(source_db, group_name="science")

    assert not any(group_dir.iterdir())


def test_import_agent_db_cleans_up_destination_on_invalid_database(
    group_dir, tmp_path
):
    source_db = tmp_path / "invalid.db"
    source_db.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid checkpoint database"):
        import_agent(source_db, group_name="science", agent_name="broken")

    assert not (group_dir / "broken").exists()
