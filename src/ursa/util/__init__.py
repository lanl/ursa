import sqlite3
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_URSA_SERDE_MODULES = [
    ("ursa.agents.planning_agent", "Plan"),
    ("ursa.agents.planning_agent", "PlanStep"),
    ("ursa.agents.execution_agent", "ReviewAssessment"),
]


def _serde() -> JsonPlusSerializer:
    """Serializer allowing the typed state models persisted by URSA agents."""
    return JsonPlusSerializer(allowed_msgpack_modules=_URSA_SERDE_MODULES)


class Checkpointer:
    @classmethod
    def from_workspace(
        cls,
        workspace: Path,
        db_dir: str = "db",
        db_name: str = "checkpointer.db",
    ) -> SqliteSaver:
        (db_path := workspace / db_dir).mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path / db_name), check_same_thread=False)
        return SqliteSaver(conn, serde=_serde())

    @classmethod
    async def async_from_workspace(
        cls,
        workspace: Path,
        db_dir: str = "db",
        db_name: str = "checkpointer.db",
    ) -> AsyncSqliteSaver:
        """Make an async SQLite checkpointer under a workspace directory."""
        (db_path := workspace / db_dir).mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path / db_name))
        return AsyncSqliteSaver(conn, serde=_serde())

    @classmethod
    def from_path(
        cls, db_path: Path, db_name: str = "checkpointer.db"
    ) -> SqliteSaver:
        """Make checkpointer sqlite db.

        Args
        ====
        * db_path: The path to the SQLite database file (e.g. ./checkpoint.db) to be created.
        """

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path / db_name), check_same_thread=False)
        return SqliteSaver(conn, serde=_serde())

    @classmethod
    async def async_from_path(
        cls, db_path: Path, db_name: str = "checkpointer.db"
    ) -> AsyncSqliteSaver:
        """Make an async SQLite checkpointer under a database path."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path / db_name))
        return AsyncSqliteSaver(conn, serde=_serde())
