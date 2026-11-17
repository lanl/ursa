from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ursa.util.secrets import get_keyring as get_keyring

_URSA_SERDE_MODULES = [
    ("ursa.agents.planning_agent", "Plan"),
    ("ursa.agents.planning_agent", "PlanStep"),
    ("ursa.agents.execution_agent", "ReviewAssessment"),
    ("ursa.util.optimization_schema", "DecisionVariableType"),
    ("ursa.util.optimization_schema", "ParameterType"),
    ("ursa.util.optimization_schema", "ObjectiveType"),
    ("ursa.util.optimization_schema", "ConstraintType"),
    ("ursa.util.optimization_schema", "NotesType"),
    ("ursa.util.optimization_schema", "ProblemSpec"),
    ("ursa.util.optimization_schema", "SolverSpec"),
]


def _serde(
    allowed_msgpack_modules: Sequence[tuple[str, str] | type] = (),
) -> JsonPlusSerializer:
    """Serializer allowing the typed state models persisted by URSA agents."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[*_URSA_SERDE_MODULES, *allowed_msgpack_modules]
    )


class Checkpointer:
    @classmethod
    def from_workspace(
        cls,
        workspace: Path,
        db_dir: str = "db",
        db_name: str = "checkpointer.db",
        *,
        allowed_msgpack_modules: Sequence[tuple[str, str] | type] = (),
    ) -> SqliteSaver:
        from langgraph.checkpoint.sqlite import SqliteSaver

        (db_path := workspace / db_dir).mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path / db_name), check_same_thread=False)
        return SqliteSaver(conn, serde=_serde(allowed_msgpack_modules))

    @classmethod
    async def async_from_workspace(
        cls,
        workspace: Path,
        db_dir: str = "db",
        db_name: str = "checkpointer.db",
        *,
        allowed_msgpack_modules: Sequence[tuple[str, str] | type] = (),
    ) -> AsyncSqliteSaver:
        """Make an async SQLite checkpointer under a workspace directory."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        (db_path := workspace / db_dir).mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path / db_name))
        return AsyncSqliteSaver(conn, serde=_serde(allowed_msgpack_modules))

    @classmethod
    def from_path(
        cls,
        db_path: Path,
        db_name: str = "checkpointer.db",
        *,
        allowed_msgpack_modules: Sequence[tuple[str, str] | type] = (),
    ) -> SqliteSaver:
        """Make checkpointer sqlite db.

        Args
        ====
        * db_path: The path to the SQLite database file (e.g. ./checkpoint.db) to be created.
        """
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path / db_name), check_same_thread=False)
        return SqliteSaver(conn, serde=_serde(allowed_msgpack_modules))

    @classmethod
    async def async_from_path(
        cls,
        db_path: Path,
        db_name: str = "checkpointer.db",
        *,
        allowed_msgpack_modules: Sequence[tuple[str, str] | type] = (),
    ) -> AsyncSqliteSaver:
        """Make an async SQLite checkpointer under a database path."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path / db_name))
        return AsyncSqliteSaver(conn, serde=_serde(allowed_msgpack_modules))
