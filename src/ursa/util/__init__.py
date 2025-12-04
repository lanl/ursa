import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


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
        return SqliteSaver(conn)

    @classmethod
    def from_path(cls, db_path: Path) -> SqliteSaver:
        """Make checkpointer sqlite db.

        Args
        ====
        * db_path: The path to the SQLite database file (e.g. ./checkpoint.db) to be created.
        """

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
