"""
sqlite_mcp.py

A small, readable MCP server that exposes a few SQLite database tools.

This file is intentionally designed as a demo:
- simple enough to understand at a glance
- useful enough to support realistic agent workflows
- not trying to be a production-grade database service

Example agent flow this server supports:
1. Create a database
2. Create a table
3. Insert rows
4. Inspect schema / list tables
5. Run a read-only SELECT query

Notes:
- Databases are stored in a local "sqlite_data" directory next to this file.
- Database names are normalized and given a ".db" suffix automatically.
- The query tool is intentionally read-only and only allows SELECT / WITH queries.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server.
# json_response=True makes structured dict/list returns cleaner for clients.
mcp = FastMCP("SQLite Demo MCP", json_response=True)


# Store demo databases in a predictable directory next to this script.
DATA_DIR = Path(__file__).resolve().parent / "sqlite_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# A conservative identifier regex for table names and column names when we
# synthesize SQL ourselves. For demo purposes, this helps avoid obviously bad
# names while keeping the code easy to follow.
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_db_name(database_name: str) -> str:
    """
    Normalize a user-provided database name into a simple filename.

    Examples:
    - "foo" -> "foo.db"
    - "foo.db" -> "foo.db"

    We keep this intentionally simple for demo readability.
    """
    name = database_name.strip()
    if not name:
        raise ValueError("Database name cannot be empty.")

    # Only keep the basename so callers cannot wander out of the demo folder.
    name = Path(name).name

    if not name.endswith(".db"):
        name = f"{name}.db"

    return name


def _db_path(database_name: str) -> Path:
    """Return the full path to a database inside the demo data directory."""
    return DATA_DIR / _normalize_db_name(database_name)


def _ensure_db_exists(database_name: str) -> Path:
    """Raise a clear error if the requested database does not exist."""
    path = _db_path(database_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Database '{_normalize_db_name(database_name)}' does not exist."
        )
    return path


def _connect(database_name: str) -> sqlite3.Connection:
    """
    Open a SQLite connection with row access by column name.

    We do not use a contextmanager here because keeping the connection lifecycle
    explicit makes the demo easier to read.
    """
    path = _db_path(database_name)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _validate_identifier(name: str, kind: str = "identifier") -> str:
    """
    Validate a simple SQL identifier such as a table name or column name.

    This is intentionally restrictive. The goal is to keep the demo safe and
    straightforward rather than support every legal SQLite identifier form.
    """
    if not SAFE_IDENTIFIER_RE.fullmatch(name):
        raise ValueError(
            f"Invalid {kind} '{name}'. Use letters, numbers, and underscores, "
            f"and do not start with a number."
        )
    return name


def _coerce_sqlite_type(type_name: str) -> str:
    """
    Normalize a type name into one of SQLite's common affinities.

    SQLite is flexible about types, but for a demo it is nice to keep things
    tidy and recognizable.
    """
    t = type_name.strip().upper()
    if t in {"INTEGER", "INT"}:
        return "INTEGER"
    if t in {"REAL", "FLOAT", "DOUBLE", "NUMERIC"}:
        return "REAL"
    if t in {"TEXT", "STRING"}:
        return "TEXT"
    if t in {"BLOB", "BYTES"}:
        return "BLOB"

    if SAFE_IDENTIFIER_RE.fullmatch(t):
        return t

    raise ValueError(f"Unsupported or invalid SQLite type '{type_name}'.")


def _quote_identifier(name: str) -> str:
    """
    Quote an identifier after validation.

    Since we validate identifiers first, quoting is mostly for correctness and
    readability.
    """
    _validate_identifier(name)
    return f'"{name}"'


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain Python dict."""
    return {key: row[key] for key in row.keys()}


def _is_readonly_query(sql: str) -> bool:
    """
    Return True only for obviously read-only queries.

    This is intentionally conservative:
    - allows SELECT ...
    - allows WITH ... where the final statement is expected to be read-only
    - rejects multi-statement SQL separated by semicolons
    """
    stripped = sql.strip()
    if not stripped:
        return False

    body = stripped[:-1].strip() if stripped.endswith(";") else stripped
    if ";" in body:
        return False

    upper = body.upper()
    return upper.startswith("SELECT") or upper.startswith("WITH")


@mcp.tool()
def list_databases() -> dict[str, Any]:
    """
    List the demo SQLite database files available to this MCP server.
    """
    databases = sorted(p.name for p in DATA_DIR.glob("*.db"))
    return {
        "data_directory": str(DATA_DIR),
        "databases": databases,
        "count": len(databases),
    }


@mcp.tool()
def create_database(
    database_name: str, overwrite: bool = False
) -> dict[str, Any]:
    """
    Create a SQLite database file in the local demo data directory.

    Args:
        database_name: Name of the database, with or without the .db suffix.
        overwrite: If True and the database already exists, delete and recreate it.

    Returns:
        A structured summary of what happened.
    """
    path = _db_path(database_name)
    existed_before = path.exists()

    if existed_before and overwrite:
        path.unlink()

    conn = sqlite3.connect(path)
    conn.close()

    return {
        "database": path.name,
        "path": str(path),
        "created": not existed_before or overwrite,
        "already_existed": existed_before and not overwrite,
        "overwritten": existed_before and overwrite,
    }


@mcp.tool()
def create_table(
    database_name: str, table_name: str, schema: str, if_not_exists: bool = True
) -> dict[str, Any]:
    """
    Create a table using a caller-provided schema fragment.

    Args:
        database_name: Target database name.
        table_name: Name of the table to create.
        schema: Column definitions inside the parentheses of CREATE TABLE.
            Example:
            'experiment_id TEXT PRIMARY KEY, A REAL, B REAL, C REAL'
        if_not_exists: If True, use CREATE TABLE IF NOT EXISTS.

    Returns:
        A summary of the operation and the SQL that was executed.

    Notes:
        This tool trusts the provided schema string more than other tools do.
        That is intentional for demo simplicity. The table name itself is still
        validated.
    """
    _ensure_db_exists(database_name)
    safe_table = _quote_identifier(
        _validate_identifier(table_name, "table name")
    )

    clause = "IF NOT EXISTS " if if_not_exists else ""
    sql = f"CREATE TABLE {clause}{safe_table} ({schema})"

    conn = _connect(database_name)
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()

    return {
        "database": _normalize_db_name(database_name),
        "table": table_name,
        "sql": sql,
        "created": True,
    }


@mcp.tool()
def list_tables(database_name: str) -> dict[str, Any]:
    """
    List user tables in a database.
    """
    _ensure_db_exists(database_name)

    conn = _connect(database_name)
    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    finally:
        conn.close()

    tables = [row["name"] for row in rows]
    return {
        "database": _normalize_db_name(database_name),
        "tables": tables,
        "count": len(tables),
    }


@mcp.tool()
def describe_table(database_name: str, table_name: str) -> dict[str, Any]:
    """
    Describe a table's columns using SQLite PRAGMA table_info.
    """
    _ensure_db_exists(database_name)
    safe_table_name = _validate_identifier(table_name, "table name")

    conn = _connect(database_name)
    try:
        rows = conn.execute(
            f'PRAGMA table_info("{safe_table_name}")'
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"Table '{table_name}' was not found in database "
            f"'{_normalize_db_name(database_name)}'."
        )

    columns = []
    for row in rows:
        columns.append({
            "cid": row["cid"],
            "name": row["name"],
            "type": row["type"],
            "notnull": bool(row["notnull"]),
            "default_value": row["dflt_value"],
            "primary_key_position": row["pk"],
        })

    return {
        "database": _normalize_db_name(database_name),
        "table": table_name,
        "columns": columns,
    }


@mcp.tool()
def create_table_from_columns(
    database_name: str,
    table_name: str,
    columns: list[dict[str, Any]],
    if_not_exists: bool = True,
) -> dict[str, Any]:
    """
    Create a table from a structured column specification.

    This is an optional convenience tool that some LLMs may find easier than
    building a raw schema string.

    Args:
        database_name: Target database name.
        table_name: Name of the table to create.
        columns: A list of column definitions. Each item should look like:
            {
              "name": "experiment_id",
              "type": "TEXT",
              "primary_key": True,
              "not_null": True,
              "unique": True
            }
        if_not_exists: If True, use CREATE TABLE IF NOT EXISTS.

    Returns:
        A summary of the operation and the generated SQL.
    """
    _ensure_db_exists(database_name)
    safe_table = _quote_identifier(
        _validate_identifier(table_name, "table name")
    )

    if not columns:
        raise ValueError("You must provide at least one column definition.")

    column_sql_parts: list[str] = []
    for col in columns:
        if "name" not in col or "type" not in col:
            raise ValueError(
                "Each column definition must include at least 'name' and 'type'."
            )

        name = _quote_identifier(
            _validate_identifier(str(col["name"]), "column name")
        )
        type_name = _coerce_sqlite_type(str(col["type"]))

        part = f"{name} {type_name}"

        if col.get("primary_key"):
            part += " PRIMARY KEY"
        if col.get("not_null"):
            part += " NOT NULL"
        if col.get("unique"):
            part += " UNIQUE"

        if "default" in col and col["default"] is not None:
            default_value = col["default"]
            if isinstance(default_value, str):
                escaped = default_value.replace("'", "''")
                part += f" DEFAULT '{escaped}'"
            elif isinstance(default_value, bool):
                part += f" DEFAULT {1 if default_value else 0}"
            else:
                part += f" DEFAULT {default_value}"

        column_sql_parts.append(part)

    schema = ", ".join(column_sql_parts)
    clause = "IF NOT EXISTS " if if_not_exists else ""
    sql = f"CREATE TABLE {clause}{safe_table} ({schema})"

    conn = _connect(database_name)
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()

    return {
        "database": _normalize_db_name(database_name),
        "table": table_name,
        "sql": sql,
        "created": True,
    }


@mcp.tool()
def insert_rows(
    database_name: str, table_name: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Insert one or more rows into a table.

    Args:
        database_name: Target database name.
        table_name: Target table name.
        rows: A list of dictionaries, one per row.
            Example:
            [
              {"experiment_id": "exp_001", "A": 1.2, "B": 3.4, "C": 5.6},
              {"experiment_id": "exp_002", "A": 7.8, "B": 9.0, "C": 1.2}
            ]

    Returns:
        A structured summary including how many rows were inserted.

    Notes:
        - All rows must have the same keys.
        - Values are passed through SQLite parameter binding.
    """
    _ensure_db_exists(database_name)
    safe_table = _quote_identifier(
        _validate_identifier(table_name, "table name")
    )

    if not rows:
        raise ValueError("You must provide at least one row to insert.")

    first_keys = list(rows[0].keys())
    if not first_keys:
        raise ValueError("Rows must contain at least one column.")

    for key in first_keys:
        _validate_identifier(key, "column name")

    expected_key_set = set(first_keys)
    for index, row in enumerate(rows):
        if set(row.keys()) != expected_key_set:
            raise ValueError(
                f"Row {index} does not have the same columns as the first row."
            )

    quoted_columns = ", ".join(_quote_identifier(k) for k in first_keys)
    placeholders = ", ".join("?" for _ in first_keys)
    sql = f"INSERT INTO {safe_table} ({quoted_columns}) VALUES ({placeholders})"
    values = [tuple(row[k] for k in first_keys) for row in rows]

    conn = _connect(database_name)
    try:
        cursor = conn.executemany(sql, values)
        conn.commit()
        inserted = cursor.rowcount if cursor.rowcount != -1 else len(rows)
    finally:
        conn.close()

    return {
        "database": _normalize_db_name(database_name),
        "table": table_name,
        "rows_inserted": inserted,
        "columns": first_keys,
    }


@mcp.tool()
def query_sql_readonly(
    database_name: str, sql: str, limit: int = 1000
) -> dict[str, Any]:
    """
    Execute a read-only SQL query and return structured rows.

    Args:
        database_name: Target database name.
        sql: A read-only SQL query. For demo safety, this must start with
            SELECT or WITH and may contain only one statement.
        limit: Maximum number of rows to return to the client.

    Returns:
        A dictionary with columns, rows, and row_count.

    Notes:
        This tool is intentionally read-only. It rejects obviously unsafe or
        mutating SQL.
    """
    _ensure_db_exists(database_name)

    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    if not _is_readonly_query(sql):
        raise ValueError(
            "Only a single read-only SELECT or WITH query is allowed."
        )

    conn = _connect(database_name)
    try:
        cursor = conn.execute(sql)
        fetched = cursor.fetchmany(limit)
        columns = (
            [desc[0] for desc in cursor.description]
            if cursor.description
            else []
        )
        rows = [_row_to_dict(row) for row in fetched]
    finally:
        conn.close()

    return {
        "database": _normalize_db_name(database_name),
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "limited_to": limit,
    }


@mcp.tool()
def delete_database(
    database_name: str, missing_ok: bool = False
) -> dict[str, Any]:
    """
    Delete a database file from the demo data directory.

    This is handy in a demo environment when you want to quickly reset state.
    """
    path = _db_path(database_name)

    if not path.exists():
        if missing_ok:
            return {
                "database": path.name,
                "deleted": False,
                "already_missing": True,
            }
        raise FileNotFoundError(f"Database '{path.name}' does not exist.")

    path.unlink()

    return {
        "database": path.name,
        "deleted": True,
    }


def main() -> None:
    """
    Start the MCP server over Streamable HTTP.

    The Python MCP SDK quick-start uses streamable-http and exposes the endpoint
    at /mcp on localhost:8000 by default.
    """
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
