"""
test_sqlite_mcp.py

A very small MCP client harness for the sqlite_mcp.py demo server.

This script assumes the SQLite MCP server is already running over
Streamable HTTP at:

    http://127.0.0.1:8000/mcp

Run the server first in one terminal:

    python sqlite_mcp.py

Then run this script from the same directory:

    python test_sqlite_mcp.py
"""

from __future__ import annotations

import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_URL = "http://127.0.0.1:8000/mcp"


def _build_client() -> MultiServerMCPClient:
    """
    Create an MCP client configuration pointing at the local SQLite demo server.
    """
    return MultiServerMCPClient({
        "sqlite_demo": {
            "transport": "streamable_http",
            "url": SERVER_URL,
        }
    })


def _find_tool(tools: list, name: str):
    """
    Find a loaded LangChain-style tool by name.
    """
    for tool in tools:
        if tool.name == name:
            return tool
    available = [tool.name for tool in tools]
    raise ValueError(f"Tool '{name}' not found. Available tools: {available}")


def _pretty_print_result(result) -> None:
    """
    Print tool results cleanly whether they arrive as dicts or JSON strings.
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass
    print(json.dumps(result, indent=2, default=str))


async def main() -> None:
    """
    Connect to the running MCP server, call a few tools, and print the results.
    """
    client = _build_client()
    tools = await client.get_tools()

    print("\nLoaded MCP tools:")
    for tool in tools:
        print(f"  - {tool.name}")

    create_database = _find_tool(tools, "create_database")
    create_table = _find_tool(tools, "create_table")
    insert_rows = _find_tool(tools, "insert_rows")
    query_sql_readonly = _find_tool(tools, "query_sql_readonly")
    list_tables = _find_tool(tools, "list_tables")
    describe_table = _find_tool(tools, "describe_table")

    print("\nCreating demo database...")
    result = await create_database.ainvoke({
        "database_name": "demo_test",
        "overwrite": True,
    })
    _pretty_print_result(result)

    print("\nCreating table...")
    result = await create_table.ainvoke({
        "database_name": "demo_test",
        "table_name": "experimental_data",
        "schema": "experiment_id TEXT PRIMARY KEY, A REAL, B REAL, C REAL",
        "if_not_exists": True,
    })
    _pretty_print_result(result)

    print("\nListing tables...")
    result = await list_tables.ainvoke({"database_name": "demo_test"})
    _pretty_print_result(result)

    print("\nDescribing table...")
    result = await describe_table.ainvoke({
        "database_name": "demo_test",
        "table_name": "experimental_data",
    })
    _pretty_print_result(result)

    print("\nInserting rows...")
    result = await insert_rows.ainvoke({
        "database_name": "demo_test",
        "table_name": "experimental_data",
        "rows": [
            {"experiment_id": "exp_001", "A": 1.1, "B": 2.2, "C": 3.3},
            {"experiment_id": "exp_002", "A": 4.4, "B": 5.5, "C": 6.6},
            {"experiment_id": "exp_003", "A": 7.7, "B": 8.8, "C": 9.9},
        ],
    })
    _pretty_print_result(result)

    print("\nQuerying rows back out...")
    result = await query_sql_readonly.ainvoke({
        "database_name": "demo_test",
        "sql": "SELECT * FROM experimental_data ORDER BY experiment_id",
        "limit": 100,
    })
    _pretty_print_result(result)

    print("\nTest completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
