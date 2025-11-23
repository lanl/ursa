# Run the LocalFS MCP Demo
This demo spins up a tiny MCP server (LocalFS) and an ExecutionAgent that:
1. asks the server for three files,
2. writes them into a local workspace,
3. (optionally) tries to append a small slice.

## Prereqs
* Python 3.11+
* Installed deps in your env:
```
pip install mcp langchain-mcp-adapters
```

(Optional for HTTP/SSE hosting: `pip install uvicorn`)

## What it runs
* Server: `scripts/mcp/localfs_mcp_server.py` (served over STDIO by default)
* Client/Agent: `scripts/mcp/run_local_mcp_agent.py`

The agent auto-seeds fixtures the first time at `tests/mcp/fixtures/`:
* `tiny.txt` → "hello world"
* `small.csv` → a,b\n1,2\n
* `tiny.bin` → 3 bytes

## One-liner (recommended)
From repo root:
```
python scripts/mcp/run_local_mcp_agent.py
```

You should see tool call logs like `fs_get_object_bytes` then `write_bytes`.
When it’s done, check the workspace:

```
ls -l workspaces/localfs_mcp_demo/data
# expect:
# tiny.txt   (11 bytes)
# small.csv  (8 bytes)
# tiny.bin   (3 bytes)
```

> Note: You’ll likely see an informational “Offset mismatch: file=11, expected=0” from the final append step. That’s fine for the demo.

## Running the server yourself (optional)
You usually don’t need this—the agent starts the stdio server for you.
But to run it manually:

STDIO:
```
python scripts/mcp/localfs_mcp_server.py --root tests/mcp/fixtures --stdio
```

SEE (HTTP) - requires `uvicorn` and an MCP build that exposes an ASGI app:
```
python scripts/mcp/localfs_mcp_server.py --root tests/mcp/fixtures --sse --host 127.0.0.1 --port 3333
```

(If your installed mcp doesn’t export SSE helpers, the script falls back to `uvicorn` or will tell you what to install.)

## Troubleshooting
* No files written / “requires data_b64 or blob_ref”: Make sure you’re running the current run_local_mcp_agent.py that parses the ToolMessage payloads and forwards data_b64 (or blob_ref if present) to write_bytes/append_bytes.
* Different MCP versions: The server script handles a few API differences (e.g., run_stdio() vs run(); SSE helper names). If you see import errors around mcp.server.sse, upgrade mcp or just use --stdio.
* Verbose tool logs:
```
RSA_AGENT_TOOL_LOG=1 python scripts/mcp/run_local_mcp_agent.py
```

# Running the Unit Tests
This demo script isn’t picked up by pytest (it doesn’t start with `test_`). To run the MCP tests:
```
pytest -q tests/mcp
```
