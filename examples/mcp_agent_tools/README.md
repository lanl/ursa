# Attach MCP tools to a Python agent

Start a local MCP server, let URSA discover its tools, and attach those tools to
a `ChatAgent`. This example keeps the server deliberately small so you can see
the complete connection before adapting it to a real service.

The workflow has two processes:

1. `laboratory_server.py` serves one `list_measurements` tool over local
   Streamable HTTP.
2. `attach_mcp_tools.py` reads `config.yaml`, initializes the configured model,
   attaches the discovered MCP tool, and asks the agent to use it.

Read the [MCP configuration guide](../../docs/configuration/mcp.md) for other
transports and authentication settings. The [Python scripts guide](../../docs/getting-started/python-scripts.md)
explains model initialization and direct agent use.

## Prepare the example

Open a terminal in this folder, install the locked environment, and set your
OpenAI key.

=== "macOS/Linux"

    ```bash
    cd examples/mcp_agent_tools
    uv sync
    export OPENAI_API_KEY="your-api-key"
    ```

=== "Windows PowerShell"

    ```powershell
    Set-Location examples\mcp_agent_tools
    uv sync
    $env:OPENAI_API_KEY = "your-api-key"
    ```

The example uses URSA's default OpenAI model. Follow [models and inference
providers](../../docs/configuration/models.md) before running it with another
provider.

## Inspect the server configuration

The client reads this MCP server definition:

```yaml
--8<-- "examples/mcp_agent_tools/config.yaml"
```

The endpoint is local and does not include authentication. Keep it bound to
your machine for this exercise.

## Start the MCP server

In the first terminal, run:

```bash
uv run laboratory_server.py
```

Leave that process running. It serves the MCP endpoint at
`http://127.0.0.1:8000/mcp`.

## Attach and use the tool

Open a second terminal in the same folder, set `OPENAI_API_KEY` there as shown
above, and run:

```bash
uv run attach_mcp_tools.py
```

The script prints the tool-to-server mapping returned by `add_mcp_tools()`, then
prints the agent's summary. Confirm that `list_measurements` is attached from
the `laboratory` server and that the answer identifies `alloy-b` as the largest
reported strength while noting that `alloy-c` was measured at another
temperature.

The client implementation is short enough to inspect in full:

```python
--8<-- "examples/mcp_agent_tools/attach_mcp_tools.py"
```

`add_mcp_tools()` accepts `tool_name="list_measurements"` or a list of names
when an agent should receive only selected server tools. The server must already
be running when discovery begins.

## Adapt the example

Add another `@mcp.tool()` function to `laboratory_server.py`, restart the server,
and run the client again. Update the prompt so the agent has a clear reason to
choose the new tool. Review the [MCP reference](../../docs/reference/mcp.md) when
you add production transports, credentials, or remote endpoints.
