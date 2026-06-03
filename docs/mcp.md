# URSA MCP Server

You can connect `ursa` as an [Model Context Protocol](https://modelcontextprotocol.io) Server
to other agentic frameworks or interfaces. To start the MCP server, run:


```shell
ursa mcp-server --transport streamable-http
```

This will start an MCP server on localhost on port 8000.

> [!WARNING]
> The MCP Server does not isolate sessions from one another. As such, using the server in a multi-user context
> is not recommended.


## MCP Inspector

After installing the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector), you can test the Ursa MCP server by running:

```shell
npx @modelcontextprotocol/inspector \
    uv run ursa mcp-server
```

Or by connecting to an existing MCP server using the `streamable-http` transport by running:

```shell
npx @modelcontextprotocol/inspector \
    --transport http \
    --server-url http://localhost:8000/mcp
```


You can test the server using curl from another terminal:


The MCP server configuration options can be seen with:
```
ursa mcp-server --help
```

The served instance of ursa can be configured via a configuration file (`ursa --config config.yaml mcp-server...`)
or command line arguments (`ursa --llm_model.model openai:gpt-5 ... mcp-server ...`).

