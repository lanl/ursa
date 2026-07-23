# Getting Started - MCP Server

URSA can run as a [Model Context Protocol](https://modelcontextprotocol.io/) server so other MCP-compatible clients or agent frameworks can use URSA capabilities.

## Prerequisites

- URSA is installed.
- You have a model configuration file.
- You understand the session-isolation warning below.

!!! warning "Session isolation"
    The URSA MCP server does not isolate sessions from one another. Do not run it as a shared multi-user service unless you have added appropriate isolation and access controls.

## Configure the served URSA instance

Create `config.yaml`:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY

workspace: ./ursa-mcp-workspace
```

## stdio transport

Use stdio when an MCP client launches URSA directly:

```bash
ursa --config config.yaml mcp-server --transport stdio
```

You can inspect it with MCP Inspector:

```bash
npx @modelcontextprotocol/inspector \
  uv run ursa --config config.yaml mcp-server --transport stdio
```

## Streamable HTTP transport

Use streamable HTTP when you want to run a local server and connect to it separately:

```bash
ursa --config config.yaml mcp-server \
  --transport streamable-http \
  --host localhost \
  --port 8000
```

Connect clients to:

```text
http://localhost:8000/mcp
```

Inspect the running server:

```bash
npx @modelcontextprotocol/inspector \
  --transport http \
  --server-url http://localhost:8000/mcp
```

## See all MCP server options

```bash
ursa mcp-server --help
```

Current options include:

```text
--transport {stdio,streamable-http}
--host HOST
--port PORT
--log_level {debug,info,notice,warning,error,critical}
```

## Where next?

- [MCP server configuration][mcp-server-configuration]
- [MCP reference][mcp-reference]
- [Sandboxing and information control][sandboxing-and-information-control]
