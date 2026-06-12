# MCP server configuration

URSA can connect to external MCP servers and can also run as an MCP server itself.

This page covers configuring external MCP servers for URSA to use. To serve URSA over MCP, see [Getting Started - MCP Server](../getting-started/mcp-server.md).

## External stdio MCP server

```yaml
mcp_servers:
  filesystem:
    transport: stdio
    command: mcp-filesystem-server
    args:
      - ./workspace
    env:
      API_KEY: ${FILESYSTEM_SERVER_API_KEY}
```

## External streamable HTTP MCP server

```yaml
mcp_servers:
  remote-tools:
    transport: streamable-http
    url: http://localhost:8000/mcp
    timeout: 60
```

## Use the config

```bash
ursa --config config.yaml
```

## Security notes

MCP servers can expose powerful tools. Only connect MCP servers that you trust, and prefer dedicated workspaces and endpoint allowlists for sensitive workflows.

See [Sandboxing and information control](../best-practices/sandboxing.md).
