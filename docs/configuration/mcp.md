# MCP server configuration

URSA can connect to external [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers and can also [run as an MCP server itself][getting-started-mcp-server].

This page covers configuring external MCP servers for URSA to use. To serve
URSA over MCP, see [Getting Started - MCP Server][getting-started-mcp-server].

!!! note
    This page assumes basic familiarity with the [Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture)

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

See [Sandboxing and information control][sandboxing-and-information-control].

## Creating an MCP server

Creating an MCP server that hosts tools is beyond the scope of the URSA documentation.
For Python servers, we recommend [FastMCP](https://gofastmcp.com).
As an MCP client, URSA can connect to any MCP server regardless of which SDK was used to implement it.
