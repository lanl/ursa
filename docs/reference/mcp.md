# MCP reference

URSA supports MCP in two directions:

1. URSA can run as an MCP server.
2. URSA can connect to external MCP servers configured in YAML.

For a guided first run, see
[Getting Started - MCP Server][getting-started-mcp-server].

For external server configuration, see
[MCP server configuration][mcp-server-configuration].

## URSA MCP server command

```bash
ursa --config config.yaml mcp-server --transport stdio
```

or:

```bash
ursa --config config.yaml mcp-server \
  --transport streamable-http \
  --host localhost \
  --port 8000
```

## Help

```bash
ursa mcp-server --help
```

## Security warning

The MCP server does not isolate sessions from one another. Do not expose it as a shared multi-user service without external isolation and access control.
