# CLI reference

Use the built-in help for the authoritative command list:

```bash
ursa --help
```

For subcommand-specific help:

```bash
ursa mcp-server --help
ursa exec --help
ursa rag-ingest --help
ursa rag-query --help
```

## Common top-level commands

```bash
ursa --config config.yaml
ursa --print-config
ursa --config config.yaml --name my-agent --group default
ursa --config config.yaml --use-web
```

## Main subcommands

Current URSA installations include subcommands for:

- running the MCP server,
- managing groups,
- managing persistent agents,
- sharing/importing agents,
- managing persistent RAG collections,
- non-interactive execution.

Use `ursa --help` to confirm the exact set in your installed version.

## Python callback API

::: ursa.cli.callbacks.HITLLogEventHandler
    options:
      show_root_heading: true
      show_source: true
