# Contributing

Contributions are welcome. This page summarizes the documentation and development workflow; see the repository's contribution files for project-specific policy details.

## Development setup

Install development dependencies in a clean environment. If the project defines dependency groups for development and documentation, prefer `uv`:

```bash
uv sync
```

If additional groups are defined in your checkout, include them as needed, for example:

```bash
uv sync --group dev --group docs
```

## Run tests and checks

Common commands include:

```bash
uv run pytest
uv run ruff check
uv run ruff format
```

Adjust commands to match the current project configuration.

## Build documentation locally

```bash
uv run mkdocs serve
```

Then open the local URL printed by MkDocs.

To build once:

```bash
uv run mkdocs build
```

## Documentation style

- Prefer short, runnable examples.
- Use YAML config files as the primary configuration path.
- Keep Getting Started pages step-by-step.
- Put exhaustive command lists in Reference or Persistence pages.
- Include warnings for execution, web access, MCP servers, and secrets where relevant.
