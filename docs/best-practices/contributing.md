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
just docs
```

Then open the local URL printed by MkDocs.

To build once:

```bash
uv run mkdocs build
```

Published documentation is versioned with Mike. To preview the versions already
present on the local `gh-pages` branch, run:

```bash
uv run mike serve
```

The documentation workflow publishes `main` after documentation changes land on
the main branch. Stable `v*` tags publish a version without the leading `v`, move
the `latest` alias to that release, and make `latest` the default site version.
Pre-release tags are not published.

## Documentation style

- Prefer short, runnable examples.
- Use YAML config files as the primary configuration path.
- Keep Getting Started pages step-by-step.
- Put exhaustive command lists in Reference or Persistence pages.
- Include warnings for execution, web access, MCP servers, and secrets where relevant.
