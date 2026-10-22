# Contributing

Contributions are welcome. This page provides guidance for contributors on how to format code, report problems, raise issues, propose changes, build documentation, etc.

## Linting and Formatting

You will need to lint and format your code using [`ruff`][2]. The easiest way
to automate linting/formatting of code prior to each commit is via the
[`pre-commit`][1] package, which simplifies management of git hooks.

Install the development dependencies from your checkout using
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

This installs Ruff, pre-commit, and the test dependencies.
Use Python 3.11 or newer and add any extras needed by your change, for example
`uv sync --group dev --extra lammps`.

To install the ruff git hook to your local `.git/hooks` folder, run
the following in the current directory:

```bash
# If using pip with venv, first activate your environment, then
pre-commit install

# If using uv
uv run pre-commit install
```

Prior to subsequent `git commit` calls, `ruff` will first lint/format code.

Instead of running git hooks, you can lint/format code manually via running the
following in the current directory:

```bash
# Lint
ruff check --fix

# Format
ruff format
```

To continually lint while developing, you can
run the following in your terminal

```bash
ruff check --watch
```

For editor (e.g., vim, VSCode) integration, see [here][4].

(You can install via ruff following instructions [here][3].)

If code is not linted/formatted, PRs will be blocked.

## Reporting problems

Report problems through [GitHub Issues](https://github.com/lanl/ursa/issues). Please include:

- the URSA and Python versions, operating system, and installation method;
- a minimal example and the steps needed to reproduce the problem;
- the expected and actual behavior;
- relevant configuration, error messages, and logs.

## Getting help

Ask usage and configuration questions in
[GitHub Issues](https://github.com/lanl/ursa/issues). Describe what you are trying
to do and what you have already tried. The
[documentation](https://lanl.github.io/ursa/) and repository's
[examples](examples/) are useful starting points.

## Proposing changes

1. For substantial changes, open an issue to discuss the approach with maintainers. Examples of substantial changes are introducing a new [agent](https://github.com/lanl/ursa/tree/main/src/ursa/agents), [workflow](https://github.com/lanl/ursa/tree/main/src/ursa/workflows), [environment](https://github.com/lanl/ursa/tree/main/src/ursa/environments), etc.
2. Fork the repository and create a branch from `main` in your fork.
3. Update relevant documentation and examples, and add
   or update tests when behavior changes.
4. Run the relevant tests and formatting checks described in this guide.
5. Open a pull request against `lanl/ursa`'s `main` branch. Explain the problem,
   the change, and how you verified it.

URSA is distributed under the [BSD-3 license](LICENSE). Ensure you have permission
to contribute the code and other materials you submit.

## Running Tests

Run tests relevant to your change, for example:

```bash
uv run pytest tests/agents/test_execution_agent
uv run ruff check
uv run ruff format --check
```

For the full test suite, install all extras and run:

```bash
uv sync --all-extras --group dev
uv run pytest
```

## Building Documentation

Install the documentation dependency group and preview changes locally:

```bash
uv sync --group dev --group docs
uv run --group docs mkdocs serve
```

Open the local URL printed by MkDocs. To build once:

```bash
uv run --group docs mkdocs build
```

- Prefer short, runnable examples and step-by-step Getting Started pages.
- Use YAML configuration examples where appropriate.
- Put detailed command lists in Reference or Persistence pages.

Documentation is versioned with Mike. The documentation workflow publishes
`main` after documentation changes land on the main branch. Stable `v*` tags
publish a version without the leading `v`, move the `latest` alias to that
release, and make `latest` the default. Pre-release tags are not published.
To preview versions already on your local `gh-pages` branch, run
`uv run --group docs mike serve`.

[1]: https://pre-commit.com
[2]: https://github.com/astral-sh/ruff
[3]: https://docs.astral.sh/ruff/installation
[4]: https://docs.astral.sh/ruff/editors/setup
