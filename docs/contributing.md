# Contributing

## Linting and Formatting

You will need to lint and format your code using [`ruff`][2]. The easiest way
to automate linting/formatting of code prior to each commit is via the
[`pre-commit`][1] package, which simplifies management of git hooks.

If using pip, `pre-commit` will be installed when you install `ursa` in
editable mode via

```bash
pip install -e .[dev]
```

If using `uv`, `pre-commit` will be installed in your default (dev)
environment.

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

## Building Documentation

If you want to edit and build documentation locally, you need to install [`mkdocs`][5] and some dependencies:

```bash
pip install mkdocs mkdocs-autorefs mkdocs-material mkdocstrings-python
```

To prototype documentation locally you can run:

```bash
mkdocs serve
```

and view the built docs at `http://127.0.0.1:8000`.

[1]: https://pre-commit.com
[2]: https://github.com/astral-sh/ruff
[3]: https://docs.astral.sh/ruff/installation
[4]: https://docs.astral.sh/ruff/editors/setup
[5]: https://www.mkdocs.org/
