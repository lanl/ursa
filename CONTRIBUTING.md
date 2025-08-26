# Contributing

## Formatting

You will need to format your code using [`ruff`][2]. The easiest way to
automate formatting of code prior to each commit is via the [`pre-commit`][1]
package, which simplifies management of git hooks.

If using pip, `pre-commit` will be installed when you install `ursa` in
editable mode via

```bash
pip install -e .[dev]
```

If using `uv`, `pre-commit` will be installed in your default (dev)
environment.

To install the ruff formatting git hook to your local `.git/hooks` folder, run
the following in the current directory:

```bash
# If using pip with venv, first activate your environment, then
pre-commit install

# If usign uv
uv run pre-commit install
```

Prior to subsequent `git commit` calls, `ruff` will first format code.

Instead of running git hooks, you can format code manually via

```bash
# In current directory:
ruff format
```

(You can install via ruff following instructions [here][3].)

If code is not formatted, PRs will be blocked.

[1]: https://pre-commit.com
[2]: https://github.com/astral-sh/ruff
[3]: https://docs.astral.sh/ruff/installation
