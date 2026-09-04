"""Validate the isolated host UI environment, then run Harbor's CLI."""

# ruff: noqa: TID251 -- this presentation preflight intentionally tests Rich.

from __future__ import annotations

from importlib.resources import files
from io import StringIO

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table


def validate_runtime() -> None:
    unicode_data = files("rich._unicode_data").joinpath("unicode17-0-0.py")
    if not unicode_data.is_file():
        raise RuntimeError(f"Rich Unicode data is missing: {unicode_data}")
    if cell_len("界") != 2:
        raise RuntimeError("Rich Unicode cell measurement is unavailable")
    table = Table("status")
    table.add_row("ready")
    Console(file=StringIO()).print(table)


def main() -> None:
    validate_runtime()
    from harbor.cli.main import app

    app()


if __name__ == "__main__":
    main()
