"""Validate Harbor task environments supported by URSA."""

from __future__ import annotations

import sys
from argparse import SUPPRESS
from pathlib import Path


def add_harbor_subcommands(subparsers) -> None:
    """Add ``ursa harbor`` integration commands."""
    from jsonargparse import ArgumentParser

    harbor_parser = ArgumentParser(
        description="Inspect URSA's Harbor integration."
    )
    subparsers.add_subcommand(
        "harbor",
        harbor_parser,
        help=harbor_parser.description,
        dest="subcommand",
    ).default_env = False
    commands = harbor_parser.add_subcommands(required=True)

    validate_parser = ArgumentParser(
        description=(
            "Validate Harbor task Dockerfiles and Compose files against "
            "URSA's Singularity environment constraints."
        )
    )
    validate_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Task files, task directories, or roots to scan recursively",
    )
    validate_parser.add_argument("--handler", default=None, help=SUPPRESS)
    commands.add_subcommand(
        "validate",
        validate_parser,
        help=validate_parser.description,
    )
    validate_parser.set_defaults(handler=validate_harbor_paths)


def validate_harbor_paths(paths: list[Path]) -> None:
    """Validate every Harbor task found at or below ``paths``."""
    from ursa.integrations.harbor_validation import (
        discover_harbor_tasks,
        validate_harbor_task,
    )

    tasks = discover_harbor_tasks(paths)
    failures = 0
    for task in tasks:
        try:
            validate_harbor_task(task)
        except Exception as exc:
            failures += 1
            print(  # noqa: T201
                f"FAIL {task}: {type(exc).__name__}: {exc}", file=sys.stderr
            )
        else:
            print(f"OK   {task}")  # noqa: T201
    if failures:
        raise SystemExit(1)
    print(f"Validated {len(tasks)} Harbor task(s).")  # noqa: T201


__all__ = ["add_harbor_subcommands", "validate_harbor_paths"]
