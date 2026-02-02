"""Shared UX / logging helpers for the Monte Carlo ExecutionAgent examples.

Goals
-----
- **Silent-by-default capable**: honor `--quiet` and `URSA_QUIET=1`.
- If stdout is **not a TTY**, default to quiet to avoid control characters.
- Optional **Rich** output when available (no hard dependency).
- Optional file logging via Python `logging`.

This module is intentionally small and self-contained so it can be vendored into
examples without depending on the URSA internals.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("", "0", "false", "no", "off")


@dataclass(frozen=True)
class UXConfig:
    quiet: bool
    is_tty: bool
    log_file: Optional[Path]

    # Rich is optional and only used when not quiet and on a TTY.
    rich_available: bool


def add_ux_args(parser: argparse.ArgumentParser) -> None:
    """Add standard UX/logging CLI args to an argparse parser."""

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all terminal output (also enabled by URSA_QUIET=1).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to a log file (JSON artifacts are still written separately).",
    )


def resolve_ux_config(
    *, args: argparse.Namespace, default_log_file: Optional[Path] = None
) -> UXConfig:
    """Resolve final UX behavior from args, env, and TTY detection."""

    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

    # Default policy:
    # - Non-TTY => quiet (no control chars / no accidental noise in CI)
    # - TTY => allow output unless user asks for quiet
    quiet_env = _env_flag("URSA_QUIET", default=False)
    quiet_default = (not is_tty) or quiet_env

    quiet = bool(getattr(args, "quiet", False)) or quiet_default

    log_file: Optional[Path]
    lf = getattr(args, "log_file", None)
    if lf is None or str(lf).strip() == "":
        log_file = default_log_file
    else:
        log_file = Path(str(lf))

    rich_available = False
    try:
        import rich  # noqa: F401

        rich_available = True
    except Exception:
        rich_available = False

    return UXConfig(
        quiet=quiet,
        is_tty=is_tty,
        log_file=log_file,
        rich_available=rich_available,
    )


def setup_file_logging(
    *, log_file: Optional[Path], level: int = logging.INFO
) -> logging.Logger:
    """Configure logging to write to a file (and only to a file).

    Returns a module-level logger you can use.
    """

    logger = logging.getLogger("example")

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate logs when re-imported.
    for h in list(root.handlers):
        root.removeHandler(h)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

        # Force UTC timestamps without touching global logging state.
        import time

        fmt.converter = time.gmtime
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Reduce verbosity from common HTTP stacks.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    return logger


class UX:
    """Tiny facade for printing/status with optional Rich."""

    def __init__(self, config: UXConfig):
        self.config = config

        self._console = None
        if (not config.quiet) and config.is_tty and config.rich_available:
            try:
                from rich.console import Console

                # force_terminal=False avoids control characters when not a TTY.
                self._console = Console(force_terminal=True, highlight=False)
            except Exception:
                self._console = None

    @property
    def quiet(self) -> bool:
        return self.config.quiet

    def print(self, message: str) -> None:
        if self.config.quiet:
            return
        if self._console is not None:
            try:
                self._console.print(message)
                return
            except Exception:
                pass
        # Plain fallback
        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def panel(self, title: str, body: str) -> None:
        if self.config.quiet:
            return
        if self._console is not None:
            try:
                from rich.panel import Panel

                self._console.print(Panel(body, title=title, expand=False))
                return
            except Exception:
                pass
        self.print(f"{title}: {body}")

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Context manager for a spinner/status (TTY + Rich only)."""

        if self.config.quiet:
            yield
            return

        if self._console is not None:
            try:
                with self._console.status(message):
                    yield
                return
            except Exception:
                pass

        # Plain fallback: print a one-liner at start, nothing at end.
        self.print(message)
        yield


def resolve_path_under(
    base: Path, maybe_relative: Optional[Path]
) -> Optional[Path]:
    """Resolve a path relative to base if it's not absolute."""

    if maybe_relative is None:
        return None
    p = maybe_relative
    if not p.is_absolute():
        p = base / p
    return p
