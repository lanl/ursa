"""Inspect and manage the running URSA installation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from argparse import SUPPRESS
from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as package_version
from pathlib import Path

TOOL_PACKAGE = "ursa-ai"
DEFAULT_GIT_REPOSITORY = "https://github.com/lanl/ursa"
_INVALID_RECEIPT = "Cannot update: uv's URSA installation receipt is invalid."
_NOT_UV_TOOL = (
    "Cannot update: this copy of URSA was not installed with `uv tool install`.\n"
    "Update it using the same mechanism that installed it."
)


def _command(commands, name, help, action):
    from jsonargparse import ArgumentParser

    parser = ArgumentParser(description=help)
    parser.add_argument(
        "--_action", dest="action", default=action, help=SUPPRESS
    )
    commands.add_subcommand(name, parser, help=help).default_env = False
    return parser


def add_self_subcommands(subparsers) -> None:
    """Register commands without inspecting the installation."""
    from jsonargparse import ArgumentParser

    parser = ArgumentParser(
        description="Inspect and manage this URSA installation.",
        epilog=(
            "Self update and modification require a uv tool installation: "
            f"`uv tool install {TOOL_PACKAGE}`"
        ),
    )
    subparsers.add_subcommand(
        "self", parser, help=parser.description, dest="subcommand"
    ).default_env = False
    commands = parser.add_subcommands(required=True)
    _command(commands, "status", "Show installation details", show_status)
    _command(
        commands,
        "update",
        "Update URSA while preserving its recipe",
        lambda _args: upgrade(),
    )
    modify = _command(
        commands, "modify", "Modify the URSA installation", _modify
    )
    modify.add_argument("--extra", action="append", default=None)
    modify.add_argument(
        "--with", dest="with_packages", action="append", default=None
    )
    modify.add_argument("--clean", action="store_true")
    source = modify.add_mutually_exclusive_group()
    source.add_argument("--version", help="Install an exact registry version")
    source.add_argument("--ref", help="Install a Git branch, tag, or commit")


def _modify(args) -> None:
    upgrade(
        extras=args.extra,
        with_packages=args.with_packages,
        version=args.version,
        ref=args.ref,
        clean=args.clean,
    )


def _read_requirements(receipt: Path) -> list[dict]:
    try:
        with receipt.open("rb") as stream:
            requirements = tomllib.load(stream)["tool"]["requirements"]
        if not isinstance(requirements, list):
            raise TypeError
        return requirements
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        raise SystemExit(_INVALID_RECEIPT) from None


def _read_requirement(receipt: Path) -> dict:
    try:
        return next(
            item
            for item in _read_requirements(receipt)
            if item.get("name") == TOOL_PACKAGE
        )
    except (AttributeError, StopIteration):
        raise SystemExit(_INVALID_RECEIPT) from None


def _running_uv_receipt(prefix: Path | None = None) -> Path | None:
    receipt = (prefix or Path(sys.prefix)) / "uv-receipt.toml"
    if not receipt.is_file():
        return None
    try:
        _read_requirement(receipt)
        return receipt
    except SystemExit:
        return None


def _uv_install() -> tuple[Path, Path]:
    if not (executable := shutil.which("uv")):
        raise SystemExit("Cannot update: uv was not found on PATH.")
    uv = Path(executable)
    result = subprocess.run(
        [str(uv), "tool", "dir"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        detail = f"\n{result.stderr.strip()}" if result.stderr.strip() else ""
        raise SystemExit(
            f"Cannot update: failed to determine uv's tool directory.{detail}"
        )
    if not (tool_dir := result.stdout.strip()):
        raise SystemExit("Cannot update: uv returned an empty tool directory.")
    current = Path(sys.prefix)
    receipt = current / "uv-receipt.toml"
    try:
        valid = (
            current.samefile(Path(tool_dir) / TOOL_PACKAGE)
            and receipt.is_file()
        )
    except OSError:
        valid = False
    if not valid:
        raise SystemExit(_NOT_UV_TOOL)
    _read_requirement(receipt)
    return uv, receipt


def _source(requirement: dict) -> str:
    if git := requirement.get("git"):
        ref = next(
            (
                requirement.get(key)
                for key in ("branch", "tag", "rev")
                if requirement.get(key)
            ),
            None,
        )
        return f" @ git+{git}{f'@{ref}' if ref else ''}"
    if url := requirement.get("url"):
        return f" @ {url}"
    if path := requirement.get("directory") or requirement.get("path"):
        return f" @ {Path(path).resolve().as_uri()}"
    return requirement.get("specifier", "")


def _format_requirement(requirement: dict) -> str:
    extras = requirement.get("extras", [])
    name = requirement["name"] + (f"[{','.join(extras)}]" if extras else "")
    return name + _source(requirement)


def show_status(_args=None) -> None:
    try:
        installed = package_version(TOOL_PACKAGE)
    except PackageNotFoundError:
        installed = "unknown"
    print(f"Version: {installed}")  # noqa: T201
    print(f"Python: {sys.version.split()[0]} ({sys.implementation.name})")  # noqa: T201
    print(f"Python path: {Path(sys.executable).resolve()}")  # noqa: T201
    print(f"Platform: {sys.platform}")  # noqa: T201
    receipt = _running_uv_receipt()
    if receipt is None:
        unavailable = "unavailable (not a uv tool installation)"
        print(f"Extras: {unavailable}")  # noqa: T201
        print(f"Additional packages: {unavailable}")  # noqa: T201
        return
    requirements = _read_requirements(receipt)
    extras = _read_requirement(receipt).get("extras", [])
    additional = [
        _format_requirement(item)
        for item in requirements
        if item.get("name") != TOOL_PACKAGE
    ]
    print(f"Extras: {', '.join(extras) if extras else 'none'}")  # noqa: T201
    print(  # noqa: T201
        f"Additional packages: {', '.join(additional) if additional else 'none'}"
    )


def _merge(
    existing: list[str], additions: list[str] | None, clean: bool
) -> list[str]:
    if any(not item for item in additions or []):
        raise SystemExit(
            "Cannot update: names and requirements cannot be empty."
        )
    return list(
        dict.fromkeys([*([] if clean else existing), *(additions or [])])
    )


def get_package_repository() -> str:
    try:
        urls = metadata(TOOL_PACKAGE).get_all("Project-URL") or []
    except PackageNotFoundError:
        return DEFAULT_GIT_REPOSITORY
    parsed = {
        label.strip().casefold(): url.strip()
        for value in urls
        for label, separator, url in [value.partition(",")]
        if separator and url.strip()
    }
    return next(
        (
            parsed[key]
            for key in ("repository", "source", "source code", "homepage")
            if key in parsed
        ),
        DEFAULT_GIT_REPOSITORY,
    )


def build_upgrade_requirement(
    receipt: Path,
    *,
    extras: list[str] | None = None,
    version: str | None = None,
    ref: str | None = None,
) -> str:
    original = _read_requirement(receipt)
    selected = original.get("extras", []) if extras is None else extras
    name = TOOL_PACKAGE + (f"[{','.join(selected)}]" if selected else "")
    if version is not None:
        if not version.strip():
            raise SystemExit("Cannot update: version cannot be empty.")
        return f"{name}=={version}"
    if ref is not None:
        if not ref.strip():
            raise SystemExit("Cannot update: ref cannot be empty.")
        return f"{name} @ git+{original.get('git') or get_package_repository()}@{ref}"
    return name + _source(original)


def upgrade(
    *,
    extras: list[str] | None = None,
    with_packages: list[str] | None = None,
    version: str | None = None,
    ref: str | None = None,
    clean: bool = False,
) -> None:
    """Replace the running process with uv update or install."""
    uv, receipt = _uv_install()
    changed = clean or any(
        value is not None for value in (extras, with_packages, version, ref)
    )
    if changed:
        selected = _merge(
            _read_requirement(receipt).get("extras", []), extras, clean
        )
        packages = _merge(
            [
                _format_requirement(item)
                for item in _read_requirements(receipt)
                if item.get("name") != TOOL_PACKAGE
            ],
            with_packages,
            clean,
        )
        requirement = build_upgrade_requirement(
            receipt,
            extras=selected,
            version=version,
            ref=ref,
        )
        argv = [
            str(uv),
            "tool",
            "install",
            "--force",
            "--reinstall",
            "--compile-bytecode",
        ]
        for package in packages:
            argv += ["--with", package]
        argv.append(requirement)
    else:
        argv = [
            str(uv),
            "tool",
            "upgrade",
            "--reinstall",
            "--compile-bytecode",
            TOOL_PACKAGE,
        ]
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(str(uv), argv)
    except OSError as exc:
        raise SystemExit(
            f"Cannot update: uv could not be started: {exc}"
        ) from None
    raise AssertionError("unreachable")
