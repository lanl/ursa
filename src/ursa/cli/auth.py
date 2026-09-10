"""Manage URSA credentials in the operating-system keyring."""

from __future__ import annotations

import getpass
from argparse import SUPPRESS
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from os import environ
from pathlib import Path
from typing import Any

import keyring
from jsonargparse import ArgumentParser, Namespace

from ursa.cli.config import (
    UrsaConfig,
    config_layers,
    deep_merge_dicts,
    load_config_file,
)
from ursa.util.secrets import SecretReference

KEYRING_SERVICE = "ursa"


def add_auth_subcommands(subparsers) -> None:
    """Add ``ursa auth`` credential-management commands."""
    auth_parser = ArgumentParser()
    subparsers.add_subcommand(
        "auth",
        auth_parser,
        help="Manage credentials in the system keyring",
        dest="subcommand",
    )
    auth_subparsers = auth_parser.add_subcommands(required=True)

    login_parser = ArgumentParser()
    source = login_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "username",
        nargs="?",
        help="Inference-provider name or keyring username",
    )
    source.add_argument(
        "--config",
        type=Path,
        help="Store every keyring-backed secret referenced by this config",
    )
    login_parser.add_argument(
        "--from-env",
        metavar="VAR",
        help="Read the secret from this environment variable instead of prompting",
    )
    login_parser.add_argument("--handler", default=None, help=SUPPRESS)
    auth_subparsers.add_subcommand(
        "login", login_parser, help="Store credentials in the system keyring"
    )
    login_parser.set_defaults(handler=login)

    list_parser = ArgumentParser()
    list_parser.add_argument(
        "--config",
        type=Path,
        help="Additional config file to merge after system and user configs",
    )
    list_parser.add_argument(
        "--show-secrets",
        action="store_true",
        default=False,
        help="Print resolved secret values",
    )
    list_parser.add_argument("--handler", default=None, help=SUPPRESS)
    auth_subparsers.add_subcommand(
        "list",
        list_parser,
        help="Check configured environment and keyring secrets",
    )
    list_parser.set_defaults(handler=list_credentials)

    # Authentication configuration must be explicit. In particular, do not
    # let jsonargparse synthesize URSA_AUTH__... settings for these parsers.
    auth_parser.default_env = False


def config_keyring_usernames(path: Path) -> list[str]:
    """Return keyring usernames referenced anywhere in an URSA config."""
    secrets = _resolved_secrets(load_config_file(path))
    return sorted({
        reference.keyring
        for _, reference in secrets
        if isinstance(reference.keyring, str)
    })


def _iter_secrets(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], SecretReference]]:
    """Yield secret templates found in an arbitrary config value."""
    value = SecretReference.maybe_validate(value)
    if isinstance(value, SecretReference):
        yield path, value
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            yield from _iter_secrets(item, (*path, str(name)))
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            yield from _iter_secrets(item, (*path, str(index)))


def _resolved_secrets(
    config: Mapping[str, Any],
) -> Iterator[tuple[tuple[str, ...], SecretReference]]:
    """Yield explicitly configured secrets after canonical resolution."""
    resolved = UrsaConfig.model_validate(config).resolve()
    resolved_by_path = dict(_iter_secrets(resolved.model_dump(mode="python")))
    for path, _ in _iter_secrets(config):
        yield path, resolved_by_path[path]


def _secret_lines(
    config: dict, show_secrets: bool = False
) -> dict[str, list[str]]:
    secrets = list(_resolved_secrets(config))
    mcp_counts = Counter(
        path[1] for path, _ in secrets if path[:1] == ("mcp_servers",)
    )
    sections: dict[str, list[str]] = {
        "Inference Providers": [],
        "MCP Servers:": [],
        "Other": [],
    }
    for path, reference in secrets:
        contextual_keyring_name = None
        if path[:1] == ("inference_providers",):
            section, name = "Inference Providers", path[1]
            contextual_keyring_name = path[1]
        elif path[:1] == ("mcp_servers",):
            section, server_name = "MCP Servers:", path[1]
            contextual_keyring_name = server_name
            suffix = ".".join(part for part in path[2:] if part != "headers")
            name = (
                f"{server_name}.{suffix}"
                if mcp_counts[server_name] > 1
                else server_name
            )
        else:
            section = "Other"
            display_path = path[:-1] if path[-1:] == ("api_key",) else path
            name = ".".join(display_path)

        source = "env" if reference.env is not None else "keyring"
        if (
            source == "keyring"
            and isinstance(reference.keyring, str)
            and reference.keyring != contextual_keyring_name
        ):
            name += f" ({reference.keyring})"
        try:
            resolved = reference.resolve()
        except (ValueError, keyring.errors.KeyringError):
            resolved = None
        line = (
            f"  {name}: {source} {'ok' if resolved is not None else 'missing'}"
        )
        if show_secrets and resolved is not None:
            line += f" = {resolved.get_secret_value()}"
        sections[section].append(line)

    return {heading: sorted(lines) for heading, lines in sections.items()}


def configured_secret_lines(
    config: Path | None = None, show_secrets: bool = False
) -> dict[str, list[str]]:
    """Build the categorized report for the active config stack."""
    merged = {}
    config_namespace = Namespace(config=config, subcommand=None)
    for layer in config_layers(config_namespace, "file+"):
        merged = deep_merge_dicts(merged, layer)
    return _secret_lines(merged, show_secrets)


def list_credentials(
    config: Path | None = None, show_secrets: bool = False
) -> None:
    """Report whether each configured external secret is present."""
    sections = configured_secret_lines(config, show_secrets)
    populated_sections = [
        (heading, references)
        for heading, references in sections.items()
        if references
    ]
    for index, (heading, references) in enumerate(populated_sections):
        if index:
            print()  # noqa: T201
        print(heading)  # noqa: T201
        for line in references:
            print(line)  # noqa: T201


def login(
    username: str | None = None,
    config: Path | None = None,
    from_env: str | None = None,
) -> None:
    """Prompt for and store one or more URSA credentials."""
    usernames = config_keyring_usernames(config) if config else [username]
    usernames = [name for name in usernames if name]
    if not usernames:
        print("No keyring-backed secrets found in the config.")  # noqa: T201
        return

    env_password: str | None = None
    if from_env is not None:
        if len(usernames) != 1:
            raise ValueError(
                "--from-env requires exactly one keyring-backed secret"
            )
        env_password = environ.get(from_env)
        if not env_password:
            raise ValueError(
                f"Environment variable '{from_env}' is not set or is empty"
            )

    for name in usernames:
        password = env_password or getpass.getpass(f"Secret for {name}: ")
        if not password:
            raise ValueError(f"Secret for '{name}' cannot be empty")
        keyring.set_password(KEYRING_SERVICE, name, password)
        print(f"Stored credential: {name}")  # noqa: T201
