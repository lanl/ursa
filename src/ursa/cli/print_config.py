from __future__ import annotations

import yaml
from jsonargparse import ArgumentParser

from ursa.cli.config import (
    merge_ursa_config,
    resolve_ursa_config,
)

VALID_PRINT_CONFIG_LEVELS = {"final", "file", "user"}
VALID_PRINT_CONFIG_STAGES = {"merged", "resolved"}


def parse_print_config_spec(spec: str | None) -> tuple[str, str] | None:
    if spec is None:
        return None
    if spec in {True, False}:
        return ("final", "resolved") if spec else None
    if spec in VALID_PRINT_CONFIG_STAGES:
        return ("final", spec)
    if "," not in spec:
        raise ValueError(
            "--print-config must be one of merged, resolved, or LEVEL[+],STAGE"
        )
    level, stage = [part.strip() for part in spec.split(",", maxsplit=1)]
    if not level:
        level = "final"
    base_level = level.removesuffix("+")
    if base_level not in VALID_PRINT_CONFIG_LEVELS:
        raise ValueError(f"Unknown print-config level '{level}'")
    if stage not in VALID_PRINT_CONFIG_STAGES:
        raise ValueError(f"Unknown print-config stage '{stage}'")
    return level, stage


def _validate_print_config_spec(spec: str) -> str:
    """Validate a print-config value while preserving its CLI representation."""
    parse_print_config_spec(spec)
    return spec


def add_print_config_argument(parser: ArgumentParser) -> None:
    """Register the validated ``--print-config`` CLI option."""
    parser.add_argument(
        "--print-config",
        nargs="?",
        const="resolved",
        default=None,
        type=_validate_print_config_spec,
        help=(
            "Print configuration and exit. Defaults to resolved output. "
            "Accepted forms: --print-config, --print-config=resolved, "
            "--print-config=merged, or --print-config=LEVEL,STAGE where "
            "LEVEL is one of final, file, user (optionally suffixed with + "
            "for cumulative input) and STAGE is merged or resolved."
        ),
    )


def print_config(cfg, overrides) -> bool:
    """Print config according to --print-config and return whether it handled output."""
    print_config_spec = parse_print_config_spec(cfg["print_config"])
    if print_config_spec is None:
        return False
    level, stage = print_config_spec
    config = merge_ursa_config(cfg, level, overrides)
    if stage == "resolved":
        config = resolve_ursa_config(config)
    print(  # noqa: T201
        yaml.safe_dump(
            config.model_dump(mode="json", context={"include_defaults": True}),
            sort_keys=False,
        )
    )
    return True
