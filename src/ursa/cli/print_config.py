from __future__ import annotations

import yaml

from ursa.cli.config import (
    UrsaConfig,
    dict_diff,
    merge_ursa_config,
    resolve_ursa_config,
)

VALID_PRINT_CONFIG_LEVELS = {"final", "file", "project", "user"}
VALID_PRINT_CONFIG_STAGES = {"merged", "resolved"}


def _drop_none(value):
    """Recursively omit null values from serialized configuration."""
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value


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


def print_config(cfg, overrides) -> bool:
    """Print config according to --print-config and return whether it handled output."""
    print_config_spec = parse_print_config_spec(cfg["print_config"])
    if print_config_spec is None:
        return False
    level, stage = print_config_spec
    config = merge_ursa_config(cfg, level, overrides)
    reference = UrsaConfig()
    if stage == "resolved":
        config = resolve_ursa_config(config)
        reference = resolve_ursa_config(reference)
    output = _drop_none(
        dict_diff(
            reference.model_dump(exclude_none=True),
            config.model_dump(exclude_none=True),
        )
    )
    print(  # noqa: T201
        yaml.safe_dump(output, sort_keys=False)
    )
    return True
