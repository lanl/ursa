from __future__ import annotations

from pathlib import Path

import yaml

from ursa.cli.config import UrsaConfig, resolve_ursa_config


VALID_PRINT_CONFIG_LEVELS = {"final", "file", "project", "user"}
VALID_PRINT_CONFIG_STAGES = {"merged", "resolved"}


def config_path_from_namespace(cfg) -> Path | None:
    """Return a root or subcommand-local config path from parsed CLI args."""
    config_path = getattr(cfg, "config", None)
    subcommand = cfg.get("subcommand", None)
    if subcommand is not None:
        cmd_cfg = cfg.get(subcommand, None)
        cmd_config_path = (
            getattr(cmd_cfg, "config", None) if cmd_cfg is not None else None
        )
        config_path = cmd_config_path or config_path
    return config_path


def config_for_level(cfg, search_paths: list[Path], level: str) -> UrsaConfig:
    """Return the merged config at a specific precedence level."""
    cfg_dict = cfg.as_dict()
    if cfg_dict.get("name") is not None:
        cfg_dict["agent_name"] = cfg_dict.pop("name")
    else:
        cfg_dict.pop("name", None)

    config = UrsaConfig()

    if level == "user":
        for config_path in search_paths:
            if config_path.parent.parent == Path.home() / ".config":
                config.update(UrsaConfig.from_file(config_path))
        return config

    if level == "project":
        for config_path in search_paths:
            if config_path == Path.cwd() / ".ursa" / "config.yaml":
                config.update(UrsaConfig.from_file(config_path))
        return config

    if level == "file":
        config_path = config_path_from_namespace(cfg)
        if config_path is not None:
            config.update(UrsaConfig.from_file(config_path))
        return config

    for config_path in search_paths:
        config.update(UrsaConfig.from_file(config_path))

    cli_updates = cfg_dict.copy()
    cli_updates.pop("subcommand", None)
    cli_updates.pop("config", None)
    cli_updates.pop("print_config", None)
    if cli_updates.get("rag_tools") is None:
        cli_updates.pop("rag_tools", None)

    config.update(UrsaConfig.model_validate(cli_updates, extra="ignore"))
    return config


def parse_print_config_spec(spec: str | None) -> tuple[str, str] | None:
    if spec is None:
        return None
    if spec in {True, False}:
        return ("final", "resolved") if spec else None
    if spec in VALID_PRINT_CONFIG_STAGES:
        return ("final", spec)
    if "," not in spec:
        raise ValueError(
            "--print-config must be one of merged, resolved, or LEVEL,STAGE"
        )
    level, stage = [part.strip() for part in spec.split(",", maxsplit=1)]
    if not level:
        level = "final"
    if level not in VALID_PRINT_CONFIG_LEVELS:
        raise ValueError(f"Unknown print-config level '{level}'")
    if stage not in VALID_PRINT_CONFIG_STAGES:
        raise ValueError(f"Unknown print-config stage '{stage}'")
    return level, stage


def print_config(cfg, search_paths: list[Path]) -> bool:
    """Print config according to --print-config and return whether it handled output."""
    print_config_spec = parse_print_config_spec(cfg["print_config"])
    if print_config_spec is None:
        return False
    level, stage = print_config_spec
    config = config_for_level(cfg, search_paths, level)
    if stage == "resolved":
        config = resolve_ursa_config(config)
    print(yaml.safe_dump(config.model_dump(), sort_keys=False))  # noqa: T201
    return True
