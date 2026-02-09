import logging
from pathlib import Path

from jsonargparse import ArgumentParser, set_parsing_settings

from ursa import __version__
from ursa.cli.config import (
    LoggingLevel,
    MCPServerConfig,
    UrsaConfig,
    deep_merge_dicts,
    dict_diff,
)

set_parsing_settings(docstring_parse_attribute_docstrings=True)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ursa",
        description="URSA: The Universal Research and Scientific Agent",
        env_prefix="URSA",
        version=__version__,
        default_env=True,
    )
    subparsers = parser.add_subcommands(required=False)

    # Default -> Launch a CLI interface
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Path to a YAML/JSON file with additional configuration. CLI Opts have priority",
    )
    parser.add_argument("--log-level", default="error", type=LoggingLevel)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the Ursa configuration and exit",
    )
    parser.add_class_arguments(UrsaConfig, help="URSA configuration")

    # Run Ursa as an MCP Server
    mcp_parser = ArgumentParser()
    mcp_parser.add_class_arguments(MCPServerConfig, help="MCP server options")
    subparsers.add_subcommand(
        "mcp-server",
        mcp_parser,
        help="[Experimental] Run URSA as an MCP server",
        dest="subcommand",
    )

    return parser


def resolve_config(cfg) -> UrsaConfig:
    """Produce the effective UrsaConfig from the parsed arguments."""
    cli_config = UrsaConfig.from_namespace(cfg)
    config_path = getattr(cfg, "config", None)
    if not config_path:
        return cli_config

    defaults = UrsaConfig().model_dump()
    cli_data = cli_config.model_dump()
    cli_overrides = dict_diff(defaults, cli_data)

    file_config = UrsaConfig.from_file(config_path)
    file_data = file_config.model_dump(mode="python")
    merged_data = deep_merge_dicts(file_data, cli_overrides)
    return UrsaConfig.model_validate(merged_data)


def main(args=None):
    parser = build_parser()
    cfg = parser.parse_args(args=args)
    ursa_config = resolve_config(cfg)

    subcommand = cfg.get("subcommand", None)
    cmd_config = cfg.get(subcommand, None) if subcommand is not None else None

    logging.basicConfig(level=getattr(cfg, "log_level", "error").upper())

    if cfg["print_config"]:
        import yaml

        print(yaml.safe_dump(ursa_config.model_dump(), sort_keys=False))
        exit(0)

    match subcommand:
        case None:
            from ursa.cli.hitl import HITL, UrsaRepl

            hitl = HITL(ursa_config)
            UrsaRepl(hitl).run()
        case "mcp-server":
            from ursa.cli.hitl import HITL

            hitl = HITL(ursa_config)
            mcp = hitl.as_mcp_server(
                host=cmd_config.host,
                port=cmd_config.port,
                log_level=cmd_config.log_level.upper(),
            )
            mcp.run(transport=cmd_config.transport)
