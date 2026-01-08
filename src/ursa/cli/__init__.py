import logging

from jsonargparse import ArgumentParser, set_parsing_settings

from ursa import __version__
from ursa.cli.config import LoggingLevel, MCPServerConfig, UrsaConfig

set_parsing_settings(docstring_parse_attribute_docstrings=True)


def main(args=None):
    parser = ArgumentParser("ursa", env_prefix="URSA_", version=__version__)
    subparsers = parser.add_subcommands(required=False)

    # Default -> Launch a cli interface
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(UrsaConfig, help="URSA Config")
    parser.add_argument("--log-level", default="error", type=LoggingLevel)

    # Run Ursa as an MCP Server
    mcp_parser = ArgumentParser()
    mcp_parser.add_class_arguments(MCPServerConfig)
    subparsers.add_subcommand(
        "mcp-server",
        mcp_parser,
        help="[Experimental] Run URSA as an MCP server",
        dest="mcp_server",
    )

    cfg = parser.parse_args(args=args)
    ursa_config = UrsaConfig.from_namespace(cfg)
    cmd_config = cfg.get(cfg.subcommand, None)
    logging.basicConfig(level=cfg.log_level.upper())
    match cfg.subcommand:
        case None:
            from ursa.cli.hitl import HITL, UrsaRepl

            ursa_config = UrsaConfig.from_namespace(cfg)
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
