import logging
from pathlib import Path

import yaml
from jsonargparse import ArgumentParser, set_parsing_settings

from ursa import __version__
from ursa.cli.agent_management import (
    add_agent_management_subcommands,
    copy_agent,
    delete_agent,
    import_agent,
    list_agents,
    save_agent,
    share_agent,
    show_agent,
)
from ursa.cli.config import (
    LoggingLevel,
    MCPServerConfig,
    UrsaConfig,
)
from ursa.cli.groups import (
    add_group_subcommands,
    create_group,
    delete_group,
    list_groups,
    show_group,
    update_group,
)
from ursa.cli.rag_management import (
    RAG_COMMANDS,
    add_rag_subcommands,
    handle_rag_command,
)
from ursa.util.http import inject_truststore_into_ssl

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
    parser.add_class_arguments(
        UrsaConfig,
        help="URSA configuration",
        skip={"agent_name", "rag_tools"},
    )
    parser.add_argument(
        "--rag-tools",
        dest="rag_tools",
        default=None,
        help="Comma-separated persisted RAG agent names to bind as tools.",
    )
    parser.add_argument(
        "--use-web",
        dest="use_web",
        action="store_true",
        default=False,
        help="Enable web-search tools for ChatAgent and ExecutionAgent.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the agent for persistence",
    )

    # Run Ursa as an MCP Server
    mcp_parser = ArgumentParser()
    mcp_parser.add_class_arguments(MCPServerConfig, help="MCP server options")
    subparsers.add_subcommand(
        "mcp-server",
        mcp_parser,
        help="[Experimental] Run URSA as an MCP server",
        dest="subcommand",
    )

    # Agent group management commands
    add_group_subcommands(subparsers)

    # Agent management commands
    add_agent_management_subcommands(subparsers)

    # Persistent RAG management commands
    add_rag_subcommands(subparsers)

    exec_parser = ArgumentParser()
    exec_parser.add_argument("prompt", type=str)
    subparsers.add_subcommand(
        "exec",
        exec_parser,
        help="Run Ursa non-interactively",
    )

    return parser


def _config_path_from_namespace(cfg) -> Path | None:
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


def resolve_config(cfg) -> UrsaConfig:
    """Produce the effective UrsaConfig from the parsed arguments."""
    cfg_dict = cfg.as_dict()
    # Change `name` to `agent_name` for consistency with agent
    #    arguments.
    # TODO: Longer term, we should make our agents use `name`
    #    as the argument for the class, but this is a problem
    #    with the current class property `name` that is used by
    #    the CLI
    if cfg_dict.get("name") is not None:
        cfg_dict["agent_name"] = cfg_dict.pop("name")
    else:
        cfg_dict.pop("name", None)

    cli_config = UrsaConfig.model_validate(cfg_dict, extra="ignore")
    config_path = _config_path_from_namespace(cfg)
    config = UrsaConfig()
    if config_path:
        config.update(UrsaConfig.from_file(config_path))
    return config.update(cli_config)


def main(args=None):
    inject_truststore_into_ssl()
    parser = build_parser()
    cfg = parser.parse_args(args=args)

    subcommand = cfg.get("subcommand", None)
    logging.basicConfig(level=getattr(cfg, "log_level", "error").upper())

    match subcommand:
        case "list-groups":
            list_groups()
            return
        case "create-group":
            cmd_config = cfg.get(subcommand, None)
            create_group(cmd_config.group_name, cmd_config.config_file)
            return
        case "delete-group":
            cmd_config = cfg.get(subcommand, None)
            delete_group(cmd_config.group_name)
            return
        case "show-group":
            cmd_config = cfg.get(subcommand, None)
            show_group(cmd_config.group_name)
            return
        case "update-group":
            cmd_config = cfg.get(subcommand, None)
            update_group(cmd_config.group_name, cmd_config.config_file)
            return
        case "list-agents":
            cmd_config = cfg.get(subcommand, None)
            list_agents(cmd_config.group)
            return
        case "show-agent":
            cmd_config = cfg.get(subcommand, None)
            show_agent(cmd_config.name, cmd_config.group)
            return
        case "delete-agent":
            cmd_config = cfg.get(subcommand, None)
            delete_agent(cmd_config.name, cmd_config.group)
            return
        case "save-agent":
            cmd_config = cfg.get(subcommand, None)
            save_agent(cmd_config.name, cmd_config.group)
            return
        case "copy-agent":
            cmd_config = cfg.get(subcommand, None)
            copy_agent(
                cmd_config.name,
                cmd_config.source_agent,
                cmd_config.group,
                cmd_config.from_group,
            )
            return
        case "share-agent":
            cmd_config = cfg.get(subcommand, None)
            share_agent(
                cmd_config.name, cmd_config.group, cmd_config.no_checkpoint
            )
            return
        case "import-agent":
            cmd_config = cfg.get(subcommand, None)
            import_agent(
                cmd_config.archive_file, cmd_config.group, cmd_config.name
            )
            return

    if subcommand in RAG_COMMANDS:
        ursa_config = resolve_config(cfg)
        if handle_rag_command(cfg, ursa_config):
            return

    ursa_config = resolve_config(cfg)
    cmd_config = cfg.get(subcommand, None) if subcommand is not None else None

    if cfg["print_config"]:
        print(yaml.safe_dump(ursa_config.model_dump(), sort_keys=False))
        exit(0)

    match subcommand:
        case None:
            from ursa.cli.hitl import HITL, UrsaRepl

            hitl = HITL(ursa_config)
            UrsaRepl(hitl).run()

        case "exec":
            from ursa.cli.hitl import HITL, UrsaRepl

            hitl = HITL(ursa_config)
            UrsaRepl(hitl).run_prompt(cmd_config.prompt)

        case "mcp-server":
            from ursa.cli.hitl import HITL

            hitl = HITL(ursa_config)
            mcp = hitl.as_mcp_server(
                host=cmd_config.host,
                port=cmd_config.port,
                log_level=cmd_config.log_level.upper(),
            )
            mcp.run(transport=cmd_config.transport)
