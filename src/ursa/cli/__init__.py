import logging
import sys
from argparse import SUPPRESS
from os import getenv
from pathlib import Path
from warnings import filterwarnings, warn

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
    merge_ursa_config,
    resolve_ursa_config,
)
from ursa.cli.groups import (
    add_group_subcommands,
    create_group,
    delete_group,
    list_groups,
    show_group,
    update_group,
)
from ursa.cli.print_config import add_print_config_argument, print_config
from ursa.cli.rag_management import (
    RAG_COMMANDS,
    RAG_METADATA_COMMANDS,
    add_rag_subcommands,
    handle_rag_command,
)
from ursa.util.http import inject_truststore_into_ssl

set_parsing_settings(docstring_parse_attribute_docstrings=True)
# NOTE [alui | 26 June, 2026]:
# Pydantic warnings occured around v0.16.0. Suppress for now.
filterwarnings("ignore", message="Pydantic serializer warnings:*")


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
        help=(
            "Path to a YAML/JSON file with additional configuration. "
            "Higher-precedence configuration layers override lower-precedence ones."
        ),
    )
    parser.add_argument("--log-level", default="error", type=LoggingLevel)
    add_print_config_argument(parser)
    parser.add_class_arguments(
        UrsaConfig,
        help="URSA configuration",
        skip={"agent_name", "rag_tools"},
    )
    parser.add_argument(
        "--llm_model.api_key_env", default=SUPPRESS, help=SUPPRESS
    )
    parser.add_argument(
        "--emb_model.api_key_env", default=SUPPRESS, help=SUPPRESS
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
        dest="agent_name",
        type=str,
        default=None,
        help="Name of the agent for persistence",
    )

    # Run Ursa as an MCP Server
    mcp_parser = ArgumentParser()
    mcp_parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help=(
            "Path to a YAML/JSON file with additional configuration "
            "(LLM model, endpoint, embedding model, MCP servers, etc.) "
            "for the URSA instance hosted by the MCP server. "
            "Higher-precedence configuration layers override lower-precedence ones."
        ),
    )
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


def resolve_config(
    cfg, overrides, cli_overrides=None, *, group: str | None = None
) -> UrsaConfig:
    """Produce the fully resolved UrsaConfig from the parsed arguments.

    This function merges configuration layers in precedence order and then
    applies derived resolution semantics such as temporary workspace
    materialization, top-level ``use_web`` promotion into ``agent_config``, and
    group-based endpoint policy enforcement. Consumers with a more specific
    effective group supply it before resolution.
    """
    merged = merge_ursa_config(
        cfg, overrides=overrides, cli_overrides=cli_overrides
    )
    if group is not None:
        merged = merged.model_copy(update={"group": group})
    return resolve_ursa_config(merged)


def _initialize_hitl(config: UrsaConfig):
    """Create the CLI controller and report provider initialization errors cleanly."""
    from ursa.cli.runtime import HITL

    try:
        return HITL(config)
    except Exception as exc:
        print(  # noqa: T201
            "Error: unable to initialize the language model. " + str(exc),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def _apply_legacy_name_env(cfg, overrides) -> None:
    """Apply deprecated ``URSA_NAME`` when no modern name override is set."""
    legacy_name = getenv("URSA_NAME")
    if legacy_name is None:
        return

    warn(
        "URSA_NAME is deprecated; use URSA_AGENT_NAME or --name instead.",
        FutureWarning,
        stacklevel=2,
    )
    if overrides.get("agent_name", None) is None:
        cfg["agent_name"] = legacy_name
        overrides["agent_name"] = legacy_name


def main(args=None):
    inject_truststore_into_ssl()
    parser = build_parser()
    cfg = parser.parse_args(args=args)
    overrides = parser.parse_args(args=[], defaults=False)
    cli_overrides = parser.parse_args(args=args, defaults=False, env=False)

    subcommand = cfg.get("subcommand", None)
    logging.basicConfig(level=getattr(cfg, "log_level", "error").upper())
    _apply_legacy_name_env(cfg, overrides)

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

    if subcommand in RAG_METADATA_COMMANDS:
        if handle_rag_command(cfg):
            return

    if subcommand in RAG_COMMANDS:
        cmd_config = cfg.get(subcommand, None)
        ursa_config = resolve_config(
            cfg, overrides, cli_overrides, group=cmd_config.group
        )
        if handle_rag_command(cfg, ursa_config):
            return

    if print_config(cfg, overrides, cli_overrides):
        exit(0)

    ursa_config = resolve_config(cfg, overrides, cli_overrides)
    cmd_config = cfg.get(subcommand, None) if subcommand is not None else None

    legacy_checkpoint = ursa_config.workspace / "db" / "checkpointer.db"
    if ursa_config.agent_name is None and legacy_checkpoint.is_file():
        # Intentionally print so this warning is visible regardless of log level.
        print(  # noqa: T201
            "\nWarning: URSA no longer restarts unnamed CLI sessions from "
            "db/checkpointer.db, and CLI checkpoint history is only persisted "
            "when --name is used.\n\nTo continue this history, from the workspace "
            "run 'ursa import-agent db/checkpointer.db --name <new agent name>', "
            "\nThen use '--name <new agent name>' for future CLI sessions.\n",
            file=sys.stderr,
        )

    match subcommand:
        case None:
            from ursa.cli.app import run_textual

            hitl = _initialize_hitl(ursa_config)
            run_textual(hitl)

        case "mcp-server":
            hitl = _initialize_hitl(ursa_config)
            mcp = hitl.as_mcp_server()

            run_kwargs = {
                "transport": cmd_config.transport,
                "log_level": cmd_config.log_level.upper(),
            }
            if cmd_config.transport != "stdio":
                run_kwargs["host"] = cmd_config.host
                run_kwargs["port"] = cmd_config.port
            mcp.run(**run_kwargs)
        case "exec":
            from ursa.cli.app import run_textual_once

            hitl = _initialize_hitl(ursa_config)
            run_textual_once(hitl, cmd_config.prompt)
        case _:
            logging.error(f"Unknown subcommand {subcommand}")
            raise NotImplementedError
