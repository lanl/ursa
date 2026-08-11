import logging
import os
import sys
from pathlib import Path
from warnings import filterwarnings

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
from ursa.cli.print_config import print_config
from ursa.cli.rag_management import (
    RAG_COMMANDS,
    add_rag_subcommands,
    handle_rag_command,
)
from ursa.util.http import inject_truststore_into_ssl

set_parsing_settings(docstring_parse_attribute_docstrings=True)
# NOTE [alui | 26 June, 2026]:
# Pydantic warnings occured around v0.16.0. Suppress for now.
filterwarnings("ignore", message="Pydantic serializer warnings:*")


def _xdg_config_search_paths() -> list[Path]:
    """Return candidate XDG config paths in increasing precedence order."""
    candidates: list[Path] = []

    config_dirs = os.getenv("XDG_CONFIG_DIRS", "/etc/xdg")
    for config_dir in config_dirs.split(":"):
        if not config_dir:
            continue
        candidates.append(
            Path(config_dir).expanduser() / "ursa" / "config.yaml"
        )

    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        candidates.append(
            Path(config_home).expanduser() / "ursa" / "config.yaml"
        )
    else:
        candidates.append(Path.home() / ".config" / "ursa" / "config.yaml")

    return candidates


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
    parser.add_argument(
        "--print-config",
        nargs="?",
        const="resolved",
        default=None,
        type=str,
        help=(
            "Print configuration and exit. Defaults to resolved output. "
            "Accepted forms: --print-config, --print-config=resolved, "
            "--print-config=merged, or --print-config=LEVEL,STAGE where "
            "LEVEL is one of final, file, project, user and STAGE is merged or resolved."
        ),
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


def _config_search_paths(cfg) -> list[Path]:
    """Return config files in increasing precedence order."""
    paths: list[Path] = []
    from ursa.cli.print_config import config_path_from_namespace

    explicit_config = config_path_from_namespace(cfg)
    implicit_paths = [
        *_xdg_config_search_paths(),
        Path.cwd() / ".ursa" / "config.yaml",
    ]
    seen: set[Path] = set()
    for candidate in implicit_paths:
        candidate = candidate.expanduser()
        if candidate == explicit_config:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            paths.append(candidate)
    if explicit_config:
        paths.append(explicit_config)
    return paths


def resolve_config(cfg) -> UrsaConfig:
    """Produce the fully resolved UrsaConfig from the parsed arguments.

    This function merges configuration layers in precedence order and then
    applies derived resolution semantics such as temporary workspace
    materialization, top-level ``use_web`` promotion into ``agent_config``,
    and group-based model endpoint policy enforcement.
    """
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

    config = UrsaConfig()
    for config_path in _config_search_paths(cfg):
        config.update(UrsaConfig.from_file(config_path))

    cli_updates = cfg_dict.copy()
    cli_updates.pop("subcommand", None)
    cli_updates.pop("config", None)
    cli_updates.pop("print_config", None)
    if cli_updates.get("rag_tools") is None:
        cli_updates.pop("rag_tools", None)

    config.update(UrsaConfig.model_validate(cli_updates, extra="ignore"))
    return resolve_ursa_config(config)


def _initialize_hitl(config: UrsaConfig):
    """Create the CLI controller and report missing OpenAI credentials cleanly."""
    from openai import OpenAIError

    from ursa.cli.hitl import HITL

    try:
        return HITL(config)
    except OpenAIError as exc:
        print(  # noqa: T201
            "Error: unable to initialize the language model. " + str(exc),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


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

    if print_config(cfg, _config_search_paths(cfg)):
        exit(0)

    ursa_config = resolve_config(cfg)
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
            from ursa.cli.hitl import UrsaRepl

            hitl = _initialize_hitl(ursa_config)
            UrsaRepl(hitl).run()

        case "mcp-server":
            from ursa.cli.hitl import HITL

            hitl = HITL(ursa_config)
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
            from ursa.cli.hitl import UrsaExec

            hitl = _initialize_hitl(ursa_config)
            UrsaExec(hitl).run([cmd_config.prompt])
        case _:
            logging.error(f"Unknown subcommand {subcommand}")
            raise NotImplementedError
