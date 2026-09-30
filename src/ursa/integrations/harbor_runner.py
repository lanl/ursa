"""Container-side entry point used by :mod:`ursa.integrations.harbor`."""

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import json
import signal
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

from rich.console import Console  # noqa: TID251

from ursa.observability.jsonl_logger import JSONLLogEventHandler


def _import_symbol(path: str) -> Any:
    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        raise ValueError("agent_import_path must use module:Class syntax")
    value: Any = importlib.import_module(module_name)
    for component in symbol_name.split("."):
        value = getattr(value, component)
    return value


def _agent_config(
    config: Any,
    agent_class: type[Any],
    *,
    use_web: bool | None = None,
) -> dict[str, Any]:
    """Select the conventional ``agent_config`` entry for an agent class."""
    name = agent_class.__name__.removesuffix("Agent")
    snake = "".join(
        f"_{char.lower()}" if char.isupper() else char for char in name
    )
    snake = snake.removeprefix("_")
    aliases = {
        "execution": "execute",
        "hypothesizer": "hypothesize",
        "planning": "plan",
        "prompting": "prompt",
    }
    key = aliases.get(snake, snake)
    options = dict(
        config.agent_config.get(key, config.agent_config.get(snake, {}))
    )
    supports_web = "use_web" in options or any(
        "use_web" in inspect.signature(base).parameters
        for base in agent_class.__mro__
    )
    if use_web is not None and supports_web:
        options["use_web"] = use_web
    return options


async def _attach_mcp_tools(agent: Any, mcp_servers: dict[str, Any]) -> None:
    if not mcp_servers:
        return
    from ursa.agents.base import AgentWithTools
    from ursa.util.mcp import start_mcp_client

    if not isinstance(agent, AgentWithTools):
        raise TypeError(
            f"{type(agent).__name__} cannot use the configured Harbor MCP servers"
        )
    await agent.add_mcp_tools(start_mcp_client(mcp_servers))


async def _close_checkpoint(checkpointer: Any) -> None:
    """Flush and close the artifact-backed SQLite checkpoint database."""
    connection = checkpointer.conn
    failure: BaseException | None = None
    try:
        await connection.commit()
        await connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException as exc:
        failure = exc
    try:
        await connection.close()
    except BaseException as exc:
        if failure is None:
            failure = exc
    join = getattr(connection, "join", None)
    if callable(join):
        try:
            await asyncio.to_thread(join)
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


async def _run(config: dict[str, Any]) -> None:
    from ursa.agents import BaseAgent
    from ursa.cli.callbacks import HITLLogEventHandler
    from ursa.cli.config import UrsaConfig, load_config_file
    from ursa.util import Checkpointer

    agent_class = _import_symbol(config["agent_import_path"])
    if not isinstance(agent_class, type) or not issubclass(
        agent_class, BaseAgent
    ):
        raise TypeError("Configured class is not an URSA BaseAgent subclass")

    ursa_config = UrsaConfig.model_validate(
        load_config_file(Path(config["config_file"]))
    )
    ursa_config.workspace = Path(config["workspace"])
    ursa_config = ursa_config.resolve()

    logs_dir = Path(config["log_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "ursa.log"
    jsonl_path = logs_dir / "ursa.jsonl"
    checkpointer = await Checkpointer.async_from_workspace(logs_dir)
    agent_options = _agent_config(
        ursa_config,
        agent_class,
        use_web=config.get("use_web"),
    )
    agent_options["checkpointer"] = checkpointer
    agent = agent_class(
        llm=ursa_config.llm_model.init_chat_model(),
        workspace=ursa_config.workspace,
        agent_name=ursa_config.agent_name or "harbor",
        group=ursa_config.group,
        thread_id=ursa_config.thread_id,
        rag_tools=ursa_config.rag_tools,
        rag_tool_embedding=(
            ursa_config.emb_model.init_embedding()
            if ursa_config.emb_model
            else None
        ),
        **agent_options,
    )

    def terminate(_signum: int, _frame: Any) -> None:
        raise SystemExit(143)

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    failure: BaseException | None = None
    try:
        await _attach_mcp_tools(agent, ursa_config.mcp_servers)
        with (
            log_path.open("a", encoding="utf-8", buffering=1) as log_file,
            jsonl_path.open("a", encoding="utf-8", buffering=1) as jsonl_file,
        ):
            callbacks = [
                HITLLogEventHandler(
                    console=Console(
                        file=log_file,
                        force_terminal=False,
                        force_interactive=False,
                        color_system=None,
                    ),
                    workspace=ursa_config.workspace,
                ),
                JSONLLogEventHandler(jsonl_file),
            ]
            output = await agent.ainvoke(
                config["instruction"],
                config={"callbacks": callbacks},
            )
        result = agent.format_result(output)
        (logs_dir / "ursa_result.out").write_text(
            result
            if isinstance(result, str)
            else json.dumps(result, default=str),
            encoding="utf-8",
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            await _close_checkpoint(checkpointer)
        except Exception:
            if failure is None:
                raise
            traceback.print_exc(file=sys.stderr)


def main(encoded: str) -> None:
    config = json.loads(base64.urlsafe_b64decode(encoded).decode())
    log_path = Path(config["log_dir"]) / "ursa.log"
    try:
        asyncio.run(_run(config))
    except BaseException:
        with suppress(OSError):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        raise
