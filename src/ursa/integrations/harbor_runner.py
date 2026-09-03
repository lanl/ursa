"""Container-side entry point used by :mod:`ursa.integrations.harbor`."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import signal
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any


def _import_symbol(path: str) -> Any:
    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        raise ValueError("agent_import_path must use module:Class syntax")
    value: Any = importlib.import_module(module_name)
    for component in symbol_name.split("."):
        value = getattr(value, component)
    return value


def _usage(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.is_file():
        return {}
    payload = json.loads(metrics_path.read_text())
    events = payload.get("llm_events", [])
    event_usage = [
        usage
        for event in events
        if (usage := event.get("metrics", {}).get("usage_rollup"))
    ]
    totals = payload.get("usage_rollup", {})
    if not totals and event_usage:
        totals = {
            key: sum(usage.get(key, 0) or 0 for usage in event_usage)
            for key in ("input_tokens", "output_tokens")
        }
    if not totals:
        # Compatibility with metrics emitted before usage was separated from
        # timing totals.
        totals = payload.get("totals", {})
    costs = payload.get("costs", {})
    return {
        "n_input_tokens": totals.get("input_tokens"),
        "n_output_tokens": totals.get("output_tokens"),
        "cost_usd": costs.get("total_usd", totals.get("total_cost")),
    }


def _agent_config(config: Any, agent_class: type[Any]) -> dict[str, Any]:
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
    return dict(
        config.agent_config.get(key, config.agent_config.get(snake, {}))
    )


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


def _export_checkpoint(agent: Any, artifacts_dir: Path) -> Path:
    """Snapshot the agent's live checkpoint database into Harbor artifacts."""
    destination = artifacts_dir / "ursa" / "checkpointer.db"
    checkpointer = getattr(agent, "checkpointer", None)
    connection = getattr(checkpointer, "conn", None)
    if connection is not None:
        database = connection.execute("PRAGMA database_list").fetchone()
        source = Path(database[2])
    else:
        source = agent.den / "db" / "checkpointer.db"
    if source.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        sqlite3.connect(source) as source_db,
        sqlite3.connect(destination) as destination_db,
    ):
        source_db.backup(destination_db)
    return destination


def _close_checkpoint(checkpointer: Any) -> None:
    """Flush and close the artifact-backed SQLite checkpoint database."""
    connection = checkpointer.conn
    failure: BaseException | None = None
    try:
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException as exc:
        failure = exc
    try:
        connection.close()
    except BaseException as exc:
        if failure is None:
            failure = exc
    if failure is not None:
        raise failure


def main(encoded: str) -> None:
    from ursa.agents import BaseAgent
    from ursa.cli.config import UrsaConfig, load_config_file
    from ursa.util import Checkpointer

    config = json.loads(base64.urlsafe_b64decode(encoded).decode())
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

    metrics_path = Path(config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(config["artifacts_dir"])
    checkpointer = Checkpointer.from_workspace(artifacts_dir, db_dir="ursa")
    agent_options = _agent_config(ursa_config, agent_class)
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
        asyncio.run(_attach_mcp_tools(agent, ursa_config.mcp_servers))
        output = agent.invoke(
            config["instruction"],
            save_json=True,
            metrics_path=str(metrics_path),
        )
        _export_checkpoint(agent, artifacts_dir)
        result = agent.format_result(output)
        sys.stdout.write(
            "URSA_HARBOR_RESULT="
            + json.dumps(
                {"result": result, **_usage(metrics_path)}, default=str
            )
            + "\n"
        )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            _close_checkpoint(checkpointer)
        except Exception:
            if failure is None:
                raise
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
