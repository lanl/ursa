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
from collections.abc import Iterator
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
    suppress,
)
from pathlib import Path
from typing import Any, TextIO


class _Tee:
    """Write text to the runner stream and its durable Harbor log."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream = stream
        self._log = log

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str | None:
        return self._stream.encoding


@contextmanager
def _capture_output(log_path: Path) -> Iterator[None]:
    """Tee runner stdout and stderr into Harbor's mounted agent logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with (
            redirect_stdout(_Tee(sys.stdout, log)),
            redirect_stderr(_Tee(sys.stderr, log)),
        ):
            yield


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


def _run(config: dict[str, Any]) -> None:
    from ursa.agents import BaseAgent
    from ursa.cli.config import UrsaConfig, load_config_file
    from ursa.util import Checkpointer
    from ursa.util.events import configure_event_logging

    configure_event_logging(rich=False)
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
        asyncio.run(_attach_mcp_tools(agent, ursa_config.mcp_servers))
        output = agent.invoke(
            config["instruction"],
            save_json=True,
            metrics_path=str(metrics_path),
        )
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


def main(encoded: str | None = None) -> None:
    if encoded is None:
        encoded = sys.argv[1]
    config = json.loads(base64.urlsafe_b64decode(encoded).decode())
    log_path = Path(config["log_path"])
    try:
        with _capture_output(log_path):
            _run(config)
    except BaseException:
        with suppress(OSError):
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        raise


if __name__ == "__main__":
    main(sys.argv[1])
