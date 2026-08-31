"""Container-side entry point used by :mod:`ursa.integrations.harbor`."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
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
    totals = payload.get("totals", {})
    if not totals:
        totals = payload.get("usage_rollup", {})
    return {
        "n_input_tokens": totals.get("input_tokens"),
        "n_output_tokens": totals.get("output_tokens"),
        "cost_usd": totals.get("total_cost"),
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


def _apply_harbor_overrides(
    config: Any, model: str | None, mcp_servers: dict[str, Any]
) -> Any:
    """Apply Harbor-owned model and MCP settings to an URSA config."""
    if model:
        provider, separator, model_name = model.partition("/")
        if not separator or not provider or not model_name:
            raise ValueError(
                "Harbor model must use inference_provider/model_name syntax"
            )
        config = config.model_merge({
            "llm_model": {
                "model": model_name,
                "inference_provider": provider,
            }
        })
    config.mcp_servers = {**config.mcp_servers, **mcp_servers}
    return config


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


def main(encoded: str) -> None:
    from ursa.agents import BaseAgent
    from ursa.cli.config import UrsaConfig, load_config_file

    config = json.loads(base64.urlsafe_b64decode(encoded).decode())
    agent_class = _import_symbol(config["agent_import_path"])
    if not isinstance(agent_class, type) or not issubclass(
        agent_class, BaseAgent
    ):
        raise TypeError("Configured class is not an URSA BaseAgent subclass")

    ursa_config = UrsaConfig.model_validate(
        load_config_file(Path(config["config_file"]))
    )
    ursa_config = _apply_harbor_overrides(
        ursa_config, config.get("model"), config.get("mcp_servers", {})
    )
    ursa_config.workspace = Path(config["workspace"])
    ursa_config = ursa_config.resolve()

    metrics_path = Path(config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    agent = agent_class(
        llm=ursa_config.llm_model.init_chat_model(),
        workspace=ursa_config.workspace,
        agent_name=ursa_config.agent_name,
        group=ursa_config.group,
        thread_id=ursa_config.thread_id,
        rag_tools=ursa_config.rag_tools,
        rag_tool_embedding=(
            ursa_config.emb_model.init_embedding()
            if ursa_config.emb_model
            else None
        ),
        **_agent_config(ursa_config, agent_class),
    )
    asyncio.run(_attach_mcp_tools(agent, ursa_config.mcp_servers))
    output = agent.invoke(
        config["instruction"], save_json=True, metrics_path=str(metrics_path)
    )
    result = agent.format_result(output)
    sys.stdout.write(
        "URSA_HARBOR_RESULT="
        + json.dumps({"result": result, **_usage(metrics_path)}, default=str)
        + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1])
