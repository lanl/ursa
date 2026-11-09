"""Invoke named workflow members in their own persistent conversation."""

import asyncio
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables.config import (
    ensure_config,
    var_child_runnable_config,
)


@contextmanager
def _invocation_kwargs(
    agent: Any, kwargs: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    if not getattr(agent, "agent_name", None):
        yield kwargs
        return

    config = ensure_config(kwargs.get("config"))
    # A named member owns its checkpoint file. Parent graph runtime objects
    # would redirect its checkpoints/storage into the parent's graph instead.
    config["configurable"] = {
        key: value
        for key, value in config.get("configurable", {}).items()
        if not key.startswith("__pregel_")
        and key
        not in {"thread_id", "checkpoint_ns", "checkpoint_id", "checkpoint_map"}
    }
    config["configurable"].update(thread_id="ursa", checkpoint_ns="")
    config["metadata"] = {
        key: value
        for key, value in config.get("metadata", {}).items()
        if not key.startswith("langgraph_")
        and key not in {"thread_id", "checkpoint_ns"}
    }
    config["metadata"]["thread_id"] = "ursa"

    # LangChain also inherits config through a ContextVar. Isolate that path
    # as well as the explicit argument, and restore it even on cancellation.
    token = var_child_runnable_config.set(config)
    try:
        yield {**kwargs, "config": config}
    finally:
        var_child_runnable_config.reset(token)


def invoke_agent(agent: Any, inputs: Any, **kwargs: Any) -> Any:
    with _invocation_kwargs(agent, kwargs) as invocation:
        return agent.invoke(inputs, **invocation)


async def ainvoke_agent(agent: Any, inputs: Any, **kwargs: Any) -> Any:
    with _invocation_kwargs(agent, kwargs) as invocation:
        ainvoke = getattr(agent, "ainvoke", None)
        if callable(ainvoke):
            result = ainvoke(inputs, **invocation)
            return await result if inspect.isawaitable(result) else result
        return await asyncio.to_thread(agent.invoke, inputs, **invocation)
