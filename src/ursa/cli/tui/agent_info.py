"""Normalized agent and tool details for the Textual agent browser."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolArgumentDetails:
    """One argument accepted by a configured tool."""

    name: str
    type_name: str
    required: bool
    description: str


@dataclass(frozen=True)
class ToolDetails:
    """Display-safe metadata for one configured tool."""

    name: str
    class_name: str
    description: str
    schema_name: str
    return_direct: bool | None
    arguments: tuple[ToolArgumentDetails, ...]
    mcp_server: str = ""


@dataclass(frozen=True)
class AgentDetails:
    """Description, configuration, and tools for one agent tab."""

    name: str
    description: str
    config: tuple[tuple[str, str], ...]
    tools: tuple[ToolDetails, ...]
    tools_loaded: bool = True
    tool_error: str = ""


def _schema_name(schema: Any) -> str:
    if schema is None:
        return "none"
    return str(
        getattr(schema, "__name__", None)
        or getattr(schema.__class__, "__name__", None)
        or schema
    )


def _schema_arguments(schema: Any) -> tuple[ToolArgumentDetails, ...]:
    schema_json = schema if isinstance(schema, Mapping) else None
    if schema_json is None:
        for method_name in ("model_json_schema", "schema"):
            method = getattr(schema, method_name, None)
            if callable(method):
                try:
                    schema_json = method()
                except Exception:  # pragma: no cover - provider schema code
                    continue
                break
    if not isinstance(schema_json, Mapping):
        return ()
    properties = schema_json.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    required = set(schema_json.get("required") or ())
    arguments = []
    for name, metadata in properties.items():
        metadata = metadata if isinstance(metadata, Mapping) else {}
        type_name = str(
            metadata.get("type")
            or metadata.get("title")
            or metadata.get("$ref")
            or "any"
        )
        arguments.append(
            ToolArgumentDetails(
                name=str(name),
                type_name=type_name.rsplit("/", 1)[-1],
                required=name in required,
                description=str(metadata.get("description") or ""),
            )
        )
    return tuple(arguments)


def _tool_details(
    name: str,
    tool: Any,
    mcp_server: str = "",
    *,
    include_schema: bool = True,
) -> ToolDetails:
    schema = None
    if include_schema:
        schema = getattr(tool, "tool_call_schema", None)
        if schema is None:
            schema = getattr(tool, "args_schema", None)
    return ToolDetails(
        name=str(getattr(tool, "name", None) or name),
        class_name=tool.__class__.__name__,
        description=str(
            getattr(tool, "description", None) or "No description available."
        ).strip(),
        schema_name=_schema_name(schema),
        return_direct=getattr(tool, "return_direct", None),
        arguments=_schema_arguments(schema) if include_schema else (),
        mcp_server=mcp_server,
    )


def _configured_tools(
    agent: Any,
    tool_sources: Mapping[str, str],
    *,
    include_schema: bool = True,
) -> tuple[ToolDetails, ...]:
    tools = getattr(agent, "tools", None)
    if isinstance(tools, Mapping):
        items = tools.items()
    elif isinstance(tools, (list, tuple)):
        items = (
            (str(getattr(tool, "name", None) or index), tool)
            for index, tool in enumerate(tools)
        )
    else:
        return ()
    return tuple(
        _tool_details(
            str(name),
            tool,
            str(tool_sources.get(str(name)) or ""),
            include_schema=include_schema,
        )
        for name, tool in sorted(items, key=lambda item: str(item[0]))
    )


def load_agent_details(runtime: Any) -> tuple[AgentDetails, ...]:
    """Snapshot configured-agent metadata without instantiating any agents."""
    details = []
    # Dict insertion order is the canonical runtime/UI order.
    for name, wrapper in runtime.agents.items():
        description = str(
            getattr(wrapper, "description", None) or "No description available."
        ).strip()
        config = tuple(
            (str(key), str(value))
            for key, value in (getattr(wrapper, "config", None) or {}).items()
        )
        actual = (
            getattr(wrapper, "_agent", None)
            if hasattr(wrapper, "_agent")
            else wrapper
        )
        details.append(
            AgentDetails(
                name=str(name),
                description=description,
                config=config,
                # Expose initialized tools immediately without invoking schema
                # generation; the selected tab enriches their arguments in its
                # off-thread hydration worker.
                tools=_configured_tools(
                    actual,
                    getattr(wrapper, "tool_sources", {}) or {},
                    include_schema=False,
                )
                if actual is not None
                else (),
                tools_loaded=False,
            )
        )
    return tuple(details)


async def load_agent_tools(runtime: Any, name: str) -> tuple[ToolDetails, ...]:
    """Instantiate one agent, if needed, and return its display-safe tools."""
    use_agent = getattr(runtime, "use_agent", None)

    async def extract(wrapper: Any) -> tuple[ToolDetails, ...]:
        remote_tools = getattr(wrapper, "remote_tools", None)
        if callable(remote_tools):
            details = await remote_tools()
            return tuple(
                ToolDetails(
                    name=str(item["name"]),
                    class_name=str(item["class_name"]),
                    description=str(item["description"]),
                    schema_name=_schema_name(item.get("schema")),
                    return_direct=item.get("return_direct"),
                    arguments=_schema_arguments(item.get("schema")),
                    mcp_server=str(item.get("mcp_server") or ""),
                )
                for item in details
            )
        actual = getattr(wrapper, "_agent", None) or wrapper
        # Schema conversion can invoke provider/Pydantic schema generation for
        # every tool, so it should not block Textual's event loop either.
        return await asyncio.to_thread(
            _configured_tools,
            actual,
            getattr(wrapper, "tool_sources", {}) or {},
        )

    async def snapshot() -> tuple[ToolDetails, ...]:
        configured = getattr(runtime, "agents", {}).get(name)
        if callable(use_agent) and callable(
            getattr(configured, "remote_tools", None)
        ):
            async with use_agent(name) as wrapper:
                return await extract(wrapper)
        return await extract(await runtime.get_agent(name))

    # Keep the lease-owning operation alive if Textual dismisses its worker:
    # the worker can clean up immediately, while this task retains the lease
    # until non-cancellable thread work has actually finished.
    operation = asyncio.create_task(snapshot())
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        operation.add_done_callback(_consume_task_exception)
        raise


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()
