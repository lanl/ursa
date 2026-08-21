"""Normalized agent and tool details for the Textual agent browser."""

from __future__ import annotations

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


def _tool_details(name: str, tool: Any, mcp_server: str = "") -> ToolDetails:
    schema = getattr(tool, "args_schema", None)
    return ToolDetails(
        name=str(getattr(tool, "name", None) or name),
        class_name=tool.__class__.__name__,
        description=str(
            getattr(tool, "description", None) or "No description available."
        ).strip(),
        schema_name=_schema_name(schema),
        return_direct=getattr(tool, "return_direct", None),
        arguments=_schema_arguments(schema),
        mcp_server=mcp_server,
    )


def _configured_tools(
    agent: Any, tool_sources: Mapping[str, str]
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
        )
        for name, tool in sorted(items, key=lambda item: str(item[0]))
    )


async def load_agent_details(hitl: Any) -> tuple[AgentDetails, ...]:
    """Load all configured agents, including tools attached during startup."""
    details = []
    get_agent = getattr(hitl, "get_agent", None)
    # Dict insertion order is the canonical UI order established in HITL.
    for name, wrapper in hitl.agents.items():
        description = str(
            getattr(wrapper, "description", None) or "No description available."
        ).strip()
        config = tuple(
            (str(key), str(value))
            for key, value in (getattr(wrapper, "config", None) or {}).items()
        )
        actual = getattr(wrapper, "_agent", None)
        tool_error = ""
        if actual is None and callable(get_agent):
            try:
                initialized = await get_agent(name)
                actual = getattr(initialized, "_agent", None) or initialized
            except Exception as exc:  # keep other agent tabs usable
                tool_error = f"{type(exc).__name__}: {exc}"
        if actual is None:
            actual = wrapper
        details.append(
            AgentDetails(
                name=str(name),
                description=description,
                config=config,
                tools=_configured_tools(
                    actual,
                    getattr(wrapper, "tool_sources", {}) or {},
                ),
                tool_error=tool_error,
            )
        )
    return tuple(details)
