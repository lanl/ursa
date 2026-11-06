import asyncio
import os
import sys
from collections.abc import Mapping
from datetime import timedelta
from typing import Annotated, Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, BeforeValidator, ValidationError

from ursa.util.http import build_mcp_httpx_async_client
from ursa.util.secrets import SecretReference


class UrsaMCPClient(MultiServerMCPClient):
    """MCP client with tool provenance and quiet stdio subprocesses."""

    def __init__(self, connections: dict[str, Any]) -> None:
        wrapped: dict[str, Any] = {}
        for name, connection in connections.items():
            connection = dict(connection)
            if connection.get("transport") == "stdio":
                command = str(connection["command"])
                args = list(connection.get("args") or [])
                connection["command"] = sys.executable
                connection["args"] = [
                    "-m",
                    "ursa.util.mcp_stdio_proxy",
                    os.devnull,
                    command,
                    *args,
                ]
            wrapped[name] = connection
        super().__init__(wrapped)


async def load_mcp_tools_with_sources(
    client: MultiServerMCPClient,
) -> tuple[list[BaseTool], dict[str, str]]:
    """Load MCP tools and retain their configured server names separately."""
    connections = getattr(client, "connections", None)
    if not isinstance(connections, Mapping):
        return await client.get_tools(), {}

    server_names = list(connections)
    tools_by_server = await asyncio.gather(
        *(
            client.get_tools(server_name=server_name)
            for server_name in server_names
        )
    )
    tools: list[BaseTool] = []
    sources: dict[str, str] = {}
    for server_name, server_tools in zip(
        server_names, tools_by_server, strict=True
    ):
        tools.extend(server_tools)
        sources.update({tool.name: server_name for tool in server_tools})
    return tools, sources


def mcp_tool_specs_from_details(
    details: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract transport-safe MCP schemas from a runner's tool description.

    Tool discovery opens a network session and can dominate child-process
    startup. The host captures the root agent's already-discovered schemas and
    passes these small, serializable specs to equivalent child agents. A child
    still builds its own client connection for actual invocations.
    """
    specs: list[dict[str, Any]] = []
    for detail in details:
        server = detail.get("mcp_server")
        name = detail.get("name")
        # Prefer the exact MCP input schema over the tool's display schema.
        # Some LangChain wrappers enrich their display schema with generated
        # titles and descriptions; those must not alter what we send back to
        # the MCP adapter.
        schema = detail.get("mcp_input_schema", detail.get("schema"))
        if not (
            isinstance(server, str)
            and server
            and isinstance(name, str)
            and name
            and isinstance(schema, Mapping)
        ):
            continue
        specs.append({
            "server": server,
            "name": name,
            "description": str(detail.get("description") or ""),
            "input_schema": dict(schema),
            **(
                {"metadata": dict(detail["mcp_metadata"])}
                if isinstance(detail.get("mcp_metadata"), Mapping)
                else {}
            ),
        })
    return specs


def mcp_tools_from_specs(
    client: MultiServerMCPClient,
    specs: list[Mapping[str, Any]],
) -> tuple[list[BaseTool], dict[str, str]]:
    """Rebuild MCP wrappers from captured schemas without rediscovery I/O."""
    connections = getattr(client, "connections", None)
    if not isinstance(connections, Mapping):
        raise TypeError("MCP client does not expose connection settings")
    tools: list[BaseTool] = []
    sources: dict[str, str] = {}
    for spec in specs:
        server = spec.get("server")
        name = spec.get("name")
        schema = spec.get("input_schema")
        if not (
            isinstance(server, str)
            and isinstance(name, str)
            and isinstance(schema, Mapping)
        ):
            raise ValueError("Invalid cached MCP tool specification")
        try:
            connection = connections[server]
        except KeyError:
            raise ValueError(
                f"Cached MCP tool {name!r} refers to unknown server {server!r}"
            ) from None
        tool = MCPTool(
            name=name,
            description=str(spec.get("description") or ""),
            inputSchema=dict(schema),
        )
        wrapped = convert_mcp_tool_to_langchain_tool(
            None,
            tool,
            connection=connection,
            server_name=server,
        )
        metadata = spec.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("Invalid cached MCP tool metadata")
            # Preserve the adapter's annotations and _meta, including hints
            # such as readOnlyHint/destructiveHint used by tool consumers.
            wrapped.metadata = dict(metadata)
        tools.append(wrapped)
        sources[wrapped.name] = server
    return tools, sources


def validate_server_parameters(config: dict):
    if not isinstance(config, dict):
        return config
    transport_hint = config.get("transport")
    payload = {k: v for k, v in config.items() if k != "transport"}
    if isinstance(headers := payload.get("headers"), dict):
        payload["headers"] = {
            name: SecretReference.maybe_validate(value)
            for name, value in headers.items()
        }
    if transport_hint == "stdio":
        return StdioServerParameters(**payload)
    elif transport_hint == "sse":
        return SseServerParameters(**payload)
    elif transport_hint in ["streamable_http", "streamable-http"]:
        return StreamableHttpParameters(**payload)
    elif transport_hint is None:
        # Let Pydantic infer (backwards compatibility)
        for candidate in (
            StdioServerParameters,
            StreamableHttpParameters,
            SseServerParameters,
        ):
            try:
                return candidate(**payload)
            except ValidationError:
                continue
        else:
            raise ValueError(
                f"Unable to determine transport for MCP server '{config}'. "
                "Provide 'transport' with one of: stdio, sse, streamable_http."
            )
    else:
        raise ValueError(
            f"Unsupported MCP transport '{transport_hint}' for server '{config}'."
        )


ServerParameters = Annotated[
    StdioServerParameters | SseServerParameters | StreamableHttpParameters,
    BeforeValidator(validate_server_parameters),
]


def parse_server_parameters(config: Any) -> ServerParameters:
    """Parse one serialized MCP server configuration."""
    if isinstance(config, BaseModel):
        return config
    return validate_server_parameters(dict(config))


def transport(sp: ServerParameters) -> str:
    if isinstance(sp, StdioServerParameters):
        return "stdio"
    elif isinstance(sp, StreamableHttpParameters):
        return "streamable_http"
    elif isinstance(sp, SseServerParameters):
        return "sse"
    else:
        raise RuntimeError("Transport for {sp} is unknown")


def start_mcp_client(
    server_configs: Mapping[str, ServerParameters | dict],
) -> UrsaMCPClient:
    client_config = {}
    for server, config in server_configs.items():
        if not isinstance(config, BaseModel):
            config = validate_server_parameters(dict(**config))
        connection = {
            **config.model_dump(),
            "transport": transport(config),
        }
        if headers := connection.get("headers"):
            connection["headers"] = {
                name: _resolve_header(value, server)
                for name, value in headers.items()
            }
        if isinstance(config, (SseServerParameters, StreamableHttpParameters)):
            connection["httpx_client_factory"] = build_mcp_httpx_async_client
        client_config[server] = connection
    return UrsaMCPClient(client_config)


def _resolve_header(value, server_name: str):
    """Resolve a typed secret template in an MCP HTTP header."""
    reference = (
        SecretReference.model_validate(value)
        if isinstance(value, dict)
        else value
    )
    if isinstance(reference, SecretReference):
        rendered = reference.bind_keyring_name(server_name).get_secret_value()
        if rendered is None:
            raise ValueError(
                f"Secret for MCP server '{server_name}' is not set"
            )
        return rendered
    return value


def serialize_server_parameters(
    config: ServerParameters,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
    exclude_unset: bool = False,
):
    """Serialize MCP parameters for configuration files and process transport."""
    values = {
        "transport": transport(config),
        **config.model_dump(
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        ),
    }
    if exclude_unset:
        allowed = set(config.model_fields_set) | {"transport"}
        values = {key: value for key, value in values.items() if key in allowed}
    for key, value in values.items():
        if isinstance(value, timedelta):
            values[key] = value.total_seconds()
    return values


# Keep the private spelling for callers predating the public config helper.
_serialize_server_config = serialize_server_parameters
