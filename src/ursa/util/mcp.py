import asyncio
import os
import sys
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters as StdioServerParameters
from mcp.client.session_group import (
    ServerParameters as ServerParameters,
)
from mcp.client.session_group import (
    SseServerParameters as SseServerParameters,
)
from mcp.client.session_group import (
    StreamableHttpParameters as StreamableHttpParameters,
)
from pydantic import ValidationError

from ursa.util.http import build_mcp_httpx_async_client
from ursa.util.secrets import SecretTemplate

_PARAMETER_TYPES = {
    "stdio": StdioServerParameters,
    "streamable_http": StreamableHttpParameters,
    "sse": SseServerParameters,
}


def parse_server_parameters(config: Any) -> ServerParameters:
    """Validate file settings with the SDK, retaining supplied SDK objects."""
    if isinstance(config, tuple(_PARAMETER_TYPES.values())):
        return config
    if not isinstance(config, dict):
        raise ValueError(
            "MCP server configuration must be a mapping or parameters"
        )
    payload = dict(config)
    kind = payload.pop("transport", None)
    if isinstance(headers := payload.get("headers"), dict):
        payload["headers"] = {
            name: SecretTemplate.maybe_validate(value)
            for name, value in headers.items()
        }
    if kind is not None:
        cls = _PARAMETER_TYPES.get(str(kind).replace("-", "_"))
        if cls is None:
            raise ValueError(f"Unsupported MCP transport {kind!r}")
        return cls(**payload)
    # Legacy inference prefers stdio, then HTTP, then SSE.
    for cls in _PARAMETER_TYPES.values():
        try:
            return cls(**payload)
        except ValidationError:
            continue
    raise ValueError("Unable to determine transport for MCP server")


validate_server_parameters = parse_server_parameters


def transport(config: ServerParameters) -> str:
    config = parse_server_parameters(config)
    return next(
        kind
        for kind, cls in _PARAMETER_TYPES.items()
        if isinstance(config, cls)
    )


def serialize_server_parameters(
    config: ServerParameters,
    *,
    exclude_defaults: bool = False,
    exclude_none: bool = True,
    exclude_unset: bool = False,
):
    config = parse_server_parameters(config)
    values = config.model_dump(
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
    )
    return {
        "transport": transport(config),
        **{
            key: value.total_seconds()
            if isinstance(value, timedelta)
            else value
            for key, value in values.items()
        },
    }


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


def start_mcp_client(
    server_configs: dict[str, ServerParameters | dict],
) -> UrsaMCPClient:
    client_config = {}
    for server, config in server_configs.items():
        config = parse_server_parameters(config)
        connection = {
            **config.model_dump(),
            "transport": transport(config),
        }
        if headers := connection.get("headers"):
            connection["headers"] = {
                name: _resolve_header(value, server)
                for name, value in headers.items()
            }
        if connection["transport"] != "stdio":
            connection["httpx_client_factory"] = build_mcp_httpx_async_client
        client_config[server] = connection
    return UrsaMCPClient(client_config)


def _resolve_header(value, server_name: str):
    """Resolve a typed secret template in an MCP HTTP header."""
    if isinstance(value, dict):
        value = SecretTemplate.model_validate(value)
    if isinstance(value, SecretTemplate):
        rendered = value.get_secret_value(server_name)
        if rendered is None:
            raise ValueError(
                f"Secret for MCP server '{server_name}' is not set"
            )
        return rendered
    return value


# Backwards-compatible private spelling retained for downstream imports.
_serialize_server_config = serialize_server_parameters
