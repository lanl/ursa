import asyncio
import os
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import timedelta
from typing import Annotated, Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)
from pydantic import BaseModel, BeforeValidator, TypeAdapter, ValidationError

from ursa.util.http import build_mcp_httpx_async_client
from ursa.util.secrets import SecretTemplate


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


def validate_server_parameters(config: dict):
    if not isinstance(config, dict):
        return config
    transport_hint = config.get("transport")
    payload = {k: v for k, v in config.items() if k != "transport"}
    if isinstance(headers := payload.get("headers"), dict):
        payload["headers"] = {
            name: SecretTemplate.maybe_validate(value)
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
        raise ValueError(
            f"Unable to determine transport for MCP server '{config}'. "
            "Provide 'transport' with one of: stdio, sse, streamable_http."
        )
    raise ValueError(
        f"Unsupported MCP transport '{transport_hint}' for server '{config}'."
    )


ServerParameters = Annotated[
    StdioServerParameters | SseServerParameters | StreamableHttpParameters,
    BeforeValidator(validate_server_parameters),
]
_SERVER_PARAMETERS_ADAPTER = TypeAdapter(ServerParameters)


def parse_server_parameters(config: Any) -> ServerParameters:
    """Parse one MCP server with the canonical runtime configuration type."""
    return _SERVER_PARAMETERS_ADAPTER.validate_python(deepcopy(config))


def transport(sp: ServerParameters) -> str:
    if isinstance(sp, StdioServerParameters):
        return "stdio"
    elif isinstance(sp, StreamableHttpParameters):
        return "streamable_http"
    elif isinstance(sp, SseServerParameters):
        return "sse"
    raise RuntimeError("Transport for {sp} is unknown")


def start_mcp_client(
    server_configs: dict[str, ServerParameters | dict],
) -> UrsaMCPClient:
    client_config = {}
    for server, config in server_configs.items():
        if not isinstance(config, BaseModel):
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
        if isinstance(config, (SseServerParameters, StreamableHttpParameters)):
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


def serialize_server_parameters(
    config: ServerParameters,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
    exclude_unset: bool = False,
):
    """Serialize MCP server parameters for a YAML/JSON config boundary."""
    serialized = {
        "transport": transport(config),
        **config.model_dump(
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
        ),
    }
    for k, v in serialized.items():
        if isinstance(v, timedelta):
            serialized[k] = v.total_seconds()
    return serialized


# Backwards-compatible private spelling retained for downstream imports.
_serialize_server_config = serialize_server_parameters
