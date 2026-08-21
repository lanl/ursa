from datetime import timedelta
from typing import Annotated

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)
from pydantic import BaseModel, BeforeValidator, ValidationError

from ursa.util.http import build_mcp_httpx_async_client
from ursa.util.secrets import SecretTemplate


def validate_server_parameters(config: dict):
    if not isinstance(config, dict):
        return config
    transport_hint = config.get("transport")
    payload = {k: v for k, v in config.items() if k != "transport"}
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
    server_configs: dict[str, ServerParameters | dict],
) -> MultiServerMCPClient:
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
    return MultiServerMCPClient(client_config)


def _resolve_header(value, server_name: str):
    """Resolve a typed secret template in an MCP HTTP header."""
    if isinstance(value, dict):
        rendered = SecretTemplate.model_validate(value).get_secret_value(
            server_name
        )
        if rendered is None:
            raise ValueError(
                f"Secret for MCP server '{server_name}' is not set"
            )
        return rendered
    return value


def _serialize_server_config(
    config: ServerParameters,
    *,
    exclude_defaults: bool = True,
    exclude_none: bool = True,
):
    """Internal: serialize MCP ServerParameters in a yaml/json compatible way"""
    config = {
        "transport": transport(config),
        **config.model_dump(
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        ),
    }
    for k, v in config.items():
        if isinstance(v, timedelta):
            config[k] = v.total_seconds()
    return config
