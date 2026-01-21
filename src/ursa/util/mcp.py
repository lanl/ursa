from datetime import timedelta
from typing import Annotated

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)
from pydantic import BaseModel, BeforeValidator, ValidationError


def validate_server_parameters(config: dict):
    if not isinstance(config, dict):
        return config
    transport_hint = config.get("transport")
    payload = {k: v for k, v in config.items() if k != "transport"}
    if transport_hint == "stdio":
        return StdioServerParameters(**payload)
    elif transport_hint == "sse":
        return SseServerParameters(**payload)
    elif transport_hint == "streamable_http":
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
        client_config[server] = {
            **config.model_dump(),
            "transport": transport(config),
        }
    return MultiServerMCPClient(client_config)


def _serialize_server_config(config: ServerParameters):
    """Internal: serialize MCP ServerParameters in a yaml/json compatible way"""
    config = {"transport": transport(config), **config.model_dump()}
    for k, v in config.items():
        if isinstance(v, timedelta):
            config[k] = v.total_seconds()
    return config
