"""Validate MCP configuration without importing clients or server runtimes.

URSA owns these configuration models so reading a config does not load the MCP
SDK. Their fields and defaults are checked against the installed SDK in tests.
The client adapter consumes their serialized values when connecting to a server.
"""

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from ursa.util.secrets import SecretTemplate


class StdioServerParameters(BaseModel):
    """Configuration for launching a local MCP process."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"


class SseServerParameters(BaseModel):
    """Configuration for an MCP SSE endpoint; timeouts are in seconds."""

    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0


class StreamableHttpParameters(BaseModel):
    """Configuration for an MCP streamable HTTP endpoint."""

    url: str
    headers: dict[str, Any] | None = None
    timeout: timedelta = timedelta(seconds=30)
    sse_read_timeout: timedelta = timedelta(minutes=5)
    terminate_on_close: bool = True


_PARAMETER_TYPES = (
    StdioServerParameters,
    StreamableHttpParameters,
    SseServerParameters,
)


def validate_server_parameters(config: Any):
    if isinstance(config, _PARAMETER_TYPES):
        return config
    if isinstance(config, BaseModel):
        # Python callers may already have SDK parameter instances. Only this
        # compatibility path needs the SDK; config files never take it.
        from mcp import StdioServerParameters as SDKStdio
        from mcp.client.session_group import (
            SseServerParameters as SDKSse,
        )
        from mcp.client.session_group import (
            StreamableHttpParameters as SDKHttp,
        )

        for cls, kind in (
            (SDKStdio, "stdio"),
            (SDKSse, "sse"),
            (SDKHttp, "streamable_http"),
        ):
            if isinstance(config, cls):
                # Preserve which fields were explicit, including secret objects.
                config = {
                    "transport": kind,
                    **{
                        name: getattr(config, name)
                        for name in config.model_fields_set
                    },
                }
                break
    if not isinstance(config, dict):
        raise ValueError(
            "MCP server configuration must be a mapping or parameters"
        )
    kind = config.get("transport")
    payload = {
        key: value for key, value in config.items() if key != "transport"
    }
    if isinstance(headers := payload.get("headers"), dict):
        payload["headers"] = {
            name: SecretTemplate.maybe_validate(value)
            for name, value in headers.items()
        }
    cls = (
        {
            "stdio": StdioServerParameters,
            "sse": SseServerParameters,
            "streamable_http": StreamableHttpParameters,
            "streamable-http": StreamableHttpParameters,
        }.get(kind)
        if isinstance(kind, str)
        else None
    )
    if cls is not None:
        return cls(**payload)
    if kind is None:
        # Preserve legacy inference: command first, then HTTP, then SSE.
        for cls in _PARAMETER_TYPES:
            try:
                return cls(**payload)
            except ValidationError:
                continue
        raise ValueError(
            "Unable to determine transport for MCP server. "
            "Provide 'transport' with one of: stdio, sse, streamable_http."
        )
    raise ValueError(f"Unsupported MCP transport {kind!r}")


ServerParameters = Annotated[
    StdioServerParameters | SseServerParameters | StreamableHttpParameters,
    BeforeValidator(validate_server_parameters),
]


def parse_server_parameters(config: Any) -> ServerParameters:
    return validate_server_parameters(config)


def transport(config: ServerParameters) -> str:
    config = parse_server_parameters(config)
    if isinstance(config, StdioServerParameters):
        return "stdio"
    if isinstance(config, StreamableHttpParameters):
        return "streamable_http"
    return "sse"


def serialize_server_parameters(
    config: ServerParameters,
    *,
    exclude_defaults: bool = True,
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
