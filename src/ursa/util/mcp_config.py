"""MCP configuration validated without importing the SDK or connecting."""

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from ursa.util.secrets import SecretTemplate


class StdioServerParameters(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"


class SseServerParameters(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0


class StreamableHttpParameters(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: timedelta = timedelta(seconds=30)
    sse_read_timeout: timedelta = timedelta(minutes=5)
    terminate_on_close: bool = True


_PARAMETER_TYPES = {
    "stdio": StdioServerParameters,
    "streamable_http": StreamableHttpParameters,
    "sse": SseServerParameters,
}


def parse_server_parameters(config: Any):
    if isinstance(config, tuple(_PARAMETER_TYPES.values())):
        return config
    if isinstance(config, BaseModel):
        # Python callers may supply SDK objects; file-based configuration
        # never needs this import. Keep effective values and secret objects.
        from mcp.client import session_group as sdk

        for kind, cls in _PARAMETER_TYPES.items():
            if isinstance(config, getattr(sdk, cls.__name__)):
                parameters = parse_server_parameters({
                    **dict(config),
                    "transport": kind,
                })
                parameters.__pydantic_fields_set__ = (
                    config.model_fields_set
                    & type(parameters).model_fields.keys()
                )
                return parameters
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
ServerParameters = Annotated[
    StdioServerParameters | SseServerParameters | StreamableHttpParameters,
    BeforeValidator(parse_server_parameters),
]


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
