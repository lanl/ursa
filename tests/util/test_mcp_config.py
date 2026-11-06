"""Configuration compatibility with the installed MCP SDK."""

from datetime import timedelta
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from ursa.cli.config import UrsaConfig
from ursa.util import mcp as sdk
from ursa.util import mcp_config as config
from ursa.util.secrets import SecretTemplate


@pytest.mark.parametrize(
    "cls",
    [
        "StdioServerParameters",
        "SseServerParameters",
        "StreamableHttpParameters",
    ],
)
def test_parameter_fields_and_defaults_match_sdk(cls):
    local_fields = getattr(config, cls).model_fields
    sdk_fields = getattr(sdk, cls).model_fields
    assert local_fields.keys() == sdk_fields.keys()
    for name, sdk_field in sdk_fields.items():
        local_field = local_fields[name]
        assert local_field.annotation == sdk_field.annotation, name
        assert local_field.is_required() == sdk_field.is_required(), name
        if sdk_field.default_factory:
            assert local_field.default_factory() == sdk_field.default_factory()
        else:
            assert local_field.default == sdk_field.default, name


SERVER_CONFIGS = [
    {"command": "mcp-server"},
    {"url": "http://localhost/mcp"},
    {"transport": "stdio", "command": "mcp-server"},
    {
        "transport": "stdio",
        "command": "mcp-server",
        "args": ["--port", "1234"],
        "env": {"TEST": "value"},
        "cwd": Path("workspace"),
        "encoding": "utf-16",
        "encoding_error_handler": "replace",
    },
    {"transport": "sse", "url": "http://localhost/sse"},
    {
        "transport": "sse",
        "url": "http://localhost/sse",
        "timeout": "2.5",
        "sse_read_timeout": 42,
        "headers": {"Authorization": {"env": "MCP_TOKEN"}},
    },
    {"transport": "streamable-http", "url": "http://localhost/mcp"},
    {
        "transport": "streamable_http",
        "url": "http://localhost/mcp",
        "timeout": 2.5,
        "sse_read_timeout": "PT2M",
        "terminate_on_close": False,
        "headers": {
            "Authorization": {"keyring": True, "template": "Bearer %s"}
        },
    },
    {
        "transport": "streamable_http",
        "url": "http://localhost/mcp",
        "timeout": timedelta(seconds=10),
        "headers": None,
    },
]


@pytest.mark.parametrize("values", SERVER_CONFIGS)
def test_parameter_validation_and_serialization_match_sdk(values):
    local = config.parse_server_parameters(values)
    original = sdk.parse_server_parameters(values)
    assert local.model_dump() == original.model_dump()
    assert local.model_fields_set == original.model_fields_set
    assert config.transport(local) == sdk.transport(original)
    for defaults, none, unset in product((True, False), repeat=3):
        flags = dict(
            exclude_defaults=defaults, exclude_none=none, exclude_unset=unset
        )
        assert config.serialize_server_parameters(local, **flags) == (
            sdk.serialize_server_parameters(original, **flags)
        )
    # Python callers can still supply SDK objects to UrsaConfig.
    converted = UrsaConfig(mcp_servers={"tools": original}).mcp_servers["tools"]
    assert type(converted) is type(local)
    assert converted.model_dump() == local.model_dump()
    assert converted.model_fields_set == local.model_fields_set
    for name, value in (getattr(local, "headers", None) or {}).items():
        if isinstance(value, SecretTemplate):
            assert isinstance(converted.headers[name], SecretTemplate)


@pytest.mark.parametrize(
    "values",
    [
        {"transport": "stdio"},
        {"transport": "stdio", "command": 42},
        {"transport": "stdio", "command": "tool", "args": "bad"},
        {
            "transport": "stdio",
            "command": "tool",
            "encoding_error_handler": "bad",
        },
        {"transport": "sse"},
        {"transport": "sse", "url": 42},
        {"transport": "sse", "url": "http://localhost/sse", "timeout": "bad"},
        {"transport": "streamable_http"},
        {
            "transport": "streamable_http",
            "url": "http://localhost/mcp",
            "timeout": "bad",
        },
    ],
)
def test_invalid_parameters_still_fail_before_connecting(values):
    with pytest.raises(ValidationError):
        sdk.parse_server_parameters(values)
    with pytest.raises(ValidationError):
        UrsaConfig(mcp_servers={"invalid": values})


def test_malformed_secret_is_rejected_before_client_construction(monkeypatch):
    def unexpected_client(_connections):
        pytest.fail("Client constructed before checking its credentials")

    monkeypatch.setattr(sdk, "UrsaMCPClient", unexpected_client)
    parsed = UrsaConfig(
        mcp_servers={
            "tools": {
                "transport": "streamable_http",
                "url": "http://localhost/mcp",
                "headers": {"Authorization": {"enb": "TOKEN"}},
            }
        }
    )
    with pytest.raises(ValidationError):
        sdk.start_mcp_client(parsed.mcp_servers)


@pytest.mark.parametrize("values", SERVER_CONFIGS)
def test_client_connections_match_sdk(monkeypatch, values):
    monkeypatch.setenv("MCP_TOKEN", "test-token")
    # Resolve keyring-backed references to a fixed value for this comparison.
    monkeypatch.setattr(
        sdk, "_resolve_header", lambda value, name: "test-token"
    )
    monkeypatch.setattr(sdk, "UrsaMCPClient", lambda connections: connections)
    local = UrsaConfig(mcp_servers={"tools": values}).mcp_servers["tools"]
    original = sdk.parse_server_parameters(values)
    assert sdk.start_mcp_client({"tools": local}) == sdk.start_mcp_client({
        "tools": original
    })
