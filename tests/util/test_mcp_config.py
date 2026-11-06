"""The lightweight configuration must retain the SDK's schema and behavior."""

from datetime import timedelta
from itertools import product

import pytest
from pydantic import Field, ValidationError

from ursa.cli.config import UrsaConfig
from ursa.util import mcp as sdk
from ursa.util import mcp_config as config
from ursa.util.secrets import SecretTemplate


class DefaultStdioParameters(sdk.StdioServerParameters):
    command: str = "tool"
    args: list[str] = Field(default_factory=lambda: ["--test"])


class DefaultSseParameters(sdk.SseServerParameters):
    url: str = "http://localhost/sse"
    timeout: float = 60


class DefaultHttpParameters(sdk.StreamableHttpParameters):
    url: str = "http://localhost/mcp"
    timeout: timedelta = timedelta(seconds=60)
    terminate_on_close: bool = False


@pytest.mark.parametrize(
    "cls", [DefaultStdioParameters, DefaultSseParameters, DefaultHttpParameters]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_sdk_subclass_defaults_survive_conversion(cls, explicit, monkeypatch):
    values = {}
    if explicit:
        values = (
            {"args": ["--override"]}
            if cls is DefaultStdioParameters
            else {"headers": {"Authorization": SecretTemplate(env="TOKEN")}}
        )
    original = cls(**values)
    local = UrsaConfig(mcp_servers={"tools": original}).mcp_servers["tools"]

    assert local.model_dump() == original.model_dump()
    assert local.model_fields_set == original.model_fields_set == set(values)
    assert local.model_dump(exclude_unset=True) == original.model_dump(
        exclude_unset=True
    )
    # Normal serialization must retain defaults that differ from the local
    # schema; exclude_unset still follows the caller's explicit fields.
    serialized = config.serialize_server_parameters(local)
    assert config.parse_server_parameters(serialized).model_dump() == (
        original.model_dump()
    )
    if "headers" in values:
        assert (
            local.headers["Authorization"] is values["headers"]["Authorization"]
        )
    monkeypatch.setenv("TOKEN", "test-token")
    monkeypatch.setattr(sdk, "UrsaMCPClient", lambda connections: connections)
    assert sdk.start_mcp_client({"tools": local}) == sdk.start_mcp_client({
        "tools": original
    })


@pytest.mark.parametrize(
    "cls, values",
    [
        (sdk.StdioServerParameters, {"command": "tool", "args": ["--test"]}),
        (
            sdk.SseServerParameters,
            {"url": "http://localhost/sse", "timeout": 2.5},
        ),
        (
            sdk.StreamableHttpParameters,
            {"url": "http://localhost/mcp", "timeout": 2.5},
        ),
    ],
)
def test_sdk_parity(cls, values, monkeypatch):
    if cls is not sdk.StdioServerParameters:
        values = dict(
            values, headers={"Authorization": SecretTemplate(env="TOKEN")}
        )
    original = cls(**values)
    local = UrsaConfig(mcp_servers={"tools": original}).mcp_servers["tools"]
    assert local.model_dump() == original.model_dump()
    assert local.model_fields_set == original.model_fields_set
    schema, sdk_schema = local.model_json_schema(), original.model_json_schema()
    assert schema["properties"] == sdk_schema["properties"]
    assert schema["required"] == sdk_schema["required"]
    for defaults, none, unset in product((True, False), repeat=3):
        flags = dict(
            exclude_defaults=defaults, exclude_none=none, exclude_unset=unset
        )
        expected = original.model_dump(**flags)
        expected = {
            k: v.total_seconds() if isinstance(v, timedelta) else v
            for k, v in expected.items()
        }
        serialized = config.serialize_server_parameters(local, **flags)
        assert serialized == {"transport": config.transport(local), **expected}
        assert (
            config.parse_server_parameters(serialized).model_dump()
            == local.model_dump()
        )
    monkeypatch.setenv("TOKEN", "test-token")
    monkeypatch.setattr(sdk, "UrsaMCPClient", lambda connections: connections)
    assert sdk.start_mcp_client({"tools": local}) == sdk.start_mcp_client({
        "tools": original
    })


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "unknown", "url": "http://localhost/mcp"},
        {"transport": "stdio"},
        {"transport": "stdio", "command": "tool", "args": "bad"},
        {"transport": "sse", "url": 123},
        {
            "transport": "streamable-http",
            "url": "http://localhost/mcp",
            "timeout": "bad",
        },
    ],
)
def test_invalid_config_fails_before_connecting(server):
    with pytest.raises(ValidationError):
        UrsaConfig(mcp_servers={"invalid": server})
