import pytest
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)

from ursa.util import mcp as mcp_mod
from ursa.util.secrets import SecretTemplate


def test_start_mcp_client_adds_httpx_factory_for_sse(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, connections):
            captured["connections"] = connections

    monkeypatch.setattr(mcp_mod, "MultiServerMCPClient", DummyClient)

    mcp_mod.start_mcp_client({
        "demo": SseServerParameters(url="https://example.com/sse")
    })

    conn = captured["connections"]["demo"]
    assert conn["transport"] == "sse"
    assert conn["httpx_client_factory"] is mcp_mod.build_mcp_httpx_async_client


def test_start_mcp_client_adds_httpx_factory_for_streamable_http(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, connections):
            captured["connections"] = connections

    monkeypatch.setattr(mcp_mod, "MultiServerMCPClient", DummyClient)

    mcp_mod.start_mcp_client({
        "demo": StreamableHttpParameters(url="https://example.com/mcp")
    })

    conn = captured["connections"]["demo"]
    assert conn["transport"] == "streamable_http"
    assert conn["httpx_client_factory"] is mcp_mod.build_mcp_httpx_async_client


def test_mcp_header_resolves_keyring_secret_template(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, connections):
            captured["connections"] = connections

    monkeypatch.setattr(mcp_mod, "MultiServerMCPClient", DummyClient)
    monkeypatch.setattr(
        "keyring.get_password",
        lambda service, username: (
            "token" if (service, username) == ("ursa", "demo") else None
        ),
    )

    mcp_mod.start_mcp_client({
        "demo": StreamableHttpParameters(
            url="https://example.com/mcp",
            headers={
                "Authorization": {
                    "keyring": True,
                    "template": "Bearer %s",
                }
            },
        )
    })

    assert captured["connections"]["demo"]["headers"] == {
        "Authorization": "Bearer token"
    }


def test_mcp_config_loading_types_secret_headers():
    config = mcp_mod.validate_server_parameters({
        "transport": "streamable-http",
        "url": "https://example.com/mcp",
        "headers": {
            "Authorization": {
                "env": "MCP_TOKEN",
                "template": "Bearer %s",
            }
        },
    })

    assert isinstance(config.headers["Authorization"], SecretTemplate)


def test_mcp_header_reports_missing_environment_secret(monkeypatch):
    monkeypatch.delenv("MCP_TOKEN", raising=False)

    with pytest.raises(ValueError, match="MCP server 'demo' is not set"):
        mcp_mod.start_mcp_client({
            "demo": StreamableHttpParameters(
                url="https://example.com/mcp",
                headers={"Authorization": {"env": "MCP_TOKEN"}},
            )
        })


def test_mcp_header_rejects_invalid_secret_mapping():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        mcp_mod.start_mcp_client({
            "demo": StreamableHttpParameters(
                url="https://example.com/mcp",
                headers={"Authorization": {"enb": "MCP_TOKEN"}},
            )
        })
