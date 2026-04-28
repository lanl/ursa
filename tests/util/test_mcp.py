from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)

from ursa.util import mcp as mcp_mod


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
