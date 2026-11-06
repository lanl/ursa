import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)

from ursa.util import mcp as mcp_mod
from ursa.util.secrets import SecretReference

DUMMY_SERVER = Path(__file__).parents[1] / "tools" / "dummy_mcp_server.py"


def test_start_mcp_client_adds_httpx_factory_for_sse(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, connections):
            captured["connections"] = connections

    monkeypatch.setattr(mcp_mod, "UrsaMCPClient", DummyClient)

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

    monkeypatch.setattr(mcp_mod, "UrsaMCPClient", DummyClient)

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

    monkeypatch.setattr(mcp_mod, "UrsaMCPClient", DummyClient)
    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(
            get_password=lambda service, username: (
                "token" if (service, username) == ("ursa", "demo") else None
            )
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

    assert isinstance(config.headers["Authorization"], SecretReference)


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


async def test_stdio_server_stderr_is_discarded(capsys):
    client = mcp_mod.start_mcp_client({
        "demo": StdioServerParameters(
            command=sys.executable,
            args=[str(DUMMY_SERVER)],
        )
    })
    tools, sources = await mcp_mod.load_mcp_tools_with_sources(client)

    assert tools
    assert sources["add"] == "demo"
    assert "dummy MCP diagnostic" not in capsys.readouterr().err


def test_cached_mcp_specs_rebuild_tools_without_tool_discovery():
    details = [
        {
            "name": "lookup",
            "description": "Look something up.",
            "schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "mcp_input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "mcp_server": "demo",
            "mcp_metadata": {
                "readOnlyHint": True,
                "_meta": {"display_name": "Remote lookup"},
            },
        },
        # Built-in tools are not MCP specs and must not be cached here.
        {"name": "read_file", "schema": {}, "mcp_server": ""},
    ]
    specs = mcp_mod.mcp_tool_specs_from_details(details)

    client = SimpleNamespace(
        connections={
            "demo": {
                "transport": "streamable_http",
                "url": "https://example.test/mcp",
            }
        }
    )
    tools, sources = mcp_mod.mcp_tools_from_specs(client, specs)

    assert [tool.name for tool in tools] == ["lookup"]
    assert sources == {"lookup": "demo"}
    assert tools[0].args_schema == specs[0]["input_schema"]
    assert tools[0].args_schema["additionalProperties"] is False
    assert tools[0].metadata == details[0]["mcp_metadata"]


def test_cached_mcp_specs_reject_unknown_server():
    client = SimpleNamespace(connections={})
    with pytest.raises(ValueError, match="unknown server"):
        mcp_mod.mcp_tools_from_specs(
            client,
            [
                {
                    "server": "missing",
                    "name": "lookup",
                    "input_schema": {"type": "object"},
                }
            ],
        )


async def test_cached_mcp_tool_invokes_through_a_fresh_session():
    client = mcp_mod.start_mcp_client({
        "demo": StdioServerParameters(
            command=sys.executable,
            args=[str(DUMMY_SERVER)],
        )
    })
    tools, sources = mcp_mod.mcp_tools_from_specs(
        client,
        [
            {
                "server": "demo",
                "name": "add",
                "description": "Add two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            }
        ],
    )

    result = await tools[0].ainvoke({
        "type": "tool_call",
        "name": "add",
        "args": {"a": 2, "b": 3},
        "id": "cached-add",
    })

    assert sources == {"add": "demo"}
    assert isinstance(result, ToolMessage)
    assert result.artifact == {"structured_content": {"result": 5}}


def test_stdio_proxy_can_redirect_stderr_to_file(tmp_path):
    stderr_path = tmp_path / "demo-mcp.log"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ursa.util.mcp_stdio_proxy",
            str(stderr_path),
            sys.executable,
            "-c",
            "import sys; print('diagnostic', file=sys.stderr)",
        ],
        check=True,
    )

    assert stderr_path.read_text() == "diagnostic\n"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows command shims are specific to Windows",
)
@pytest.mark.parametrize("extension", [".cmd", ".bat"])
def test_stdio_proxy_launches_windows_command_shim(tmp_path, extension):
    shim = tmp_path / f"npx{extension}"
    shim.write_text("@echo off\r\necho shim diagnostic 1>&2\r\nexit /b 0\r\n")
    stderr_path = tmp_path / "shim.log"
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ursa.util.mcp_stdio_proxy",
            str(stderr_path),
            "npx",
        ],
        check=True,
        env=env,
    )

    assert stderr_path.read_text().strip() == "shim diagnostic"


async def test_upstream_adapter_puts_structured_content_in_tool_artifact():
    client = mcp_mod.start_mcp_client({
        "demo": StdioServerParameters(
            command=sys.executable,
            args=[str(DUMMY_SERVER)],
        )
    })
    tools = await client.get_tools(server_name="demo")
    add = next(tool for tool in tools if tool.name == "add")
    result = await add.ainvoke({
        "type": "tool_call",
        "name": "add",
        "args": {"a": 2, "b": 3},
        "id": "add-call",
    })

    assert isinstance(result, ToolMessage)
    assert result.artifact == {"structured_content": {"result": 5}}
