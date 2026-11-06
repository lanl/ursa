import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)
from pydantic import Field

from ursa.cli.config import UrsaConfig, config_file_value
from ursa.util import mcp as mcp_mod
from ursa.util.secrets import SecretTemplate

DUMMY_SERVER = Path(__file__).parents[1] / "tools" / "dummy_mcp_server.py"


class DefaultStdioParameters(StdioServerParameters):
    command: str = "tool"
    args: list[str] = Field(default_factory=lambda: ["--test"])


class DefaultSseParameters(SseServerParameters):
    url: str = "http://localhost/sse"
    timeout: float = 60


class DefaultHttpParameters(StreamableHttpParameters):
    url: str = "http://localhost/mcp"
    timeout: timedelta = timedelta(seconds=60)
    terminate_on_close: bool = False


@pytest.mark.parametrize(
    "cls", [DefaultStdioParameters, DefaultSseParameters, DefaultHttpParameters]
)
def test_sdk_defaults_survive_config_round_trip(cls, monkeypatch):
    original = cls()
    config = UrsaConfig(mcp_servers={"tools": original})
    assert config.mcp_servers["tools"] is original
    assert mcp_mod.parse_server_parameters(original) is original
    assert original.model_fields_set == set()

    saved = config_file_value(config)["mcp_servers"]["tools"]
    restored = mcp_mod.parse_server_parameters(saved)
    assert restored.model_dump() == original.model_dump()
    monkeypatch.setattr(
        mcp_mod, "UrsaMCPClient", lambda connections: connections
    )
    assert mcp_mod.start_mcp_client({"tools": saved}) == (
        mcp_mod.start_mcp_client(config.mcp_servers)
    )


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
def test_invalid_parameters_fail_before_connecting(server, monkeypatch):
    def unexpected_client(connections):
        pytest.fail("Invalid parameters reached the MCP client")

    monkeypatch.setattr(mcp_mod, "UrsaMCPClient", unexpected_client)
    with pytest.raises(ValueError):
        mcp_mod.start_mcp_client({"invalid": server})


@pytest.mark.parametrize(
    "parameters,transport",
    [
        (SseServerParameters, "sse"),
        (StreamableHttpParameters, "streamable_http"),
    ],
)
def test_start_mcp_client_adds_httpx_factory(
    monkeypatch, parameters, transport
):
    monkeypatch.setattr(
        mcp_mod, "UrsaMCPClient", lambda connections: connections
    )
    connections = mcp_mod.start_mcp_client({
        "demo": parameters(url="https://example.com/mcp")
    })

    conn = connections["demo"]
    assert conn["transport"] == transport
    assert conn["httpx_client_factory"] is mcp_mod.build_mcp_httpx_async_client


def test_mcp_header_resolves_keyring_secret_template(monkeypatch):
    monkeypatch.setattr(
        mcp_mod, "UrsaMCPClient", lambda connections: connections
    )
    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(
            get_password=lambda service, username: (
                "token" if (service, username) == ("ursa", "demo") else None
            )
        ),
    )

    connections = mcp_mod.start_mcp_client({
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

    assert connections["demo"]["headers"] == {"Authorization": "Bearer token"}


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


@pytest.mark.parametrize("from_config", [False, True])
async def test_upstream_adapter_puts_structured_content_in_tool_artifact(
    from_config,
):
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(DUMMY_SERVER)],
    )
    if from_config:
        server = UrsaConfig.model_validate({
            "mcp_servers": {
                "demo": {"transport": "stdio", **server.model_dump()}
            }
        }).mcp_servers["demo"]
    client = mcp_mod.start_mcp_client({"demo": server})
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
