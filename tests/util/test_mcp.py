import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters
from mcp.client.session_group import (
    SseServerParameters,
    StreamableHttpParameters,
)

from ursa.util import mcp as mcp_mod

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
