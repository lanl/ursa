"""Transport and policy tests for Harbor's Unix-socket network proxy."""

import asyncio
import os
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest
from aiohttp import web

pytest.importorskip("harbor")

from ursa.integrations.harbor.network_proxy import (
    HostProxy,
    authority,
    bridge_server,
    destination,
    host_server,
)

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX") or os.name == "nt",
    reason="Proxy requires Unix sockets and permissions",
)


@pytest.fixture
def tmp_path():
    # macOS's default pytest paths can exceed AF_UNIX's path length limit.
    with TemporaryDirectory(prefix="ursa-proxy-", dir="/tmp") as directory:
        yield Path(directory)


@asynccontextmanager
async def upstream(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def proxy(tmp_path, *, allowed_hosts=("127.0.0.1",)):
    directory = tmp_path / "proxy"
    directory.mkdir(mode=0o700)
    path = directory / "proxy.sock"
    async with (
        host_server(path, allowed_hosts),
        bridge_server(path, 0) as server,
    ):
        assert path.stat().st_mode & 0o777 == 0o600
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    assert not path.exists()


@pytest.mark.parametrize(
    "rule",
    [
        "example.com",
        "*.example.com:443",
        "http://example.com:80",
        "user@example.com:443",
        "example.com:443/path",
        "example.com:0",
        "example.com:65536",
        ".:443",
    ],
)
def test_invalid_allowlist_rule(rule):
    with pytest.raises(ValueError):
        authority(rule)


def test_exact_normalized_destinations():
    assert authority("EXAMPLE.com.:443") == ("example.com", 443)
    assert authority("[::1]:8080") == ("::1", 8080)
    assert destination("https://example.com/v1") == ("example.com", 443)
    with pytest.raises(ValueError):
        destination("http://example.com:0/")


async def test_requests_forward_streamed_body_and_errors(tmp_path):
    async def echo(request):
        assert request.headers.get("X-Remove") is None
        assert request.headers["Authorization"] == "Bearer local-test"
        return web.Response(
            status=429,
            body=await request.read(),
            headers={"Retry-After": "2", "X-Upstream-Host": request.host},
        )

    async def chunks():
        yield b'{"messages":['
        yield b'{"role":"user","content":"hello"}]}'

    async with upstream(echo) as origin, proxy(tmp_path) as url:
        async with httpx.AsyncClient(proxy=url, trust_env=False) as client:
            response = await client.post(
                origin + "/v1/chat/completions",
                content=chunks(),
                headers={
                    "Host": "not-allowed.example",
                    "Connection": "X-Remove",
                    "X-Remove": "ignored",
                    "Authorization": "Bearer local-test",
                },
            )
        assert response.status_code == 429
        assert response.headers["retry-after"] == "2"
        assert response.headers["x-upstream-host"] == origin.removeprefix(
            "http://"
        )
        assert response.json()["messages"][0]["content"] == "hello"


async def test_sse_is_delivered_before_upstream_finishes(tmp_path):
    release = asyncio.Event()

    async def stream(request):
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(b'data: {"delta":"hello"}\n\n')
        await release.wait()
        await response.write(b"data: [DONE]\n\n")
        return response

    async with (
        upstream(stream) as origin,
        proxy(tmp_path) as url,
    ):
        try:
            async with httpx.AsyncClient(proxy=url, trust_env=False) as client:
                async with client.stream(
                    "GET", origin + "/v1/chat/completions"
                ) as response:
                    lines = response.aiter_lines()
                    assert (
                        await asyncio.wait_for(anext(lines), 2)
                        == 'data: {"delta":"hello"}'
                    )
                    release.set()
                    assert "data: [DONE]" in [line async for line in lines]
        finally:
            release.set()


async def test_forward_proxy_allowlist_and_redirects(tmp_path):
    contacted = []

    async def allowed(request):
        contacted.append(request.path)
        return web.Response(
            status=302, headers={"Location": "http://blocked.invalid/"}
        )

    async with (
        upstream(allowed) as origin,
        proxy(tmp_path) as url,
    ):
        async with httpx.AsyncClient(proxy=url, trust_env=False) as client:
            response = await client.get(origin + "/redirect")
            assert response.status_code == 302
            response = await client.get(
                origin + "/redirect", follow_redirects=True
            )
            assert response.status_code == 403
            assert (
                await client.head("http://blocked.invalid/")
            ).status_code == 403
            assert (await client.get("http://localhost:1/")).status_code == 403
        assert contacted == ["/redirect", "/redirect"]


async def test_connect_tunnel_denies_other_hosts_and_handles_half_close(
    tmp_path,
):
    async def echo(reader, writer):
        writer.write((await reader.read()).upper())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server, proxy(tmp_path) as url:
        for host, status in [("localhost", b"403"), ("127.0.0.1", b"200")]:
            reader, writer = await asyncio.open_connection(*destination(url))
            writer.write(
                f"CONNECT {host}:{port} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n\r\n".encode()
            )
            await writer.drain()
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
            assert status in headers.split(b"\r\n")[0]
            if status == b"200":
                writer.write(b"hello")
                await writer.drain()
                writer.write_eof()
                assert await asyncio.wait_for(reader.read(), 2) == b"HELLO"
            writer.close()
            await writer.wait_closed()


async def test_host_refuses_open_directory_and_existing_socket(tmp_path):
    directory = tmp_path / "shared"
    directory.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="0700"):
        async with host_server(directory / "proxy.sock", ["example.com"]):
            pytest.fail("Should not expose proxy through an open directory")
    directory.chmod(0o700)
    path = directory / "proxy.sock"
    async with host_server(path, ["example.com"]):
        with pytest.raises(FileExistsError):
            async with host_server(path, ["example.com"]):
                pytest.fail("Should not replace a live socket")


@pytest.mark.parametrize(
    "host, allowed",
    [
        ("example.com", True),
        ("sub.example.com", False),
        ("one.allowed.com", True),
        ("nested.one.allowed.com", True),
        ("allowed.com", False),
        ("notallowed.com", False),
        ("one.allowed.com.bad", False),
        ("127.0.0.1", True),
        ("::1", True),
        ("127.0.0.2", False),
    ],
)
def test_harbor_host_rules_allow_all_ports(host, allowed):
    policy = HostProxy(
        None, ["example.com", "*.allowed.com", "127.0.0.1", "::1"]
    )
    for port in (1, 80, 443, 65535):
        assert policy.permits((host, port)) is allowed


async def test_empty_allowlist_denies_every_destination(tmp_path):
    async with proxy(tmp_path, allowed_hosts=[]) as url:
        async with httpx.AsyncClient(proxy=url, trust_env=False) as client:
            assert (await client.get("http://localhost/")).status_code == 403


@pytest.mark.parametrize("stop_bridge", [False, True])
async def test_stopping_proxy_revokes_existing_tunnel(tmp_path, stop_bridge):
    connected = asyncio.Event()
    disconnected = asyncio.Event()

    async def peer(reader, writer):
        connected.set()
        try:
            assert await reader.read() == b""
        finally:
            writer.close()
            await writer.wait_closed()
            disconnected.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server, AsyncExitStack() as host, AsyncExitStack() as bridge:
        path = tmp_path / "host.sock"
        await host.enter_async_context(host_server(path, ["127.0.0.1"]))
        listener = await bridge.enter_async_context(bridge_server(path))
        reader, writer = await asyncio.open_connection(
            *listener.sockets[0].getsockname()
        )
        writer.write(
            f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
        )
        await writer.drain()
        assert b"200" in await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), 2
        )
        await asyncio.wait_for(connected.wait(), 2)
        await asyncio.wait_for((bridge if stop_bridge else host).aclose(), 2)
        assert await asyncio.wait_for(reader.read(), 2) == b""
        await asyncio.wait_for(disconnected.wait(), 2)
        writer.close()
        await writer.wait_closed()
