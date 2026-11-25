"""Transport and policy tests for Harbor's Unix-socket network proxy."""

import asyncio
import ipaddress
import os
import socket
import struct
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import httpx
import pytest
from aiohttp import web

pytest.importorskip("harbor")

from ursa.integrations.harbor import network_proxy
from ursa.integrations.harbor.network_proxy import (
    HostProxy,
    SocksProxy,
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
async def proxy(tmp_path, *, allowed_hosts=("127.0.0.1",), upstream_proxy=None):
    directory = tmp_path / "proxy"
    directory.mkdir(mode=0o700)
    path = directory / "proxy.sock"
    async with (
        host_server(path, allowed_hosts, upstream_proxy=upstream_proxy),
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


async def test_forwarding_does_not_add_cookies_or_content_negotiation(tmp_path):
    requests = []

    async def endpoint(request):
        requests.append(dict(request.headers))
        return web.Response(
            text="ok", headers={"Set-Cookie": "session=first; Path=/"}
        )

    async with upstream(endpoint) as origin, proxy(tmp_path) as url:
        for extra in (
            "",
            "",
            "Cookie: session=explicit\r\nAccept-Encoding: identity\r\n",
        ):
            reader, writer = await asyncio.open_connection(*destination(url))
            writer.write(
                f"GET {origin}/ HTTP/1.1\r\nHost: localhost\r\n{extra}\r\n".encode()
            )
            await writer.drain()
            assert (
                b"200"
                in (await asyncio.wait_for(reader.read(), 2)).split(b"\r\n", 1)[
                    0
                ]
            )
            writer.close()
            await writer.wait_closed()
    for request in requests[:2]:
        assert "Cookie" not in request
        assert "Accept-Encoding" not in request
        assert "User-Agent" not in request
    assert requests[2]["Cookie"] == "session=explicit"
    assert requests[2]["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("stop_bridge", [False, True])
async def test_shutdown_aborts_output_when_client_stops_reading(
    tmp_path, stop_bridge
):
    streaming = asyncio.Event()

    async def large_response(request):
        response = web.StreamResponse()
        await response.prepare(request)
        try:
            for i in range(2000):
                await response.write(b"x" * 65536)
                if i == 20:
                    streaming.set()
        except ConnectionError:
            pass
        return response

    async with (
        upstream(large_response) as origin,
        AsyncExitStack() as host,
        AsyncExitStack() as bridge,
    ):
        path = tmp_path / "host.sock"
        await host.enter_async_context(host_server(path, ["127.0.0.1"]))
        if stop_bridge:
            listener = await bridge.enter_async_context(bridge_server(path))
            reader, writer = await asyncio.open_connection(
                *listener.sockets[0].getsockname(), limit=1
            )
        else:
            reader, writer = await asyncio.open_unix_connection(path, limit=1)
        try:
            writer.write(
                f"GET {origin}/ HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
            )
            await writer.drain()
            await asyncio.wait_for(streaming.wait(), 3)
            # Fill the peer's receive buffer, as well as StreamReader's buffer.
            await asyncio.sleep(0.1)
            await asyncio.wait_for(
                (bridge if stop_bridge else host).aclose(), 3
            )
        finally:
            writer.close()
            await writer.wait_closed()


async def test_dedicated_model_forward_preserves_stream_bytes(tmp_path):
    async def echo(reader, writer):
        try:
            writer.write((await reader.read()).upper())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    peer = await asyncio.start_server(echo, "127.0.0.1", 0)
    target = f"127.0.0.1:{peer.sockets[0].getsockname()[1]}"
    path = tmp_path / "host.sock"
    async with (
        peer,
        host_server(path, ["127.0.0.1"]),
        bridge_server(path, target=target) as bridge,
    ):
        reader, writer = await asyncio.open_connection(
            *bridge.sockets[0].getsockname()
        )
        writer.write(b"tls-or-http")
        await writer.drain()
        writer.write_eof()
        assert await asyncio.wait_for(reader.read(), 2) == b"TLS-OR-HTTP"
        writer.close()
        await writer.wait_closed()


@asynccontextmanager
async def socks_server(
    *, auth=None, reject=False, fragment=False, stall=None, disconnected=None
):
    """SOCKS fixture resolves test-only names which the client cannot resolve."""
    calls, tasks = [], set()

    async def stall_until_disconnect(reader, stage):
        if stall != stage:
            return False
        assert await reader.read() == b""
        if disconnected is not None:
            disconnected.set()
        return True

    async def reply(writer, data):
        if fragment:
            for byte in data:
                writer.write(bytes([byte]))
                await writer.drain()
                await asyncio.sleep(0.001)
        else:
            writer.write(data)
            await writer.drain()

    async def copy(reader, writer):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        calls.append(None)
        peer_writer = None
        transfers = []
        try:
            version, count = await reader.readexactly(2)
            assert version == 5
            methods = await reader.readexactly(count)
            if await stall_until_disconnect(reader, "greeting"):
                return
            method = 2 if auth else 0
            assert method in methods
            await reply(writer, bytes([5, method]))
            if auth:
                version, length = await reader.readexactly(2)
                assert version == 1
                username = await reader.readexactly(length)
                password = await reader.readexactly(
                    (await reader.readexactly(1))[0]
                )
                valid = (username.decode(), password.decode()) == auth
                if await stall_until_disconnect(reader, "authentication"):
                    return
                await reply(writer, bytes([1, 0 if valid else 1]))
                if not valid:
                    return
            version, command, reserved, kind = await reader.readexactly(4)
            assert (version, command, reserved) == (5, 1, 0)
            if kind == 3:
                host = (
                    await reader.readexactly((await reader.readexactly(1))[0])
                ).decode()
            else:
                assert kind in (1, 4)
                host = str(
                    ipaddress.ip_address(
                        await reader.readexactly(4 if kind == 1 else 16)
                    )
                )
            port = int.from_bytes(await reader.readexactly(2), "big")
            calls.append((host, port))
            if await stall_until_disconnect(reader, "connect"):
                return
            if reject:
                await reply(writer, b"\x05\x05\x00\x01" + bytes(6))
                return
            peer_reader, peer_writer = await asyncio.open_connection(
                "127.0.0.1", port
            )
            await reply(writer, b"\x05\x00\x00\x01" + bytes(6))
            transfers = [
                asyncio.create_task(copy(reader, peer_writer)),
                asyncio.create_task(copy(peer_reader, writer)),
            ]
            await asyncio.gather(*transfers)
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            for transfer in transfers:
                transfer.cancel()
            await asyncio.gather(*transfers, return_exceptions=True)
            if peer_writer is not None:
                peer_writer.close()
            writer.close()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    credentials = (
        ""
        if auth is None
        else f"{quote(auth[0], safe='')}:{quote(auth[1], safe='')}@"
    )
    url = (
        f"socks5h://{credentials}127.0.0.1:{server.sockets[0].getsockname()[1]}"
    )
    try:
        yield url, calls
    finally:
        server.close()
        for task in tuple(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


@pytest.mark.parametrize("auth", [None, ("user@lab", "secret:password")])
@pytest.mark.parametrize("scheme", ["socks5", "socks5h"])
@pytest.mark.parametrize("allowed_hosts", [["only-at-proxy.invalid"], None])
@pytest.mark.parametrize("fragment", [False, True])
async def test_http_and_connect_use_upstream_socks_with_remote_dns(
    tmp_path, auth, scheme, allowed_hosts, fragment
):
    async def allowed(request):
        assert request.headers.get("Proxy-Authorization") is None
        return web.Response(text="via socks")

    async with (
        upstream(allowed) as origin,
        socks_server(auth=auth, fragment=fragment) as (socks, calls),
    ):
        socks = socks.replace("socks5h:", scheme + ":")
        port = destination(origin)[1]
        target = f"http://only-at-proxy.invalid:{port}/"
        async with proxy(
            tmp_path,
            allowed_hosts=allowed_hosts,
            upstream_proxy=socks,
        ) as url:
            async with httpx.AsyncClient(proxy=url, trust_env=False) as client:
                response = await client.get(target)
                assert (
                    response.status_code == 200 and response.text == "via socks"
                )
            reader, writer = await asyncio.open_connection(*destination(url))
            writer.write(
                f"CONNECT only-at-proxy.invalid:{port} HTTP/1.1\r\nHost: only-at-proxy.invalid\r\n\r\n".encode()
            )
            await writer.drain()
            assert b"200" in await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), 2
            )
            writer.write(
                b"GET / HTTP/1.1\r\nHost: only-at-proxy.invalid\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            assert b"via socks" in await asyncio.wait_for(reader.read(), 2)
            writer.close()
            await writer.wait_closed()
        assert [call for call in calls if call] == [
            ("only-at-proxy.invalid", port)
        ] * 2


@pytest.mark.parametrize("method", ["GET", "CONNECT"])
@pytest.mark.parametrize("failure", ["policy", "auth", "refused"])
async def test_socks_failures_never_fall_back_to_direct_access(
    tmp_path, method, failure
):
    contacted = []

    async def forbidden(request):
        contacted.append(request.path)
        return web.Response(text="direct access must not happen")

    async with (
        upstream(forbidden) as origin,
        socks_server(
            auth=("user", "password"), reject=failure == "refused"
        ) as (socks, calls),
    ):
        if failure == "auth":
            socks = socks.replace("password", "wrong-password")
        allowed = [] if failure == "policy" else ["127.0.0.1"]
        async with proxy(
            tmp_path, allowed_hosts=allowed, upstream_proxy=socks
        ) as url:
            reader, writer = await asyncio.open_connection(*destination(url))
            target = (
                origin if method == "GET" else origin.removeprefix("http://")
            )
            writer.write(
                f"{method} {target} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 3)
            assert (
                b"403" if failure == "policy" else b"502"
            ) in response.split(b"\r\n", 1)[0]
            assert b"password" not in response and b"user" not in response
            writer.close()
            await writer.wait_closed()
        assert contacted == []
        if failure == "policy":
            assert calls == []


@pytest.mark.parametrize("host", ["only-at-proxy.invalid", "127.0.0.1", "::1"])
async def test_socks_connect_handles_fragmented_replies_and_half_close(host):
    async def echo(reader, writer):
        writer.write((await reader.read()).upper())
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with (
        server,
        socks_server(auth=("user", "password"), fragment=True) as (url, calls),
    ):
        reader, writer = await asyncio.wait_for(
            SocksProxy(url).connect(host, port), 2
        )
        writer.write(b"hello")
        await writer.drain()
        writer.write_eof()
        assert await asyncio.wait_for(reader.read(), 2) == b"HELLO"
        writer.close()
        await writer.wait_closed()
        assert calls[-1] == (host, port)


@pytest.mark.parametrize(
    "url",
    [
        "http://user:private@localhost:1080",
        "socks4://localhost:1080",
        "socks5h://localhost",
        "socks5h://user:private@localhost:99999",
        "socks5h://localhost:1080/path",
        "socks5h://user@localhost:1080",
        "socks5h://us%C3%A9r:private@localhost:1080",
        "socks5h://user:privat%C3%A9@localhost:1080",
        "socks5h://usér:private@localhost:1080",
        "socks5h://user:privaté@localhost:1080",
    ],
)
def test_invalid_socks_configuration_does_not_disclose_credentials(url):
    with pytest.raises(ValueError) as error:
        SocksProxy(url)
    assert "private" not in str(error.value)
    assert url not in str(error.value)
    assert type(error.value) is ValueError


@pytest.mark.parametrize("method", ["GET", "CONNECT"])
@pytest.mark.parametrize("stage", ["greeting", "authentication", "connect"])
async def test_socks_handshake_deadline_returns_502_and_closes_connection(
    tmp_path, monkeypatch, method, stage
):
    monkeypatch.setattr(network_proxy, "CONNECT_TIMEOUT", 0.15)
    disconnected = asyncio.Event()
    async with (
        socks_server(
            auth=("user", "password"), stall=stage, disconnected=disconnected
        ) as (socks, calls),
        proxy(tmp_path, allowed_hosts=None, upstream_proxy=socks) as url,
    ):
        reader, writer = await asyncio.open_connection(*destination(url))
        target = (
            "http://only-at-proxy.invalid/"
            if method == "GET"
            else "only-at-proxy.invalid:443"
        )
        writer.write(
            f"{method} {target} HTTP/1.1\r\nHost: only-at-proxy.invalid\r\n\r\n".encode()
        )
        await writer.drain()
        try:
            response = await asyncio.wait_for(reader.read(), 2)
            assert response.startswith(b"HTTP/1.1 502 ")
            assert b"password" not in response
            assert calls, "request never reached the SOCKS server"
            await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.parametrize("method", ["GET", "CONNECT"])
async def test_socks_handshake_deadline_does_not_limit_response_stream(
    tmp_path, monkeypatch, method
):
    monkeypatch.setattr(network_proxy, "CONNECT_TIMEOUT", 0.15)

    async def streaming(request):
        # Both the initial response and the body may outlive negotiation.
        await asyncio.sleep(0.25)
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"first")
        await asyncio.sleep(0.25)
        await response.write(b"last")
        return response

    async with (
        upstream(streaming) as origin,
        socks_server(auth=("user", "password"), fragment=True) as (
            socks,
            calls,
        ),
        proxy(tmp_path, allowed_hosts=None, upstream_proxy=socks) as url,
    ):
        reader, writer = await asyncio.open_connection(*destination(url))
        host, port = destination(origin)
        try:
            if method == "CONNECT":
                writer.write(
                    f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()
                )
                await writer.drain()
                assert b"200" in await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), 2
                )
            target = origin if method == "GET" else "/"
            writer.write(
                f"GET {target} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 3)
            assert response.startswith(b"HTTP/1.1 200 ")
            assert b"first" in response and b"last" in response
            assert calls[-1] == (host, port)
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.parametrize("method", ["GET", "CONNECT"])
async def test_socks_reset_during_negotiation_returns_502(
    tmp_path, method, caplog
):
    reset = asyncio.Event()

    async def reset_connection(reader, writer):
        await reader.readexactly(2)
        writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        writer.transport.abort()
        reset.set()

    server = await asyncio.start_server(reset_connection, "127.0.0.1", 0)
    socks = f"socks5h://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    async with (
        server,
        proxy(tmp_path, allowed_hosts=None, upstream_proxy=socks) as url,
    ):
        reader, writer = await asyncio.open_connection(*destination(url))
        target = (
            "http://only-at-proxy.invalid/"
            if method == "GET"
            else "only-at-proxy.invalid:443"
        )
        writer.write(
            f"{method} {target} HTTP/1.1\r\nHost: only-at-proxy.invalid\r\n\r\n".encode()
        )
        await writer.drain()
        try:
            response = await asyncio.wait_for(reader.read(), 2)
            assert reset.is_set()
            assert response.startswith(b"HTTP/1.1 502 ")
            assert "Unhandled exception" not in caplog.text
        finally:
            writer.close()
            await writer.wait_closed()
