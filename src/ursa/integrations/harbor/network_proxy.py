"""Allowlisted HTTP(S) over a Unix socket for isolated Harbor environments.

The environment owns the host server and starts this file's stdlib-only bridge
inside its network namespace. Keep third-party imports out of the bridge path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import os
import signal
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit


def authority(value: str) -> tuple[str, int]:
    """Parse a destination authority, including bracketed IPv6 addresses."""
    parsed = urlsplit(f"//{value}")
    if (
        not parsed.hostname
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Expected an exact HOST:PORT (no URL, path, or wildcard)"
        )
    host = parsed.hostname.rstrip(".").lower()
    if not host:
        raise ValueError("Invalid hostname")
    try:
        host = str(ipaddress.ip_address(host))
    except ValueError:
        host = host.encode("idna").decode("ascii")
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in host):
            raise ValueError("Invalid hostname") from None
    return host, parsed.port


def destination(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "Expected an HTTP(S) URL without credentials or fragment"
        )
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return authority(f"{host}:{port}")


def end_to_end_headers(headers):
    # Strip both standard hop-by-hop headers and Connection-nominated fields.
    excluded = {
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"proxy-connection",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
        b"host",
    }
    for name, value in headers:
        if name.lower() == b"connection":
            excluded.update(part.strip().lower() for part in value.split(b","))
    return [(k, v) for k, v in headers if k.lower() not in excluded]


async def close(writer):
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


async def relay(reader, writer, peer_reader, peer_writer):
    """Bounded-buffer, bidirectional forwarding, including TCP half-closes."""

    async def copy(source, target):
        while data := await source.read(65536):
            target.write(data)
            await target.drain()
        if target.can_write_eof():
            target.write_eof()
            await target.drain()

    tasks = [
        asyncio.create_task(copy(reader, peer_writer)),
        asyncio.create_task(copy(peer_reader, writer)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close(peer_writer)


class HostProxy:
    def __init__(self, client, allowed_hosts):
        self.client = client
        self.allowed_hosts = allowed_hosts
        self.tasks = set()
        self.closing = False

    def permits(self, endpoint):
        host, _port = endpoint
        return any(
            host == entry
            or (entry.startswith("*.") and host.endswith(entry[1:]))
            for entry in self.allowed_hosts
        )

    async def stop(self):
        self.closing = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle(self, reader, writer):
        import h11
        import httpx

        if self.closing:
            await close(writer)
            return
        task = asyncio.current_task()
        self.tasks.add(task)
        protocol = h11.Connection(h11.SERVER, max_incomplete_event_size=65536)
        response_started = False

        async def event():
            while True:
                result = protocol.next_event()
                if result is not h11.NEED_DATA:
                    return result
                data = await asyncio.wait_for(reader.read(65536), timeout=300)
                protocol.receive_data(data)

        async def send(value):
            writer.write(protocol.send(value))
            await writer.drain()

        try:
            # Read exactly the headers: CONNECT payload must stay in reader.
            headers = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=15
            )
            protocol.receive_data(headers)
            request = await event()
            if not isinstance(request, h11.Request):
                raise ValueError("Expected an HTTP request")
            method = request.method.decode("ascii")
            target = request.target.decode("ascii")
            if method == "CONNECT":
                endpoint = authority(target)
            else:
                endpoint = destination(target)
            if not self.permits(endpoint):
                message = b"Destination is not allowlisted\n"
                await send(
                    h11.Response(
                        status_code=403,
                        headers=[
                            (b"connection", b"close"),
                            (b"content-length", str(len(message)).encode()),
                        ],
                    )
                )
                response_started = True
                if method != "HEAD":
                    await send(h11.Data(data=message))
                await send(h11.EndOfMessage())
                return
            if method == "CONNECT":
                peer_reader, peer_writer = await asyncio.wait_for(
                    asyncio.open_connection(*endpoint), timeout=10
                )
                try:
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                    response_started = True
                    await relay(reader, writer, peer_reader, peer_writer)
                finally:
                    await close(peer_writer)
                return

            async def body():
                while True:
                    item = await event()
                    if isinstance(item, h11.EndOfMessage):
                        return
                    if not isinstance(item, h11.Data):
                        raise ValueError("Incomplete HTTP request body")
                    yield item.data

            if any(
                k.lower() == b"expect" and v.lower() == b"100-continue"
                for k, v in request.headers
            ):
                await send(
                    h11.InformationalResponse(status_code=100, headers=[])
                )
            async with self.client.stream(
                method,
                target,
                headers=end_to_end_headers(request.headers),
                content=body(),
                follow_redirects=False,
            ) as response:
                response_headers = end_to_end_headers(response.headers.raw)
                response_headers.append((b"connection", b"close"))
                await send(
                    h11.Response(
                        status_code=response.status_code,
                        headers=response_headers,
                    )
                )
                response_started = True
                async for chunk in response.aiter_raw():
                    await send(h11.Data(data=chunk))
                await send(h11.EndOfMessage())
        except (
            ValueError,
            OSError,
            TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            h11.ProtocolError,
            httpx.HTTPError,
            httpx.InvalidURL,
        ) as exc:
            if not response_started:
                # Do not expose upstream URLs, credentials, or response bodies.
                status = (
                    400
                    if isinstance(
                        exc, (ValueError, h11.ProtocolError, httpx.InvalidURL)
                    )
                    else 502
                )
                writer.write(
                    f"HTTP/1.1 {status} Proxy Error\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
                )
                with contextlib.suppress(OSError):
                    await writer.drain()
        finally:
            try:
                await close(writer)
            finally:
                self.tasks.discard(task)


@contextlib.asynccontextmanager
async def host_server(socket_path: Path, allowed_hosts):
    import httpx

    socket_path = socket_path.absolute()
    socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = socket_path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("Socket directory must be owned by you and mode 0700")
    if socket_path.exists() or socket_path.is_symlink():
        raise FileExistsError(socket_path)
    async with httpx.AsyncClient(
        verify=ssl.create_default_context(),
        trust_env=False,
        timeout=httpx.Timeout(300, connect=10),
    ) as client:
        proxy = HostProxy(client, allowed_hosts)
        server = await asyncio.start_unix_server(
            proxy.handle, path=socket_path, limit=65536
        )
        os.chmod(socket_path, 0o600)
        inode = socket_path.stat().st_ino
        try:
            yield server
        finally:
            # Python 3.13 waits for accepted connections in wait_closed();
            # revoke them before waiting for the listening server to finish.
            server.close()
            await proxy.stop()
            await server.wait_closed()
            if socket_path.exists() and socket_path.stat().st_ino == inode:
                socket_path.unlink()


@contextlib.asynccontextmanager
async def bridge_server(socket_path: Path, port=0):
    tasks = set()
    closing = False

    async def handle(reader, writer):
        if closing:
            await close(writer)
            return
        task = asyncio.current_task()
        tasks.add(task)
        try:
            peer_reader, peer_writer = await asyncio.open_unix_connection(
                socket_path
            )
            await relay(reader, writer, peer_reader, peer_writer)
        except OSError:
            pass
        finally:
            try:
                await close(writer)
            finally:
                tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    try:
        yield server
    finally:
        closing = True
        server.close()
        for task in tuple(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


async def run(args):
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    async with bridge_server(args.socket, args.port) as server:
        port = server.sockets[0].getsockname()[1]
        sys.stdout.write(json.dumps({"port": port}) + "\n")
        sys.stdout.flush()
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
