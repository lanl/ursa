"""Regression tests for nested worker output routing."""

import asyncio
import json
import sys
from io import StringIO

import pytest

from ursa.runtime.worker import (
    _OUTPUT,
    ThreadWorker,
    authenticate,
    read_packet,
    serve,
    write_packet,
)


@pytest.mark.parametrize("close_order", [(0, 1), (1, 0)])
def test_nested_thread_workers_restore_host_streams(monkeypatch, close_order):
    stdout, stderr = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    messages = []
    workers = [ThreadWorker(messages.append), ThreadWorker(messages.append)]

    token = _OUTPUT.set(messages.append)
    try:
        sys.stdout.write("agent output\n")
    finally:
        _OUTPUT.reset(token)
    sys.stdout.write("host output")
    assert messages == [{"type": "log", "level": 20, "message": "agent output"}]
    assert stdout.getvalue() == "host output"

    first, second = close_order
    workers[first].close()
    assert sys.stdout is workers[second].output
    assert sys.stderr is workers[second].errors
    workers[second].close()
    assert sys.stdout is stdout
    assert sys.stderr is stderr


@pytest.mark.asyncio
async def test_serve_restores_thread_worker_streams(monkeypatch, tmp_path):
    stdout, stderr = StringIO(), StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    key = bytes(range(32))
    hello = asyncio.get_running_loop().create_future()

    async def host(reader, writer):
        try:
            await authenticate(reader, writer, key, server=True)
            hello.set_result(await read_packet(reader))
            await write_packet(writer, {"command": "close"})
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(host, "127.0.0.1", 0)
    rendezvous = tmp_path / "connection.json"
    rendezvous.write_text(
        json.dumps({
            "host": "127.0.0.1",
            "port": server.sockets[0].getsockname()[1],
            "key": key.hex(),
        })
    )
    try:
        await asyncio.wait_for(serve(rendezvous), 2)
        assert (await asyncio.wait_for(hello, 1))["hostname"]
    finally:
        server.close()
        await server.wait_closed()
    assert sys.stdout is stdout
    assert sys.stderr is stderr
