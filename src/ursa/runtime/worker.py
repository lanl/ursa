"""Thread workers and the authenticated, multiplexed worker wire protocol.

Run with ``python -m ursa.runtime.worker --rendezvous PATH``. MPI and Slurm
launch independent copies of this entry point; no MPI Python binding is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import pickle
import queue
import secrets
import socket
import ssl
import struct
import sys
import threading
from pathlib import Path

from .manager import AgentProcess, _ProcessOutputForwarder

LOGGER = logging.getLogger(__name__)
_OUTPUT = contextvars.ContextVar("ursa_thread_output", default=None)
_EOF = object()
_MAX_PACKET = 128 * 1024 * 1024


async def authenticate(reader, writer, key: bytes, *, server: bool) -> None:
    """Mutually authenticate before accepting any serialized Python objects."""
    async with asyncio.timeout(10):
        nonce = secrets.token_bytes(32)
        if server:
            writer.write(nonce)
            await writer.drain()
            peer = await reader.readexactly(32)
            signature = await reader.readexactly(32)
            expected = hmac.digest(
                key, b"client" + nonce + peer, hashlib.sha256
            )
            if not hmac.compare_digest(signature, expected):
                raise PermissionError("Invalid worker credentials")
            writer.write(
                hmac.digest(key, b"server" + peer + nonce, hashlib.sha256)
            )
            await writer.drain()
        else:
            peer = await reader.readexactly(32)
            writer.write(
                nonce
                + hmac.digest(key, b"client" + peer + nonce, hashlib.sha256)
            )
            await writer.drain()
            signature = await reader.readexactly(32)
            expected = hmac.digest(
                key, b"server" + nonce + peer, hashlib.sha256
            )
            if not hmac.compare_digest(signature, expected):
                raise PermissionError("Invalid host credentials")


async def read_packet(reader):
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if size > _MAX_PACKET:
        raise ValueError("Worker packet exceeds 128 MiB")
    return pickle.loads(await reader.readexactly(size))


async def write_packet(writer, packet) -> None:
    data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > _MAX_PACKET:
        raise ValueError("Worker packet exceeds 128 MiB")
    writer.write(struct.pack("!I", len(data)) + data)
    await writer.drain()


class Mailbox:
    """Blocking connection facade used by the existing agent dispatcher."""

    def __init__(self, send):
        self.send = send
        self.incoming = queue.Queue()
        self.pending = None
        self.closed = False

    def poll(self, timeout=0):
        if self.pending is None:
            try:
                self.pending = self.incoming.get(timeout=timeout)
            except queue.Empty:
                return False
        return True

    def recv(self):
        value = (
            self.pending if self.pending is not None else self.incoming.get()
        )
        self.pending = None
        if value is _EOF:
            self.incoming.put(_EOF)
            raise EOFError
        return value

    def close(self):
        self.closed = True
        self.incoming.put_nowait(_EOF)


class HostMailbox:
    """Host receives must not occupy one default-executor thread per agent."""

    def __init__(self, send):
        self.send = send
        self.incoming = asyncio.Queue()

    def close(self):
        self.incoming.put_nowait(_EOF)

    async def arecv(self):
        value = await self.incoming.get()
        if value is _EOF:
            self.incoming.put_nowait(_EOF)
            raise EOFError
        return value


class _ThreadOutput:
    """Leave host/TUI output alone; capture writes in agent thread contexts."""

    def __init__(self, original, level):
        self.original, self.level = original, level

    def write(self, value):
        send = _OUTPUT.get()
        if send is None:
            return self.original.write(value)
        if value.strip():
            send({"type": "log", "level": self.level, "message": value.strip()})
        return len(value)

    def flush(self):
        if _OUTPUT.get() is None:
            self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class AgentThread:
    """One event loop per agent; cancellation never blocks the control loop."""

    def __init__(self, spec, state, interval, send):
        self.spec, self.state, self.interval = spec, state, interval
        self.send = send
        self.connection = Mailbox(send)
        self.loop = None
        self.task = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"ursa-agent-{spec.agent_id}", daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.connection.close()
        if self.loop is not None and self.task is not None:
            try:
                self.loop.call_soon_threadsafe(self.task.cancel)
            except RuntimeError:
                pass

    def _run(self):
        token = _OUTPUT.set(self.send)
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        runner = AgentProcess(
            self.spec,
            self.state,
            self.connection,
            None,
            self.interval,
        )

        try:
            self.task = loop.create_task(runner.run(isolated=False))
            if self.stopped.is_set():
                self.task.cancel()
            loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass
        finally:
            self.stopped.set()
            self.connection.close()
            self.send({"type": "thread_exited"})
            # A synchronous user tool may not cooperate with cancellation.
            # Do not wait for it on the control path or tear down other agents.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            _OUTPUT.reset(token)


class ThreadWorker:
    """Dispatch host packets to independently cancellable agent loops."""

    def __init__(self, send):
        self.send = send
        self.agents: dict[str, AgentThread] = {}
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.output = _ThreadOutput(self.stdout, logging.INFO)
        self.errors = _ThreadOutput(self.stderr, logging.WARNING)
        sys.stdout, sys.stderr = self.output, self.errors

    def dispatch(self, packet):
        identity = packet["agent"]
        command = packet["command"]
        if command == "start":

            def send(message):
                self.send({"agent": identity, "message": message})

            self.agents[identity] = AgentThread(
                packet["spec"], packet["state"], packet["interval"], send
            )
        elif command == "stop":
            agent = self.agents.pop(identity, None)
            if agent is not None:
                agent.stop()
        else:
            agent = self.agents.get(identity)
            if agent is not None:
                agent.connection.incoming.put(packet["message"])

    def close(self):
        for agent in self.agents.values():
            agent.stop()
        self.agents.clear()
        if sys.stdout is self.output:
            sys.stdout = self.stdout
        if sys.stderr is self.errors:
            sys.stderr = self.stderr


async def serve(rendezvous: Path) -> None:
    settings = json.loads(rendezvous.read_text())
    tls = (
        ssl.create_default_context(cafile=settings["certificate"])
        if settings.get("certificate")
        else None
    )
    if tls is not None:
        tls.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    reader, writer = await asyncio.open_connection(
        settings["host"], settings["port"], ssl=tls
    )
    await authenticate(
        reader, writer, bytes.fromhex(settings["key"]), server=False
    )
    await write_packet(
        writer, {"pid": os.getpid(), "hostname": socket.gethostname()}
    )
    outgoing = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def send(packet):
        if not loop.is_closed():
            loop.call_soon_threadsafe(outgoing.put_nowait, packet)

    worker = ThreadWorker(send)
    output = _ProcessOutputForwarder(outgoing, "worker")
    output.install()

    async def write():
        while True:
            await write_packet(writer, await outgoing.get())

    async def read():
        while True:
            packet = await read_packet(reader)
            if packet["command"] == "close":
                return
            worker.dispatch(packet)

    tasks = [asyncio.create_task(write()), asyncio.create_task(read())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    finally:
        worker.close()
        output.restore()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendezvous", required=True, type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.rendezvous))
    except (EOFError, ConnectionError, asyncio.IncompleteReadError):
        LOGGER.info("Runtime host disconnected")


if __name__ == "__main__":
    main()
