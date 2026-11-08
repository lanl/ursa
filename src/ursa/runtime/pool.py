"""Bounded worker placement with elastic local processes and fixed launch jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
import secrets
import socket
import ssl
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

from ursa.util.async_utils import cancel_tasks

from .manager import AgentProcessExitedError, AgentStatus
from .worker import (
    HostMailbox,
    ThreadWorker,
    authenticate,
    read_packet,
    write_packet,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSnapshot:
    worker_id: str
    hostname: str
    pid: int
    agents: tuple[str, ...]
    pending: int


class _AgentHandle:
    """Logical process facade: the manager retains its existing lifecycle API."""

    def __init__(self, worker, spec, state, interval):
        self.worker, self.spec = worker, spec
        self.identity = uuid4().hex
        self.pid = worker.pid
        self.heartbeat = SimpleNamespace(value=monotonic())
        self.done = threading.Event()
        self.connection = HostMailbox(
            lambda message: self._send("message", message=message)
        )
        worker.agents[self.identity] = self
        self._send("start", spec=spec, state=state, interval=interval)

    def _send(self, command, **payload):
        if self.done.is_set() or self.worker.closed:
            raise BrokenPipeError("Worker is no longer available")
        packet = {"agent": self.identity, "command": command, **payload}
        self.worker.send(pickle.loads(pickle.dumps(packet)))

    def receive(self, message):
        if self.done.is_set():
            return
        if message["type"] == "heartbeat":
            self.heartbeat.value = monotonic()
        elif message["type"] == "thread_exited":
            self.finish()
        else:
            self.connection.incoming.put_nowait(message)

    def finish(self):
        if not self.done.is_set() and not self.worker.closed:
            try:
                self._send("stop")
            except BrokenPipeError:
                pass
        self.done.set()
        self.connection.close()
        self.worker.agents.pop(self.identity, None)
        self.worker.last_used = monotonic()

    def is_alive(self):
        return not self.done.is_set()

    def join(self, timeout=None):
        self.done.wait(timeout)

    terminate = finish
    kill = terminate
    close = terminate


class _Worker:
    def __init__(self, pool, pid, hostname, writer=None):
        self.pool, self.pid, self.hostname = pool, pid, hostname
        self.identity = uuid4().hex
        self.writer = writer
        self.outgoing = asyncio.Queue()
        self.agents = {}
        self.closed = False
        self.last_used = monotonic()
        self.local = (
            ThreadWorker(self.receive_threadsafe) if writer is None else None
        )

    def send(self, packet):
        if self.closed:
            raise BrokenPipeError("Worker disconnected")
        self.pool.loop.call_soon_threadsafe(self.outgoing.put_nowait, packet)

    def receive_threadsafe(self, packet):
        if not self.pool.loop.is_closed():
            self.pool.loop.call_soon_threadsafe(self.receive, packet)

    def receive(self, packet):
        if packet.get("type") == "log":
            LOGGER.log(
                packet.get("level", logging.INFO),
                "[%s] %s",
                self.hostname,
                packet["message"],
            )
        elif handle := self.agents.get(packet.get("agent")):
            handle.receive(packet["message"])

    async def write(self):
        while True:
            packet = await self.outgoing.get()
            if self.local is not None:
                self.local.dispatch(packet)
            else:
                await write_packet(self.writer, packet)

    async def read(self, reader):
        while True:
            self.receive(await read_packet(reader))

    def close(self):
        self.closed = True
        for handle in tuple(self.agents.values()):
            handle.finish()
        if self.local is not None:
            self.local.close()
        if self.writer is not None:
            self.writer.close()


class WorkerPool:
    def __init__(self, config, manager):
        self.config, self.manager = config, manager
        self.workers = {}
        self.processes = []
        self.tasks = set()
        self.lock = asyncio.Lock()
        self.preferred = {}
        self.retiring = set()
        self.loop = None
        self.server = None
        self.directory = None
        self.maintenance = None
        self.closed = False

    def _task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            LOGGER.error("Worker service failed", exc_info=error)

    async def _serve(self, worker, reader=None):
        tasks = [asyncio.create_task(worker.write())]
        if reader is not None:
            tasks.append(asyncio.create_task(worker.read(reader)))
        try:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                await task
        except (ConnectionError, asyncio.IncompleteReadError, EOFError):
            LOGGER.info("Worker %s disconnected", worker.identity)
        except Exception:
            LOGGER.exception("Worker transport failed")
        finally:
            worker.close()
            self.workers.pop(worker.identity, None)
            await cancel_tasks(*tasks)

    async def _accept(self, reader, writer):
        try:
            await authenticate(reader, writer, self.key, server=True)
            async with asyncio.timeout(10):
                hello = await read_packet(reader)
            if self.closed or len(self.workers) >= self.config.max_workers:
                raise ValueError("Worker pool is full or closed")
            worker = _Worker(
                self, int(hello["pid"]), str(hello["hostname"]), writer
            )
            self.workers[worker.identity] = worker
            await self._serve(worker, reader)
        except Exception:
            LOGGER.exception("Worker connection rejected")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _initialize(self):
        self.loop = asyncio.get_running_loop()
        if self.config.backend == "thread":
            worker = _Worker(self, os.getpid(), socket.gethostname())
            self.workers[worker.identity] = worker
            self._task(self._serve(worker))
        else:
            await self._start_process_workers()
        self.maintenance = asyncio.create_task(self._maintain())

    async def _start_process_workers(self):
        self.key = secrets.token_bytes(32)
        tls = None
        if self.config.tls_certificate:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(
                self.config.tls_certificate, self.config.tls_key
            )
        self.server = await asyncio.start_server(
            lambda reader, writer: self._task(self._accept(reader, writer)),
            self.config.bind_host,
            0,
            ssl=tls,
        )
        root = self.config.rendezvous_dir or (
            Path.cwd() if self.config.launch_command else None
        )
        self.directory = tempfile.TemporaryDirectory(
            prefix="ursa-workers-", dir=root
        )
        self.rendezvous = Path(self.directory.name) / "connection.json"
        address = self.config.advertise_host or self.config.bind_host
        if address in {"0.0.0.0", "::"}:
            address = socket.getfqdn()
        descriptor = os.open(
            self.rendezvous, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "host": address,
                    "port": self.server.sockets[0].getsockname()[1],
                    "key": self.key.hex(),
                    "certificate": (
                        str(self.config.tls_certificate.resolve())
                        if self.config.tls_certificate
                        else None
                    ),
                },
                stream,
            )
        await self._launch(
            self.config.max_workers
            if self.config.launch_command
            else self.config.min_workers
        )

    async def _launch(self, count):
        if not self.config.launch_command:
            connected = {worker.pid for worker in self.workers.values()}
            await asyncio.gather(
                *(
                    self._stop_launcher(process)
                    for process in self.processes
                    if process.returncode is None
                    and process.pid not in connected
                )
            )
        self.processes = [p for p in self.processes if p.returncode is None]
        if not self.config.launch_command and (
            len(self.processes) + count > self.config.max_workers
        ):
            raise AgentProcessExitedError(
                "Retiring workers have not released the process limit"
            )
        target = len(self.workers) + count
        command = self.config.launch_command or [
            sys.executable,
            "-m",
            "ursa.runtime.worker",
            "--rendezvous",
            "{rendezvous}",
        ]
        launch_count = 1 if self.config.launch_command else count
        launched = [
            await self._launch_command(command, count)
            for _ in range(launch_count)
        ]
        async with asyncio.timeout(self.manager.startup_timeout):
            while len(self.workers) < target:
                if any(process.returncode is not None for process in launched):
                    raise AgentProcessExitedError(
                        "Worker launch command exited before workers connected"
                    )
                await asyncio.sleep(0.02)

    async def _launch_command(self, command, count):
        values = {
            "workers": str(count),
            "python": sys.executable,
            "rendezvous": str(self.rendezvous),
        }
        process = await asyncio.create_subprocess_exec(
            *(part.format_map(values) for part in command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.processes.append(process)
        self._task(self._logs(process))
        return process

    async def _logs(self, process):
        while line := await process.stdout.read(4096):
            LOGGER.info(
                "Worker launcher: %s", line.decode(errors="replace").rstrip()
            )

    async def start(self, spec, state, interval):
        if spec.environment is not None or spec.working_directory is not None:
            raise ValueError(
                "Per-agent environment/cwd requires the process backend; threaded agents share process globals"
            )
        # Match process serialization/isolation at the admission boundary.
        spec, state = pickle.loads(pickle.dumps((spec, state)))
        async with self.lock:
            if self.closed:
                raise AgentProcessExitedError("Worker pool is closed")
            if self.loop is None:
                try:
                    await self._initialize()
                except BaseException:
                    await self.aclose()
                    raise
            if not self.workers:
                if self.config.launch_command:
                    raise AgentProcessExitedError(
                        "All externally launched workers exited"
                    )
                await self._launch(self.config.min_workers)
            preferred = self.preferred.pop(spec.agent_id, None)
            worker = self.workers.get(preferred) or min(
                self._available_workers(), key=self._load
            )
            return _AgentHandle(worker, spec, state, interval)

    def _available_workers(self):
        return [
            worker
            for worker in self.workers.values()
            if worker.identity not in self.retiring
        ] or list(self.workers.values())

    def _load(self, worker):
        return (
            sum(
                len(self.manager._entry(h.spec.agent_id).pending)
                for h in worker.agents.values()
            ),
            len(worker.agents),
        )

    def snapshots(self):
        return [
            WorkerSnapshot(
                w.identity,
                w.hostname,
                w.pid,
                tuple(h.spec.agent_id for h in w.agents.values()),
                self._load(w)[0],
            )
            for w in self.workers.values()
        ]

    async def _maintain(self):
        while True:
            await asyncio.sleep(self.config.rebalance_interval)
            try:
                await self.rebalance()
            except Exception:
                LOGGER.exception("Worker rebalancing failed")

    async def rebalance(self):
        """Resize local workers and move only idle agents at committed boundaries."""
        if self.closed or self.config.backend == "thread":
            return
        await self._resize()
        await self._move_idle_agents()
        if not self.config.launch_command:
            await self._retire_empty_workers()

    async def _resize(self):
        async with self.lock:
            count = sum(len(worker.agents) for worker in self.workers.values())
            wanted = min(
                self.config.max_workers,
                max(
                    self.config.min_workers,
                    math.ceil(count / self.config.agents_per_worker),
                ),
            )
            if not self.config.launch_command and wanted > len(self.workers):
                await self._launch(wanted - len(self.workers))
            excess = (
                max(0, len(self.workers) - wanted)
                if not self.config.launch_command
                else 0
            )
            self.retiring = {
                worker.identity
                for worker in sorted(self.workers.values(), key=self._load)[
                    :excess
                ]
            }

    async def _move_idle_agents(self):
        # Lifecycle locks exclude new invocations while an idle agent migrates.
        # Never replay running requests or cancel work to improve placement.
        for entry in tuple(self.manager._agents.values()):
            if self.closed or len(self.workers) < 2:
                break
            async with entry.lifecycle_lock:
                process = entry.process
                if (
                    not isinstance(process, _AgentHandle)
                    or entry.status is not AgentStatus.READY
                    or entry.pending
                    or entry.background_tasks
                    or not entry.has_state
                ):
                    continue
                source = process.worker
                target = min(self._available_workers(), key=self._load)
                if source is target or (
                    source.identity not in self.retiring
                    and len(source.agents) <= len(target.agents) + 1
                ):
                    continue
                self.preferred[entry.spec.agent_id] = target.identity
                try:
                    await self.manager._kill_locked(entry)
                    await self.manager._start_locked(entry)
                finally:
                    self.preferred.pop(entry.spec.agent_id, None)

    async def _retire_empty_workers(self):
        async with self.lock:
            for worker in tuple(self.workers.values()):
                if len(self.workers) <= self.config.min_workers:
                    return
                if worker.agents or (
                    monotonic() - worker.last_used < self.config.idle_timeout
                ):
                    continue
                await write_packet(worker.writer, {"command": "close"})
                worker.close()
                self.workers.pop(worker.identity, None)

    async def stop_scheduling(self):
        self.closed = True
        if self.maintenance is not None:
            await cancel_tasks(self.maintenance)

    async def aclose(self):
        await self.stop_scheduling()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for worker in tuple(self.workers.values()):
            worker.close()
        await asyncio.gather(
            *(self._stop_launcher(process) for process in self.processes)
        )
        await cancel_tasks(*self.tasks)
        if self.directory is not None:
            self.directory.cleanup()

    @staticmethod
    async def _stop_launcher(process):
        # Let disconnected workers exit normally before escalating the launcher.
        for signal in (None, process.terminate, process.kill):
            if signal is not None and process.returncode is None:
                try:
                    signal()
                except ProcessLookupError:
                    pass
            try:
                async with asyncio.timeout(2):
                    await process.wait()
                return
            except TimeoutError:
                continue
        LOGGER.error(
            "Worker launcher %s did not exit after being killed", process.pid
        )
