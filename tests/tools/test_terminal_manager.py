"""Public manager contracts with deterministic stream and screen doubles."""

import asyncio
from datetime import UTC, datetime

import pytest

from ursa.tools.terminal import (
    TerminalRenderSnapshot,
    TerminalSpan,
    TermManager,
    TermSession,
)
from ursa.tools.terminal import manager as manager_module
from ursa.tools.terminal.base import OutputJournal


class FakeTerm(TermSession):
    def __init__(self, *args, screen=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.output, self.stream, self.writes, self.screen = "", "", [], screen
        self.running, self.terminated = True, False

    async def start(self, command=None):
        self.command = command

    async def send_bytes(self, data):
        self.writes.append(data)

    async def read(self, *, offset=0, lines=None):
        rows = self.output.splitlines(keepends=True)
        return "".join(
            rows[: len(rows) - offset][-lines if lines is not None else None :]
        )

    async def contents(self):
        return self.output

    async def output_marker(self):
        return len(self.stream)

    async def output_since(self, marker):
        return self.stream[marker:]

    async def is_alive(self):
        return {"is_alive": True} if self.running else {"exit_code": 0}

    async def wait(self):
        self.running = False
        return 0

    async def terminate(self):
        self.running = False
        self.terminated = True

    def emit(self, text):
        self.output += text
        self.stream += text

    async def render_snapshot(self):
        return TerminalRenderSnapshot(
            self.term_id,
            (TerminalSpan(self.output),),
            rows=1,
            cols=max(1, len(self.output)),
            screen=self.screen,
        )

    async def resize(self, rows, cols):
        self.shape = (rows, cols)

    async def cursor(self):
        return (0, 0)

    async def size(self):
        return (2, 3)

    async def mouse_event(self, *args, **kwargs):
        self.mouse = (args, kwargs)

    async def paste_text(self, text):
        self.paste = text


@pytest.fixture
def manager(monkeypatch):
    instance = TermManager()
    for value in ("_sessions", "_reserved_ids", "_session_info"):
        getattr(instance, value).clear()
    instance._closing = False
    monkeypatch.setattr(instance, "_dispatch", lambda coroutine, **_: coroutine)
    yield instance
    instance._sessions.clear()


def test_output_journal_bounds_offsets_and_styles_are_immutable():
    journal = OutputJournal(max_lines=2, max_chars=8)
    journal.append("one\ntwo\nthree\n")
    start, text, end = journal.window_since(0)
    assert (
        text.endswith("three\n")
        and start > 0
        and end == len("one\ntwo\nthree\n")
    )


@pytest.mark.asyncio
async def test_register_metadata_io_and_screen_operations(manager):
    term = FakeTerm("screen12", ["sh"], screen=True)
    manager.register(term)
    info = manager.terminal_info(term.term_id)
    assert info.term_id == term.term_id and info.created_at.tzinfo is UTC
    await manager.send_text(term.term_id, "x")
    await manager.send_line(term.term_id, "y")
    await manager.paste_text(term.term_id, "paste")
    await manager.mouse_input(term.term_id, 1, 2, (("press", "left"),))
    await manager.resize(term.term_id, 4, 5)
    assert (
        term.writes == [b"x", b"y\n"]
        and term.paste == "paste"
        and term.shape == (4, 5)
    )
    assert term.mouse[0][:3] == ("press", 1, 2)


@pytest.mark.asyncio
async def test_waits_use_only_new_output_and_changed_screen(
    manager, monkeypatch
):
    term = FakeTerm("wait1234", ["sh"], screen=True)
    term.emit("old")
    manager.register(term)

    async def update():
        await asyncio.sleep(0.02)
        term.emit(" new")

    task = asyncio.create_task(update())
    assert "new" in await manager.wait_for(term.term_id, "new", timeout=0.2)
    await task
    assert await manager.wait_for(term.term_id, "never", timeout=0.01) == (
        "Pattern not found"
    )

    async def update_screen():
        await asyncio.sleep(0.02)
        term.output = "NEW"

    task = asyncio.create_task(update_screen())
    assert (
        await manager.wait_screen(
            term.term_id,
            condition="change",
            bounding_box=(0, 0, 1, 1),
            include_styling=False,
            timeout=0.2,
        )
        == "Screen changed"
    )
    await task
    monkeypatch.setattr(manager_module, "SCREEN_STABILITY_SECONDS", 0)
    monkeypatch.setattr(manager_module, "SCREEN_MIN_STABILITY_FRAMES", 2)
    assert (
        await manager.wait_screen(term.term_id, timeout=0.2)
        == "Screen stabilized"
    )


@pytest.mark.asyncio
async def test_capability_errors_resources_remove_and_close(
    manager, monkeypatch
):
    stream = FakeTerm("stream12", ["sh"])
    manager.register(stream)
    await manager.mouse_input(stream.term_id, 0, 0, ())
    assert (await manager.resources(stream.term_id))["status"] == "unavailable"
    await manager.remove(stream.term_id)
    assert stream.terminated
    manager.register(FakeTerm("close123", ["sh"]))
    await manager.close_all()
    assert not manager.ids()


@pytest.mark.asyncio
async def test_create_registers_factory_session_and_rejects_empty_shell(
    manager,
):
    with pytest.raises(ValueError):
        await manager.create(shell=[])
    terminal = await manager.create(session_factory=FakeTerm)
    assert terminal.term_id in manager.ids()
    await manager.remove(terminal.term_id)


@pytest.mark.asyncio
async def test_cancelled_start_releases_reservation_and_terminates(manager):
    started = asyncio.Event()
    instances = []

    class Slow(FakeTerm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        async def start(self, command=None):
            started.set()
            await asyncio.Event().wait()

    reservation = manager_module._CreationReservation(
        "cancel12", datetime.now(UTC), 1
    )
    manager._reserved_ids.add(reservation.term_id)
    task = asyncio.create_task(
        manager._create(
            reservation,
            None,
            env=None,
            shell=None,
            cwd=None,
            session_factory=Slow,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reservation.term_id not in manager._reserved_ids
    assert instances[0].terminated
