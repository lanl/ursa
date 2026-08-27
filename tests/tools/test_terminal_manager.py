import asyncio
import re
import threading
from dataclasses import FrozenInstanceError
from datetime import UTC
from pathlib import Path

import pytest

from ursa.tools.terminal import TERM_ID_LENGTH, TermManager, TermSession


class FakeTerm(TermSession):
    def __init__(self, *args, output: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self.started_with = None
        self.writes: list[bytes] = []
        self.terminated = False
        self.running = True

    async def start(self, command=None):
        self.started_with = command

    async def send_bytes(self, data: bytes):
        self.writes.append(data)

    async def read(self, *, offset=0, lines=None):
        split = self.output.splitlines(keepends=True)
        end = len(split) - offset if offset else len(split)
        start = 0 if lines is None else max(0, end - lines)
        return "".join(split[start:end])

    async def contents(self):
        return self.output

    async def is_alive(self):
        return {"is_alive": True} if self.running else {"exit_code": 0}

    async def wait(self):
        self.running = False
        return 0

    async def terminate(self):
        self.terminated = True
        self.running = False


@pytest.fixture
def manager(monkeypatch):
    instance = TermManager()
    instance._sessions.clear()
    instance._reserved_ids.clear()
    instance._creation_reservations.clear()
    instance._session_info.clear()
    instance._closing = False

    async def inline(coroutine):
        return await coroutine

    monkeypatch.setattr(instance, "_dispatch", inline)
    yield instance
    instance._sessions.clear()
    instance._reserved_ids.clear()
    instance._creation_reservations.clear()
    instance._session_info.clear()
    instance._closing = False


@pytest.fixture
def real_manager():
    instance = TermManager()
    instance._sessions.clear()
    instance._reserved_ids.clear()
    instance._creation_reservations.clear()
    instance._session_info.clear()
    instance._closing = False
    yield instance
    instance._sessions.clear()
    instance._reserved_ids.clear()
    instance._creation_reservations.clear()
    instance._session_info.clear()
    instance._closing = False


async def test_session_convenience_sends_and_validates_keycodes():
    terminal = FakeTerm("abcdefgh", ["bash"])
    await terminal.send_text("é")
    await terminal.send_line("hello")
    await terminal.send_keycode(3)
    assert terminal.writes == ["é".encode(), b"hello\n", b"\x03"]
    with pytest.raises(ValueError, match="between 0 and 255"):
        await terminal.send_keycode(256)


def test_session_rejects_empty_shell():
    with pytest.raises(ValueError, match="shell"):
        FakeTerm("abcdefgh", [])


def test_manager_is_singleton_and_generates_alphanumeric_ids(manager):
    assert manager is TermManager()
    ids = {manager.new_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(value) == TERM_ID_LENGTH for value in ids)
    assert all(re.fullmatch(r"[A-Za-z0-9]+", value) for value in ids)


def test_terminal_metadata_is_immutable_and_oldest_to_newest(manager):
    first = FakeTerm("first123", ["bash"])
    second = FakeTerm("second12", ["bash"])
    manager.register(first)
    manager.register(second)

    infos = manager.terminals()
    assert tuple(info.term_id for info in infos) == ("first123", "second12")
    assert infos[0].creation_order < infos[1].creation_order
    assert infos[0].created_at.tzinfo is UTC
    assert infos[0].backend == "fake"
    assert infos[0].supports_screen is False
    assert {"read", "contents", "send_text"} <= infos[0].capabilities
    assert {"resize", "cursor", "size"}.isdisjoint(infos[0].capabilities)
    assert manager.terminal_info("second12") is infos[1]
    with pytest.raises(FrozenInstanceError):
        infos[0].backend = "changed"


def test_terminal_metadata_detects_screen_backend_and_removal(manager):
    class ScreenTerm(FakeTerm):
        async def resize(self, rows, cols):
            pass

        async def cursor(self):
            return (0, 0)

        async def size(self):
            return (24, 80)

    terminal = ScreenTerm("screen12", ["bash"])
    manager.register(terminal)
    info = manager.terminal_info(terminal.term_id)
    assert info.backend == "screen"
    assert info.supports_screen is True
    assert {"resize", "cursor", "size"} <= info.capabilities

    asyncio.run(manager.remove(terminal.term_id, terminate=False))
    assert manager.terminals() == ()
    with pytest.raises(KeyError, match="unknown terminal"):
        manager.terminal_info(terminal.term_id)


def test_new_id_retries_registered_and_reserved_collisions(
    manager, monkeypatch
):
    manager.register(FakeTerm("aaaaaaaa", ["bash"]))
    manager._reserved_ids.add("bbbbbbbb")
    characters = iter("aaaaaaaabbbbbbbbcccccccc")
    monkeypatch.setattr(
        "ursa.tools.terminal.manager.secrets.choice",
        lambda alphabet: next(characters),
    )
    assert manager.new_id() == "cccccccc"


async def test_create_passes_configuration_and_registers(
    manager, tmp_path: Path
):
    made = []

    def factory(*args, **kwargs):
        terminal = FakeTerm(*args, **kwargs)
        made.append(terminal)
        return terminal

    result = await manager.create(
        ["printf", "hello"],
        env={"SPECIAL": "yes"},
        shell=["custom-shell", "-i"],
        cwd=tmp_path,
        session_factory=factory,
    )
    assert result is made[0]
    assert result.started_with == ["printf", "hello"]
    assert result.shell == ["custom-shell", "-i"]
    assert result.env == {"SPECIAL": "yes"}
    assert result.cwd == tmp_path
    assert manager.get(result.term_id) is result
    assert manager.ids() == (result.term_id,)


async def test_create_cleans_up_when_start_fails(manager):
    class BrokenTerm(FakeTerm):
        async def start(self, command=None):
            raise RuntimeError("boom")

    made = []

    def factory(*args, **kwargs):
        made.append(BrokenTerm(*args, **kwargs))
        return made[-1]

    with pytest.raises(RuntimeError, match="boom"):
        await manager.create(session_factory=factory)
    assert made[0].terminated is True
    assert manager.ids() == ()


async def test_failed_start_and_cleanup_remains_reachable(manager):
    class BrokenTerm(FakeTerm):
        async def start(self, command=None):
            raise RuntimeError("start failed")

        async def terminate(self):
            raise OSError("cleanup failed")

    with pytest.raises(BaseExceptionGroup) as caught:
        await manager.create(session_factory=BrokenTerm)
    assert [str(error) for error in caught.value.exceptions] == [
        "start failed",
        "cleanup failed",
    ]
    (term_id,) = manager.ids()
    assert manager.get(term_id).__class__ is BrokenTerm
    assert manager.terminal_info(term_id).term_id == term_id


async def test_registry_remove_close_and_unknown_errors(manager):
    first = FakeTerm("first123", ["bash"])
    second = FakeTerm("second12", ["bash"])
    manager.register(first)
    manager.register(second)
    with pytest.raises(ValueError, match="already registered"):
        manager.register(first)
    await manager.remove(first.term_id, terminate=False)
    assert first.terminated is False
    await manager.close_all()
    assert second.terminated is True
    assert manager.ids() == ()
    with pytest.raises(KeyError, match="unknown terminal"):
        manager.get("missing")
    with pytest.raises(KeyError, match="unknown terminal"):
        await manager.remove("missing")


async def test_close_all_drains_admitted_create_and_gates_new_ones(
    real_manager,
):
    manager = real_manager
    started = threading.Event()
    release = threading.Event()

    class SlowTerm(FakeTerm):
        async def start(self, command=None):
            started.set()
            await asyncio.to_thread(release.wait)

    creating = asyncio.create_task(manager.create(session_factory=SlowTerm))
    await asyncio.to_thread(started.wait)
    closing = asyncio.create_task(manager.close_all())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="closing"):
        await manager.create(session_factory=FakeTerm)
    with pytest.raises(RuntimeError, match="closing"):
        manager.register(FakeTerm("too-late", ["bash"]))
    release.set()
    terminal = await creating
    await closing
    assert terminal.terminated is True
    assert manager.ids() == ()


async def test_creation_recency_is_request_order_not_start_completion(
    real_manager,
):
    manager = real_manager
    releases = [threading.Event(), threading.Event()]
    starts = [threading.Event(), threading.Event()]
    made = []

    class OrderedTerm(FakeTerm):
        async def start(self, command=None):
            index = self.index
            starts[index].set()
            await asyncio.to_thread(releases[index].wait)

    def factory(*args, **kwargs):
        terminal = OrderedTerm(*args, **kwargs)
        terminal.index = len(made)
        made.append(terminal)
        return terminal

    first = asyncio.create_task(manager.create(session_factory=factory))
    await asyncio.to_thread(starts[0].wait)
    second = asyncio.create_task(manager.create(session_factory=factory))
    await asyncio.to_thread(starts[1].wait)
    releases[1].set()
    await second
    releases[0].set()
    await first

    assert [info.term_id for info in manager.terminals()] == [
        made[0].term_id,
        made[1].term_id,
    ]


async def test_close_all_retains_and_propagates_cancelled_termination(manager):
    class CancelledCleanupTerm(FakeTerm):
        async def terminate(self):
            raise asyncio.CancelledError("backend cancelled")

    terminal = CancelledCleanupTerm("cancel12", ["bash"])
    manager.register(terminal)
    with pytest.raises(asyncio.CancelledError):
        await manager.close_all()
    assert manager.get(terminal.term_id) is terminal


async def test_manager_io_methods_forward_to_session(manager):
    terminal = FakeTerm("forward1", ["bash"], output="abc")
    manager.register(terminal)
    await manager.send_bytes(terminal.term_id, b"x")
    await manager.send_text(terminal.term_id, "y")
    await manager.send_line(terminal.term_id, "z")
    await manager.send_keycode(terminal.term_id, 4)
    assert terminal.writes == [b"x", b"y", b"z\n", b"\x04"]
    assert await manager.read(terminal.term_id) == "abc"
    assert await manager.is_alive(terminal.term_id) == {"is_alive": True}


def test_registry_access_is_safe_across_threads_and_event_loops(manager):
    errors = []

    def worker(number):
        try:
            terminal = FakeTerm(f"thread{number}", ["bash"])
            manager.register(terminal)
            assert manager.get(terminal.term_id) is terminal
            asyncio.run(manager.remove(terminal.term_id))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(number,)) for number in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert manager.ids() == ()


async def test_wait_for_returns_matching_line_and_absolute_offset(manager):
    terminal = FakeTerm(
        "wait1234", ["bash"], output="before\nstatus: READY now\nafter\n"
    )
    manager.register(terminal)
    result = await manager.wait_for(terminal.term_id, r"READY")
    assert result == "status: READY now\nOffset: 15"


async def test_wait_for_observes_new_output(manager):
    terminal = FakeTerm("wait1234", ["bash"], output="starting\n")
    manager.register(terminal)

    async def update():
        await asyncio.sleep(0.01)
        terminal.output += "done 42\n"

    task = asyncio.create_task(update())
    assert await manager.wait_for(terminal.term_id, r"done \d+", 0.5) == (
        "done 42\nOffset: 9"
    )
    await task


async def test_wait_for_timeout_and_timeout_validation(manager):
    terminal = FakeTerm("wait1234", ["bash"], output="no match")
    manager.register(terminal)
    assert (
        await manager.wait_for(terminal.term_id, "never", 0)
        == "Pattern not found"
    )
    with pytest.raises(ValueError, match="must not be negative"):
        await manager.wait_for(terminal.term_id, "x", -0.1)
    with pytest.raises(ValueError, match="cannot exceed"):
        await manager.wait_for(terminal.term_id, "x", 10**6)


def test_default_shell_unix_and_windows_fallback(monkeypatch):
    monkeypatch.setattr("ursa.tools.terminal.manager.os.name", "posix")
    monkeypatch.setattr(
        "ursa.tools.terminal.manager.shutil.which", lambda name: "/x/bash"
    )
    assert TermManager.default_shell() == ["/x/bash"]

    monkeypatch.setattr("ursa.tools.terminal.manager.os.name", "nt")
    monkeypatch.setattr(
        TermManager, "_find_git_bash", staticmethod(lambda: None)
    )
    monkeypatch.setattr(
        "ursa.tools.terminal.manager.shutil.which",
        lambda name: "C:/PowerShell/pwsh.exe" if name == "pwsh" else None,
    )
    assert TermManager.default_shell() == ["C:/PowerShell/pwsh.exe"]


def test_default_shell_prefers_git_bash_on_windows(monkeypatch):
    monkeypatch.setattr("ursa.tools.terminal.manager.os.name", "nt")
    monkeypatch.setattr(
        TermManager,
        "_find_git_bash",
        staticmethod(lambda: "C:/Program Files/Git/bin/bash.exe"),
    )
    assert TermManager.default_shell() == ["C:/Program Files/Git/bin/bash.exe"]


def test_find_git_bash_prefers_path_lookup(monkeypatch):
    monkeypatch.setattr(
        "ursa.tools.terminal.manager.shutil.which",
        lambda executable: "C:/Git/bash.exe",
    )
    assert TermManager._find_git_bash() == "C:/Git/bash.exe"


@pytest.mark.parametrize("screen", [False, True])
def test_manager_capabilities_follow_selected_backend(monkeypatch, screen):
    monkeypatch.setattr(
        TermManager,
        "_default_backend",
        staticmethod(lambda: (FakeTerm, screen)),
    )
    capabilities = TermManager.supported_capabilities()
    assert capabilities >= {
        "contents",
        "is_alive",
        "read",
        "send_bytes",
        "send_keycode",
        "send_line",
        "send_text",
        "wait",
        "wait_for",
    }
    assert TermManager.supports_screen() is screen
    for capability in {"resize", "cursor", "size"}:
        assert (capability in capabilities) is screen
    assert TermManager._default_factory() is FakeTerm


def test_capabilities_follow_default_backend_screen_support(monkeypatch):
    monkeypatch.setattr(
        TermManager,
        "_default_backend",
        staticmethod(lambda: (FakeTerm, False)),
    )
    assert TermManager.supports_screen() is False
    assert {"read", "contents", "send_text"} <= (
        TermManager.supported_capabilities()
    )
    assert {"resize", "cursor", "size"}.isdisjoint(
        TermManager.supported_capabilities()
    )

    monkeypatch.setattr(
        TermManager,
        "_default_backend",
        staticmethod(lambda: (FakeTerm, True)),
    )
    assert TermManager.supports_screen() is True
    assert {"resize", "cursor", "size"} <= (
        TermManager.supported_capabilities()
    )


def test_manager_marshals_operations_from_distinct_event_loop_threads(
    real_manager,
):
    manager = real_manager

    class LoopTerm(FakeTerm):
        async def send_bytes(self, data):
            self.calls.append((
                threading.get_ident(),
                asyncio.get_running_loop(),
            ))
            await super().send_bytes(data)

    terminal = LoopTerm("threads1", ["bash"])
    terminal.calls = []
    manager.register(terminal)

    def invoke(value):
        asyncio.run(manager.send_text(terminal.term_id, value))

    threads = [
        threading.Thread(target=invoke, args=(str(index),))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(terminal.calls) == 2
    assert {call[0] for call in terminal.calls} == {manager._owner_thread.ident}
    assert {call[1] for call in terminal.calls} == {manager._owner_loop}


async def test_remove_retains_session_when_termination_fails(manager):
    class BrokenCleanupTerm(FakeTerm):
        async def terminate(self):
            raise RuntimeError("cleanup failed")

    terminal = BrokenCleanupTerm("cleanup1", ["bash"])
    manager.register(terminal)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await manager.remove(terminal.term_id)
    assert manager.get(terminal.term_id) is terminal


async def test_cancelled_create_reclaims_session_on_owner_loop(real_manager):
    manager = real_manager

    class SlowTerm(FakeTerm):
        async def start(self, command=None):
            await asyncio.sleep(0.03)
            await super().start(command)

    made = []

    def factory(*args, **kwargs):
        terminal = SlowTerm(*args, **kwargs)
        made.append(terminal)
        return terminal

    task = asyncio.create_task(manager.create("ok", session_factory=factory))
    await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.ids() == ()
    assert made[0].started_with == "ok"
    assert made[0].terminated is True
