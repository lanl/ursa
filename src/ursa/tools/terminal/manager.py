"""Thread-safe registry and lifecycle manager for terminal sessions."""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import shutil
import string
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .base import (
    TERM_ID_LENGTH,
    TERM_TIMEOUT,
    TerminalRenderSnapshot,
    TermSession,
)

SessionFactory = Callable[..., TermSession]
TERM_BACKEND_ENV = "URSA_TERM_BACKEND"
_BACKEND_PREFERENCES = frozenset({"auto", "ghostty", "process"})

_STREAM_CAPABILITIES = frozenset({
    "contents",
    "is_alive",
    "read",
    "send_bytes",
    "send_keycode",
    "send_line",
    "send_text",
    "wait",
    "wait_for",
})
_SCREEN_CAPABILITIES = frozenset({"cursor", "resize", "size"})


@dataclass(frozen=True, slots=True)
class TermInfo:
    """Immutable registry metadata for a terminal session.

    ``creation_order`` is the authoritative recency key.  ``created_at`` is
    supplied for display only, since wall clocks can move or have insufficient
    resolution to distinguish terminals created close together.
    """

    term_id: str
    backend: str
    created_at: datetime
    creation_order: int
    capabilities: frozenset[str]
    supports_screen: bool


@dataclass(frozen=True, slots=True)
class _CreationReservation:
    term_id: str
    created_at: datetime
    creation_order: int


class TermManager:
    """Own all live terminal sessions.

    The singleton construction and registry are protected by synchronous
    locks, making lookup safe across event-loop threads.  No lock is held
    while awaiting backend I/O.
    """

    _instance: TermManager | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> TermManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._sessions = {}
                cls._instance._session_info = {}
                cls._instance._creation_counter = 0
                cls._instance._reserved_ids = set()
                cls._instance._creation_reservations = {}
                cls._instance._closing = False
                cls._instance._sessions_lock = threading.RLock()
                cls._instance._loop_ready = threading.Event()
                cls._instance._owner_thread = threading.Thread(
                    target=cls._instance._run_owner_loop,
                    name="ursa-terminal-manager",
                    daemon=True,
                )
                cls._instance._owner_thread.start()
                cls._instance._loop_ready.wait()
        return cls._instance

    def _run_owner_loop(self) -> None:
        self._owner_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._owner_loop)
        self._loop_ready.set()
        self._owner_loop.run_forever()

    async def _dispatch(self, coroutine):
        """Run backend work on the manager's dedicated event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self._owner_loop:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        # Caller cancellation must not strand a half-created or half-closed
        # session.  The owner loop completes lifecycle operations regardless.
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _dispatch_create(self, coroutine) -> TermSession:
        """Dispatch creation and reclaim a session abandoned by cancellation."""
        if asyncio.get_running_loop() is self._owner_loop:
            return await coroutine

        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # A cancelled caller cannot receive the successfully created ID.
            # Let atomic creation finish, then reclaim its session.
            try:
                terminal = await asyncio.shield(wrapped)
            except BaseException:
                pass
            else:
                cleanup = asyncio.run_coroutine_threadsafe(
                    self._remove(terminal.term_id, terminate=True),
                    self._owner_loop,
                )
                try:
                    await asyncio.shield(asyncio.wrap_future(cleanup))
                except BaseException:
                    # Failed termination leaves the session registered and
                    # reachable for a later cleanup attempt.
                    pass
            raise

    async def _dispatch_close(self) -> None:
        """Run close atomically even when its caller is cancelled."""
        if asyncio.get_running_loop() is self._owner_loop:
            await self._close_all()
            return
        future = asyncio.run_coroutine_threadsafe(
            self._close_all(), self._owner_loop
        )
        wrapped = asyncio.wrap_future(future)
        try:
            await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # Do not reopen the creation gate while close is still mutating the
            # registry.  Consume its eventual result, then preserve cancellation.
            try:
                await asyncio.shield(wrapped)
            except BaseException:
                pass
            raise

    @classmethod
    def default_shell(cls) -> list[str]:
        """Return the platform's preferred interactive shell command."""
        if os.name != "nt":
            return [shutil.which("bash") or "/bin/bash"]

        git_bash = cls._find_git_bash()
        if git_bash is not None:
            return [git_bash]
        powershell = (
            shutil.which("pwsh")
            or shutil.which("powershell")
            or "powershell.exe"
        )
        return [powershell]

    @staticmethod
    def _find_git_bash() -> str | None:
        direct = shutil.which("bash")
        if direct:
            return direct
        candidates = (
            Path(os.environ.get("ProgramFiles", "")) / "Git/bin/bash.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Git/bin/bash.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs/Git/bin/bash.exe",
        )
        return next((str(path) for path in candidates if path.is_file()), None)

    def new_id(self) -> str:
        """Return an unused random eight-character alphanumeric ID."""
        alphabet = string.ascii_letters + string.digits
        while True:
            term_id = "".join(
                secrets.choice(alphabet) for _ in range(TERM_ID_LENGTH)
            )
            with self._sessions_lock:
                if (
                    term_id not in self._sessions
                    and term_id not in self._reserved_ids
                ):
                    return term_id

    async def create(
        self,
        command: str | list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        shell: list[str] | None = None,
        cwd: str | Path | None = None,
        session_factory: SessionFactory | None = None,
    ) -> TermSession:
        """Construct, start, and register a terminal session."""
        with self._sessions_lock:
            if self._closing:
                raise RuntimeError("terminal manager is closing")
            term_id = self.new_id()
            self._creation_counter += 1
            reservation = _CreationReservation(
                term_id=term_id,
                created_at=datetime.now(UTC),
                creation_order=self._creation_counter,
            )
            self._reserved_ids.add(term_id)
            self._creation_reservations[term_id] = reservation
        return await self._dispatch_create(
            self._create(
                reservation,
                command,
                env=env,
                shell=shell,
                cwd=cwd,
                session_factory=session_factory,
            )
        )

    async def _create(
        self,
        reservation: _CreationReservation,
        command: str | list[str] | None,
        *,
        env: dict[str, str] | None,
        shell: list[str] | None,
        cwd: str | Path | None,
        session_factory: SessionFactory | None,
    ) -> TermSession:
        factory = session_factory or self._default_factory()
        term_id = reservation.term_id
        terminal: TermSession | None = None
        try:
            terminal = factory(
                term_id,
                shell or self.default_shell(),
                env=env,
                cwd=cwd,
            )
            await terminal.start(command)
        except BaseException as start_error:
            cleanup_error: BaseException | None = None
            try:
                if terminal is not None:
                    await terminal.terminate()
            except BaseException as error:
                cleanup_error = error
            with self._sessions_lock:
                self._reserved_ids.discard(term_id)
                self._creation_reservations.pop(term_id, None)
                if terminal is not None and cleanup_error is not None:
                    self._sessions[term_id] = terminal
                    self._record_info(terminal, reservation)
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "terminal startup and cleanup failed",
                    [start_error, cleanup_error],
                )
            raise
        assert terminal is not None
        with self._sessions_lock:
            self._reserved_ids.discard(term_id)
            self._creation_reservations.pop(term_id, None)
            self._sessions[term_id] = terminal
            self._record_info(terminal, reservation)
        return terminal

    @staticmethod
    def _session_capabilities(terminal: TermSession) -> frozenset[str]:
        capabilities = _STREAM_CAPABILITIES
        session_type = type(terminal)
        if all(
            getattr(session_type, operation)
            is not getattr(TermSession, operation)
            for operation in ("cursor", "resize", "size")
        ):
            capabilities |= _SCREEN_CAPABILITIES
        return capabilities

    @staticmethod
    def _backend_name(terminal: TermSession) -> str:
        name = type(terminal).__name__
        if name.endswith("Term"):
            name = name[:-4]
        return name.lower()

    def _record_info(
        self,
        terminal: TermSession,
        reservation: _CreationReservation | None = None,
    ) -> None:
        """Record metadata while ``_sessions_lock`` is held."""
        if reservation is None:
            self._creation_counter += 1
            reservation = _CreationReservation(
                term_id=terminal.term_id,
                created_at=datetime.now(UTC),
                creation_order=self._creation_counter,
            )
        capabilities = self._session_capabilities(terminal)
        self._session_info[terminal.term_id] = TermInfo(
            term_id=terminal.term_id,
            backend=self._backend_name(terminal),
            created_at=reservation.created_at,
            creation_order=reservation.creation_order,
            capabilities=capabilities,
            supports_screen=_SCREEN_CAPABILITIES <= capabilities,
        )

    @staticmethod
    def _backend_preference() -> str:
        """Return and validate the configured backend preference."""
        preference = os.environ.get(TERM_BACKEND_ENV, "auto").strip().lower()
        if preference not in _BACKEND_PREFERENCES:
            choices = ", ".join(sorted(_BACKEND_PREFERENCES))
            raise ValueError(f"{TERM_BACKEND_ENV} must be one of: {choices}")
        return preference

    @staticmethod
    def _ghostty_backend() -> tuple[SessionFactory, bool]:
        """Return Ghostty when its native dependency is usable."""
        if os.name == "nt":
            raise RuntimeError("Ghostty is unsupported on Windows")
        from .ghostty import GhosttyTerm, PyGhosttyTerminal

        if PyGhosttyTerminal is None:
            raise RuntimeError("pyghostty is unavailable")
        return GhosttyTerm, True

    @classmethod
    def _default_backend(cls) -> tuple[SessionFactory, bool]:
        """Return the configured factory and its screen support."""
        preference = cls._backend_preference()
        if preference == "process":
            from .process import ProcessTerm

            return ProcessTerm, False
        if preference == "ghostty":
            return cls._ghostty_backend()
        try:
            return cls._ghostty_backend()
        except (ImportError, OSError, RuntimeError):
            pass
        from .process import ProcessTerm

        return ProcessTerm, False

    @classmethod
    def _default_factory(cls) -> SessionFactory:
        return cls._default_backend()[0]

    @classmethod
    def backend_status(cls) -> str:
        """Return a human-readable description of the selected backend.

        ProcessTerm is deliberately identified as a fallback so the status UI
        does not imply that screen rendering and resize support are active.
        Detection failures are reported instead of breaking the status modal.
        """
        try:
            factory, supports_screen = cls._default_backend()
        except Exception as exc:
            return f"Unavailable ({type(exc).__name__}: {exc})"
        name = factory.__name__
        if name.endswith("Term"):
            name = name[:-4]
        preference = cls._backend_preference()
        if preference != "auto":
            return f"{name.capitalize()} (forced by {TERM_BACKEND_ENV})"
        if supports_screen:
            return f"{name.capitalize()} (preferred)"
        reason = (
            "Ghostty is unsupported on Windows"
            if os.name == "nt"
            else "pyghostty is unavailable"
        )
        return f"{name.capitalize()} (fallback: {reason})"

    @classmethod
    def supported_capabilities(cls) -> frozenset[str]:
        """Return operations supported by the usable default backend.

        This follows the same selection path as :meth:`create` without
        constructing a session, including native ``pyghostty`` availability.
        """
        capabilities = _STREAM_CAPABILITIES
        _, supports_screen = cls._default_backend()
        if supports_screen:
            capabilities |= _SCREEN_CAPABILITIES
        return capabilities

    @classmethod
    def supports_screen(cls) -> bool:
        """Return whether the default backend provides screen semantics."""
        return _SCREEN_CAPABILITIES <= cls.supported_capabilities()

    def register(self, terminal: TermSession) -> None:
        """Register an already-created session, primarily for integrations."""
        with self._sessions_lock:
            if self._closing:
                raise RuntimeError("terminal manager is closing")
            if (
                terminal.term_id in self._sessions
                or terminal.term_id in self._reserved_ids
            ):
                raise ValueError(
                    f"terminal already registered: {terminal.term_id}"
                )
            self._sessions[terminal.term_id] = terminal
            self._record_info(terminal)

    def get(self, term_id: str) -> TermSession:
        """Return a session or raise ``KeyError`` for an unknown ID."""
        with self._sessions_lock:
            try:
                return self._sessions[term_id]
            except KeyError:
                raise KeyError(f"unknown terminal: {term_id}") from None

    def ids(self) -> tuple[str, ...]:
        """Return a stable snapshot of registered IDs."""
        with self._sessions_lock:
            return tuple(self._sessions)

    def terminals(self) -> tuple[TermInfo, ...]:
        """Return an immutable oldest-to-newest terminal metadata snapshot."""
        with self._sessions_lock:
            infos = (
                self._session_info[term_id]
                for term_id in self._sessions
                if term_id in self._session_info
            )
            return tuple(sorted(infos, key=lambda info: info.creation_order))

    def terminal_info(self, term_id: str) -> TermInfo:
        """Return immutable metadata for one registered terminal."""
        with self._sessions_lock:
            if term_id not in self._sessions:
                raise KeyError(f"unknown terminal: {term_id}")
            return self._session_info[term_id]

    async def send_bytes(self, term_id: str, data: bytes) -> None:
        """Send raw bytes to a registered terminal."""
        await self._dispatch(self.get(term_id).send_bytes(data))

    async def send_text(self, term_id: str, text: str) -> None:
        """Send UTF-8 text to a registered terminal."""
        await self._dispatch(self.get(term_id).send_text(text))

    async def send_line(self, term_id: str, text: str) -> None:
        """Send text and a trailing newline to a registered terminal."""
        await self._dispatch(self.get(term_id).send_line(text))

    async def send_keycode(self, term_id: str, keycode: int) -> None:
        """Send a byte-valued keycode to a registered terminal."""
        await self._dispatch(self.get(term_id).send_keycode(keycode))

    async def read(
        self,
        term_id: str,
        *,
        offset: int = 0,
        lines: int | None = None,
    ) -> str:
        """Read a registered terminal's output."""
        return await self._dispatch(
            self.get(term_id).read(offset=offset, lines=lines)
        )

    async def is_alive(self, term_id: str) -> dict[str, bool | int]:
        """Return a registered terminal's process state."""
        return await self._dispatch(self.get(term_id).is_alive())

    async def wait(self, term_id: str) -> int:
        """Wait for a registered terminal to exit."""
        return await self._dispatch(self.get(term_id).wait())

    async def contents(self, term_id: str) -> str:
        """Return a registered terminal's complete output or scrollback."""
        return await self._dispatch(self.get(term_id).contents())

    async def render_snapshot(self, term_id: str) -> TerminalRenderSnapshot:
        """Return immutable display state for a registered terminal."""
        return await self._dispatch(self.get(term_id).render_snapshot())

    async def resize(self, term_id: str, rows: int, cols: int) -> None:
        """Resize a registered terminal."""
        await self._dispatch(self.get(term_id).resize(rows, cols))

    async def cursor(self, term_id: str) -> tuple[int, int]:
        """Return a registered terminal's cursor position."""
        return await self._dispatch(self.get(term_id).cursor())

    async def size(self, term_id: str) -> tuple[int, int]:
        """Return a registered terminal's dimensions."""
        return await self._dispatch(self.get(term_id).size())

    async def remove(self, term_id: str, *, terminate: bool = True) -> None:
        """Unregister a terminal and optionally terminate it."""
        await self._dispatch(self._remove(term_id, terminate=terminate))

    async def _remove(self, term_id: str, *, terminate: bool) -> None:
        with self._sessions_lock:
            terminal = self._sessions.get(term_id)
        if terminal is None:
            raise KeyError(f"unknown terminal: {term_id}")
        if terminate:
            await terminal.terminate()
        with self._sessions_lock:
            if self._sessions.get(term_id) is terminal:
                del self._sessions[term_id]
                self._session_info.pop(term_id, None)

    async def close_all(self) -> None:
        """Terminate and unregister every session."""
        with self._sessions_lock:
            if self._closing:
                raise RuntimeError("terminal manager is already closing")
            self._closing = True
        try:
            await self._dispatch_close()
        finally:
            with self._sessions_lock:
                self._closing = False

    async def _close_all(self) -> None:
        # Creates admitted before close_all set the gate may still be queued on
        # the owner loop or awaiting backend startup.  Yield until each has
        # either registered a reachable session or failed cleanly.
        while True:
            with self._sessions_lock:
                if not self._reserved_ids:
                    break
            await asyncio.sleep(0)
        with self._sessions_lock:
            terminals = tuple(self._sessions.values())
        results = await asyncio.gather(
            *(terminal.terminate() for terminal in terminals),
            return_exceptions=True,
        )
        with self._sessions_lock:
            for terminal, result in zip(terminals, results, strict=True):
                if not isinstance(result, BaseException):
                    if self._sessions.get(terminal.term_id) is terminal:
                        del self._sessions[terminal.term_id]
                        self._session_info.pop(terminal.term_id, None)
        failures = [
            result for result in results if isinstance(result, BaseException)
        ]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "multiple terminals failed to close", failures
            )

    async def wait_for(
        self,
        term_id: str,
        pattern: str,
        timeout: float | None = None,
    ) -> str:
        """Wait for a regex, returning its matching line and offset."""
        return await self._dispatch(
            self._wait_for(term_id, pattern, timeout=timeout)
        )

    async def _wait_for(
        self,
        term_id: str,
        pattern: str,
        *,
        timeout: float | None,
    ) -> str:
        wait_timeout = TERM_TIMEOUT if timeout is None else timeout
        if wait_timeout < 0:
            raise ValueError("timeout must not be negative")
        if wait_timeout > 2 * TERM_TIMEOUT:
            maximum = 2 * TERM_TIMEOUT
            raise ValueError(f"timeout cannot exceed {maximum:g} seconds")
        regex = re.compile(pattern)
        terminal = self.get(term_id)
        deadline = asyncio.get_running_loop().time() + wait_timeout
        while True:
            contents = await terminal.contents()
            match = regex.search(contents)
            if match is not None:
                line_start = contents.rfind("\n", 0, match.start()) + 1
                line_end = contents.find("\n", match.end())
                if line_end < 0:
                    line_end = len(contents)
                line = contents[line_start:line_end]
                return f"{line}\nOffset: {match.start()}"
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return "Pattern not found"
            await asyncio.sleep(min(0.05, remaining))


term_manager = TermManager()
