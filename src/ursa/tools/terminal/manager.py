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
from pathlib import Path

from .base import TERM_ID_LENGTH, TERM_TIMEOUT, TermSession

SessionFactory = Callable[..., TermSession]

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
                cls._instance._reserved_ids = set()
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
        return await self._dispatch_create(
            self._create(
                command,
                env=env,
                shell=shell,
                cwd=cwd,
                session_factory=session_factory,
            )
        )

    async def _create(
        self,
        command: str | list[str] | None,
        *,
        env: dict[str, str] | None,
        shell: list[str] | None,
        cwd: str | Path | None,
        session_factory: SessionFactory | None,
    ) -> TermSession:
        factory = session_factory or self._default_factory()
        with self._sessions_lock:
            term_id = self.new_id()
            self._reserved_ids.add(term_id)
        terminal: TermSession | None = None
        try:
            terminal = factory(
                term_id,
                shell or self.default_shell(),
                env=env,
                cwd=cwd,
            )
            await terminal.start(command)
        except BaseException:
            try:
                if terminal is not None:
                    await terminal.terminate()
            except BaseException:
                pass
            with self._sessions_lock:
                self._reserved_ids.discard(term_id)
            raise
        assert terminal is not None
        with self._sessions_lock:
            self._reserved_ids.discard(term_id)
            self._sessions[term_id] = terminal
        return terminal

    @staticmethod
    def _default_backend() -> tuple[SessionFactory, bool]:
        """Return the usable default factory and its screen support."""
        if os.name != "nt":
            try:
                from .ghostty import GhosttyTerm, PyGhosttyTerminal

                if PyGhosttyTerminal is not None:
                    return GhosttyTerm, True
            except (ImportError, OSError):
                pass
        from .process import ProcessTerm

        return ProcessTerm, False

    @staticmethod
    def _default_factory() -> SessionFactory:
        return TermManager._default_backend()[0]

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
            if (
                terminal.term_id in self._sessions
                or terminal.term_id in self._reserved_ids
            ):
                raise ValueError(
                    f"terminal already registered: {terminal.term_id}"
                )
            self._sessions[terminal.term_id] = terminal

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

    async def close_all(self) -> None:
        """Terminate and unregister every session."""
        await self._dispatch(self._close_all())

    async def _close_all(self) -> None:
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
        failures = [
            result for result in results if isinstance(result, Exception)
        ]
        if failures:
            raise failures[0]

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
