"""Thread-safe registry and lifecycle manager for terminal sessions."""

from __future__ import annotations

import asyncio
import atexit
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
from typing import Any

from .base import (
    TERM_ID_LENGTH,
    TERM_MAX_BYTES,
    TERM_MAX_SESSIONS,
    TERM_TIMEOUT,
    TerminalRenderSnapshot,
    TermSession,
)
from .resources import collect_resources
from .screenshot import screen_comparison_key

SessionFactory = Callable[..., TermSession]
TERM_BACKEND_ENV = "URSA_TERM_BACKEND"
_BACKEND_PREFERENCES = frozenset({"auto", "ghostty", "process"})

_STREAM_CAPABILITIES = frozenset({
    "contents",
    "is_alive",
    "read",
    "resources",
    "send_bytes",
    "send_line",
    "send_text",
    "wait",
    "wait_for",
})
_SCREEN_CAPABILITIES = frozenset({
    "cursor",
    "resize",
    "size",
    "wait_screen",
})
_MOUSE_CAPABILITIES = frozenset({"mouse"})
_PASTE_CAPABILITIES = frozenset({"paste"})
SCREEN_STABILITY_SECONDS = 1.0
SCREEN_STABILITY_FRAMES = 10
SCREEN_MIN_STABILITY_FRAMES = 2
WAIT_DEFAULT_MULTIPLIER = 5
WAIT_MAX_MULTIPLIER = 10


@dataclass(frozen=True, slots=True)
class TermInfo:
    """Immutable registry metadata; ``creation_order`` defines recency."""

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
    """Thread-safe singleton registry for sessions owned by one event loop."""

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

    async def _dispatch(self, coroutine, *, cancellable: bool = False) -> Any:
        """Run backend work on the owner loop with the requested cancellation policy."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is self._owner_loop:
            return await coroutine
        future = asyncio.run_coroutine_threadsafe(coroutine, self._owner_loop)
        wrapped = asyncio.wrap_future(future)
        if not cancellable:
            # Lifecycle work must survive caller cancellation.
            return await asyncio.shield(wrapped)
        try:
            return await wrapped
        except asyncio.CancelledError:
            future.cancel()
            raise

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
            self._ensure_capacity_locked()
            term_id = self.new_id()
            self._creation_counter += 1
            reservation = _CreationReservation(
                term_id=term_id,
                created_at=datetime.now(UTC),
                creation_order=self._creation_counter,
            )
            self._reserved_ids.add(term_id)
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

    def _ensure_capacity_locked(self) -> None:
        active = len(self._sessions) + len(self._reserved_ids)
        if active >= TERM_MAX_SESSIONS:
            raise RuntimeError(
                f"terminal limit reached ({TERM_MAX_SESSIONS}); "
                "close a terminal before opening another"
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
        factory = session_factory or self._default_backend()[0]
        term_id = reservation.term_id
        terminal: TermSession | None = None
        try:
            terminal = factory(
                term_id,
                self.default_shell() if shell is None else shell,
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
            self._sessions[term_id] = terminal
            self._record_info(terminal, reservation)
        return terminal

    @staticmethod
    def _session_capabilities(terminal: TermSession) -> frozenset[str]:
        session_type = type(terminal)
        screen = all(
            getattr(session_type, operation)
            is not getattr(TermSession, operation)
            for operation in ("cursor", "resize", "size")
        )
        capabilities = TermManager._capabilities(session_type, screen)
        if TermManager._factory_supports_mouse(session_type):
            capabilities |= _MOUSE_CAPABILITIES
        if TermManager._factory_supports_paste(session_type):
            capabilities |= _PASTE_CAPABILITIES
        return capabilities

    @staticmethod
    def _capabilities(
        factory: SessionFactory, supports_screen: bool
    ) -> frozenset[str]:
        capabilities = _STREAM_CAPABILITIES
        if supports_screen:
            capabilities |= _SCREEN_CAPABILITIES
            if TermManager._factory_supports_mouse(factory):
                capabilities |= _MOUSE_CAPABILITIES
            if TermManager._factory_supports_paste(factory):
                capabilities |= _PASTE_CAPABILITIES
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
        if preference != "process":
            try:
                return cls._ghostty_backend()
            except (ImportError, OSError, RuntimeError):
                if preference == "ghostty":
                    raise
        from .process import ProcessTerm

        return ProcessTerm, False

    @classmethod
    def backend_status(cls) -> str:
        """Describe the selected backend without raising detection failures."""
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
        """Return capabilities of the usable default backend."""
        factory, supports_screen = cls._default_backend()
        return cls._capabilities(factory, supports_screen)

    @staticmethod
    def _factory_supports_mouse(factory: SessionFactory) -> bool:
        return (
            getattr(factory, "mouse_event", TermSession.mouse_event)
            is not TermSession.mouse_event
        )

    @staticmethod
    def _factory_supports_paste(factory: SessionFactory) -> bool:
        return (
            getattr(factory, "paste_text", TermSession.paste_text)
            is not TermSession.paste_text
        )

    @classmethod
    def supports_screen(cls) -> bool:
        return _SCREEN_CAPABILITIES <= cls.supported_capabilities()

    @classmethod
    def supports_mouse(cls) -> bool:
        return "mouse" in cls.supported_capabilities()

    @classmethod
    def supports_paste(cls) -> bool:
        return "paste" in cls.supported_capabilities()

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
            self._ensure_capacity_locked()
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

    async def _call(
        self, term_id: str, operation: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke one backend operation on the owner loop."""
        return await self._dispatch(
            getattr(self.get(term_id), operation)(*args, **kwargs)
        )

    async def send_bytes(self, term_id: str, data: bytes) -> None:
        """Send raw bytes to a registered terminal."""
        await self._call(term_id, "send_bytes", data)

    async def send_text(self, term_id: str, text: str) -> None:
        """Send UTF-8 text to a registered terminal."""
        await self._call(term_id, "send_text", text)

    async def send_line(self, term_id: str, text: str) -> None:
        """Send text and a trailing newline to a registered terminal."""
        await self._call(term_id, "send_line", text)

    async def read(
        self,
        term_id: str,
        *,
        offset: int = 0,
        lines: int | None = None,
    ) -> str:
        """Read a registered terminal's output."""
        return await self._call(term_id, "read", offset=offset, lines=lines)

    async def is_alive(self, term_id: str) -> dict[str, bool | int]:
        """Return a registered terminal's process state."""
        return await self._call(term_id, "is_alive")

    async def resources(self, term_id: str) -> dict[str, int | float | str]:
        """Sample a terminal's local process tree without sending it input."""
        return await self._dispatch(
            self._resources(self.get(term_id)), cancellable=True
        )

    @staticmethod
    async def _resources(terminal: TermSession) -> dict[str, int | float | str]:
        result: dict[str, int | float | str] = {
            "term_id": terminal.term_id,
            "scope": "process_tree",
        }
        identity = terminal.process_identity
        if identity is not None:
            result["pid"] = identity.pid
        state = await terminal.is_alive()
        if "exit_code" in state:
            return TermManager._exited_resources(result, state)
        if identity is None:
            return {
                **result,
                "status": "unavailable",
                "reason": "backend has no captured process identity",
            }
        sampled = await collect_resources(identity)
        # Lifecycle state is authoritative, including an exit or concurrent
        # close during sampling. Do not return old counters as live usage.
        state = await terminal.is_alive()
        if "exit_code" in state:
            return TermManager._exited_resources(result, state)
        return {**result, **sampled}

    @staticmethod
    def _exited_resources(
        result: dict[str, int | float | str], state: dict[str, bool | int]
    ) -> dict[str, int | float | str]:
        return {**result, "status": "exited", "exit_code": state["exit_code"]}

    async def wait(self, term_id: str) -> int:
        return await self._call(term_id, "wait")

    async def contents(self, term_id: str) -> str:
        return await self._call(term_id, "contents")

    async def render_snapshot(self, term_id: str) -> TerminalRenderSnapshot:
        return await self._call(term_id, "render_snapshot")

    async def resize(self, term_id: str, rows: int, cols: int) -> None:
        await self._call(term_id, "resize", rows, cols)

    async def cursor(self, term_id: str) -> tuple[int, int]:
        return await self._call(term_id, "cursor")

    async def size(self, term_id: str) -> tuple[int, int]:
        return await self._call(term_id, "size")

    async def paste_text(self, term_id: str, text: str) -> None:
        terminal = self.get(term_id)
        if "paste" not in self.terminal_info(term_id).capabilities:
            raise NotImplementedError(
                "paste input requires a Ghostty-backed terminal"
            )
        await self._dispatch(terminal.paste_text(text))

    async def mouse_input(
        self,
        term_id: str,
        row: int,
        col: int,
        events: tuple[tuple[str, str | None], ...],
        *,
        modifiers: frozenset[str] = frozenset(),
    ) -> None:
        """Send encoded mouse input to a screen-backed terminal cell."""
        terminal = self.get(term_id)
        if "mouse" not in self.terminal_info(term_id).capabilities:
            raise NotImplementedError(
                "mouse input requires a screen-backed terminal"
            )
        if row < 0 or col < 0:
            raise ValueError("mouse row and col must be non-negative")
        rows, cols = await self._dispatch(terminal.size())
        if row >= rows or col >= cols:
            raise ValueError(
                f"mouse position ({row}, {col}) is outside the "
                f"{rows} by {cols} terminal"
            )
        await self._dispatch(
            self._mouse_input(terminal, row, col, events, modifiers=modifiers)
        )

    @staticmethod
    async def _mouse_input(
        terminal: TermSession,
        row: int,
        col: int,
        events: tuple[tuple[str, str | None], ...],
        *,
        modifiers: frozenset[str],
    ) -> None:
        actions = {"press", "release", "motion"}
        buttons = {
            "left",
            "middle",
            "right",
            "wheel_up",
            "wheel_down",
            "wheel_left",
            "wheel_right",
        }
        for action, button in events:
            if action not in actions:
                raise ValueError(f"unsupported mouse action: {action}")
            if button is not None and button not in buttons:
                raise ValueError(f"unsupported mouse button: {button}")
            if action != "motion" and button is None:
                raise ValueError(f"mouse {action} requires a button")
        await terminal.mouse_events(row, col, events, modifiers=modifiers)

    async def remove(self, term_id: str, *, terminate: bool = True) -> None:
        await self._dispatch(self._remove(term_id, terminate=terminate))

    async def _remove(self, term_id: str, *, terminate: bool) -> None:
        with self._sessions_lock:
            terminal = self._sessions.get(term_id)
        if terminal is None:
            raise KeyError(f"unknown terminal: {term_id}")
        if terminate:
            await terminal.terminate()
        self._unregister(terminal)

    def _unregister(self, terminal: TermSession) -> None:
        """Remove *terminal* only if it is still the registered instance."""
        with self._sessions_lock:
            if self._sessions.get(terminal.term_id) is terminal:
                del self._sessions[terminal.term_id]
                self._session_info.pop(terminal.term_id, None)

    async def close_all(self) -> None:
        with self._sessions_lock:
            if self._closing:
                raise RuntimeError("terminal manager is already closing")
            self._closing = True
        try:
            await self._dispatch_close()
        finally:
            with self._sessions_lock:
                self._closing = False

    def close_all_sync(self, *, timeout: float = 10.0) -> None:
        with self._sessions_lock:
            if not self._sessions and not self._reserved_ids:
                return
            if self._closing:
                return
            self._closing = True
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._close_all(), self._owner_loop
            )
            future.result(timeout=timeout)
        finally:
            with self._sessions_lock:
                self._closing = False

    def terminate_all_at_exit(self) -> None:
        with self._sessions_lock:
            terminals = tuple(self._sessions.values())
        for terminal in terminals:
            try:
                terminal.terminate_at_exit()
            except BaseException:
                continue
            self._unregister(terminal)

    async def _close_all(self) -> None:
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
        for terminal, result in zip(terminals, results, strict=True):
            if not isinstance(result, BaseException):
                self._unregister(terminal)
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
        return await self._dispatch(
            self._wait_for(term_id, pattern, timeout=timeout), cancellable=True
        )

    async def _wait_for(
        self,
        term_id: str,
        pattern: str,
        *,
        timeout: float | None,
    ) -> str:
        wait_timeout = _wait_timeout(timeout)
        regex = re.compile(pattern)
        terminal = self.get(term_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout
        if wait_timeout <= 0:
            return "Pattern not found"
        try:
            marker = await asyncio.wait_for(
                terminal.output_marker(), timeout=wait_timeout
            )
        except TimeoutError:
            return "Pattern not found"
        cursor = marker
        contents = ""
        content_offset = marker
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return "Pattern not found"
            try:
                chunk_start, chunk, chunk_end = await asyncio.wait_for(
                    terminal.output_window(cursor), timeout=remaining
                )
            except TimeoutError:
                return "Pattern not found"
            if loop.time() >= deadline:
                return "Pattern not found"
            match = None
            if chunk_start > cursor:
                contents = ""
                content_offset = chunk_start
            cursor = chunk_end
            if chunk:
                contents += chunk
                if len(contents) > TERM_MAX_BYTES:
                    removed = len(contents) - TERM_MAX_BYTES
                    contents = contents[removed:]
                    content_offset += removed
                match = _last_overlapping_match(regex, contents)
            if match is not None:
                line_start = contents.rfind("\n", 0, match.start()) + 1
                line_end = contents.find("\n", match.end())
                if line_end < 0:
                    line_end = len(contents)
                line = contents[line_start:line_end]
                return f"{line}\nOffset: {content_offset + match.start()}"
            remaining = deadline - loop.time()
            if remaining <= 0:
                return "Pattern not found"
            await asyncio.sleep(min(0.05, remaining))

    async def wait_screen(
        self,
        term_id: str,
        *,
        condition: str = "stable",
        bounding_box: tuple[int, int, int, int] | None = None,
        include_styling: bool = True,
        timeout: float | None = None,
    ) -> str:
        return await self._dispatch(
            self._wait_screen(
                term_id,
                condition=condition,
                bounding_box=bounding_box,
                include_styling=include_styling,
                timeout=timeout,
            ),
            cancellable=True,
        )

    async def _wait_screen(
        self,
        term_id: str,
        *,
        condition: str,
        bounding_box: tuple[int, int, int, int] | None,
        include_styling: bool,
        timeout: float | None,
    ) -> str:
        if condition not in {"stable", "change"}:
            raise ValueError("condition must be 'stable' or 'change'")
        wait_timeout = _wait_timeout(timeout)
        _validate_bounding_box(bounding_box)

        terminal = self.get(term_id)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + wait_timeout
        if wait_timeout <= 0:
            return _screen_timeout_result(condition)
        try:
            initial = await asyncio.wait_for(
                terminal.render_snapshot(), timeout=wait_timeout
            )
        except TimeoutError:
            return _screen_timeout_result(condition)
        if not initial.screen:
            raise NotImplementedError(
                "screen waiting requires a screen-backed terminal"
            )
        previous = screen_comparison_key(initial, bounding_box, include_styling)
        stable_since = loop.time()
        stable_frames = 1
        while True:
            now = loop.time()
            remaining = deadline - now
            if remaining <= 0:
                return _screen_timeout_result(condition)
            sleep_for = min(0.05, remaining)
            if condition == "stable":
                sleep_for = min(
                    sleep_for,
                    max(0, SCREEN_STABILITY_SECONDS - (now - stable_since)),
                )
            await asyncio.sleep(sleep_for)
            remaining = deadline - loop.time()
            if remaining <= 0:
                return _screen_timeout_result(condition)
            try:
                snapshot = await asyncio.wait_for(
                    terminal.render_snapshot(), timeout=remaining
                )
            except TimeoutError:
                return _screen_timeout_result(condition)
            current = screen_comparison_key(
                snapshot,
                bounding_box,
                include_styling,
                allow_clipped_box=True,
            )
            now = loop.time()
            if now >= deadline:
                return _screen_timeout_result(condition)
            if current != previous:
                if condition == "change":
                    return "Screen changed"
                previous = current
                stable_since = now
                stable_frames = 1
            elif condition == "stable":
                stable_frames += 1
                enough_frames = stable_frames >= SCREEN_MIN_STABILITY_FRAMES
                if enough_frames and (
                    stable_frames >= SCREEN_STABILITY_FRAMES
                    or now - stable_since >= SCREEN_STABILITY_SECONDS
                ):
                    return "Screen stabilized"


def _last_overlapping_match(
    regex: re.Pattern[str], contents: str
) -> re.Match[str] | None:
    """Return the rightmost match, including matches that overlap."""
    match = None
    position = 0
    while position <= len(contents):
        candidate = regex.search(contents, position)
        if candidate is None:
            break
        match = candidate
        position = candidate.start() + 1
    return match


def _wait_timeout(timeout: float | None) -> float:
    """Validate one bounded terminal wait timeout."""
    value = (
        WAIT_DEFAULT_MULTIPLIER * TERM_TIMEOUT if timeout is None else timeout
    )
    if value < 0:
        raise ValueError("timeout must not be negative")
    maximum = WAIT_MAX_MULTIPLIER * TERM_TIMEOUT
    if value > maximum:
        raise ValueError(f"timeout cannot exceed {maximum:g} seconds")
    return value


def _validate_bounding_box(
    bounding_box: tuple[int, int, int, int] | None,
) -> None:
    if bounding_box is None:
        return
    top, left, bottom, right = bounding_box
    if min(bounding_box) < 0:
        raise ValueError("bounding-box coordinates must not be negative")
    if bottom <= top or right <= left:
        raise ValueError("bounding box must have positive height and width")


def _screen_timeout_result(condition: str) -> str:
    return (
        "Screen did not stabilize"
        if condition == "stable"
        else "Screen did not change"
    )


term_manager = TermManager()


def _close_terminals_at_exit() -> None:
    try:
        term_manager.terminate_all_at_exit()
    except BaseException:
        # Interpreter teardown may already have dismantled backend modules.
        # Backend destructors remain as the final best-effort fallback.
        pass


atexit.register(_close_terminals_at_exit)
