"""Unix PTY terminal session rendered by :mod:`pyghostty`.

``pyghostty`` is a headless VT emulator rather than a process manager.  This
backend owns both halves: a shell attached to a pseudoterminal and a Ghostty
terminal which consumes the bytes read from the PTY master.
"""

from __future__ import annotations

import asyncio
import errno
import os
import pty
import shlex
import signal
import struct
from contextlib import suppress
from typing import Any

from .base import TermSession

try:
    from pyghostty import Terminal as PyGhosttyTerminal
except (
    ImportError,
    OSError,
) as exc:  # The wheel may exist without its native library.
    PyGhosttyTerminal = None  # type: ignore[assignment,misc]
    _PYGHOSTTY_IMPORT_ERROR: BaseException | None = exc
else:
    _PYGHOSTTY_IMPORT_ERROR = None


class GhosttyTerm(TermSession):
    """A shell running in a Unix PTY, with screen state managed by Ghostty."""

    def __init__(
        self,
        term_id: str,
        shell: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        rows: int = 24,
        cols: int = 80,
        scrollback: int = 10_000,
    ) -> None:
        super().__init__(term_id, shell, env=env, cwd=cwd)
        if PyGhosttyTerminal is None:
            raise RuntimeError(
                "GhosttyTerm requires a working pyghostty installation"
            ) from _PYGHOSTTY_IMPORT_ERROR
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        self._terminal: Any = PyGhosttyTerminal(
            cols=cols, rows=rows, scrollback=scrollback
        )
        self._pid: int | None = None
        self._return_code: int | None = None
        self._master_fd: int | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._pump_error: BaseException | None = None
        self._io_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._terminate_task: asyncio.Task[None] | None = None
        self._closed = False
        self._terminal_closed = False
        self._cached_text = ""
        self._cached_contents = ""

    async def start(self, command: str | list[str] | None = None) -> None:
        if self._pid is not None:
            raise RuntimeError("terminal session has already been started")
        if self._closed:
            raise RuntimeError("terminal session is closed")

        child_env = os.environ.copy()
        child_env.update(self.env)
        child_env.setdefault("TERM", "xterm-256color")
        argv = self.shell
        if command is not None:
            command_text = (
                command if isinstance(command, str) else shlex.join(command)
            )
            argv = [*self.shell, "-c", command_text]
        pid, master_fd = pty.fork()
        if pid == 0:
            try:
                if self.cwd is not None:
                    os.chdir(self.cwd)
                os.execvpe(argv[0], argv, child_env)
            except BaseException:
                os._exit(127)

        self._pid = pid
        self._master_fd = master_fd
        os.set_blocking(master_fd, False)
        rows, cols = await self.size()
        self._set_winsize(master_fd, rows, cols)
        self._reader_task = asyncio.create_task(
            self._pump_output(), name=f"ursa-term-reader-{self.term_id}"
        )
        self._wait_task = asyncio.create_task(
            self._wait_process(), name=f"ursa-term-wait-{self.term_id}"
        )

    async def _pump_output(self) -> None:
        assert self._master_fd is not None
        loop = asyncio.get_running_loop()
        fd = self._master_fd
        try:
            while True:
                ready = loop.create_future()

                def mark_ready() -> None:
                    if not ready.done():
                        ready.set_result(None)

                loop.add_reader(fd, mark_ready)
                try:
                    await ready
                finally:
                    loop.remove_reader(fd)
                while True:
                    try:
                        data = os.read(fd, 65_536)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        if exc.errno == errno.EIO:  # Linux PTY end-of-file.
                            return
                        raise
                    if not data:
                        return
                    async with self._io_lock:
                        self._terminal.feed(data)
        except asyncio.CancelledError:
            raise
        finally:
            await asyncio.shield(self._close_master(fd))

    async def _close_master(self, fd: int) -> None:
        async with self._write_lock:
            async with self._io_lock:
                if self._master_fd != fd:
                    return
                self._master_fd = None
                with suppress(OSError):
                    os.close(fd)

    async def _wait_process(self) -> int:
        assert self._pid is not None
        _, status = await asyncio.to_thread(os.waitpid, self._pid, 0)
        code = os.waitstatus_to_exitcode(status)
        self._return_code = code
        if self._reader_task is not None:
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                self._pump_error = exc
        return code

    async def send_bytes(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        async with self._write_lock:
            async with self._io_lock:
                fd = self._master_fd
                stopped = self._return_code is not None or self._closed
            if fd is None or stopped:
                raise RuntimeError("terminal session is not running")
            view = memoryview(data)
            while view:
                try:
                    written = os.write(fd, view)
                except BlockingIOError:
                    await self._wait_writable(fd)
                    continue
                except OSError as exc:
                    if exc.errno in (errno.EBADF, errno.EIO, errno.EPIPE):
                        raise RuntimeError(
                            "terminal session is not running"
                        ) from exc
                    raise
                if written == 0:
                    await self._wait_writable(fd)
                    continue
                view = view[written:]

    @staticmethod
    async def _wait_writable(fd: int) -> None:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()

        def mark_ready() -> None:
            if not ready.done():
                ready.set_result(None)

        loop.add_writer(fd, mark_ready)
        try:
            await ready
        finally:
            loop.remove_writer(fd)

    async def read(self, *, offset: int = 0, lines: int | None = None) -> str:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if lines is not None and lines < 0:
            raise ValueError("lines must be non-negative")
        if lines is None and offset == 0:
            return await self.text()
        contents = await self.contents()
        all_lines = contents.splitlines()
        end = len(all_lines) - offset
        if lines is None:
            start = 0
        else:
            start = max(0, end - lines)
        return "\n".join(all_lines[start : max(0, end)])

    async def contents(self) -> str:
        """Return full scrollback plus the visible screen."""
        async with self._io_lock:
            return (
                self._cached_contents
                if self._terminal_closed
                else self._terminal.contents()
            )

    async def text(self) -> str:
        """Return the plain text of the currently visible screen."""
        async with self._io_lock:
            return (
                self._cached_text
                if self._terminal_closed
                else self._terminal.text()
            )

    async def is_alive(self) -> dict[str, bool | int]:
        if self._pid is None:
            return {"is_alive": False}
        if self._return_code is None:
            return {"is_alive": True}
        return {"exit_code": self._return_code}

    async def wait(self) -> int:
        if self._wait_task is None:
            raise RuntimeError("terminal session has not been started")
        return await asyncio.shield(self._wait_task)

    async def terminate(self) -> None:
        async with self._lifecycle_lock:
            if self._terminate_task is None:
                self._terminate_task = asyncio.create_task(
                    self._terminate(), name=f"ursa-term-close-{self.term_id}"
                )
            task = self._terminate_task
        await asyncio.shield(task)

    async def _terminate(self) -> None:
        self._closed = True
        try:
            pid = self._pid
            if pid is not None and self._return_code is None:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.wait(), timeout=2.0)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(pid, signal.SIGKILL)
                    await self.wait()
            elif self._wait_task is not None:
                await self.wait()
        finally:
            reader = self._reader_task
            if reader is not None:
                if not reader.done():
                    reader.cancel()
                try:
                    await reader
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    if self._pump_error is None:
                        self._pump_error = exc

            fd = self._master_fd
            if fd is not None:
                await self._close_master(fd)

            async with self._io_lock:
                if not self._terminal_closed:
                    with suppress(BaseException):
                        self._cached_text = self._terminal.text()
                    with suppress(BaseException):
                        self._cached_contents = self._terminal.contents()
                    with suppress(BaseException):
                        self._terminal.close()
                    self._terminal_closed = True

    async def resize(self, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        async with self._io_lock:
            if self._closed or self._terminal_closed:
                raise RuntimeError("terminal session is closed")
            self._terminal.resize(cols, rows)
            fd = self._master_fd
            if fd is not None:
                try:
                    self._set_winsize(fd, rows, cols)
                except OSError as exc:
                    if exc.errno not in (errno.EBADF, errno.EIO):
                        raise

    async def cursor(self) -> tuple[int, int]:
        async with self._io_lock:
            if self._terminal_closed:
                raise RuntimeError("terminal session is closed")
            col, row = self._terminal.cursor
            return row, col

    async def size(self) -> tuple[int, int]:
        async with self._io_lock:
            if self._terminal_closed:
                raise RuntimeError("terminal session is closed")
            cols, rows = self._terminal.size
            return rows, cols

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        import fcntl
        import termios

        fcntl.ioctl(
            fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0)
        )


__all__ = ["GhosttyTerm"]
