"""Subprocess-backed terminal sessions.

This backend deliberately provides only stream semantics.  It is useful when a
real terminal emulator is unavailable, but cannot report a cursor position or
render a screen in the way the Ghostty backend can.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .base import TermSession


class ProcessTerm(TermSession):
    """A terminal-like shell backed by :class:`subprocess.Popen`.

    Standard output and standard error share a named temporary file.  Opening
    a fresh reader for each call to :meth:`read` avoids moving the descriptor
    used by the child and works on platforms without ``os.pread``.
    """

    def __init__(
        self,
        term_id: str,
        shell: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> None:
        super().__init__(term_id, shell, env=env, cwd=cwd)
        self._process: subprocess.Popen[bytes] | None = None
        self._output_path: Path | None = None
        self._output_writer = None
        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._send_lock = asyncio.Lock()
        self._termination_task: asyncio.Task[None] | None = None

    async def start(self, command: str | list[str] | None = None) -> None:
        """Start the configured shell and optionally submit an initial command."""
        command = self._interactive_python_command(command)
        command_text = None
        if command is not None:
            command_text = (
                command
                if isinstance(command, str)
                else (
                    self._powershell_command(command)
                    if self._is_powershell(self.shell)
                    else self._format_command(command)
                )
            )
        await asyncio.to_thread(self._start_sync, command_text)

    @classmethod
    def _interactive_python_command(
        cls, command: str | list[str] | None
    ) -> str | list[str] | None:
        """Make an exact bare Python launch behave as a pipe-backed REPL."""
        if command is None:
            return None
        if isinstance(command, list):
            if len(command) == 1 and cls._is_python_executable(command[0]):
                return [*command, "-i"]
            return command

        try:
            words = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            return command
        if len(words) == 1 and cls._is_python_executable(words[0].strip("'\"")):
            return f"{command} -i"
        return command

    @staticmethod
    def _is_python_executable(value: str) -> bool:
        executable = value.replace("\\", "/").rsplit("/", 1)[-1]
        return (
            re.fullmatch(
                r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?",
                executable,
                re.IGNORECASE,
            )
            is not None
        )

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return shlex.join(command)

    @staticmethod
    def _is_powershell(shell: list[str]) -> bool:
        executable = shell[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
        return executable in {
            "powershell",
            "powershell.exe",
            "pwsh",
            "pwsh.exe",
        }

    @staticmethod
    def _powershell_command(command: list[str]) -> str:
        """Render argv as PowerShell single-quoted literal arguments."""
        if not command:
            return ""

        def quote(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        return "& " + " ".join(quote(value) for value in command)

    def _start_sync(self, command: str | None) -> None:
        with self._state_lock:
            if self._process is not None:
                raise RuntimeError(
                    f"terminal {self.term_id!r} is already started"
                )

            output = tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f"ursa-term-{self.term_id}-", delete=False
            )
            environment = os.environ.copy()
            environment.update(self.env)
            # ProcessTerm deliberately uses ordinary pipes/files instead of a
            # pseudo-terminal.  Python therefore block-buffers its output by
            # default, which makes an interactive REPL appear silent until it
            # exits.  Make child Python processes stream output while honoring
            # an explicit value supplied by the caller or parent environment.
            environment.setdefault("PYTHONUNBUFFERED", "1")
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            argv = self.shell
            if command is not None:
                command_flag = (
                    "-Command" if self._is_powershell(self.shell) else "-c"
                )
                argv = [*self.shell, command_flag, command]

            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    cwd=self.cwd,
                    env=environment,
                    bufsize=0,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
            except BaseException:
                path = Path(output.name)
                output.close()
                path.unlink(missing_ok=True)
                raise

            self._output_path = Path(output.name)
            self._output_writer = output
            self._process = process

    async def send_bytes(self, data: bytes) -> None:
        """Write raw bytes to the shell's standard input."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        async with self._send_lock:
            await asyncio.to_thread(self._send_sync, data)

    def _send_sync(self, data: bytes) -> None:
        with self._state_lock:
            process = self._require_started()
            if process.poll() is not None or process.stdin is None:
                raise BrokenPipeError(
                    f"terminal {self.term_id!r} is not running"
                )
            stdin = process.stdin
        try:
            with self._write_lock:
                remaining = memoryview(data)
                while remaining:
                    written = stdin.write(remaining)
                    if not written:
                        raise BrokenPipeError
                    remaining = remaining[written:]
                stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise BrokenPipeError(
                f"terminal {self.term_id!r} is not running"
            ) from exc

    async def read(self, *, offset: int = 0, lines: int | None = None) -> str:
        """Read output, optionally selecting lines back from the end.

        ``offset`` skips that many trailing lines.  ``lines`` then limits the
        result to the preceding number of lines.  With neither argument the
        complete captured stream is returned.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if lines is not None and lines < 0:
            raise ValueError("lines must be non-negative")
        return await asyncio.to_thread(self._read_sync, offset, lines)

    def _read_sync(self, offset: int, lines: int | None) -> str:
        with self._state_lock:
            self._require_started()
            path = self._output_path
        assert path is not None
        data = path.read_bytes().decode(errors="replace")
        if offset == 0 and lines is None:
            return data

        output_lines = data.splitlines(keepends=True)
        end = len(output_lines) - offset if offset else len(output_lines)
        start = 0 if lines is None else max(0, end - lines)
        return "".join(output_lines[start:end])

    async def is_alive(self) -> dict[str, bool | int]:
        """Return running state, or the process exit code once complete."""
        return await asyncio.to_thread(self._is_alive_sync)

    def _is_alive_sync(self) -> dict[str, bool | int]:
        with self._state_lock:
            return_code = self._require_started().poll()
        if return_code is None:
            return {"is_alive": True}
        return {"exit_code": return_code}

    async def wait(self) -> int:
        """Wait for the shell process to exit and return its exit code."""
        with self._state_lock:
            process = self._require_started()
        return await asyncio.to_thread(process.wait)

    async def terminate(self) -> None:
        """Terminate the shell, escalating to a kill when it does not stop."""
        if self._termination_task is None:
            self._termination_task = asyncio.create_task(
                asyncio.to_thread(self._terminate_sync)
            )
        await asyncio.shield(self._termination_task)

    def _terminate_sync(self) -> None:
        with self._state_lock:
            process = self._process
            if process is not None:
                self._stop_process(process)
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if self._output_writer is not None:
                self._output_writer.close()
                self._output_writer = None
            if self._output_path is not None:
                self._output_path.unlink(missing_ok=True)
                self._output_path = None

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = time.monotonic() + 2
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()

    def _require_started(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError(
                f"terminal {self.term_id!r} has not been started"
            )
        return self._process

    def __del__(self) -> None:
        """Stop an orphaned child and remove its private capture file."""
        try:
            process = self._process
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if self._output_writer is not None:
                self._output_writer.close()
            if self._output_path is not None:
                self._output_path.unlink(missing_ok=True)
        except (OSError, AttributeError):
            pass
