"""Interfaces and shared values for persistent terminal sessions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

TERM_TIMEOUT: Final[float] = float(os.getenv("URSA_TERM_TIMEOUT", "10"))
TERM_MAX_BYTES: Final[int] = int(os.getenv("URSA_TERM_MAX_BYTES", "20000"))
TERM_MAX_LINES: Final[int] = int(os.getenv("URSA_TERM_MAX_LINES", "200"))
TERM_ID_LENGTH: Final[int] = 8


class TermSession(ABC):
    """An asynchronous shell session owned by :class:`TermManager`.

    Backends are responsible for serializing access to their streams and
    terminal emulator.  The manager only serializes changes to its registry.
    """

    def __init__(
        self,
        term_id: str,
        shell: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if not shell:
            raise ValueError("shell must contain at least one argument")
        self.term_id = term_id
        self.shell = list(shell)
        self.env = dict(env or {})
        self.cwd = Path(cwd) if cwd is not None else None

    @abstractmethod
    async def start(self, command: str | list[str] | None = None) -> None:
        """Start the shell, optionally executing *command* in it."""

    @abstractmethod
    async def send_bytes(self, data: bytes) -> None:
        """Write raw bytes to the terminal input."""

    async def send_text(self, text: str) -> None:
        """Write text to the terminal using UTF-8."""
        await self.send_bytes(text.encode())

    async def send_line(self, text: str) -> None:
        """Write text followed by a newline."""
        await self.send_text(f"{text}\n")

    async def send_keycode(self, keycode: int) -> None:
        """Write a single byte keycode to the terminal."""
        if not 0 <= keycode <= 255:
            raise ValueError("keycode must be between 0 and 255")
        await self.send_bytes(bytes((keycode,)))

    @abstractmethod
    async def read(self, *, offset: int = 0, lines: int | None = None) -> str:
        """Read output, optionally selecting lines back from the end."""

    async def contents(self) -> str:
        """Return the complete terminal output or scrollback."""
        return await self.read()

    @abstractmethod
    async def is_alive(self) -> dict[str, bool | int]:
        """Return ``is_alive`` or the completed process's ``exit_code``."""

    @abstractmethod
    async def wait(self) -> int:
        """Wait for the shell process and return its exit code."""

    @abstractmethod
    async def terminate(self) -> None:
        """Terminate the shell and release backend-owned resources."""

    async def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal, if supported by the backend."""
        del rows, cols
        raise NotImplementedError("this terminal backend cannot be resized")

    async def cursor(self) -> tuple[int, int]:
        """Return the cursor row and column, if supported."""
        raise NotImplementedError("this terminal backend has no cursor")

    async def size(self) -> tuple[int, int]:
        """Return terminal rows and columns, if supported."""
        raise NotImplementedError("this terminal backend has no screen size")
