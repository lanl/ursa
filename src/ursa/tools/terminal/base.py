"""Interfaces and shared values for persistent terminal sessions."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .resources import ProcessIdentity

TERM_TIMEOUT: Final[float] = float(os.getenv("URSA_TERM_TIMEOUT", "10"))
TERM_MAX_BYTES: Final[int] = int(os.getenv("URSA_TERM_MAX_BYTES", "20000"))
TERM_MAX_LINES: Final[int] = int(os.getenv("URSA_TERM_MAX_LINES", "200"))
TERM_SCROLLBACK_LINES: Final[int] = 2_000
TERM_CAPTURE_MAX_BYTES: Final[int] = 100 * 1024 * 1024
TERM_OUTPUT_JOURNAL_MAX_CHARS: Final[int] = 8 * 1024 * 1024
TERM_MAX_SESSIONS: Final[int] = 16
TERM_ID_LENGTH: Final[int] = 8


class OutputJournal:
    """Bounded decoded output with monotonic absolute character offsets."""

    def __init__(
        self,
        *,
        max_lines: int = TERM_SCROLLBACK_LINES,
        max_chars: int = TERM_OUTPUT_JOURNAL_MAX_CHARS,
    ) -> None:
        if max_lines <= 0 or max_chars <= 0:
            raise ValueError("journal limits must be positive")
        self._max_lines = max_lines
        self._max_chars = max_chars
        self._lines: deque[str] = deque()
        self._partial = ""
        self._retained_chars = 0
        self._end_offset = 0

    @property
    def end_offset(self) -> int:
        """Return the total decoded character count, including evicted text."""
        return self._end_offset

    @property
    def start_offset(self) -> int:
        """Return the absolute offset of the oldest retained character."""
        return self._end_offset - self._retained_chars

    @property
    def retained_lines(self) -> int:
        """Return the number of complete or partial retained lines."""
        return len(self._lines) + bool(self._partial)

    @property
    def retained_chars(self) -> int:
        return self._retained_chars

    def append(self, text: str) -> None:
        """Append decoded text and evict the oldest excess output."""
        if not text:
            return
        combined = self._partial + text
        pieces = combined.split("\n")
        self._partial = pieces.pop()
        self._lines.extend(f"{piece}\n" for piece in pieces)
        self._retained_chars += len(text)
        self._end_offset += len(text)
        self._trim_lines()
        self._trim_chars()

    def window_since(self, marker: int) -> tuple[int, str, int]:
        """Return retained output at or after *marker* and absolute bounds."""
        if marker < 0:
            raise ValueError("output marker must be non-negative")
        start = max(marker, self.start_offset)
        end = self._end_offset
        if start >= end:
            return end, "", end

        skip = start - self.start_offset
        chunks: list[str] = []
        for piece in (*self._lines, self._partial):
            if skip >= len(piece):
                skip -= len(piece)
                continue
            chunks.append(piece[skip:])
            skip = 0
        return start, "".join(chunks), end

    def _trim_lines(self) -> None:
        while self.retained_lines > self._max_lines and self._lines:
            removed = self._lines.popleft()
            self._retained_chars -= len(removed)

    def _trim_chars(self) -> None:
        excess = self._retained_chars - self._max_chars
        while excess > 0 and self._lines:
            first = self._lines[0]
            if len(first) <= excess:
                self._lines.popleft()
                removed = len(first)
            else:
                self._lines[0] = first[excess:]
                removed = excess
            self._retained_chars -= removed
            excess -= removed
        if excess > 0:
            self._partial = self._partial[excess:]
            self._retained_chars -= excess


@dataclass(frozen=True, slots=True)
class TerminalStyle:
    """Renderer-neutral visual attributes for terminal text."""

    foreground: tuple[int, int, int] | None = None
    background: tuple[int, int, int] | None = None
    bold: bool = False
    italic: bool = False
    faint: bool = False
    blink: bool = False
    underline: bool = False
    underline_kind: int = 0
    underline_color: tuple[int, int, int] | None = None
    reverse: bool = False
    conceal: bool = False
    strike: bool = False
    overline: bool = False


@dataclass(frozen=True, slots=True)
class TerminalSpan:
    """A run of terminal text sharing one style."""

    text: str
    style: TerminalStyle = TerminalStyle()
    cells: int | None = None


@dataclass(frozen=True, slots=True)
class TerminalRenderSnapshot:
    """Immutable terminal display state safe to move across event loops.

    Screen-backed snapshots have exact ``rows`` and ``cols``. Stream-backed
    snapshots leave both dimensions unset so a consumer can reflow them to its
    own viewport.
    """

    term_id: str
    spans: tuple[TerminalSpan, ...]
    rows: int | None = None
    cols: int | None = None
    screen: bool = False
    cursor: tuple[int, int] | None = None


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
        self._process_identity: ProcessIdentity | None = None

    @property
    def process_identity(self) -> ProcessIdentity | None:
        """Launch-time process identity, or None for an untracked backend.

        Backends capture this before polling/waiting can reap the child. The
        immutable identity is retained after exit; it must never be refreshed
        from a PID that may since have been reused.
        """
        return self._process_identity

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

    @abstractmethod
    async def read(self, *, offset: int = 0, lines: int | None = None) -> str:
        """Read output, optionally selecting lines back from the end."""

    async def contents(self) -> str:
        """Return the complete terminal output or scrollback."""
        return await self.read()

    async def output_marker(self) -> int:
        """Return a monotonic character offset into the backend output stream."""
        return len(await self.contents())

    async def output_since(self, marker: int) -> str:
        """Return output emitted at or after a marker from ``output_marker``."""
        if marker < 0:
            raise ValueError("output marker must be non-negative")
        return (await self.contents())[marker:]

    async def output_window(self, marker: int) -> tuple[int, str, int]:
        """Return new output and its absolute start and end offsets."""
        output = await self.output_since(marker)
        return marker, output, marker + len(output)

    async def render_snapshot(self) -> TerminalRenderSnapshot:
        """Return immutable display data for a view-only renderer."""
        return TerminalRenderSnapshot(
            term_id=self.term_id,
            spans=(TerminalSpan(await self.read()),),
        )

    @abstractmethod
    async def is_alive(self) -> dict[str, bool | int]:
        """Return ``is_alive`` or the completed process's ``exit_code``."""

    @abstractmethod
    async def wait(self) -> int:
        """Wait for the shell process and return its exit code."""

    @abstractmethod
    async def terminate(self) -> None:
        """Terminate the shell and release backend-owned resources."""

    def terminate_at_exit(self) -> None:
        """Synchronously terminate during interpreter teardown, if supported."""
        raise NotImplementedError

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

    async def mouse_event(
        self,
        action: str,
        row: int,
        col: int,
        *,
        button: str | None = None,
        modifiers: frozenset[str] = frozenset(),
    ) -> None:
        """Send one mouse event, if supported by the terminal backend."""
        del action, row, col, button, modifiers
        raise NotImplementedError("this terminal backend has no mouse input")

    async def paste_text(self, text: str) -> None:
        """Paste literal text according to the terminal's negotiated mode."""
        del text
        raise NotImplementedError("this terminal backend has no paste input")

    async def mouse_events(
        self,
        row: int,
        col: int,
        events: tuple[tuple[str, str | None], ...],
        *,
        modifiers: frozenset[str] = frozenset(),
    ) -> None:
        """Send an ordered batch of mouse events."""
        for action, button in events:
            await self.mouse_event(
                action, row, col, button=button, modifiers=modifiers
            )
