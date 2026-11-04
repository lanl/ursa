"""Ghostty backend contracts, using the binding-free emulator double."""

import os
import signal
import subprocess
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from ursa.tools.terminal import ghostty
from ursa.tools.terminal.screenshot import (
    settled_screen_snapshot,
    terminal_snapshot_to_png,
)

REAL_BINDING = ghostty.PyGhosttyTerminal
pytestmark = pytest.mark.skipif(
    REAL_BINDING is None, reason="pyghostty unavailable"
)


class FakeTerminal:
    def __init__(self, *, cols, rows, scrollback):
        self.size, self.cursor, self.scrollback = (
            (cols, rows),
            (5, 3),
            scrollback,
        )
        self.fed, self.closed, self.resize_calls = bytearray(), False, []

    def feed(self, data):
        self.fed.extend(data)

    def contents(self):
        return self.fed.decode(errors="replace")

    def text(self):
        return self.contents()

    def style(self, col, row):
        return {
            "fg": ("rgb", (10 + col, 20, 30)),
            "bg": ("palette", 1),
            "bold": row == 0,
            "italic": False,
            "faint": False,
            "blink": False,
            "underline": 0,
            "inverse": False,
            "invisible": False,
            "strikethrough": False,
            "overline": False,
        }

    def resize(self, cols, rows):
        self.resize_calls.append((cols, rows))
        self.size = (cols, rows)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_binding(monkeypatch):
    monkeypatch.setattr(ghostty, "PyGhosttyTerminal", FakeTerminal)


def test_constructor_limits_and_visible_decoder(monkeypatch):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"], scrollback=10_000)
    assert terminal._terminal.size == (80, 24)
    assert terminal._terminal.scrollback == 2_000
    with pytest.raises(ValueError):
        ghostty.GhosttyTerm("bad", ["sh"], rows=0)
    monkeypatch.setattr(ghostty, "PyGhosttyTerminal", None)
    with pytest.raises(RuntimeError):
        ghostty.GhosttyTerm("missing", ["sh"])
    decoder = ghostty._VisibleOutputDecoder()
    assert decoder.decode(b"\x1b[31mREADY\x1b[0m\n") == "READY\n"
    assert decoder.decode(b"\x1b]0;hidden\x1b\\visible") == "visible"


@pytest.mark.asyncio
async def test_screen_read_render_resize_and_png(monkeypatch):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"], rows=2, cols=8)
    terminal._terminal.feed(b"one\ntwo")
    terminal._master_fd = 9
    winsizes = []
    monkeypatch.setattr(
        terminal, "_set_winsize", lambda *args: winsizes.append(args)
    )
    assert await terminal.read(lines=1) == "two"
    snapshot = await terminal.render_snapshot()
    assert snapshot.screen and (snapshot.rows, snapshot.cols) == (2, 8)
    assert snapshot.spans[0].style.foreground == (10, 20, 30)
    await terminal.resize(3, 9)
    assert terminal._terminal.resize_calls == [(9, 3)] and winsizes == [
        (9, 3, 9)
    ]
    terminal._master_fd = None
    image = Image.open(BytesIO(terminal_snapshot_to_png(snapshot)))
    assert image.width > 0 and image.height > 0
    await terminal.terminate()
    assert terminal._terminal.closed


@pytest.mark.asyncio
async def test_input_mouse_paste_and_validation(monkeypatch):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    sent = []

    async def send(data):
        sent.append(data)

    monkeypatch.setattr(terminal, "send_bytes", send)
    monkeypatch.setattr(
        terminal,
        "_encode_mouse_event",
        lambda *_args, **_kwargs: b"mouse",
    )
    await terminal.mouse_event("press", 1, 2, button="left")
    assert sent == [b"mouse"]
    terminal._terminal._t = [None]
    monkeypatch.setattr(
        ghostty, "_ghostty_ffi", SimpleNamespace(new=lambda _: [True])
    )
    monkeypatch.setattr(
        ghostty,
        "_ghostty_lib",
        SimpleNamespace(ghostty_terminal_mode_get=lambda *_: 0),
    )
    monkeypatch.setattr(ghostty, "_ghostty_check", lambda *_: None)
    await terminal.paste_text("paste")
    assert sent[-1] == b"\x1b[200~paste\x1b[201~"
    with pytest.raises(ValueError):
        await terminal.resize(0, 1)


@pytest.mark.asyncio
async def test_screen_settling_observes_change_and_timeout():
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    terminal._terminal.feed(b"before")

    class Provider:
        async def render_snapshot(self, _term_id):
            return await terminal.render_snapshot()

    settled = await settled_screen_snapshot(
        Provider(), terminal.term_id, timeout=0.02, interval=0
    )
    assert "before" in "".join(span.text for span in settled.spans)


@pytest.mark.skipif(os.name == "nt", reason="PTY smoke is Unix-only")
def test_native_backend_smoke_when_binding_is_available():
    if REAL_BINDING is None:
        pytest.skip("pyghostty unavailable")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio\n"
            "from ursa.tools.terminal.ghostty import GhosttyTerm\n"
            "async def main():\n"
            "    terminal = GhosttyTerm('native123', ['/bin/sh'])\n"
            "    try:\n"
            "        await terminal.start('printf native')\n"
            "        assert await terminal.wait() == 0\n"
            "        assert 'native' in await terminal.contents()\n"
            "    finally:\n"
            "        await terminal.terminate()\n"
            "asyncio.run(main())\n",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == -signal.SIGILL:
        pytest.skip("pyghostty wheel requires unsupported CPU instructions")
    assert result.returncode == 0, result.stderr
