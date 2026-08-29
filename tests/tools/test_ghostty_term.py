import asyncio
import os
from io import BytesIO

import pytest
from PIL import Image

from ursa.tools.terminal import ghostty
from ursa.tools.terminal.screenshot import (
    settled_screen_snapshot,
    terminal_snapshot_to_png,
)


class FakeGhosttyTerminal:
    def __init__(self, *, cols, rows, scrollback):
        self.size = (cols, rows)
        self.cursor = (5, 3)
        self.scrollback = scrollback
        self.fed = bytearray()
        self.closed = False
        self.resize_calls = []

    def feed(self, data):
        self.fed.extend(data)

    def contents(self):
        return self.fed.decode(errors="replace")

    def text(self):
        return f"SCREEN:{self.contents()}"

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


@pytest.fixture
def fake_ghostty(monkeypatch):
    monkeypatch.setattr(ghostty, "PyGhosttyTerminal", FakeGhosttyTerminal)


def test_ghostty_requires_library_and_valid_dimensions(
    monkeypatch, fake_ghostty
):
    with pytest.raises(ValueError, match="positive"):
        ghostty.GhosttyTerm("ghost123", ["/bin/sh"], rows=0)
    monkeypatch.setattr(ghostty, "PyGhosttyTerminal", None)
    with pytest.raises(RuntimeError, match="requires a working pyghostty"):
        ghostty.GhosttyTerm("ghost123", ["/bin/sh"])


def test_ghostty_default_screen_is_120_columns_by_40_rows(fake_ghostty):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])

    assert terminal._terminal.size == (120, 40)


def test_visible_output_decoder_strips_split_color_and_title_sequences():
    decoder = ghostty._VisibleOutputDecoder()
    assert decoder.decode(b"\x1b[31") == ""
    assert decoder.decode(b"mREADY\x1b[0m\n") == "READY\n"
    assert decoder.decode(b"\x1b]0;secret") == ""
    assert decoder.decode(b" title\x1b\\clean") == "clean"
    assert decoder.decode(b"\x1b(Btext") == "text"
    assert decoder.decode("\x9b31mRED\x9b0m".encode()) == "RED"
    assert decoder.decode("\x9d0;hidden\x9cshown".encode()) == "shown"
    assert decoder.decode(b"\x9b31mRAW\x9b0m") == "RAW"
    assert decoder.decode(b"\x9d0;raw title\x9cvisible") == "visible"
    assert decoder.decode("؛".encode()) == "؛"


async def test_ghostty_screen_primitives_and_read_slicing(
    fake_ghostty, monkeypatch
):
    terminal = ghostty.GhosttyTerm(
        "ghost123", ["/bin/sh"], rows=20, cols=70, scrollback=123
    )
    terminal._terminal.feed(b"one\ntwo\nthree\nfour")
    assert await terminal.read() == "SCREEN:one\ntwo\nthree\nfour"
    assert await terminal.contents() == "one\ntwo\nthree\nfour"
    assert await terminal.read(lines=2) == "three\nfour"
    assert await terminal.read(offset=1, lines=2) == "two\nthree"
    assert await terminal.read(offset=2) == "one\ntwo"
    assert await terminal.text() == "SCREEN:one\ntwo\nthree\nfour"
    assert await terminal.cursor() == (3, 5)
    assert await terminal.size() == (20, 70)

    resize_ioctl = []
    terminal._master_fd = 99
    monkeypatch.setattr(
        terminal,
        "_set_winsize",
        lambda fd, rows, cols: resize_ioctl.append((fd, rows, cols)),
    )
    await terminal.resize(30, 100)
    assert terminal._terminal.resize_calls == [(100, 30)]
    assert resize_ioctl == [(99, 30, 100)]
    assert await terminal.size() == (30, 100)


async def test_ghostty_render_snapshot_has_exact_grid_and_cell_styles(
    fake_ghostty,
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"], rows=2, cols=12)
    terminal._terminal.text = lambda: "ab\nc"

    rendered = await terminal.render_snapshot()

    assert rendered.term_id == "ghost123"
    assert rendered.screen is True
    assert (rendered.rows, rendered.cols) == (2, 12)
    assert "".join(span.text for span in rendered.spans) == (
        "ab" + " " * 10 + "\n" + "c" + " " * 11
    )
    assert rendered.spans[0].style.foreground == (10, 20, 30)
    assert rendered.spans[0].style.background == (205, 49, 49)
    assert rendered.spans[0].style.bold is True
    assert rendered.spans[1].style.foreground == (11, 20, 30)
    assert rendered.spans[-1].style.bold is False

    await terminal.terminate()
    assert terminal._terminal.closed
    assert await terminal.render_snapshot() == rendered


async def test_ghostty_validates_operations(fake_ghostty):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    with pytest.raises(ValueError, match="offset"):
        await terminal.read(offset=-1)
    with pytest.raises(ValueError, match="lines"):
        await terminal.read(lines=-1)
    with pytest.raises(ValueError, match="positive"):
        await terminal.resize(-1, 20)
    with pytest.raises(TypeError, match="bytes"):
        await terminal.send_bytes("bad")
    with pytest.raises(RuntimeError, match="not running"):
        await terminal.send_bytes(b"x")
    with pytest.raises(RuntimeError, match="not been started"):
        await terminal.wait()
    assert await terminal.is_alive() == {"is_alive": False}


@pytest.mark.parametrize("rows,cols", [(0, 1), (1, 0), (-1, 1), (1, -1)])
async def test_ghostty_resize_rejects_each_nonpositive_dimension(
    fake_ghostty, rows, cols
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    with pytest.raises(ValueError, match="rows and cols must be positive"):
        await terminal.resize(rows, cols)


async def test_ghostty_resize_accepts_one_and_handles_stale_pty_errors(
    fake_ghostty, monkeypatch
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    terminal._master_fd = 42
    for expected_errno in (ghostty.errno.EBADF, ghostty.errno.EIO):
        monkeypatch.setattr(
            terminal,
            "_set_winsize",
            lambda *args, value=expected_errno: (_ for _ in ()).throw(
                OSError(value, "stale pty")
            ),
        )
        await terminal.resize(1, 1)
    assert terminal._terminal.resize_calls == [(1, 1), (1, 1)]


async def test_ghostty_resize_propagates_unexpected_os_error(
    fake_ghostty, monkeypatch
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    terminal._master_fd = 42

    def fail(*args):
        raise OSError(ghostty.errno.EPERM, "denied")

    monkeypatch.setattr(terminal, "_set_winsize", fail)
    with pytest.raises(OSError, match="denied"):
        await terminal.resize(1, 1)


async def test_ghostty_send_retries_partial_and_blocking_writes(
    fake_ghostty, monkeypatch
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    terminal._master_fd = 42
    terminal._pid = 123
    writes = []

    def partial_write(fd, data):
        writes.append((fd, bytes(data)))
        if len(writes) == 1:
            raise BlockingIOError
        return min(2, len(data))

    monkeypatch.setattr(ghostty.os, "write", partial_write)
    writable_waits = []

    async def immediately_writable(fd):
        writable_waits.append(fd)

    monkeypatch.setattr(terminal, "_wait_writable", immediately_writable)
    await terminal.send_bytes(b"abcde")
    assert writes == [
        (42, b"abcde"),
        (42, b"abcde"),
        (42, b"cde"),
        (42, b"e"),
    ]
    assert writable_waits == [42]


@pytest.mark.skipif(os.name == "nt", reason="Ghostty backend uses Unix PTYs")
async def test_ghostty_pumps_pty_output_and_caches_on_close(
    fake_ghostty, tmp_path
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"], cwd=tmp_path)
    marker = await terminal.output_marker()
    await terminal.start("printf '\\033[31mghost-output\\033[0m\\n'; exit 4")
    assert await terminal.wait() == 4
    emitted = await terminal.output_since(marker)
    assert "ghost-output" in emitted
    assert "\x1b" not in emitted
    assert "ghost-output" in await terminal.read()
    assert await terminal.is_alive() == {"exit_code": 4}
    with pytest.raises(RuntimeError, match="not running"):
        await terminal.send_bytes(b"too late")
    await terminal.terminate()
    assert terminal._terminal.closed is True
    assert "ghost-output" in await terminal.read()
    assert "ghost-output" in await terminal.text()
    await terminal.terminate()  # idempotent


@pytest.mark.skipif(os.name == "nt", reason="Ghostty backend uses Unix PTYs")
async def test_ghostty_feed_failure_does_not_prevent_cleanup(
    fake_ghostty, tmp_path, monkeypatch
):
    terminal = ghostty.GhosttyTerm("feedfail", ["/bin/sh"], cwd=tmp_path)

    def broken_feed(data):
        del data
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(terminal._terminal, "feed", broken_feed)
    await terminal.start("printf output")
    assert isinstance(await terminal.wait(), int)
    assert isinstance(terminal._pump_error, RuntimeError)
    assert str(terminal._pump_error) == "renderer failed"

    await asyncio.gather(terminal.terminate(), terminal.terminate())
    assert terminal._terminal.closed is True
    assert terminal._terminal_closed is True
    assert terminal._master_fd is None
    await terminal.terminate()


@pytest.mark.skipif(os.name == "nt", reason="Ghostty backend uses Unix PTYs")
async def test_ghostty_write_backpressure_does_not_block_output_pump(
    fake_ghostty, tmp_path
):
    terminal = ghostty.GhosttyTerm("pressure", ["/bin/sh"], cwd=tmp_path)
    command = (
        "stty raw -echo; "
        "i=0; while [ $i -lt 2000 ]; do echo output-$i; i=$((i+1)); done; "
        "dd iflag=fullblock bs=20000 count=1 2>/dev/null | wc -c"
    )
    await terminal.start(command)
    for _ in range(100):
        if "output-0" in await terminal.contents():
            break
        await asyncio.sleep(0.01)
    await asyncio.wait_for(terminal.send_bytes(b"x" * 20_000), timeout=5)
    assert await asyncio.wait_for(terminal.wait(), timeout=5) == 0
    assert "output-1999" in await terminal.contents()
    assert "20000" in await terminal.contents()
    await terminal.terminate()


async def test_ghostty_closed_screen_operations_fail(fake_ghostty):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    terminal._closed = True
    terminal._terminal_closed = True
    with pytest.raises(RuntimeError, match="closed"):
        await terminal.resize(20, 80)
    with pytest.raises(RuntimeError, match="closed"):
        await terminal.cursor()
    with pytest.raises(RuntimeError, match="closed"):
        await terminal.size()


async def test_ghostty_mouse_batches_do_not_interleave(
    fake_ghostty, monkeypatch
):
    terminal = ghostty.GhosttyTerm("ghost123", ["/bin/sh"])
    calls = []

    async def record(action, row, col, *, button, modifiers):
        calls.append((action, row, col, button, modifiers))
        await asyncio.sleep(0)

    monkeypatch.setattr(terminal, "_send_mouse_event", record)
    first = (("press", "left"), ("release", "left"))
    second = (("press", "right"), ("release", "right"))

    await asyncio.gather(
        terminal.mouse_events(1, 2, first),
        terminal.mouse_events(3, 4, second, modifiers=frozenset({"ctrl"})),
    )

    assert calls in (
        [
            ("press", 1, 2, "left", frozenset()),
            ("release", 1, 2, "left", frozenset()),
            ("press", 3, 4, "right", frozenset({"ctrl"})),
            ("release", 3, 4, "right", frozenset({"ctrl"})),
        ],
        [
            ("press", 3, 4, "right", frozenset({"ctrl"})),
            ("release", 3, 4, "right", frozenset({"ctrl"})),
            ("press", 1, 2, "left", frozenset()),
            ("release", 1, 2, "left", frozenset()),
        ],
    )


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_mouse_events_use_negotiated_sgr_protocol():
    terminal = ghostty.GhosttyTerm("realmouse", ["/bin/sh"], rows=40, cols=120)
    sent = []

    async def capture(data):
        sent.append(data)

    terminal.send_bytes = capture
    try:
        await terminal.mouse_event("press", 2, 4, button="left")
        assert sent == []

        terminal._terminal.feed(b"\x1b[?1003h\x1b[?1006h")
        await terminal.mouse_event("press", 2, 4, button="left")
        await terminal.mouse_event("motion", 2, 4)
        await terminal.mouse_event("release", 2, 4, button="left")
        await terminal.mouse_event("motion", 2, 4)
        await terminal.mouse_event(
            "press",
            2,
            4,
            button="wheel_down",
            modifiers=frozenset({"ctrl"}),
        )
    finally:
        await terminal.terminate()

    assert sent == [
        b"\x1b[<0;5;3M",
        b"\x1b[<32;5;3M",
        b"\x1b[<0;5;3m",
        b"\x1b[<35;5;3M",
        b"\x1b[<81;5;3M",
    ]
    assert terminal._mouse_encoder is None


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_disabled_release_clears_held_drag_button():
    terminal = ghostty.GhosttyTerm("mousestate", ["/bin/sh"])
    sent = []

    async def capture(data):
        sent.append(data)

    terminal.send_bytes = capture
    try:
        terminal._terminal.feed(b"\x1b[?1003h\x1b[?1006h")
        await terminal.mouse_event("press", 2, 4, button="left")

        terminal._terminal.feed(b"\x1b[?1003l\x1b[?1006l")
        await terminal.mouse_event("release", 2, 4, button="left")

        terminal._terminal.feed(b"\x1b[?1003h\x1b[?1006h")
        await terminal.mouse_event("motion", 2, 4)
    finally:
        await terminal.terminate()

    assert sent == [b"\x1b[<0;5;3M", b"\x1b[<35;5;3M"]


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_scrollback_differs_from_visible_screen(tmp_path):
    terminal = ghostty.GhosttyTerm(
        "realtext", ["/bin/sh"], cwd=tmp_path, rows=4, cols=40
    )
    command = "i=0; while [ $i -lt 10 ]; do echo line-$i; i=$((i+1)); done"
    await terminal.start(command)
    assert await terminal.wait() == 0
    visible = await terminal.read()
    contents = await terminal.contents()
    assert "line-0" not in visible
    assert "line-0" in contents
    assert "line-9" in visible
    assert await terminal.read(lines=2) == "line-8\nline-9"
    await terminal.terminate()
    assert "line-0" in await terminal.contents()


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_ansi_cursor_and_resize(tmp_path):
    terminal = ghostty.GhosttyTerm(
        "realansi", ["/bin/sh"], cwd=tmp_path, rows=6, cols=20
    )
    await terminal.start("printf '\\033[3;5H\\033[31mX\\033[0m'")
    assert await terminal.wait() == 0
    assert "X" in await terminal.read()
    assert await terminal.cursor() == (2, 5)
    await terminal.resize(8, 30)
    assert await terminal.size() == (8, 30)
    await terminal.terminate()


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_screenshot_settles_initial_python_output(
    tmp_path,
):
    terminal = ghostty.GhosttyTerm(
        "realshot", ["/bin/sh"], cwd=tmp_path, rows=24, cols=80
    )

    class Manager:
        async def render_snapshot(self, term_id):
            assert term_id == "realshot"
            return await terminal.render_snapshot()

    manager = Manager()
    await terminal.start("python")
    try:
        rendered = await settled_screen_snapshot(manager, "realshot")
        text = "".join(span.text for span in rendered.spans)
        assert "Python" in text
        assert ">>>" in text
        png = terminal_snapshot_to_png(rendered)
        with Image.open(BytesIO(png)) as image:
            assert image.width > 0
            assert image.height > 0
            assert len(set(image.convert("RGB").getdata())) > 1
    finally:
        await terminal.terminate()


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_snapshot_uses_grid_graphemes_and_live_palette():
    terminal = ghostty.GhosttyTerm("realgrid", ["/bin/sh"], rows=1, cols=16)
    terminal._terminal.feed(
        "👩\u200d💻👍🏽\x1b]4;1;#010203\x07\x1b[38;5;1mX\x1b[4:3;58:2::4:5:6mY"
    )

    rendered = await terminal.render_snapshot()
    text = "".join(span.text for span in rendered.spans)
    assert text.startswith("👩\u200d💻👍🏽XY")
    assert sum(span.cells or 0 for span in rendered.spans) == 16
    x_span = next(span for span in rendered.spans if "X" in span.text)
    y_span = next(span for span in rendered.spans if "Y" in span.text)
    assert x_span.style.foreground == (1, 2, 3)
    assert y_span.style.underline_kind == 3
    assert y_span.style.underline_color == (4, 5, 6)
    await terminal.terminate()


@pytest.mark.skipif(
    ghostty.PyGhosttyTerminal is None or os.name == "nt",
    reason="requires native pyghostty on Unix",
)
async def test_real_ghostty_exec_failure_and_concurrent_close(tmp_path):
    terminal = ghostty.GhosttyTerm(
        "realerr1", ["/definitely/missing-shell"], cwd=tmp_path
    )
    await terminal.start("ignored")
    assert await terminal.wait() == 127
    await asyncio.gather(
        terminal.terminate(), terminal.terminate(), terminal.terminate()
    )
    assert terminal._terminal_closed
