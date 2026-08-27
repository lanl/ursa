from datetime import UTC, datetime
from io import BytesIO

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Static

from tests.cli._app_fakes import FakeHITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.image_export import (
    MAX_TEXTUAL_PNG_SCALE,
    textual_app_to_png,
    textual_screenshot_to_png,
)
from ursa.cli.tui.widgets import TermsScreen
from ursa.tools.terminal import (
    TerminalRenderSnapshot,
    TerminalSpan,
    TerminalStyle,
    TermInfo,
)


def test_textual_screenshot_to_png_preserves_dimensions_and_color():
    svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8">
  <rect width="12" height="8" fill="#1a2b3c"/>
</svg>
"""

    png = textual_screenshot_to_png(svg)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(png)) as image:
        assert image.size == (12, 8)
        assert image.convert("RGB").getpixel((5, 5)) == (26, 43, 60)


def test_textual_screenshot_to_png_supports_high_dpi_scale():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="7" height="5"/>'

    with Image.open(BytesIO(textual_screenshot_to_png(svg, scale=2))) as image:
        assert image.size == (14, 10)


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_textual_screenshot_to_png_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        textual_screenshot_to_png("<svg/>", scale=scale)


def test_textual_screenshot_to_png_caps_scale():
    with pytest.raises(ValueError, match="scale must not exceed"):
        textual_screenshot_to_png("<svg/>", scale=MAX_TEXTUAL_PNG_SCALE + 1)


def test_textual_screenshot_to_png_caps_svg_bytes(monkeypatch):
    monkeypatch.setattr("ursa.cli.tui.image_export.MAX_TEXTUAL_SVG_BYTES", 10)

    with pytest.raises(ValueError, match="10-byte safety limit"):
        textual_screenshot_to_png("<svg><!-- large --></svg>")


def test_textual_screenshot_to_png_caps_pixels_before_rasterizing(monkeypatch):
    monkeypatch.setattr("ursa.cli.tui.image_export.MAX_TEXTUAL_PNG_PIXELS", 99)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'

    with pytest.raises(ValueError, match="99-pixel safety limit"):
        textual_screenshot_to_png(svg)


def test_textual_screenshot_to_png_rejects_empty_svg():
    with pytest.raises(ValueError, match="svg must not be empty"):
        textual_screenshot_to_png("")


def test_textual_screenshot_to_png_rejects_non_string_svg():
    with pytest.raises(TypeError, match="svg must be a string"):
        textual_screenshot_to_png(b"<svg/>")  # type: ignore[arg-type]


async def test_textual_app_to_png_rasterizes_live_compositor():
    class StyledApp(App[None]):
        CSS = "Static { color: #ff0000; background: #000000; }"

        def compose(self) -> ComposeResult:
            yield Static("styled terminal text")

    app = StyledApp()
    async with app.run_test(size=(30, 6)):
        png = textual_app_to_png(app, title="Terminal", simplify=False)

    with Image.open(BytesIO(png)) as image:
        assert image.width > 0
        assert image.height > 0


async def test_textual_app_to_png_preserves_live_terms_screen_style(tmp_path):
    red_style = TerminalStyle(foreground=(255, 0, 0), bold=True)
    green_style = TerminalStyle(foreground=(0, 255, 0), bold=True)
    render_snapshot = TerminalRenderSnapshot(
        term_id="styled00",
        spans=(
            TerminalSpan("████ red ", red_style),
            TerminalSpan("████ green", green_style),
        ),
        rows=6,
        cols=28,
        screen=True,
    )

    class SnapshotManager:
        async def render_snapshot(self, term_id):
            assert term_id == "styled00"
            return render_snapshot

    terminal = TermInfo(
        term_id="styled00",
        backend="GhosttyTerm",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        creation_order=1,
        capabilities=frozenset({"read"}),
        supports_screen=True,
    )
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            TermsScreen([terminal], manager=SnapshotManager())  # type: ignore[arg-type]
        )
        await pilot.pause()
        png = textual_app_to_png(app, simplify=False)

    with Image.open(BytesIO(png)) as image:
        assert image.size == (994, 636)
        colors = set(image.convert("RGB").getdata())
        assert any(
            red > 150 and red > green * 2 and red > blue * 2
            for red, green, blue in colors
        )
        assert any(
            green > 150 and green > red * 2 and green > blue * 2
            for red, green, blue in colors
        )
