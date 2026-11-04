from io import BytesIO

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Static

from ursa.cli.tui.image_export import (
    MAX_TEXTUAL_PNG_SCALE,
    terminal_snapshot_to_png,
    textual_app_to_png,
    textual_screenshot_to_png,
)
from ursa.tools.terminal import (
    TerminalRenderSnapshot,
    TerminalSpan,
    TerminalStyle,
)


def test_textual_screenshot_to_png_validates_scale_and_preserves_color():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8"><rect width="12" height="8" fill="#1a2b3c"/></svg>'
    with Image.open(BytesIO(textual_screenshot_to_png(svg, scale=2))) as image:
        assert image.size == (24, 16)
        assert image.convert("RGB").getpixel((5, 5)) == (26, 43, 60)
    for scale in (0, -1, float("inf"), MAX_TEXTUAL_PNG_SCALE + 1):
        with pytest.raises(ValueError):
            textual_screenshot_to_png(svg, scale=scale)


def test_terminal_snapshot_to_png_keeps_styled_unicode_grid():
    accent = TerminalStyle(
        foreground=(0, 240, 255), background=(70, 10, 90), bold=True
    )
    default = TerminalStyle(foreground=(230, 230, 230), background=(0, 0, 0))
    first = "界 🐍 e\N{COMBINING ACUTE ACCENT} ┌─┐"
    spans = [
        TerminalSpan(first, accent, cells=16),
        TerminalSpan(" " * 64, default),
    ]
    for _ in range(23):
        spans.extend((TerminalSpan("\n"), TerminalSpan(" " * 80, default)))
    png = terminal_snapshot_to_png(
        TerminalRenderSnapshot(
            "Unicode1", tuple(spans), rows=24, cols=80, screen=True
        )
    )
    with Image.open(BytesIO(png)) as image:
        assert image.width > 500 and image.height > 300
        assert any(
            blue > 150 and green > 150 and red < 100
            for red, green, blue in image.convert("RGB").getdata()
        )


async def test_textual_app_to_png_rasterizes_live_compositor():
    class StyledApp(App[None]):
        CSS = "Static { color: #ff0000; background: #000000; }"

        def compose(self) -> ComposeResult:
            yield Static("styled terminal text")

    app = StyledApp()
    async with app.run_test(size=(30, 6)):
        png = textual_app_to_png(app, title="Terminal", simplify=False)
    with Image.open(BytesIO(png)) as image:
        assert image.width > 0 and image.height > 0
