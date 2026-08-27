# ruff: noqa: TID251

"""Raster image export for Textual screens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ursa.tools.terminal.screenshot import (
    MAX_PNG_PIXELS as MAX_TEXTUAL_PNG_PIXELS,
)
from ursa.tools.terminal.screenshot import (
    MAX_PNG_SCALE as MAX_TEXTUAL_PNG_SCALE,
)
from ursa.tools.terminal.screenshot import (
    MAX_SVG_BYTES as MAX_TEXTUAL_SVG_BYTES,
)
from ursa.tools.terminal.screenshot import (
    svg_to_png,
    terminal_snapshot_to_png,
)

if TYPE_CHECKING:
    from textual.app import App


def textual_screenshot_to_png(svg: str, *, scale: float = 1.0) -> bytes:
    """Rasterize an SVG document as PNG bytes.

    PyMuPDF is already a core URSA dependency and delegates SVG layout to
    MuPDF.  A scale greater than one is useful for high-DPI image output.
    """
    return svg_to_png(svg, scale=scale)


def textual_app_to_png(
    app: App[object],
    *,
    title: str | None = None,
    simplify: bool = True,
    scale: float = 1.0,
) -> bytes:
    """Export the currently composed Textual screen as PNG bytes.

    The app must be running, as required by Textual's screenshot API.  This
    captures the same compositor output shown to the user, including terminal
    colors and text attributes rendered by its widgets.
    """
    svg = app.export_screenshot(title=title, simplify=simplify)
    return textual_screenshot_to_png(svg, scale=scale)


__all__ = [
    "MAX_TEXTUAL_PNG_PIXELS",
    "MAX_TEXTUAL_PNG_SCALE",
    "MAX_TEXTUAL_SVG_BYTES",
    "textual_app_to_png",
    "textual_screenshot_to_png",
    "terminal_snapshot_to_png",
]
