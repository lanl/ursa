"""Raster image export for Textual screens."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pymupdf

if TYPE_CHECKING:
    from textual.app import App

MAX_TEXTUAL_SVG_BYTES = 16 * 1024 * 1024
MAX_TEXTUAL_PNG_PIXELS = 64_000_000
MAX_TEXTUAL_PNG_SCALE = 8.0


def textual_screenshot_to_png(svg: str, *, scale: float = 1.0) -> bytes:
    """Rasterize an SVG document as PNG bytes.

    PyMuPDF is already a core URSA dependency and delegates SVG layout to
    MuPDF.  A scale greater than one is useful for high-DPI image output.
    """
    if not isinstance(svg, str):
        raise TypeError("svg must be a string")
    if not svg:
        raise ValueError("svg must not be empty")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if scale > MAX_TEXTUAL_PNG_SCALE:
        raise ValueError(f"scale must not exceed {MAX_TEXTUAL_PNG_SCALE:g}")
    svg_bytes = svg.encode()
    if len(svg_bytes) > MAX_TEXTUAL_SVG_BYTES:
        raise ValueError(
            "SVG screenshot exceeds the "
            f"{MAX_TEXTUAL_SVG_BYTES}-byte safety limit"
        )

    with pymupdf.open(stream=svg_bytes, filetype="svg") as document:
        page = document[0]
        output_width = math.ceil(page.rect.width * scale)
        output_height = math.ceil(page.rect.height * scale)
        if output_width * output_height > MAX_TEXTUAL_PNG_PIXELS:
            raise ValueError(
                "PNG output exceeds the "
                f"{MAX_TEXTUAL_PNG_PIXELS}-pixel safety limit"
            )
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), alpha=False
        )
        return pixmap.tobytes("png")


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
]
