# ruff: noqa: TID251

"""Styled terminal snapshot rendering and screenshot stabilization."""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from collections.abc import Iterator
from io import StringIO
from typing import Protocol

import pymupdf
from rich.cells import cell_len
from rich.color import Color
from rich.console import Console
from rich.style import Style
from rich.text import Text

from .base import TerminalRenderSnapshot, TerminalStyle

MAX_SVG_BYTES = 16 * 1024 * 1024
MAX_PNG_PIXELS = 64_000_000
MAX_PNG_SCALE = 8.0
SCREENSHOT_SETTLE_TIMEOUT = 0.75
SCREENSHOT_SETTLE_INTERVAL = 0.05


class SnapshotProvider(Protocol):
    async def render_snapshot(self, term_id: str) -> TerminalRenderSnapshot: ...


def rich_style(style: TerminalStyle) -> Style:
    """Translate a renderer-neutral terminal style to Rich."""
    foreground = Color.from_rgb(*style.foreground) if style.foreground else None
    background = Color.from_rgb(*style.background) if style.background else None
    return Style(
        color=foreground,
        bgcolor=background,
        bold=style.bold,
        dim=style.faint,
        italic=style.italic,
        underline=style.underline,
        underline2=style.underline_kind == 2,
        blink=style.blink,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        overline=style.overline,
    )


def snapshot_text(snapshot: TerminalRenderSnapshot) -> Text:
    """Create the styled Rich representation of a terminal snapshot."""
    output = Text(
        no_wrap=snapshot.screen, overflow="crop" if snapshot.screen else "fold"
    )
    for span in snapshot.spans:
        text = span.text
        if span.cells is not None:
            text += " " * max(0, span.cells - cell_len(text))
        output.append(text, rich_style(span.style))
    return output


def screen_comparison_key(
    snapshot: TerminalRenderSnapshot,
    bounding_box: tuple[int, int, int, int] | None,
    include_styling: bool,
    *,
    allow_clipped_box: bool = False,
) -> tuple[object, ...]:
    """Return comparable display cells for a styled screen region."""
    if not snapshot.screen or snapshot.rows is None or snapshot.cols is None:
        raise NotImplementedError(
            "screen waiting requires a screen-backed terminal"
        )
    top, left, bottom, right = bounding_box or (
        0,
        0,
        snapshot.rows,
        snapshot.cols,
    )
    box_is_clipped = bottom > snapshot.rows or right > snapshot.cols
    if box_is_clipped and not allow_clipped_box:
        raise ValueError("bounding box exceeds the terminal screen dimensions")
    cells: dict[tuple[int, int], tuple[str, object]] = {}
    row = 0
    col = 0
    last_cells: tuple[tuple[int, int], ...] = ()
    for span in snapshot.spans:
        consumed = 0
        for grapheme in _graphemes(span.text):
            if grapheme == "\n":
                row += 1
                col = 0
                last_cells = ()
                continue
            width = cell_len(grapheme)
            if width == 0:
                for position in last_cells:
                    previous, style = cells[position]
                    cells[position] = (previous + grapheme, style)
                continue
            last_cells = tuple((row, col + offset) for offset in range(width))
            for position in last_cells:
                cells[position] = (grapheme, span.style)
            col += width
            consumed += width
        if span.cells is not None:
            padding = max(0, span.cells - consumed)
            for cell_col in range(col, col + padding):
                cells[(row, cell_col)] = (" ", span.style)
            col += padding
    selected = tuple(
        (row_index, col_index, text, style)
        if include_styling
        else (row_index, col_index, text)
        for (row_index, col_index), (text, style) in sorted(cells.items())
        if top <= row_index < bottom and left <= col_index < right
    )
    dimensions: tuple[object, ...] = (
        (snapshot.rows, snapshot.cols)
        if bounding_box is None or box_is_clipped
        else ()
    )
    cursor: tuple[object, ...] = ()
    if include_styling and snapshot.cursor is not None:
        cursor_row, cursor_col = snapshot.cursor
        if top <= cursor_row < bottom and left <= cursor_col < right:
            cursor = (("cursor", cursor_row, cursor_col),)
    return (*dimensions, *selected, *cursor)


def _graphemes(text: str) -> Iterator[str]:
    """Yield practical terminal grapheme clusters without splitting emoji."""
    cluster = ""
    regional_count = 0
    for character in text:
        codepoint = ord(character)
        category = unicodedata.category(character)
        is_regional = 0x1F1E6 <= codepoint <= 0x1F1FF
        joins_previous = (
            category in {"Mc", "Me", "Mn"}
            or 0xFE00 <= codepoint <= 0xFE0F
            or 0x1F3FB <= codepoint <= 0x1F3FF
            or character == "\u200d"
            or cluster.endswith("\u200d")
            or (is_regional and regional_count % 2 == 1)
        )
        if cluster and not joins_previous:
            yield cluster
            cluster = ""
            regional_count = 0
        cluster += character
        regional_count = regional_count + 1 if is_regional else 0
    if cluster:
        yield cluster


def inline_svg_styles(svg: str) -> str:
    """Inline Rich's class styles for rasterizers that ignore SVG CSS."""
    styles = dict(re.findall(r"\.([\w-]+)\s*\{\s*([^{}]+?)\s*\}", svg))

    def inline(match: re.Match[str]) -> str:
        class_name = match.group(1)
        style = styles.get(class_name)
        if style is None:
            return match.group(0)
        return f'class="{class_name}" style="{style.strip()}"'

    return re.sub(r'class="([\w-]+)"', inline, svg)


def svg_to_png(svg: str, *, scale: float = 1.0) -> bytes:
    """Rasterize an SVG document to bounded PNG bytes."""
    if not isinstance(svg, str):
        raise TypeError("svg must be a string")
    if not svg:
        raise ValueError("svg must not be empty")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if scale > MAX_PNG_SCALE:
        raise ValueError(f"scale must not exceed {MAX_PNG_SCALE:g}")
    svg_bytes = svg.encode()
    if len(svg_bytes) > MAX_SVG_BYTES:
        raise ValueError(
            f"SVG screenshot exceeds the {MAX_SVG_BYTES}-byte safety limit"
        )
    with pymupdf.open(stream=svg_bytes, filetype="svg") as document:
        page = document[0]
        width = math.ceil(page.rect.width * scale)
        height = math.ceil(page.rect.height * scale)
        if width * height > MAX_PNG_PIXELS:
            raise ValueError(
                f"PNG output exceeds the {MAX_PNG_PIXELS}-pixel safety limit"
            )
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), alpha=False
        )
        return pixmap.tobytes("png")


def terminal_snapshot_to_png(snapshot: TerminalRenderSnapshot) -> bytes:
    """Render an exact screen-backed terminal snapshot as a styled PNG."""
    if not snapshot.screen or snapshot.rows is None or snapshot.cols is None:
        raise ValueError("snapshot must contain a screen with dimensions")
    output = StringIO()
    console = Console(
        file=output,
        record=True,
        width=snapshot.cols,
        height=snapshot.rows,
        color_system="truecolor",
        force_terminal=True,
    )
    console.print(snapshot_text(snapshot), end="")
    svg = console.export_svg(title=f"Terminal {snapshot.term_id}")
    return svg_to_png(inline_svg_styles(svg))


async def settled_screen_snapshot(
    provider: SnapshotProvider,
    term_id: str,
    *,
    timeout: float = SCREENSHOT_SETTLE_TIMEOUT,
    interval: float = SCREENSHOT_SETTLE_INTERVAL,
) -> TerminalRenderSnapshot:
    """Wait briefly for initial PTY output and a stable rendered screen."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    previous = await provider.render_snapshot(term_id)
    if not previous.screen:
        return previous
    saw_content = bool("".join(span.text for span in previous.spans).strip())
    while loop.time() < deadline:
        await asyncio.sleep(interval)
        current = await provider.render_snapshot(term_id)
        has_content = bool("".join(span.text for span in current.spans).strip())
        if has_content and saw_content and current == previous:
            return current
        previous = current
        saw_content = has_content
    return previous
