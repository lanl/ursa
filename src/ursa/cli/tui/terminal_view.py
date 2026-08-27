# ruff: noqa: TID251

"""View-only Textual rendering for managed terminal sessions."""

from __future__ import annotations

from rich.cells import cell_len
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual import events
from textual.timer import Timer
from textual.widgets import Static

from ursa.tools.terminal.base import TerminalRenderSnapshot, TerminalStyle
from ursa.tools.terminal.manager import TermManager, term_manager


def _rich_style(style: TerminalStyle) -> Style:
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
        output.append(text, _rich_style(span.style))
    return output


class TerminalView(Static):
    """Live, non-interactive display of one managed terminal.

    Ghostty screens retain their emulated row and column dimensions and do
    not reflow. Process-backed streams instead occupy the available pane and
    soft-wrap long lines.
    """

    DEFAULT_CSS = """
    TerminalView {
        width: 100%;
        height: 100%;
        border: round $accent;
        overflow: hidden;
        content-align: left bottom;
    }
    """

    can_focus = False

    def __init__(
        self,
        term_id: str,
        *,
        manager: TermManager | None = None,
        refresh_interval: float = 0.25,
        id: str | None = None,
    ) -> None:
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        super().__init__("Loading terminal…", expand=True, markup=False, id=id)
        self.term_id = term_id
        self.manager = manager or term_manager
        self.refresh_interval = refresh_interval
        self._refresh_timer: Timer | None = None
        self._snapshot_pending = False
        self._screen_snapshot = False
        self._latest_snapshot: TerminalRenderSnapshot | None = None

    def on_mount(self) -> None:
        """Start periodically refreshing from the manager's owner loop."""
        self.request_snapshot()
        self._refresh_timer = self.set_interval(
            self.refresh_interval, self.request_snapshot
        )

    def on_unmount(self) -> None:
        """Stop the refresh timer once its tab or modal is removed."""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def on_show(self) -> None:
        """Refresh immediately when a tab becomes visible."""
        self.request_snapshot()

    def on_resize(self, _event: events.Resize) -> None:
        """Repaint streams at the new viewport width."""
        self.refresh(layout=True)
        if not self._screen_snapshot:
            self.call_after_refresh(self._render_stream_tail)

    def request_snapshot(self) -> None:
        """Schedule a fresh immutable snapshot without blocking Textual."""
        if (
            not self.is_mounted
            or any(
                not widget.display or not widget.visible
                for widget in self.ancestors_with_self
            )
            or self._snapshot_pending
        ):
            return
        self._snapshot_pending = True
        try:
            self.run_worker(self._update_snapshot(), exit_on_error=False)
        except BaseException:
            self._snapshot_pending = False
            raise

    async def _update_snapshot(self) -> None:
        try:
            try:
                snapshot = await self.manager.render_snapshot(self.term_id)
            except KeyError:
                self.update("Terminal session no longer exists.")
                return
            except Exception as exc:  # Keep backend failures visible.
                self.update(f"Unable to render terminal: {exc}")
                return
            self._apply_snapshot(snapshot)
        finally:
            self._snapshot_pending = False

    def _apply_snapshot(self, snapshot: TerminalRenderSnapshot) -> None:
        self._screen_snapshot = snapshot.screen
        self._latest_snapshot = snapshot
        if snapshot.screen:
            self.update(snapshot_text(snapshot))
            assert snapshot.cols is not None and snapshot.rows is not None
            # Textual dimensions include the border. Keep the emulated grid's
            # rows and columns exact within that surrounding chrome.
            self.styles.width = snapshot.cols + 2
            self.styles.height = snapshot.rows + 2
        else:
            self.styles.width = "100%"
            self.styles.height = "100%"
            self.call_after_refresh(self._render_stream_tail)

    def _render_stream_tail(self) -> None:
        """Wrap process output and retain the newest viewport-sized tail."""
        snapshot = self._latest_snapshot
        if snapshot is None or snapshot.screen:
            return
        rendered = snapshot_text(snapshot)
        if not rendered.plain:
            # Pipe-backed interactive programs commonly suppress their prompt
            # when stdout isn't a TTY.  Keep the modal informative instead of
            # presenting an apparently broken, completely blank pane.
            self.update("No output has been captured from this terminal yet.")
            return
        width = max(1, self.content_size.width)
        height = max(1, self.content_size.height)
        lines = rendered.wrap(
            self.app.console,
            width,
            overflow="fold",
        )
        tail = Text("\n").join(lines[-height:])
        tail.no_wrap = False
        tail.overflow = "fold"
        self.update(tail)


__all__ = ["TerminalView", "snapshot_text"]
