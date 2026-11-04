# ruff: noqa: TID251

"""View-only Textual rendering for managed terminal sessions."""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.timer import Timer
from textual.widgets import Static

from ursa.tools.terminal.base import TerminalRenderSnapshot
from ursa.tools.terminal.manager import TermManager, term_manager
from ursa.tools.terminal.screenshot import snapshot_text


class TerminalView(Static):
    """Live, non-interactive display of one managed terminal."""

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
        self.term_id, self.manager = term_id, manager or term_manager
        self.refresh_interval = refresh_interval
        self._refresh_timer: Timer | None = None
        self._snapshot_pending = self._screen_snapshot = False
        self._latest_snapshot: TerminalRenderSnapshot | None = None

    def on_mount(self) -> None:
        self.request_snapshot()
        self._refresh_timer = self.set_interval(
            self.refresh_interval, self.request_snapshot
        )

    def on_unmount(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def on_show(self) -> None:
        self.request_snapshot()

    def on_resize(self, _event: events.Resize) -> None:
        self.refresh(layout=True)
        if not self._screen_snapshot:
            self.call_after_refresh(self._render_stream_tail)

    def request_snapshot(self) -> None:
        """Schedule a snapshot only while this view is visible."""
        if (
            self._snapshot_pending
            or not self.is_mounted
            or any(
                not widget.display or not widget.visible
                for widget in self.ancestors_with_self
            )
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
            self._apply_snapshot(
                await self.manager.render_snapshot(self.term_id)
            )
        except KeyError:
            self.update("Terminal session no longer exists.")
        except Exception as error:
            self.update(f"Unable to render terminal: {error}")
        finally:
            self._snapshot_pending = False

    def _apply_snapshot(self, snapshot: TerminalRenderSnapshot) -> None:
        self._latest_snapshot = snapshot
        self._screen_snapshot = snapshot.screen
        if not snapshot.screen:
            self.styles.width = self.styles.height = "100%"
            self.call_after_refresh(self._render_stream_tail)
            return
        self.update(snapshot_text(snapshot))
        assert snapshot.cols is not None and snapshot.rows is not None
        # Account for this widget's border while retaining the emulated grid.
        self.styles.width, self.styles.height = (
            snapshot.cols + 2,
            snapshot.rows + 2,
        )

    def _render_stream_tail(self) -> None:
        snapshot = self._latest_snapshot
        if snapshot is None or snapshot.screen:
            return
        rendered = snapshot_text(snapshot)
        if not rendered.plain:
            self.update("No output has been captured from this terminal yet.")
            return
        lines = rendered.wrap(
            self.app.console, max(1, self.content_size.width), overflow="fold"
        )
        tail = Text("\n").join(lines[-max(1, self.content_size.height) :])
        tail.no_wrap, tail.overflow = False, "fold"
        self.update(tail)


__all__ = ["TerminalView", "snapshot_text"]
