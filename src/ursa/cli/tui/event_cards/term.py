# ruff: noqa: TID251

"""Compact, live presentation for managed terminal tool calls."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Static

from ursa.cli.tui.event_cards.base import EventCard
from ursa.cli.tui.terminal_view import TerminalView
from ursa.tools.terminal.manager import TermManager, term_manager

_TERM_ID = re.compile(r"[A-Za-z0-9]{8}")
_TERM_RESULT = re.compile(r"Terminal ID:\s*([A-Za-z0-9]{8})")


def terminal_id(value: Any) -> str | None:
    """Extract a session ID from tool arguments or the stable result format."""
    if isinstance(value, ToolMessage):
        value = value.content
    if isinstance(value, Mapping):
        candidate = value.get("term_id")
        if isinstance(candidate, str) and _TERM_ID.fullmatch(candidate):
            return candidate
        return (
            terminal_id(value.get("text"))
            if value.get("type") == "text"
            else None
        )
    if isinstance(value, list | tuple):
        return next(
            (found for item in value if (found := terminal_id(item))), None
        )
    if isinstance(value, str) and (
        match := _TERM_RESULT.fullmatch(value.strip())
    ):
        return match.group(1)
    return None


class TermCard(EventCard):
    """One live card shared by all calls against a terminal session."""

    def __init__(
        self,
        key: str,
        term_id: str | None = None,
        *,
        manager: TermManager | None = None,
    ) -> None:
        super().__init__(key, "Terminal")
        self.term_id, self.manager = term_id, manager or term_manager
        self.call_count = 0
        self.last_line = "Starting terminal…"
        self._tail_timer: Timer | None = None
        self._view: TerminalView | None = None

    def compose(self) -> ComposeResult:
        yield Static(classes="term-card-title")
        yield Static(self.last_line, classes="term-card-tail")
        yield Vertical(classes="term-card-live hidden")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self._render_summary()
        self._tail_timer = self.set_interval(0.25, self._refresh_tail)
        self._mount_view()
        self._update_expand_hint()

    def on_unmount(self) -> None:
        if self._tail_timer is not None:
            self._tail_timer.stop()
            self._tail_timer = None

    def record_call(self, tool: str, payload: Mapping[str, Any]) -> None:
        phase = str(payload.get("phase") or "")
        if phase in {"", "start"}:
            self.call_count += 1
        if phase == "error" or payload.get("status") == "error":
            self.last_line = str(payload.get("error") or "Terminal call failed")
        elif tool == "term" and phase == "end" and self.term_id is None:
            result = payload.get("tool_message", payload.get("result"))
            lines = str(
                result.content
                if isinstance(result, ToolMessage)
                else result or ""
            ).splitlines()
            if lines:
                self.last_line = lines[-1]
        self._render_summary()

    def _mount_view(self) -> None:
        if self._view is None and self.term_id is not None:
            self._view = TerminalView(self.term_id, manager=self.manager)
            self.query_one(".term-card-live", Vertical).mount(self._view)

    async def _refresh_tail(self) -> None:
        if self.term_id is None:
            return
        try:
            contents = await self.manager.contents(self.term_id)
        except KeyError:
            self.last_line = "Terminal session no longer exists."
        except RuntimeError:
            return
        else:
            self.last_line = next(
                (
                    line
                    for line in reversed(contents.splitlines())
                    if line.strip()
                ),
                "(terminal is empty)",
            )
        self._render_summary()

    def _render_summary(self) -> None:
        if self.is_mounted:
            identity = self.term_id or "starting"
            suffix = (
                f" · {self.call_count} calls" if self.call_count != 1 else ""
            )
            self.query_one(".term-card-title", Static).update(
                f"▣ Terminal {identity}{suffix}"
            )
            self.query_one(".term-card-tail", Static).update(self.last_line)

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        if self.is_mounted:
            self.query_one(".term-card-tail").set_class(expanded, "hidden")
            self.query_one(".term-card-live").set_class(not expanded, "hidden")
            if expanded and self._view is not None:
                self._view.request_snapshot()
        self._update_expand_hint()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.set_expanded(not self.expanded)


__all__ = ["TermCard", "terminal_id"]
