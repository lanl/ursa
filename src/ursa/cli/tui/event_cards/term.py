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

TERM_TOOLS = frozenset({
    "term",
    "term_send_bytes",
    "term_send_text",
    "term_send_line",
    "term_send_key",
    "term_read",
    "term_is_alive",
    "term_wait_for",
    "term_wait_screen",
    "term_click",
    "term_mouse_down",
    "term_mouse_up",
    "term_hover",
    "term_scroll",
    "term_resize",
    "term_cursor",
    "term_size",
    "term_screenshot",
})
_TERM_RESULT = re.compile(r"Terminal ID:\s*([A-Za-z0-9]{8})")


def terminal_id(value: Any) -> str | None:
    """Extract a terminal ID from tool input or the documented result text."""
    if isinstance(value, ToolMessage):
        value = value.content
    if isinstance(value, Mapping):
        candidate = value.get("term_id")
        if isinstance(candidate, str) and re.fullmatch(
            r"[A-Za-z0-9]{8}", candidate
        ):
            return candidate
        if value.get("type") == "text":
            return terminal_id(value.get("text"))
        return None
    if isinstance(value, list | tuple):
        for nested in value:
            if found := terminal_id(nested):
                return found
        return None
    if not isinstance(value, str):
        return None
    # Avoid treating arbitrary output as an ID. Tool results intentionally
    # identify sessions with this exact stable format.
    match = _TERM_RESULT.fullmatch(value.strip())
    return match.group(1) if match else None


class TermCard(EventCard):
    """One card shared by every tool call against a terminal session."""

    def __init__(
        self,
        key: str,
        term_id: str | None = None,
        *,
        manager: TermManager | None = None,
    ) -> None:
        super().__init__(key, "Terminal")
        self.term_id = term_id
        self.manager = manager or term_manager
        self.call_count = 0
        self._call_tools: list[str] = []
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
        if self.term_id:
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
            self._call_tools.append(tool)
        failed = phase == "error" or payload.get("status") == "error"
        if failed:
            self.last_line = str(payload.get("error") or "Terminal call failed")
        elif tool == "term" and phase == "end" and self.term_id is None:
            result = payload.get("tool_message", payload.get("result"))
            if isinstance(result, ToolMessage):
                result = result.content
            lines = str(result or "").splitlines()
            if lines:
                self.last_line = lines[-1]
        self._render_summary()

    def associate(self, term_id: str) -> None:
        if self.term_id == term_id:
            return
        self.term_id = term_id
        self.last_line = "Terminal session started"
        self._render_summary()
        if self.is_mounted:
            self._mount_view()

    def absorb(self, other: TermCard) -> None:
        """Merge callback state from a provisional card into this session."""
        if other is self:
            return
        if set(self._call_tools) == {"term"} and set(other._call_tools) == {
            "term"
        }:
            # Duplicate launch callback lifecycles can have distinct run IDs,
            # but one manager-issued terminal ID proves they were one launch.
            self.call_count = max(self.call_count, other.call_count)
        else:
            self.call_count += other.call_count
            self._call_tools.extend(other._call_tools)
        if self.last_line in {
            "Starting terminal…",
            "Terminal session started",
        } and other.last_line not in {
            "Starting terminal…",
            "Terminal session started",
        }:
            self.last_line = other.last_line
        self._render_summary()

    def _mount_view(self) -> None:
        if self._view is not None or self.term_id is None:
            return
        self._view = TerminalView(self.term_id, manager=self.manager)
        self.query_one(".term-card-live", Vertical).mount(self._view)

    async def _refresh_tail(self) -> None:
        if self.term_id is None:
            return
        try:
            contents = await self.manager.contents(self.term_id)
        except KeyError:
            self.last_line = "Terminal session no longer exists."
            self._render_summary()
            return
        except RuntimeError:
            return
        lines = contents.splitlines()
        self.last_line = next(
            (line for line in reversed(lines) if line.strip()),
            "(terminal is empty)",
        )
        self._render_summary()

    def _render_summary(self) -> None:
        if not self.is_mounted:
            return
        identity = self.term_id or "starting"
        suffix = f" · {self.call_count} calls" if self.call_count != 1 else ""
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


__all__ = ["TERM_TOOLS", "TermCard", "terminal_id"]
