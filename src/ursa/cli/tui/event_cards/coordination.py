# ruff: noqa: TID251

"""One compact, disk-backed coordination timeline per conversation turn."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from functools import partial
from typing import Any
from uuid import uuid4

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from ursa.cli.tui.event_cards.base import EventCard
from ursa.cli.tui.event_cards.hierarchy import AgentToolCard
from ursa.cli.tui.event_cards.tools import tool_event_input
from ursa.cli.tui.message_store import MessageStore


class CoordinationCard(EventCard):
    """Absorb every spawn/send/wait/stop call, retaining only two previews."""

    PAGE_SIZE = 20
    can_focus = True
    BINDINGS = [
        Binding("enter,space", "toggle", "Expand coordination", show=False),
    ]

    _ACTION_NAMES = {
        "spawn_agent": "Spawn agent",
        "send_message": "Send message",
        "wait_for": "Wait for",
        "stop_agent": "Stop agent",
    }

    @staticmethod
    def _recent_target(
        tool: str, target: str, target_count: int | None = None
    ) -> str:
        """Keep the two-line rollup scannable for broadcasts and waits."""
        if tool in {"send_message", "wait_for"}:
            # Zero is intentionally distinct from a missing legacy count: a
            # selector such as @everyone expands at delivery time, so its
            # concrete recipient count is not available in this compact row.
            count = (
                target_count
                if target_count is not None
                else len(target.split(", "))
            )
            if count > 1:
                return f"{count} agents"
        return target

    def __init__(self, key: str, store: MessageStore) -> None:
        super().__init__(key, "Coordination")
        self.store = store
        self.turn_id = uuid4().hex
        self.total = 0
        self._offset = 0
        self._details_lock = asyncio.Lock()
        self._reveal_after_expand = False

    def compose(self) -> ComposeResult:
        yield Static("Coordination", classes="coordination-summary")
        with Vertical(classes="coordination-recent"):
            yield Static("", classes="coordination-action-preview")
            yield Static("", classes="coordination-action-preview")
        with Vertical(classes="coordination-details hidden"):
            with Horizontal(classes="coordination-navigation"):
                yield Button("← Newer", classes="coordination-newer")
                yield Static("", classes="coordination-page")
                yield Button("Older →", classes="coordination-older")
            yield Vertical(classes="coordination-actions")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self._update_expand_hint()

    async def update_action(self, payload: Mapping[str, Any]) -> None:
        previous_total = self.total
        phase = str(payload.get("phase") or "")
        completed = phase in {"end", "error"}
        failed = phase == "error" or payload.get("status") == "error"
        output = payload.get("error")
        if output is None:
            output = payload.get("tool_message", payload.get("result"))
        failed = failed or getattr(output, "status", None) == "error"
        await self.store.record_action(
            self.turn_id,
            str(payload.get("_run_id") or uuid4().hex),
            str(payload["tool"]),
            tool_event_input(payload),
            output=AgentToolCard._text_content(output),
            completed=completed,
            failed=failed,
        )
        counts, recent = await self.store.action_summary(self.turn_id)
        self.total = counts["total"]
        # ``actions`` is newest-first. Hold a reader's historical page in
        # place when a new action arrives, rather than silently replacing its
        # oldest visible row with the new action.
        if self.expanded and self._offset:
            self._offset += max(0, self.total - previous_total)
        parts = [
            f"Coordination · {self.total} action{'s' if self.total != 1 else ''}"
        ]
        if counts["running"]:
            parts.append(f"{counts['running']} active")
        elif not counts["failed"]:
            parts.append("complete")
        if counts["failed"]:
            count = counts["failed"]
            parts.append(
                f"{count} action{'s' if count != 1 else ''} need"
                f"{'s' if count == 1 else ''} attention"
            )
        self.done = not counts["running"]
        self.set_class(bool(counts["failed"]), "coordination-needs-attention")
        self.query_one(".coordination-summary", Static).update(
            " · ".join(parts)
        )
        lines = []
        for row in recent:
            state = "✗" if row["failed"] else "✓" if row["completed"] else "…"
            action = self._ACTION_NAMES[row["tool"]]
            result = f" — {row['preview']}" if row["completed"] else ""
            target = self._recent_target(
                row["tool"], row["target"], row.get("target_count")
            )
            prefix = "Needs attention · " if row["failed"] else ""
            lines.append(f"{state} {prefix}{action} → {target}{result}")
        for index, widget in enumerate(
            self.query(".coordination-action-preview")
        ):
            widget.update(Text(lines[index] if index < len(lines) else ""))
        if self.expanded:
            await self._refresh_details()

    def set_expanded(self, expanded: bool) -> None:
        self._reveal_after_expand = (
            expanded and not self.expanded and self.is_mounted
        )
        self.expanded = expanded
        self._offset = 0
        if self.is_mounted:
            self.query_one(".coordination-details").set_class(
                not expanded, "hidden"
            )
            self.run_worker(self._refresh_details)
        self._update_expand_hint()

    async def _refresh_details(self) -> None:
        async with self._details_lock:
            if not self.is_mounted:
                return
            body = self.query_one(".coordination-actions", Vertical)
            if not self.expanded:
                await body.remove_children()
                return
            rows = await self.store.actions(
                self.turn_id, offset=self._offset, limit=self.PAGE_SIZE
            )
            if not self.expanded or not self.is_mounted:
                return
            existing = {card.key: card for card in body.query(AgentToolCard)}
            wanted = {str(row["id"]) for row in rows}
            if (
                existing
                and rows
                and min(map(int, existing)) > min(map(int, wanted))
            ):
                await body.remove_children()
                existing = {}
            else:
                for key, card in existing.items():
                    if key not in wanted:
                        await card.remove()
            cards = []
            for row in reversed(rows):
                card = existing.get(str(row["id"]))
                if card is None:
                    card = AgentToolCard(
                        str(row["id"]),
                        row["tool"],
                        json.loads(row["input_preview"]),
                        details_loader=partial(
                            self.store.action_body, self.turn_id, row["id"]
                        ),
                    )
                    cards.append(card)
                if row["completed"] and not card.completed:
                    card.complete(row["preview"], failed=bool(row["failed"]))
            if cards:
                await body.mount(*cards)
            self.query_one(".coordination-newer", Button).disabled = (
                self._offset == 0
            )
            self.query_one(".coordination-older", Button).disabled = (
                self._offset + len(rows) >= self.total
            )
            self.query_one(".coordination-page", Static).update(
                f"{self.total - self._offset - len(rows) + 1}–{self.total - self._offset} of {self.total}"
            )
            if self._reveal_after_expand:
                self.call_after_refresh(self._reveal_expanded_card)

    def _reveal_expanded_card(self) -> None:
        """Reveal user-expanded controls without following later updates."""
        if not self._reveal_after_expand or not self.expanded:
            return
        self._reveal_after_expand = False
        conversation = self.app.query_one("#conversation", VerticalScroll)
        conversation.release_anchor()
        self.scroll_visible(animate=False, top=True, immediate=True)

    @on(Button.Pressed, ".coordination-older")
    async def older(self, event: Button.Pressed) -> None:
        event.stop()
        self._offset += self.PAGE_SIZE
        await self._refresh_details()

    @on(Button.Pressed, ".coordination-newer")
    async def newer(self, event: Button.Pressed) -> None:
        event.stop()
        self._offset = max(0, self._offset - self.PAGE_SIZE)
        await self._refresh_details()
