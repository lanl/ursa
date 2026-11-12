"""A sliding window of widgets backed by the card projection.

Only this window participates in Textual layout. Paging retains overlapping
widgets and restores a visible card's screen coordinate after pruning, so
scrolling through long or variably sized cards doesn't require measuring the
rest of the transcript.
"""

from __future__ import annotations

import asyncio
from math import ceil
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from ursa.cli.tui.event_cards import EventCard
from ursa.cli.tui.event_cards.card_codec import apply_card, decode_card
from ursa.cli.tui.event_cards.card_store import CardRecord, CardStore
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import (
    ActivityIndicator,
    MessageCard,
    ToolMessage,
    WelcomeBanner,
)

if TYPE_CHECKING:
    from ursa.cli.tui.app import UrsaTextualApp


class StoredTurnView(Turn):
    """A visible fragment of a turn, with no callback/conversation history."""

    def compose(self) -> ComposeResult:
        yield Vertical(classes="events hidden")
        yield ActivityIndicator()
        yield Static("", classes="turn-end-marker")


class CardViewport:
    def __init__(self, app: UrsaTextualApp, store: CardStore) -> None:
        self.app = app
        self.store = store
        self.records: list[CardRecord] = []
        self.widgets: dict[int, Widget] = {}
        self.turns: dict[int, StoredTurnView] = {}
        self.lock = asyncio.Lock()
        self.follow_tail = True
        self.floor = 0
        self.average_height = 3.0
        self.pending = False
        self.loading = False
        self.has_older = False
        self.has_newer = False
        self.resume_position = 0
        self.closed = False
        self.expansion_generation = 0

    @property
    def conversation(self):
        return self.app.query_one("#conversation")

    @property
    def limit(self) -> int:
        height = self.conversation.size.height or self.app.size.height
        return max(10, 5 * ceil(height / max(1, self.average_height)))

    def request_refresh(self) -> None:
        if not self.pending and not self.closed and not self.app._exit:
            self.pending = True
            self.app.set_timer(1 / 30, self._start_refresh)

    def _start_refresh(self) -> None:
        if not self.closed and not self.app._exit:
            # An async timer callback is awaited by the app's message pump.
            # Mounting a window there delays input, even if each frame is fast.
            self.conversation.run_worker(self.refresh(), group="card-viewport")

    def schedule_page(self, direction: int) -> None:
        if not self.closed and not self.loading:
            self.conversation.run_worker(
                self.page(direction), group="card-viewport"
            )

    def _anchor(self) -> tuple[Widget, int] | None:
        top = self.conversation.content_region.y
        bottom = self.conversation.content_region.bottom
        visible = [
            widget
            for widget in self.widgets.values()
            if widget.region.bottom > top and widget.region.y < bottom
        ]
        if not visible:
            return None
        widget = min(visible, key=lambda item: item.region.y)
        return widget, widget.region.y

    async def refresh(
        self, *, before: int | None = None, after: int | None = None
    ) -> None:
        async with self.lock:
            if self.closed or self.app._exit:
                return
            self.pending = False
            self.loading = True
            try:
                await self._refresh(before=before, after=after)
            finally:
                self.loading = False

    async def _refresh(self, *, before: int | None, after: int | None) -> None:
        anchor = self._anchor() if not self.follow_tail else None
        if (
            before is None
            and after is None
            and not self.follow_tail
            and self.records
        ):
            after = self.records[0].position - 1
        records = await asyncio.to_thread(
            self.store.window,
            self.limit,
            before=before,
            after=after,
            floor=self.floor,
        )
        if not records:
            return
        preferences = await asyncio.to_thread(
            self.store.view_state, [record.position for record in records]
        )
        positions = {record.position for record in records}
        previous = {record.position: record for record in self.records}
        # Work and memory here are bounded by the viewport, including removal.
        for position in list(self.widgets):
            if position not in positions:
                widget = self.widgets.pop(position)
                if isinstance(widget, EventCard):
                    old = previous[position]
                    self.turns[old.turn_id].cards.pop(widget.key, None)
                if not isinstance(widget, ActivityIndicator):
                    await widget.remove()
        turn_ids = {record.turn_id for record in records}
        for turn_id in list(self.turns):
            if turn_id not in turn_ids:
                await self.turns.pop(turn_id).remove()
        conversation = self.conversation
        for record in records:
            turn = self.turns.get(record.turn_id)
            if turn is None:
                turn = StoredTurnView("", self.app.hitl.workspace)
                self.turns[record.turn_id] = turn
                following = next(
                    (
                        self.turns[item.turn_id]
                        for item in records
                        if item.turn_id > record.turn_id
                        and item.turn_id in self.turns
                    ),
                    None,
                )
                await conversation.mount(turn, before=following)
                turn.query_one(ActivityIndicator).add_class("hidden")
            old = previous.get(record.position)
            widget = self.widgets.get(record.position)
            if widget is not None and old is not None and old.id == record.id:
                continue
            kind = record.card.get("type")
            if widget is not None and isinstance(widget, EventCard):
                apply_card(widget, record.card)
                continue
            if kind == "message":
                widget = MessageCard(
                    record.card["role"], record.card["content"]
                )
                before_widget = (
                    turn.query_one(".events")
                    if record.card["role"] == "user"
                    else turn.query_one(".turn-end-marker")
                )
                await turn.mount(widget, before=before_widget)
                if record.card["role"] == "user":
                    turn.prompt = record.card["content"]
            elif kind == "notice":
                widget = ToolMessage(record.card["content"])
                await turn.mount(
                    widget, before=turn.query_one(".turn-end-marker")
                )
            elif kind == "activity":
                widget = turn.query_one(ActivityIndicator)
                turn.token_usage = record.card["tokens"]
                widget.finish(
                    elapsed=record.card["elapsed"], tokens=record.card["tokens"]
                )
            else:
                active = self.app.current_turn
                widget = decode_card(
                    record.card,
                    interrupted=active is None
                    or active.turn_id != record.turn_id,
                )
                events = turn.query_one(".events", Vertical)
                events.remove_class("hidden")
                events.add_class("has-events")
                next_widget = next(
                    (
                        self.widgets[item.position]
                        for item in records
                        if item.turn_id == record.turn_id
                        and item.position > record.position
                        and item.position in self.widgets
                        and isinstance(self.widgets[item.position], EventCard)
                    ),
                    None,
                )
                await events.mount(widget, before=next_widget)
                turn.cards[widget.key] = widget
                widget.refresh_content()
                preference = preferences.get(record.position, {})
                expanded = (
                    preference["expanded"]
                    if preference.get("generation") == self.expansion_generation
                    else self.app.card_details_expanded
                )
                widget.set_expanded(expanded)
                widget.expansion_changed = (
                    lambda expanded,
                    position=record.position: self.app.call_later(
                        self.save_expanded, position, expanded
                    )
                )
            self.widgets[record.position] = widget
        self.records = records
        active = self.app.current_turn
        for turn_id, view in self.turns.items():
            if (
                active is not None
                and active.turn_id == turn_id
                and not active.finished
            ):
                activity = view.query_one(ActivityIndicator)
                activity.remove_class("hidden")
                activity.set_class(not view.cards, "activity-after-user")
                activity.update_message(active.activity)
                view.token_usage = active.token_usage
        # Two indexed single-row seeks, not COUNT(*) or an all-history scan.
        older, newer = await asyncio.gather(
            asyncio.to_thread(
                self.store.window,
                1,
                before=records[0].position,
                floor=self.floor,
            ),
            asyncio.to_thread(
                self.store.window,
                1,
                after=records[-1].position,
                floor=self.floor,
            ),
        )
        self.has_older, self.has_newer = bool(older), bool(newer)
        for banner in self.app.query(WelcomeBanner):
            banner.display = not self.has_older
        if anchor and anchor[0] in self.widgets.values():
            widget, old_y = anchor

            def restore():
                if widget.is_mounted:
                    conversation.scroll_to(
                        y=conversation.scroll_y + widget.region.y - old_y,
                        animate=False,
                        immediate=True,
                        force=True,
                        release_anchor=False,
                    )

            self.app.call_after_refresh(restore)
        elif self.follow_tail:
            self.app.call_after_refresh(
                self.app._anchor_conversation_if_overflowing
            )
        self.app.call_after_refresh(self._measure)

    def _measure(self) -> None:
        if self.closed:
            return
        heights = [
            widget.outer_size.height
            for widget in self.widgets.values()
            if widget.display and widget.outer_size.height > 0
        ]
        if heights:
            # Tall cards may fill a screen on their own; retain enough neighbors
            # to page in either direction even when one is expanded.
            self.average_height = max(1, min(8, sum(heights) / len(heights)))
        self.remember_position()

    async def save_expanded(self, position: int, expanded: bool) -> None:
        if not self.closed:
            await asyncio.to_thread(
                self.store.save_view_state,
                position,
                expanded=expanded,
                generation=self.expansion_generation,
            )

    def remember_position(self) -> None:
        anchor = self._anchor()
        if anchor:
            self.resume_position = next(
                position
                for position, widget in self.widgets.items()
                if widget is anchor[0]
            )

    async def page(self, direction: int) -> None:
        if self.loading or not self.records:
            return
        if direction < 0 and self.has_older:
            self.follow_tail = False
            cursor = self.records[max(1, len(self.records) // 2)].position
            await self.refresh(before=cursor)
        elif direction > 0 and self.has_newer:
            self.follow_tail = False
            cursor = self.records[len(self.records) // 2].position
            await self.refresh(after=cursor)

    def scrolled(self, old: float, new: float) -> None:
        if not self.closed:
            self.remember_position()
        if self.loading or self.follow_tail or not self.records:
            return
        conversation = self.conversation
        margin = conversation.size.height
        if new < old and new < margin and self.has_older:
            self.schedule_page(-1)
        elif new > old and new > conversation.max_scroll_y - margin:
            if self.has_newer:
                self.schedule_page(1)
            elif new >= conversation.max_scroll_y:
                self.follow_tail = True
                conversation.anchor()

    async def end(self) -> None:
        self.follow_tail = True
        await self.refresh()
        self.conversation.scroll_end(animate=False)
        self.conversation.anchor()
        self.follow_tail = True

    async def home(self) -> None:
        self.follow_tail = False
        await self.refresh(after=self.floor)
        self.app.call_after_refresh(
            self.conversation.scroll_home, animate=False, immediate=True
        )

    async def clear(self) -> None:
        async with self.lock:
            latest = await asyncio.to_thread(self.store.window, 1)
            self.floor = latest[-1].position if latest else 0
            self.records.clear()
            self.widgets.clear()
            self.turns.clear()
            self.follow_tail = True
            await asyncio.to_thread(self.store.save_state, floor=self.floor)
