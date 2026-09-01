"""Conversation-turn state and event-to-card orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ursa.cli.tui.event_cards import (
    AgentEventCard,
    ArtifactCard,
    EditCard,
    EventCard,
    ExceptionCard,
    FileActivityCard,
    PlanCard,
    RunCommandCard,
    SearchEventCard,
    TermCard,
    ToolCallCard,
)
from ursa.cli.tui.event_cards.term import terminal_id
from ursa.cli.tui.helpers import (
    AGENT_LABELS,
    FILE_TOOLS,
    SEARCH_TOOLS,
    SUMMARY_GROUP_GRACE_SECONDS,
)
from ursa.cli.tui.widgets import ActivityIndicator, MessageCard
from ursa.util.rendering import event_artifacts


class Turn(Static):
    def __init__(self, prompt: str, workspace: Path) -> None:
        super().__init__(classes="turn")
        self.prompt = prompt
        self.workspace = Path(workspace).resolve()
        self.cards: dict[str, EventCard] = {}
        self.started_at = monotonic()
        self.token_usage = 0
        self._command_count = 0
        self._commands: list[RunCommandCard] = []
        self._commands_by_id: dict[str, RunCommandCard] = {}
        self._commands_overlapped = False
        self._tool_calls_by_id: dict[str, ToolCallCard] = {}
        self._terms_by_id: dict[str, TermCard] = {}
        self._edits_by_id: dict[str, EditCard] = {}
        self._summary_count = 0
        self._summary_cards: dict[str, EventCard] = {}
        self._summary_deadlines: dict[str, float] = {}
        self._summary_timers: dict[str, Any] = {}
        self._plan_cards: list[PlanCard] = []
        self._current_event_at = self.started_at
        # LangChain may invoke callbacks for parallel tools concurrently. Keep
        # the read/decide/mount/update sequence atomic so those callbacks all
        # observe the same current summary group.
        self._event_lock = asyncio.Lock()
        self.card_details_expanded = False

    def compose(self) -> ComposeResult:
        yield MessageCard("user", self.prompt)
        yield Vertical(classes="events hidden")
        activity = ActivityIndicator()
        activity.add_class("activity-after-user")
        yield activity
        yield Static("", classes="turn-end-marker")

    async def event(self, payload: dict[str, Any]) -> None:
        async with self._event_lock:
            await self._event(payload)

    async def _event(self, payload: dict[str, Any]) -> None:
        received_at = payload.get("_received_at")
        self._current_event_at = (
            float(received_at)
            if isinstance(received_at, int | float)
            else monotonic()
        )
        activity = (
            payload.get("message")
            or payload.get("reasoning_content")
            or payload.get("reasoning_summary")
            or payload.get("reasoning")
            or payload.get("thinking")
            or payload.get("stage")
        )
        if activity:
            self.update_activity(str(activity))
        tool = str(payload.get("tool") or "")
        agent = str(payload.get("agent") or "")
        stage = str(payload.get("stage") or "")
        path = str(payload.get("path") or payload.get("filename") or "").strip()
        if path:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace / candidate
            candidate = candidate.resolve()
            try:
                path = str(candidate.relative_to(self.workspace))
            except ValueError:
                path = str(candidate)
        old = payload.get("old_code")
        new = payload.get("new_code")
        if tool == "run_command":
            await self._run_command_event(payload)
            return
        if tool == "term" or tool.startswith("term_"):
            await self._term_event(payload)
            return
        artifacts = event_artifacts(payload)
        if artifacts:
            artifact_key = self._next_summary_key("artifact")
            await self._replace_or_mount(
                artifact_key, ArtifactCard(artifact_key, artifacts)
            )
        if agent in {"PlanningAgent", "planner"} and stage in {
            "generate",
            "generate_result",
            "reflect",
            "reflect_result",
        }:
            await self._plan_event(payload)
            return
        edit_card = None
        additions = payload.get("additions")
        deletions = payload.get("deletions")
        outcome = self._file_outcome(payload)
        edit_id = str(payload.get("_run_id") or "")
        if tool in {"edit_code", "write_code", "write_code_with_repo"} and (
            outcome is not None
        ):
            existing_edit = self._edits_by_id.get(edit_id)
            if existing_edit is None:
                matching_edits = [
                    candidate
                    for candidate in self.cards.values()
                    if isinstance(candidate, EditCard)
                    and candidate.path == path
                ]
                if len(matching_edits) == 1:
                    existing_edit = matching_edits[0]
            if existing_edit is not None:
                existing_edit.set_outcome(*outcome)
                return
        code = payload.get("code")
        change: tuple[str, str] | None = None
        if path:
            if tool in {"write_code", "write_code_with_repo"} and isinstance(
                code, str
            ):
                change = ("", code)
            elif tool == "edit_code" and (old is not None or new is not None):
                change = (str(old or ""), str(new or ""))
        if change is not None:
            has_edit_heading = any(
                isinstance(candidate, EditCard)
                or (
                    isinstance(candidate, FileActivityCard)
                    and bool(candidate.files["Editing"])
                )
                for candidate in self.cards.values()
            )
            edit_card = EditCard(
                path,
                *change,
                show_heading=not has_edit_heading,
            )
        if edit_card is not None:
            edit_key = self._next_summary_key("edit")
            await self._replace_or_mount(edit_key, edit_card)
            if edit_id:
                self._edits_by_id[edit_id] = edit_card
            return

        if tool in {"edit_code", "write_code", "write_code_with_repo"} and any(
            isinstance(candidate, EditCard) and candidate.path == path
            for candidate in self.cards.values()
        ):
            # A follow-up callback without diff content belongs to the rich
            # edit row already mounted for this file.
            return

        if tool in FILE_TOOLS and path:
            operation = FILE_TOOLS[tool]
            if outcome is not None:
                card = self._latest_file_card(operation, path)
                if card is None:
                    key = self._next_summary_key("files")
                    card = FileActivityCard(key)
                    card.add_file(operation, path)
                    await self._replace_or_mount(key, card)
                state, detail = outcome
                card.record_outcome(operation, path, state, detail)
                return
            summary_kind = f"files:{operation}"
            self._prepare_summary(summary_kind)
            card = (
                self._summary_cards.get(summary_kind)
                if self._can_reuse_summary(summary_kind)
                and isinstance(
                    self._summary_cards.get(summary_kind),
                    FileActivityCard,
                )
                else None
            )
            if card is None:
                key = self._next_summary_key("files")
                card = FileActivityCard(key)
                await self._replace_or_mount(key, card)
            self._mark_summary(summary_kind, card)
            card.add_file(
                operation,
                path,
                additions=additions if isinstance(additions, int) else None,
                deletions=deletions if isinstance(deletions, int) else None,
            )
            return

        if tool and tool not in SEARCH_TOOLS and agent not in AGENT_LABELS:
            await self._default_tool_event(payload)
            return

        label = str(payload.get("message") or stage or tool or "Event")
        source = str(agent or tool or "agent")
        summary_kind = f"progress:{source}"
        card_type: type[EventCard] = (
            AgentEventCard
            if agent in AGENT_LABELS
            else SearchEventCard
            if tool in SEARCH_TOOLS
            else EventCard
        )
        self._prepare_summary(summary_kind)
        card = (
            self._summary_cards.get(summary_kind)
            if self._can_reuse_summary(summary_kind)
            and type(self._summary_cards.get(summary_kind)) is card_type
            else None
        )
        if card is None:
            key = self._next_summary_key("progress")
            card = (
                AgentEventCard(key, agent)
                if agent in AGENT_LABELS
                else SearchEventCard(key, tool)
                if tool in SEARCH_TOOLS
                else EventCard(key, f"◌ {source}")
            )
            await self._replace_or_mount(key, card)
        self._mark_summary(summary_kind, card)
        if isinstance(card, AgentEventCard):
            card.update_event(payload)
        elif isinstance(card, SearchEventCard):
            card.update_event(payload)
        else:
            detail = payload.get("error") or payload.get("preview")
            card.add(label, str(detail) if detail else None)

    async def _plan_event(self, payload: Mapping[str, Any]) -> None:
        """Update the live card for one planning/review revision."""
        stage = str(payload.get("stage") or "")
        if stage == "generate" or not self._plan_cards:
            for previous in self._plan_cards:
                previous.set_expanded(False)
            plan_key = self._next_summary_key("plan")
            plan = PlanCard(plan_key, len(self._plan_cards) + 1)
            self._plan_cards.append(plan)
            await self._replace_or_mount(plan_key, plan)
        else:
            plan = self._plan_cards[-1]

        if stage == "generate":
            plan.set_drafting()
        elif stage == "generate_result" and isinstance(
            payload.get("steps"), list
        ):
            plan.set_plan(payload["steps"])
        elif stage == "reflect":
            plan.set_reviewing()
        elif stage == "reflect_result":
            plan.finish_review(
                bool(payload.get("approved")),
                str(payload.get("reason") or ""),
            )

    async def _default_tool_event(self, payload: Mapping[str, Any]) -> None:
        run_id = str(payload.get("_run_id") or "")
        card = self._tool_calls_by_id.get(run_id) if run_id else None
        phase = str(payload.get("phase") or "")
        if card is None:
            ignored = {
                "tool",
                "phase",
                "result",
                "error",
                "status",
                "tool_message",
                "_run_id",
                "_received_at",
            }
            tool_input = {
                key: value
                for key, value in payload.items()
                if key not in ignored
            }
            key = self._next_summary_key("tool")
            card = ToolCallCard(key, str(payload.get("tool")), tool_input)
            await self._replace_or_mount(key, card)
            if run_id:
                self._tool_calls_by_id[run_id] = card
        if phase in {"end", "error"}:
            card.update_event(payload)

    async def _term_event(self, payload: Mapping[str, Any]) -> None:
        """Route terminal events by concrete session id only."""
        tool = str(payload.get("tool") or "term")
        run_id = str(payload.get("_run_id") or "")
        result = payload.get("tool_message", payload.get("result"))
        result_id = terminal_id(result)
        term_id = terminal_id(payload) or result_id

        if term_id is None:
            await self._default_tool_event(payload)
            return

        card = self._terms_by_id.get(term_id)
        if card is None:
            pending = self._tool_calls_by_id.get(run_id) if run_id else None
            key = (
                pending.key
                if pending is not None and pending.tool == "term"
                else self._next_summary_key("term")
            )
            card = TermCard(key, term_id)
            await self._replace_or_mount(key, card)
            self._terms_by_id[term_id] = card
        card.record_call(tool, payload)

    def _latest_file_card(
        self, operation: str, path: str
    ) -> FileActivityCard | None:
        for card in reversed(list(self.cards.values())):
            if (
                isinstance(card, FileActivityCard)
                and path in card.files[operation]
            ):
                return card
        return None

    @staticmethod
    def _file_outcome(payload: Mapping[str, Any]) -> tuple[str, str] | None:
        phase = str(payload.get("phase") or "")
        result = str(payload.get("result") or payload.get("error") or "")
        failed = phase == "error" or payload.get("status") == "error"
        failed = failed or result.casefold().startswith("failed")
        if failed:
            return "failed", result
        if result.casefold().startswith("no changes made"):
            return "unchanged", result
        return None

    def _next_summary_key(self, prefix: str) -> str:
        self._summary_count += 1
        return f"{prefix}:{self._summary_count}"

    def _can_reuse_summary(self, kind: str) -> bool:
        card = self._summary_cards.get(kind)
        deadline = self._summary_deadlines.get(kind)
        return (
            card is not None
            and not card.done
            and deadline is not None
            and self._current_event_at < deadline
        )

    def _prepare_summary(self, kind: str) -> None:
        deadline = self._summary_deadlines.get(kind)
        if deadline is not None and self._current_event_at >= deadline:
            self._finalize_summary(kind)

    def _mark_summary(self, kind: str, card: EventCard) -> None:
        self._summary_cards[kind] = card
        deadline = self._current_event_at + SUMMARY_GROUP_GRACE_SECONDS
        self._summary_deadlines[kind] = deadline
        timer = self._summary_timers.pop(kind, None)
        if timer is not None:
            timer.stop()
        delay = max(0, deadline - monotonic())
        self._summary_timers[kind] = self.set_timer(
            delay,
            lambda: self._finalize_summary(kind, card),
        )

    def _finalize_summary(
        self,
        kind: str,
        expected_card: EventCard | None = None,
    ) -> None:
        card = self._summary_cards.get(kind)
        if expected_card is not None and card is not expected_card:
            return
        timer = self._summary_timers.pop(kind, None)
        if timer is not None:
            timer.stop()
        if card is not None:
            card.mark_done()
        self._summary_cards.pop(kind, None)
        self._summary_deadlines.pop(kind, None)

    def _finalize_summaries(self) -> None:
        for kind in list(self._summary_cards):
            self._finalize_summary(kind)

    async def _run_command_event(self, payload: dict[str, Any]) -> None:
        command = str(payload.get("query") or "").strip()
        command_id = str(payload.get("_command_id") or "")
        key = f"command:{command_id}" if command_id else ""
        card = self._commands_by_id.get(command_id) if command_id else None
        if card is None and str(payload.get("phase") or "") != "start":
            matching = [
                candidate
                for candidate in self._commands
                if candidate.command == command and not candidate.completed
            ]
            if len(matching) == 1:
                # Some callback providers assign custom tool events a child
                # run ID instead of the ID used by on_tool_start. A unique
                # active command match is safe to correlate and avoids a
                # duplicate card stuck in its initial state.
                card = matching[0]
                if command_id:
                    self._commands_by_id[command_id] = card
        if card is None:
            if self._commands_overlapped and not any(
                command.completed is False for command in self._commands
            ):
                # A new solitary command starts a fresh layout batch. Preserve
                # compact history from an earlier overlapping batch without
                # forcing all later commands to remain compact forever.
                for previous in self._commands:
                    previous.force_compact = True
                self._commands_overlapped = False
            self._command_count += 1
            key = key or f"command:{self._command_count}"
            card = RunCommandCard(key, command or "(command unavailable)")
            await self._replace_or_mount(key, card)
            self._commands.append(card)
            if command_id:
                self._commands_by_id[command_id] = card
        assert isinstance(card, RunCommandCard)
        card.update_event(payload)
        self._update_command_layout()

    def _update_command_layout(self) -> None:
        active = [
            command for command in self._commands if not command.completed
        ]
        if len(active) > 1:
            self._commands_overlapped = True
        detailed: set[RunCommandCard] = set()
        if self._commands_overlapped:
            detailed = set()
        elif len(active) == 1:
            detailed.add(active[0])

        for command_card in self._commands:
            command_card.set_multi_command(
                command_card not in detailed or command_card.force_compact
            )
            command_card.set_output_expanded(self.card_details_expanded)

    def update_activity(self, message: str) -> None:
        self.query_one(ActivityIndicator).update_message(message)

    def add_tokens(self, count: int) -> None:
        self.token_usage += count

    def finish_activity(self, *, succeeded: bool = True) -> None:
        for plan in self._plan_cards:
            plan.finish_pending_review(succeeded=succeeded)
        self._finalize_summaries()
        self.query_one(ActivityIndicator).finish(
            elapsed=monotonic() - self.started_at,
            tokens=self.token_usage,
        )

    async def _replace_or_mount(self, key: str, card: EventCard) -> None:
        previous = self.cards.get(key)
        if previous is not None:
            await previous.remove()
        self.cards[key] = card
        events = self.query_one(".events", Vertical)
        events.add_class("has-events")
        events.remove_class("hidden")
        self.query_one(ActivityIndicator).remove_class("activity-after-user")
        await events.mount(card)
        if self.card_details_expanded:
            card.set_expanded(True)

    async def add_response(self, response: str) -> None:
        message = MessageCard("assistant", response)
        await self.mount(message, before=self.query_one(".turn-end-marker"))

    async def add_exception(
        self, error: BaseException, traceback: str
    ) -> ExceptionCard:
        """Add an expandable failure card without discarding traceback data."""
        key = self._next_summary_key("exception")
        card = ExceptionCard(key, error, traceback)
        await self._replace_or_mount(key, card)
        return card

    def set_card_details_expanded(self, expanded: bool) -> None:
        self.card_details_expanded = expanded
        for card in self.cards.values():
            card.set_expanded(expanded)
