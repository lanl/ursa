"""Convert callbacks to durable cards before notifying the viewport.

Only unfinished magnetic groups and commands live in memory. Correlation of
late results uses indexed aliases in SQLite, including results outside the
viewport. No event handler walks the conversation history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

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
    ToolCallCard,
)
from ursa.cli.tui.event_cards.card_codec import (
    decode_card,
    encode_card,
    finish_card,
)
from ursa.cli.tui.event_cards.card_store import CardStore
from ursa.cli.tui.helpers import (
    AGENT_LABELS,
    FILE_TOOLS,
    SEARCH_TOOLS,
    SUMMARY_GROUP_GRACE_SECONDS,
    file_outcome,
)
from ursa.util.rendering import event_artifacts

# Bound even a continually refreshed bundle. Long streams become several
# independently addressable bundles rather than one ever-growing JSON value.
BUNDLE_LIMIT = 32


@dataclass
class Bundle:
    card: EventCard
    deadline: float
    members: list[int] = field(default_factory=list)


class CardStream:
    def __init__(self, store: CardStore, workspace: Path, prompt: str) -> None:
        self.store = store
        self.workspace = workspace.resolve()
        self.prompt = prompt
        self.started_at = monotonic()
        self.token_usage = 0
        self.turn_id = 0
        self.groups: dict[str, Bundle] = {}
        self.commands: dict[str, RunCommandCard] = {}
        self.pending_tools: set[str] = set()
        self.revision = 0
        self.plan: PlanCard | None = None
        self.has_edit_heading = False
        self.sequence = 0
        self.activity = "Thinking…"
        self.finished = False
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.turn_id = await asyncio.to_thread(
            self.store.start_turn,
            {
                "version": 1,
                "type": "message",
                "role": "user",
                "content": self.prompt,
            },
        )

    def _key(self, kind: str) -> str:
        self.sequence += 1
        return f"{kind}:{self.sequence}"

    def _save(self, card: EventCard, **kwargs) -> int:
        return self.store.append(
            self.turn_id, card.key, encode_card(card), **kwargs
        )

    def _find(self, alias: str) -> EventCard | None:
        record = self.store.linked(self.turn_id, alias)
        return decode_card(record.card) if record else None

    def _link(self, alias: str, card: EventCard) -> None:
        self.store.link(self.turn_id, alias, card.key)

    async def _write(self, action: Callable, *args):
        """Serialize mutations and commit them outside the UI thread."""

        def commit():
            with self.store.transaction():
                return action(*args)

        async with self.lock:
            return await asyncio.to_thread(commit)

    async def event(self, payload: dict[str, Any]) -> None:
        await self._write(self._event, dict(payload))

    def _event(self, payload: dict[str, Any]) -> None:
        received_at = payload.get("_received_at")
        now = (
            float(received_at)
            if isinstance(received_at, (int, float))
            else monotonic()
        )
        self._expire(now)
        self._convert(payload, now)

    def _commit(self, kind: str) -> None:
        bundle = self.groups.pop(kind)
        bundle.card.mark_done()
        bundle.card.bundle = {"members": bundle.members}
        self._save(bundle.card, record_type="commit")

    def _expire(self, now: float) -> bool:
        expired = [
            kind
            for kind, bundle in self.groups.items()
            if now >= bundle.deadline
        ]
        for kind in expired:
            self._commit(kind)
        return bool(expired)

    async def expire(self) -> bool:
        return await self._write(lambda: self._expire(monotonic()))

    def _bundle(
        self, kind: str, individual: EventCard, now: float, update
    ) -> None:
        bundle = self.groups.get(kind)
        if bundle is not None and len(bundle.members) >= BUNDLE_LIMIT:
            self._commit(kind)
            bundle = None
        # Each incoming card is immediately durable, even while its bundle is
        # still open. The head is a bounded snapshot for cheap partial reads.
        member_id = self._save(individual, visible=False)
        if bundle is None:
            bundle = Bundle(individual, now)
            self.groups[kind] = bundle
        else:
            update(bundle.card)
        bundle.members.append(member_id)
        bundle.deadline = now + SUMMARY_GROUP_GRACE_SECONDS
        self._save(bundle.card, record_type="snapshot")

    def _convert(self, payload: dict[str, Any], now: float) -> None:
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
        phase = str(payload.get("phase") or "")
        if tool == "run_command":
            self._command(payload)
            return
        if artifacts := event_artifacts(payload):
            self._save(ArtifactCard(self._key("artifact"), artifacts))
        if agent in {"PlanningAgent", "planner"} and stage in {
            "generate",
            "generate_result",
            "reflect",
            "reflect_result",
        }:
            self._plan(payload)
            return
        path = str(payload.get("path") or payload.get("filename") or "").strip()
        if path:
            candidate = Path(path).expanduser()
            candidate = (self.workspace / candidate).resolve()
            try:
                path = str(candidate.relative_to(self.workspace))
            except ValueError:
                path = str(candidate)
        run_id = str(payload.get("_run_id") or "")
        outcome = file_outcome(payload)
        if tool in FILE_TOOLS and path:
            edit = None
            if tool in {"edit_code", "write_code", "write_code_with_repo"}:
                edit = self._find(f"edit-run:{run_id}") if run_id else None
                edit = edit or self._find(f"edit-path:{path}")
            if isinstance(edit, EditCard):
                if outcome:
                    edit.set_outcome(*outcome)
                    self._save(edit)
                    return
                if phase in {"end", "error"}:
                    return
            code = payload.get("code")
            old, new = payload.get("old_code"), payload.get("new_code")
            change = None
            if tool in {"write_code", "write_code_with_repo"} and isinstance(
                code, str
            ):
                change = ("", code)
            elif tool == "edit_code" and (old is not None or new is not None):
                change = (str(old or ""), str(new or ""))
            if change is not None:
                card = EditCard(
                    path, *change, show_heading=not self.has_edit_heading
                )
                card.key = self._key("edit")
                self.has_edit_heading = True
                if outcome:
                    card.set_outcome(*outcome)
                self._save(card)
                self._link(f"edit-path:{path}", card)
                if run_id:
                    self._link(f"edit-run:{run_id}", card)
                return
            if edit is not None:
                return
            operation = FILE_TOOLS[tool]
            kind = f"files:{operation}"
            if outcome:
                card = self._find(f"file-run:{run_id}") if run_id else None
                card = card or self._find(f"file:{operation}:{path}")
                if not isinstance(card, FileActivityCard):
                    card = FileActivityCard(self._key("files"))
                    card.add_file(operation, path)
                # Prefer the live object when the result updates an open group.
                bundle = self.groups.get(kind)
                if bundle and bundle.card.key == card.key:
                    card = bundle.card
                card.record_outcome(operation, path, *outcome)
                self._save(card)
                return

            def add_file(card):
                card.add_file(
                    operation,
                    path,
                    additions=payload.get("additions"),
                    deletions=payload.get("deletions"),
                )

            card = FileActivityCard(self._key("files"))
            add_file(card)
            self._bundle(kind, card, now, add_file)
            card = self.groups[kind].card
            self._link(f"file:{operation}:{path}", card)
            if run_id:
                self._link(f"file-run:{run_id}", card)
            if operation == "Editing":
                self.has_edit_heading = True
            return
        if tool and tool not in SEARCH_TOOLS and agent not in AGENT_LABELS:
            card = self._find(f"tool:{run_id}") if run_id else None
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
                card = ToolCallCard(
                    self._key("tool"),
                    tool,
                    {
                        key: value
                        for key, value in payload.items()
                        if key not in ignored
                    },
                )
            card.update_event(payload)
            self._save(card, record_type="commit" if card.completed else "card")
            if card.completed:
                self.pending_tools.discard(card.key)
            else:
                self.pending_tools.add(card.key)
            if run_id:
                self._link(f"tool:{run_id}", card)
            return
        source = agent or tool or "agent"
        key = self._key("progress")
        card = (
            AgentEventCard(key, agent)
            if agent in AGENT_LABELS
            else SearchEventCard(key, tool)
            if tool in SEARCH_TOOLS
            else EventCard(key, f"◌ {source}")
        )

        def update(card):
            if isinstance(card, (AgentEventCard, SearchEventCard)):
                card.update_event(payload)
            else:
                detail = payload.get("error") or payload.get("preview")
                card.add(
                    str(payload.get("message") or stage or tool or "Event"),
                    str(detail) if detail else None,
                )

        update(card)
        self._bundle(f"progress:{source}", card, now, update)

    def _command(self, payload: dict[str, Any]) -> None:
        call_id = payload.get("tool_call_id")
        if call_id in (None, ""):
            raise ValueError("run_command event missing tool_call_id")
        key = f"command:{call_id}"
        card = self.commands.get(key)
        if card is None:
            record = self.store.find(self.turn_id, key)
            card = (
                decode_card(record.card)
                if record
                else RunCommandCard(
                    key, str(payload.get("query") or "(command unavailable)")
                )
            )
        self.commands[key] = card
        if isinstance(payload.get("content_ref"), dict):
            card.content_ref = payload["content_ref"]
        if card.content_ref is not None:
            # A referenced buffer must not be copied into every record.
            if "result" in payload:
                payload = {**payload, "result": ""}
            if "artifacts" in payload:
                payload = {**payload, "artifacts": []}
        card.update_event(payload)
        overlapped = (
            sum(not item.completed for item in self.commands.values()) > 1
        )
        # Iteration is over concurrently active commands, never past commands.
        for command in self.commands.values():
            if overlapped or command.completed or command.force_compact:
                command.multi_command = True
            self._save_command(command)
        if card.completed:
            self.commands.pop(key)

    def _save_command(self, command: RunCommandCard) -> None:
        member = self._save(command)
        if command.completed:
            record = self.store.find(self.turn_id, command.key)
            command.bundle = {"position": record.position, "through": member}
            self._save(command, record_type="commit")

    def _plan(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage"))
        if stage == "generate" or self.plan is None:
            self.revision += 1
            self.plan = PlanCard(self._key("plan"), self.revision)
        plan = self.plan
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
                bool(payload.get("approved")), str(payload.get("reason") or "")
            )
        self._save(plan)

    def update_activity(self, message: str) -> None:
        self.activity = " ".join(message.split())[-500:]

    def add_tokens(self, count: int) -> None:
        self.token_usage += count

    async def finish(self, response: str, *, succeeded: bool = True) -> None:
        await self._write(self._finish, response, succeeded)
        self.finished = True
        self.commands.clear()
        self.pending_tools.clear()
        self.plan = None

    def _finish(self, response: str, succeeded: bool) -> None:
        for kind in list(self.groups):
            self._commit(kind)
        for command in self.commands.values():
            finish_card(command)
            self._save_command(command)
        for key in self.pending_tools:
            record = self.store.find(self.turn_id, key)
            tool = decode_card(record.card, interrupted=True)
            self._save(tool, record_type="commit")
        if self.plan:
            finish_card(self.plan, succeeded=succeeded)
            self._save(self.plan)
        self.store.append(
            self.turn_id,
            "activity",
            {
                "version": 1,
                "type": "activity",
                "tokens": self.token_usage,
                "elapsed": monotonic() - self.started_at,
            },
        )
        self.store.append(
            self.turn_id,
            "assistant",
            {
                "version": 1,
                "type": "message",
                "role": "assistant",
                "content": response,
            },
        )

    async def add_exception(self, error: BaseException, traceback: str) -> None:
        await self._write(
            lambda: self._save(
                ExceptionCard(self._key("exception"), error, traceback)
            )
        )
