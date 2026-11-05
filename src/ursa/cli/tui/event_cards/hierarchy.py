"""Purpose-built cards for agents delegating work to one another."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Markdown, Static

from ursa.cli.tui.event_cards.base import EventCard
from ursa.cli.tui.widgets import ActivityIndicator


class AgentToolCard(EventCard):
    """Show the intent and outcome of a hierarchy-management tool call."""

    _PRESENTATION = {
        "spawn_agent": ("✦", "Delegating work", "subagent"),
        "send_message": ("↗", "Messaging agent", "recipient"),
        "list_agents": ("⌘", "Inspecting agents", "hierarchy"),
    }

    def __init__(self, key: str, tool: str, tool_input: Any) -> None:
        icon, title, subject = self._PRESENTATION[tool]
        super().__init__(key, f"{icon} {title}")
        self.tool = tool
        self.tool_input = tool_input if isinstance(tool_input, Mapping) else {}
        self.subject = subject
        self.output = ""
        self.completed = False
        self.failed = False
        self._frame = 0
        self._spinner_timer = None

    @property
    def target(self) -> str:
        if self.tool == "spawn_agent":
            name = str(self.tool_input.get("name") or "new subagent")
            agent_type = str(self.tool_input.get("agent_type") or "").strip()
            return f"{name} ({agent_type})" if agent_type else name
        if self.tool == "send_message":
            return str(self.tool_input.get("target") or "another agent")
        return "agent hierarchy"

    @property
    def task(self) -> str:
        return str(self.tool_input.get("message") or "").strip()

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="agent-tool-title")
        with Horizontal(classes="agent-tool-summary"):
            yield Static(
                ActivityIndicator.FRAMES[0], classes="agent-tool-state"
            )
            yield Static(self._summary(), classes="agent-tool-preview")
        with Vertical(classes="agent-tool-details hidden"):
            yield Static("Task", classes="agent-tool-detail-title task-title")
            yield Markdown(
                self.task or "_No task provided._", classes="agent-tool-task"
            )
            yield Static(
                "Result", classes="agent-tool-detail-title result-title hidden"
            )
            yield Markdown("", classes="agent-tool-result hidden")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.08, self._advance_spinner)
        self._render_card()
        self._update_expand_hint()

    def _advance_spinner(self) -> None:
        if not self.is_mounted or self.completed:
            return
        self.query_one(".agent-tool-state", Static).update(
            ActivityIndicator.FRAMES[self._frame]
        )
        self._frame = (self._frame + 1) % len(ActivityIndicator.FRAMES)

    def _summary(self) -> str:
        if not self.completed:
            if self.tool == "spawn_agent":
                return f"Starting {self.target}"
            if self.tool == "send_message":
                return f"Waiting for {self.target}"
            return "Reading the active hierarchy"
        if self.failed:
            return f"Could not reach {self.target}"
        if self.tool == "spawn_agent":
            return f"{self.target} completed its initial task"
        if self.tool == "send_message":
            return f"Reply received from {self.target}"
        count = len([line for line in self.output.splitlines() if line.strip()])
        return f"{count} agent{'s' if count != 1 else ''} in the hierarchy"

    @staticmethod
    def _text_content(value: Any) -> str:
        if isinstance(value, ToolMessage):
            value = value.content
        if isinstance(value, list):
            parts = [
                item
                if isinstance(item, str)
                else item.get("text")
                if isinstance(item, Mapping) and item.get("type") == "text"
                else None
                for item in value
            ]
            if all(isinstance(part, str) for part in parts):
                return "\n".join(parts)
        return str(value or "")

    def complete(self, output: Any, *, failed: bool = False) -> None:
        self.output = self._text_content(output)
        self.completed = True
        self.failed = failed
        self.mark_done()
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self._render_card()

    def update_event(self, payload: Mapping[str, Any]) -> None:
        phase = str(payload.get("phase") or "")
        if phase not in {"end", "error"}:
            return
        failed = phase == "error" or payload.get("status") == "error"
        self.complete(
            payload.get("error")
            if failed
            else payload.get("tool_message", payload.get("result")),
            failed=failed,
        )

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self._render_card()
        self._update_expand_hint()

    def _render_card(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(".agent-tool-state", Static).update(
            "✗"
            if self.failed
            else "✓"
            if self.completed
            else ActivityIndicator.FRAMES[self._frame]
        )
        self.query_one(".agent-tool-preview", Static).update(self._summary())
        details = self.query_one(".agent-tool-details", Vertical)
        details.set_class(not self.expanded, "hidden")
        result_title = self.query_one(".result-title", Static)
        result = self.query_one(".agent-tool-result", Markdown)
        result_title.set_class(not self.completed, "hidden")
        result.set_class(not self.completed, "hidden")
        if self.completed:
            result.update(self.output or "_No response returned._")
