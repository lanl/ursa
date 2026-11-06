# ruff: noqa: TID251

"""Purpose-built cards for agents delegating work to one another."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Markdown, Static

from ursa.cli.tui.event_cards.base import EventCard
from ursa.cli.tui.widgets import ActivityIndicator


class AgentToolCard(EventCard):
    """Show the intent and outcome of a hierarchy-management tool call."""

    _PRESENTATION = {
        "spawn_agent": ("✦", "Delegating work", "subagent"),
        "send_message": ("↗", "Messaging agent", "recipient"),
        "stop_agent": ("■", "Stopping agent", "agent"),
        "wait_for": ("◌", "Waiting for agents", "agent"),
        "list_agents": ("⌘", "Inspecting agents", "hierarchy"),
    }
    can_focus = True
    BINDINGS = [
        Binding("enter,space", "toggle", "Expand action", show=False),
    ]

    def __init__(
        self,
        key: str,
        tool: str,
        tool_input: Any,
        *,
        details_loader: Callable[[], Awaitable[tuple[dict[str, Any], str]]]
        | None = None,
    ) -> None:
        icon, title, subject = self._PRESENTATION[tool]
        super().__init__(key, f"{icon} {title}")
        self.tool = tool
        self.tool_input = tool_input if isinstance(tool_input, Mapping) else {}
        self.subject = subject
        self.output = ""
        self._details_loader = details_loader
        self._details_loaded = False
        self._compact_input = self.tool_input
        self._compact_output = ""
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
            targets = self.targets
            if len(targets) == 1:
                return targets[0]
            return f"{len(targets)} agents"
        if self.tool == "wait_for":
            targets = self.targets
            if len(targets) == 1:
                return targets[0]
            return f"{len(targets)} agents"
        if self.tool == "stop_agent":
            return str(self.tool_input.get("target") or "another agent")
        return "agent hierarchy"

    @property
    def targets(self) -> list[str]:
        """Return one or more display-safe message targets."""
        value = self.tool_input.get("target")
        if isinstance(value, list):
            targets = [
                str(target).strip() for target in value if str(target).strip()
            ]
            return targets or ["other agents"]
        return [str(value or "another agent")]

    @property
    def task(self) -> str:
        return str(self.tool_input.get("message") or "").strip()

    @property
    def request(self) -> str:
        """Describe the actual tool request, including non-message actions."""
        target = self.target
        if self.tool == "spawn_agent":
            name = str(self.tool_input.get("name") or "new subagent")
            agent_type = str(self.tool_input.get("agent_type") or "").strip()
            details = f"**Subagent:** `{name}`"
            if agent_type:
                details += f"\n\n**Agent type:** `{agent_type}`"
            if self.task:
                details += f"\n\n**Initial task**\n\n{self.task}"
            return details
        if self.tool == "send_message":
            return f"**Recipient:** `{target}`\n\n{self.task or '_Empty message._'}"
        if self.tool == "wait_for":
            wait_all = bool(self.tool_input.get("wait_all"))
            timeout = self.tool_input.get("timeout_seconds")
            details = (
                f"**Watching:** `{target}`\n\n"
                f"**Return:** {'all agents finish' if wait_all else 'first agent finishes'}"
            )
            if timeout is not None:
                details += f"\n\n**Timeout:** {timeout} seconds"
            return details
        if self.tool == "stop_agent":
            return f"**Stopping:** `{target}` and its descendants"
        return "_No request details available._"

    @property
    def request_title(self) -> str:
        return "Message" if self.tool == "send_message" else "Request"

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="agent-tool-title")
        with Horizontal(classes="agent-tool-summary"):
            yield Static(
                ActivityIndicator.FRAMES[0], classes="agent-tool-state"
            )
            # Tool results can include arbitrary agent messages (including
            # bracketed JSON from a batched wait). Keep this compact preview
            # as literal text instead of Textual markup.
            yield Static(Text(self._summary()), classes="agent-tool-preview")
        with Vertical(classes="agent-tool-details hidden"):
            yield Static(
                self.request_title, classes="agent-tool-detail-title task-title"
            )
            yield Markdown(
                "" if self._details_loader else self.request,
                classes="agent-tool-task",
            )
            yield Static(
                "Result", classes="agent-tool-detail-title result-title hidden"
            )
            yield Markdown("", classes="agent-tool-result hidden")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        if not self.completed:
            self._spinner_timer = self.set_interval(0.08, self._advance_spinner)
        self._render_card()
        self._update_expand_hint()

    def _advance_spinner(self) -> None:
        if (
            not self.is_mounted
            or self.completed
            or self.screen is not self.app.screen
        ):
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
                return f"Queueing for {self.target}"
            if self.tool == "wait_for":
                return f"Watching {self.target} for completion"
            if self.tool == "stop_agent":
                return f"Stopping {self.target} and its subagents"
            return "Reading the active hierarchy"
        if self.failed:
            return f"Could not reach {self.target}"
        if self.tool == "spawn_agent":
            return f"{self.target} started with its initial task"
        if self.tool == "send_message":
            if len(self.targets) > 1:
                return f"Messages queued for {len(self.targets)} agents"
            return f"Message queued for {self.target}"
        if self.tool == "wait_for":
            return self.output or f"Stopped waiting for {self.target}"
        if self.tool == "stop_agent":
            return self.output or f"Stopped {self.target} and its subagents"
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
        self._compact_output = self.output
        self._details_loaded = False
        self.completed = True
        self.failed = failed
        self.mark_done()
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self._render_card()
        if self._details_loader and self.expanded and self.is_mounted:
            self._request_details()

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
        if self._details_loader and not expanded:
            self.tool_input = self._compact_input
            self.output = self._compact_output
            self._details_loaded = False
        self._render_card()
        self._update_expand_hint()
        if self._details_loader and expanded and self.is_mounted:
            self._request_details()

    def action_toggle(self) -> None:
        """Expand or collapse the action from the keyboard."""
        self.set_expanded(not self.expanded)

    def _update_expand_hint(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(".event-expand-hint", Static).update(
            "Click or Enter to collapse"
            if self.expanded
            else "Click or Enter to expand"
        )

    def _request_details(self) -> None:
        self.run_worker(
            self._load_details, exclusive=True, group="agent-tool-details"
        )

    async def _load_details(self) -> None:
        if self._details_loader is None:
            return
        tool_input, output = await self._details_loader()
        if self.expanded and self.is_mounted:
            self.tool_input = tool_input
            self.output = output
            self._details_loaded = True
            self._render_card()

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
        self.query_one(".agent-tool-preview", Static).update(
            Text(self._summary())
        )
        details = self.query_one(".agent-tool-details", Vertical)
        details.set_class(not self.expanded, "hidden")
        result_title = self.query_one(".result-title", Static)
        result = self.query_one(".agent-tool-result", Markdown)
        result_title.set_class(not self.completed, "hidden")
        result.set_class(not self.completed, "hidden")
        task = self.query_one(".agent-tool-task", Markdown)
        if self._details_loader and not self._details_loaded:
            task.update("_Loading…_" if self.expanded else "")
            result.update("_Loading…_" if self.expanded else "")
            return
        task.update(self.request)
        if self.completed:
            result.update(self.output or "_No response returned._")
