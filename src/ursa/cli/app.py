# ruff: noqa: TID251

"""Textual front end for URSA's human-in-the-loop runner."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from rich.cells import cell_len, chop_cells
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from ursa.agents.base import URSA_VERSION
from ursa.cli.callbacks import HITLLogEventHandler
from ursa.cli.runtime import HITL
from ursa.cli.tips import random_tip
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.util.rendering import event_artifacts, render_event_artifacts

SUMMARY_GROUP_GRACE_SECONDS = 3.0
FILE_TOOLS = {
    "read_file": "Reading",
    "write_code": "Editing",
    "write_code_with_repo": "Editing",
    "edit_code": "Editing",
}

SEARCH_TOOLS = {
    "run_arxiv_search",
    "run_osti_search",
    "run_web_search",
}

AGENT_LABELS = {
    "ExecutionAgent": ("⚙️", "Execute"),
    "DeepReviewAgent": ("🔎", "Deep Review"),
    "HypothesizerAgent": ("💡", "Hypothesize"),
    "LammpsAgent": ("⚛️", "LAMMPS"),
    "PlanningAgent": ("🗺️", "Plan"),
    "executor": ("⚙️", "Execute"),
    "deep_review": ("🔎", "Deep Review"),
    "hypothesizer": ("💡", "Hypothesize"),
    "lammps": ("⚛️", "LAMMPS"),
    "planner": ("🗺️", "Plan"),
}

COMMAND_CHOICES = {
    "agents": "Configured agents, descriptions, and options",
    "status": "Tokens, models, endpoints, group, and MCP servers",
    "keymap": "Complete keyboard map",
}


def _endpoint(value: Any) -> str:
    """Return a concise endpoint label for a model-like object."""
    if value is None:
        return "none"
    for attribute in ("base_url", "api_base", "openai_api_base"):
        if endpoint := getattr(value, attribute, None):
            return str(endpoint)
    return "default"


def _embedding_name(hitl: HITL) -> str:
    embedding = getattr(hitl, "embedding", None)
    if embedding is None:
        return "none"
    for attribute in ("model_name", "model"):
        if value := getattr(embedding, attribute, None):
            return str(value)
    return type(embedding).__name__


def _route_prompt(hitl: HITL, prompt: str) -> tuple[str, str]:
    """Route a leading ``#agent`` macro, defaulting to chat."""
    first, separator, rest = prompt.partition(" ")
    if first.startswith("#") and first[1:] in hitl.agents:
        return first[1:], rest if separator else ""
    return "chat", prompt


def _fuzzy_match(query: str, candidate: str) -> bool:
    """Return whether all query characters occur in order in candidate."""
    return _fuzzy_score(query, candidate) is not None


def _field_fuzzy_score(query: str, value: str) -> int | None:
    """Score a fuzzy subsequence, favoring compact and early matches."""
    query = query.casefold()
    value = value.casefold()
    if not query:
        return 0
    positions: list[int] = []
    start = 0
    for character in query:
        position = value.find(character, start)
        if position < 0:
            return None
        positions.append(position)
        start = position + 1
    span = positions[-1] - positions[0] + 1
    score = 1000 - positions[0] * 4 - (span - len(query)) * 8
    if query == value:
        score += 3000
    elif value.startswith(query):
        score += 2000
    elif query in value:
        score += 1000
    return score


def _fuzzy_score(query: str, candidate: str) -> int | None:
    """Rank matches, strongly preferring a picker's primary name field."""
    primary, separator, description = candidate.partition(" — ")
    primary_score = _field_fuzzy_score(query, primary)
    description_score = (
        _field_fuzzy_score(query, description) if separator else None
    )
    scores = []
    if primary_score is not None:
        scores.append(10_000 + primary_score)
    if description_score is not None:
        scores.append(description_score)
    return max(scores) if scores else None


def _token_usage(value: Any) -> int:
    """Extract total token usage from common LangChain response shapes."""
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if item is None or id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, dict):
            for key in ("total_tokens", "total_token_count"):
                count = item.get(key)
                if isinstance(count, int):
                    return count
            return max((visit(child) for child in item.values()), default=0)
        if isinstance(item, (list, tuple)):
            return max((visit(child) for child in item), default=0)
        for attribute in (
            "llm_output",
            "usage_metadata",
            "response_metadata",
            "generations",
            "message",
        ):
            if hasattr(item, attribute):
                count = visit(getattr(item, attribute))
                if count:
                    return count
        return 0

    return visit(value)


def _model_name(hitl: HITL) -> str:
    model = hitl.model
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    return type(model).__name__


def _reasoning_trace(chunk: Any) -> str | None:
    """Extract provider-published reasoning summaries from an LLM chunk."""

    def text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return " ".join(
                text(value.get(key))
                for key in (
                    "text",
                    "content",
                    "summary",
                    "reasoning",
                    "thinking",
                )
                if value.get(key)
            )
        if isinstance(value, (list, tuple)):
            return " ".join(filter(None, (text(item) for item in value)))
        return ""

    values = [chunk, getattr(chunk, "message", None)]
    for value in values:
        mappings = [value] if isinstance(value, Mapping) else []
        for attribute in ("additional_kwargs", "response_metadata"):
            mapping = getattr(value, attribute, None)
            if isinstance(mapping, Mapping):
                mappings.append(mapping)
        for mapping in mappings:
            for key in (
                "reasoning_content",
                "reasoning_summary",
                "reasoning",
                "thinking",
            ):
                if trace := " ".join(text(mapping.get(key)).split()):
                    return trace[-500:]

        content = getattr(value, "content", None)
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if str(block.get("type", "")).casefold() in {
                    "reasoning",
                    "reasoning_summary",
                    "thinking",
                }:
                    if trace := " ".join(text(block).split()):
                        return trace[-500:]
    return None


class PromptArea(TextArea):
    """A multiline editor whose bare Enter submits the current prompt."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self) -> None:
        super().__init__(
            language="markdown",
            soft_wrap=True,
            tab_behavior="indent",
            placeholder="Ask URSA…  (@ files, # agents, Shift+Enter newline)",
            id="prompt",
        )
        self.prompt_history: list[str] = []
        self._history_index: int | None = None

    def _remember(self, text: str) -> None:
        if text and (
            not self.prompt_history or self.prompt_history[-1] != text
        ):
            self.prompt_history.append(text)
        self._history_index = None

    def _load_history(self, index: int) -> None:
        self._history_index = index
        self.load_text(self.prompt_history[index])
        self.move_cursor((
            len(self.document.lines) - 1,
            len(self.document.lines[-1]),
        ))

    async def _on_key(self, event: events.Key) -> None:
        if event.key in {"alt+left", "meta+left", "alt+b"}:
            event.prevent_default()
            event.stop()
            self.action_cursor_word_left()
            return
        if event.key in {"alt+right", "meta+right", "alt+f"}:
            event.prevent_default()
            event.stop()
            self.action_cursor_word_right()
            return
        if event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            self._remember(self.text)
            self._history_index = len(self.prompt_history)
            self.load_text("")
            return
        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self._remember(text)
                self.post_message(self.Submitted(text))
            return
        if (
            event.key == "up"
            and self.prompt_history
            and (not self.text or self.cursor_location[0] == 0)
        ):
            event.prevent_default()
            event.stop()
            index = (
                len(self.prompt_history)
                if self._history_index is None
                else self._history_index
            )
            self._load_history(max(0, index - 1))
            return
        if event.key == "down" and self._history_index is not None:
            event.prevent_default()
            event.stop()
            next_index = self._history_index + 1
            if next_index < len(self.prompt_history):
                self._load_history(next_index)
            else:
                self._history_index = None
                self.load_text("")
            return
        await super()._on_key(event)


class HotlistScreen(ModalScreen[str | None]):
    """Fuzzy-searchable picker overlaid above the prompt."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title: str, candidates: Sequence[str]) -> None:
        super().__init__()
        self.picker_title = title
        self.candidates = list(candidates)
        self.matches = list(candidates)

    def compose(self) -> ComposeResult:
        with Vertical(id="hotlist"):
            yield Static(self.picker_title, classes="hotlist-title")
            yield Input(placeholder="fzf search…", id="hotlist-query")
            yield OptionList(
                *(
                    Option(candidate, id=str(index))
                    for index, candidate in enumerate(self.matches)
                ),
                id="hotlist-options",
            )

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._highlight_first()

    def on_key(self, event: events.Key) -> None:
        options = self.query_one(OptionList)
        if event.key == "down":
            event.prevent_default()
            event.stop()
            options.action_cursor_down()
        elif event.key == "up":
            event.prevent_default()
            event.stop()
            options.action_cursor_up()
        elif event.key == "enter" and options.option_count:
            event.prevent_default()
            event.stop()
            options.action_select()

    @on(Input.Changed)
    def filter_options(self, event: Input.Changed) -> None:
        ranked = []
        for index, candidate in enumerate(self.candidates):
            score = _fuzzy_score(event.value, candidate)
            if score is not None:
                ranked.append((-score, index, candidate))
        ranked.sort()
        self.matches = [candidate for _, _, candidate in ranked]
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options(self.matches)
        self._highlight_first()

    def _highlight_first(self) -> None:
        options = self.query_one(OptionList)
        options.highlighted = 0 if options.option_count else None

    @on(OptionList.OptionSelected)
    def select_option(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_cancel(self) -> None:
        self.dismiss(None)


class InformationScreen(ModalScreen[None]):
    """Scrollable command output displayed without leaving the application."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self.screen_title = title
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="information"):
            yield Static(self.screen_title, id="information-title")
            yield VerticalScroll(Markdown(self.content), id="information-body")

    def action_close(self) -> None:
        self.dismiss(None)


class WelcomeBanner(Vertical):
    """URSA logo, active configuration snapshot, and a concise usage tip."""

    LOGO = r"""  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/"""

    def __init__(self, hitl: HITL) -> None:
        super().__init__(id="welcome")
        self.hitl = hitl
        workspace = Path(self.hitl.workspace).resolve()
        try:
            self.workspace_text = f"~/{workspace.relative_to(Path.home())}"
        except ValueError:
            self.workspace_text = str(workspace)
        self.version_text = f"v{URSA_VERSION}"
        self.tip = random_tip()

    @staticmethod
    def _fit_middle(text: str, width: int) -> str:
        if width <= 0 or cell_len(text) <= width:
            return text
        if width == 1:
            return "…"
        available = width - 1
        left = available // 3
        right = available - left
        prefix = chop_cells(text, left)[0]
        suffix = chop_cells(text[::-1], right)[0][::-1]
        return f"{prefix}…{suffix}"

    def _fit_metadata(self) -> None:
        version = self.query_one("#welcome-version", Static)
        workspace_row = self.query_one("#welcome-workspace-row")
        workspace = self.query_one("#welcome-workspace", Static)
        version.update(
            self._fit_middle(self.version_text, version.content_region.width)
        )
        row_width = workspace_row.content_region.width
        inline = (
            cell_len("Workspace") + 2 + cell_len(self.workspace_text)
            <= row_width
        )
        workspace_row.set_class(inline, "workspace-inline")
        workspace_row.set_class(not inline, "workspace-stacked")
        workspace_width = row_width - 11 if inline else row_width
        workspace.update(self._fit_middle(self.workspace_text, workspace_width))

    def on_mount(self) -> None:
        self._fit_metadata()

    def on_resize(self) -> None:
        self._fit_metadata()

    def compose(self) -> ComposeResult:
        embedding = getattr(self.hitl, "embedding", None)
        embedding_text = "none"
        if embedding is not None:
            embedding_text = (
                f"{_embedding_name(self.hitl)} ({_endpoint(embedding)})"
            )
        snapshot = "\n".join([
            f"LLM        {_model_name(self.hitl)} ({_endpoint(self.hitl.model)})",
            f"Embedding  {embedding_text}",
            f"Group      {getattr(self.hitl, 'group', None) or 'default'}",
        ])
        with Horizontal(id="welcome-top"):
            with Vertical(id="welcome-logo"):
                with Vertical(id="welcome-logo-stack"):
                    yield Static(self.LOGO, id="welcome-logo-art")
                    yield Static(self.version_text, id="welcome-version")
            with Vertical(id="welcome-config"):
                with Vertical(id="welcome-workspace-row"):
                    yield Static("Workspace", id="welcome-workspace-label")
                    yield Static(
                        self.workspace_text,
                        id="welcome-workspace",
                    )
                yield Static(snapshot, id="welcome-config-values")
        yield Static(
            f"Tip: {self.tip}",
            id="welcome-tip",
        )


class MessageCard(Static):
    def __init__(self, role: str, content: str) -> None:
        super().__init__(classes=f"message-card {role}")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "assistant":
            yield Static("URSA", classes="message-role")
        yield Markdown(self.content, classes="message-body")

    @on(Markdown.TableOfContentsUpdated)
    def remove_trailing_markdown_margin(
        self, event: Markdown.TableOfContentsUpdated
    ) -> None:
        blocks = list(event.markdown.children)
        if blocks:
            blocks[-1].styles.margin = 0


class ActivityIndicator(Horizontal):
    """Animated, event-driven status for one conversation turn."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        super().__init__(classes="activity")
        self._frame = 0
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Static(self.FRAMES[0], classes="activity-spinner")
        yield Static("Thinking…", classes="activity-text")
        yield Static("", classes="activity-done-mark")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.08, self._advance)

    def _advance(self) -> None:
        self.query_one(".activity-spinner", Static).update(
            self.FRAMES[self._frame]
        )
        self._frame = (self._frame + 1) % len(self.FRAMES)

    def update_message(self, message: str) -> None:
        message = " ".join(str(message).split())
        if message:
            self.query_one(".activity-text", Static).update(message[-500:])

    def finish(self, *, elapsed: float, tokens: int) -> None:
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("")
        if elapsed <= 30:
            self.query_one(".activity-text", Static).update("")
            self.query_one(".activity-done-mark", Static).update("")
            self.remove_class("done")
            self.add_class("hidden")
            return
        seconds = ceil(elapsed)
        if seconds < 60:
            duration = f"{seconds}s"
        else:
            minutes, seconds = divmod(seconds, 60)
            duration = f"{minutes}m {seconds:02d}s"
        self.query_one(".activity-text", Static).update(
            f"Done in {duration} and {tokens:,} tokens"
        )
        self.query_one(".activity-done-mark", Static).update("✓")
        self.remove_class("hidden")
        self.add_class("done")


class EventCard(Static):
    """A live, compact summary of one event stream."""

    def __init__(self, key: str, label: str) -> None:
        super().__init__(classes="event-card")
        self.key = key
        self.label = label
        self.lines: list[str] = []
        self.details: list[str] = []
        self.expanded = False
        self.done = False

    def compose(self) -> ComposeResult:
        yield Markdown("", classes="event-summary")
        yield Static("", classes="event-card-done")

    def on_mount(self) -> None:
        self.refresh_content()

    def mark_done(self) -> None:
        if self.done:
            return
        self.done = True
        self.add_class("summary-done")
        markers = list(self.query(".event-card-done"))
        if markers:
            markers[0].update("✓")

    def add(self, summary: str, detail: str | None = None) -> None:
        if summary and summary not in self.lines:
            self.lines.append(summary)
        if detail:
            self.details.append(detail)
        self.refresh_content()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.refresh_content()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.set_expanded(not self.expanded)

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        visible = self.lines if self.expanded else self.lines[-6:]
        omitted = len(self.lines) - len(visible)
        body = [f"**{self.label}**"]
        if omitted:
            body.append(f"_{omitted} earlier items hidden_  ")
        body.extend(f"- {line}" for line in visible)
        if self.expanded:
            body.extend(f"\n```text\n{detail}\n```" for detail in self.details)
        self.query_one(Markdown).update("\n".join(body))


class PlanCard(EventCard):
    """A live draft/review timeline for one plan revision."""

    SPINNER_FRAMES = (".", "..", "...")

    def __init__(self, key: str, revision: int) -> None:
        super().__init__(key, "🗺️ Planning")
        self.revision = revision
        self.steps: list[str] = []
        self.state = "drafting"
        self.review_reason = ""
        self.expanded = False
        self._frame = 0
        self._spinner_timer = None

    def compose(self) -> ComposeResult:
        yield Markdown("", classes="event-summary")
        yield Static("Click to expand", classes="plan-expand-hint")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.3, self._advance_spinner)
        self.refresh_content()

    def on_resize(self) -> None:
        self.refresh_content()

    def _advance_spinner(self) -> None:
        if self.state not in {"drafting", "reviewing"}:
            if self._spinner_timer is not None:
                self._spinner_timer.pause()
            return
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        self.refresh_content()

    def _resume_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.resume()

    def set_drafting(self) -> None:
        self.state = "drafting"
        self._resume_spinner()
        self.refresh_content()

    def set_plan(self, steps: Sequence[Any]) -> None:
        self.steps = [
            self._step_text(index, step) for index, step in enumerate(steps, 1)
        ]
        self.state = "reviewing"
        self._resume_spinner()
        self.refresh_content()

    def set_reviewing(self) -> None:
        self.state = "reviewing"
        self._resume_spinner()
        self.refresh_content()

    def finish_review(self, approved: bool, reason: str = "") -> None:
        self.state = "complete" if approved else "revision_needed"
        self.review_reason = "" if approved else reason.strip()
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self.refresh_content()

    def finish_pending_review(self, *, succeeded: bool) -> None:
        if self.state != "reviewing":
            return
        if succeeded:
            self.finish_review(True)
            return
        self.state = "revision_needed"
        self.review_reason = "Planning stopped before review completed."
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self.refresh_content()

    @staticmethod
    def _step_text(index: int, step: Any) -> str:
        if not isinstance(step, Mapping):
            dump = getattr(step, "model_dump", None)
            step = dump() if callable(dump) else {"name": str(step)}
        name = str(step.get("name") or f"Step {index}")
        description = " ".join(str(step.get("description") or "").split())
        return f"{index}. {name}" + (f": {description}" if description else "")

    @staticmethod
    def _truncate_middle(text: str, width: int) -> str:
        if cell_len(text) <= width:
            return text
        marker = " … truncated … "
        available = width - cell_len(marker)
        left = (available + 1) // 2
        right = available // 2
        prefix = chop_cells(text, left)[0]
        suffix = chop_cells(text[::-1], right)[0][::-1]
        return f"{prefix} _… truncated …_ {suffix}"

    def _step_width(self) -> int:
        """Use the complete rendered row, minus Markdown's list indentation."""
        markdown_width = self.query_one(Markdown).content_size.width
        content_width = markdown_width or max(0, self.content_size.width - 4)
        return max(40, content_width - 4)

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        if self.expanded or len(self.steps) <= 4:
            visible = self.steps
        else:
            hidden = len(self.steps) - 4
            visible = [
                self.steps[0],
                self.steps[1],
                f"_… {hidden} middle step{'s' if hidden != 1 else ''} hidden …_",
                self.steps[-2],
                self.steps[-1],
            ]
        if not self.expanded:
            width = self._step_width()
            visible = [self._truncate_middle(step, width) for step in visible]

        body = [f"**{self.label}**  "]
        status_indent = " "
        if not self.steps:
            spinner = self.SPINNER_FRAMES[self._frame]
            body.append(f"{status_indent}✍️ Drafting Plan{spinner}")
        else:
            plan_label = (
                "Initial Plan" if self.revision == 1 else "Revised Plan"
            )
            body.append(f"{status_indent}📄 **{plan_label}**")
            body.append("")
            for step in visible:
                if "middle step" in step:
                    body.extend(["", f"{status_indent}{step}", ""])
                else:
                    body.append(step)
            body.append("")
            if self.state == "reviewing":
                spinner = self.SPINNER_FRAMES[self._frame]
                body.append(f"{status_indent}📋 Reviewing{spinner}")
            elif self.state == "revision_needed":
                body.append(f"{status_indent}❌ Plan needs another revision")
            elif self.state == "complete":
                body.append(f"{status_indent}✅ 📋 Plan is complete")
            if self.expanded and self.review_reason:
                body.extend([
                    "",
                    f"{status_indent}**Revision feedback**",
                    *(f"> {line}" for line in self.review_reason.splitlines()),
                ])

        expandable = bool(
            self.steps and not self.expanded and len(self.steps) > 4
        )
        self.query_one(".plan-expand-hint").set_class(not expandable, "hidden")
        self.query_one(Markdown).update("\n".join(body))


class ArtifactCard(EventCard):
    """Rich-rendered structured artifacts emitted by an agent or tool."""

    def __init__(self, key: str, artifacts: list[Mapping[str, Any]]) -> None:
        super().__init__(key, "Artifact")
        self.artifacts = artifacts

    def compose(self) -> ComposeResult:
        yield Static(
            render_event_artifacts(self.artifacts), classes="event-summary"
        )
        yield Static("", classes="event-card-done")


class AgentEventCard(EventCard):
    """Specialized live summary for non-file agent progress."""

    def __init__(self, key: str, agent: str) -> None:
        icon, label = AGENT_LABELS.get(agent, ("◌", agent or "Agent"))
        super().__init__(key, f"{icon} {label}")

    @staticmethod
    def _stage_icon(agent: str, stage: str, payload: Mapping[str, Any]) -> str:
        if agent in {"PlanningAgent", "planner"}:
            if stage == "reflect_result":
                return "✅" if payload.get("approved") else "🔁"
            return {"generate": "📐", "generate_result": "🗺️"}.get(stage, "📋")
        if agent in {"HypothesizerAgent", "hypothesizer"}:
            return {
                "generate": "✨",
                "generate_result": "💡",
                "critique": "🔬",
                "critique_result": "🧪",
                "competitor": "🧭",
                "competitor_result": "🗣️",
                "finalize": "🛠️",
                "finalize_result": "⭐",
                "summarize": "📝",
                "summarize_result": "📚",
            }.get(stage, "💡")
        if agent in {"LammpsAgent", "lammps"}:
            return {
                "author_input": "📝",
                "choose_potential": "🧲",
                "fix_input": "🛠️",
                "run": "▶",
                "run_result": ("✅" if payload.get("returncode") == 0 else "✖"),
                "summarize_potential": "🔬",
                "summarize_results": "📊",
            }.get(stage, "⚛️")
        return "⚙️"

    def update_event(self, payload: Mapping[str, Any]) -> None:
        agent = str(payload.get("agent") or "")
        message = str(payload.get("message") or payload.get("stage") or "Event")
        stage = str(payload.get("stage") or "")
        detail = payload.get("preview")
        if stage == "reflect_result":
            detail = payload.get("reason")
        elif stage == "choose_potential" and payload.get("phase") == "end":
            detail = "\n".join(
                filter(
                    None,
                    (
                        f"Potential: {payload.get('potential_id')}",
                        f"Index: {payload.get('chosen_index')}",
                        str(payload.get("rationale") or ""),
                    ),
                )
            )
        elif stage == "run" and payload.get("phase") == "error":
            detail = payload.get("error_output") or payload.get("error")
        elif stage == "fix_input" and (
            payload.get("old_code") is not None
            or payload.get("new_code") is not None
        ):
            detail = EditCard._diff(
                str(payload.get("old_code") or ""),
                str(payload.get("new_code") or ""),
            )[2]
        if output_path := payload.get("output_path"):
            output_detail = f"Output: {output_path}"
            detail = f"{detail}\n{output_detail}" if detail else output_detail
        icon = self._stage_icon(agent, stage, payload)
        self.add(f"{icon} {message}", str(detail) if detail else None)


class SearchEventCard(EventCard):
    """Live search status with query and result-size details."""

    def __init__(self, key: str, tool: str) -> None:
        label = {
            "run_arxiv_search": "arXiv Search",
            "run_osti_search": "OSTI Search",
            "run_web_search": "Web Search",
        }.get(tool, "Search")
        super().__init__(key, f"🔎 {label}")

    def update_event(self, payload: Mapping[str, Any]) -> None:
        message = str(payload.get("message") or "Searching")
        query = str(payload.get("query") or "").strip()
        phase = str(payload.get("phase") or "")
        icon = "✖" if phase == "error" else "✓" if phase == "end" else "🔎"
        summary = f"{icon} {message}" + (f": {query}" if query else "")
        detail = (
            payload.get("error")
            or payload.get("reason")
            or payload.get("preview")
        )
        if isinstance(payload.get("result_chars"), int):
            size = f"{payload['result_chars']:,} result characters"
            detail = f"{detail}\n{size}" if detail else size
        self.add(summary, str(detail) if detail else None)


class FileActivityCard(EventCard):
    """Group touched files by the operation performed on them."""

    SECTIONS = ("Reading", "Editing")

    def __init__(self, key: str = "files") -> None:
        super().__init__(key, "◫ Files")
        self.files: dict[str, dict[str, tuple[int | None, int | None]]] = {
            section: {} for section in self.SECTIONS
        }
        self.outcomes: dict[tuple[str, str], tuple[str, str]] = {}

    def compose(self) -> ComposeResult:
        yield Static("", classes="event-summary file-summary")
        yield Static("", classes="event-card-done")

    def add_file(
        self,
        operation: str,
        path: str,
        *,
        additions: int | None = None,
        deletions: int | None = None,
    ) -> None:
        current = self.files[operation].get(path)
        if current is not None and additions is None and deletions is None:
            return
        self.files[operation][path] = (additions, deletions)
        self.refresh_content()

    def record_outcome(
        self, operation: str, path: str, state: str, detail: str = ""
    ) -> None:
        self.outcomes[(operation, path)] = (state, detail)
        self.refresh_content()

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        output = Text()
        reading = self.files["Reading"]
        if reading:
            output.append("📖 Reading: ", style="bold")
            for index, path in enumerate(reading):
                if index:
                    output.append(", ")
                output.append(path, style="cyan")

        editing = self.files["Editing"]
        if editing:
            if reading:
                output.append("\n")
            output.append("✍️ Editing", style="bold")
            path_width = max(map(len, editing))
            addition_width = max(
                len(f"+{additions if additions is not None else '?'}")
                for additions, _ in editing.values()
            )
            for path, (additions, deletions) in editing.items():
                addition = f"+{additions if additions is not None else '?'}"
                deletion = f"-{deletions if deletions is not None else '?'}"
                output.append("\n- ")
                output.append(path, style="cyan")
                output.append(" " * (path_width - len(path) + 3))
                output.append(addition.rjust(addition_width), style="green")
                output.append(" ")
                output.append(deletion, style="red")
        for (operation, path), (state, detail) in self.outcomes.items():
            if reading or editing or output:
                output.append("\n")
            icon = "✖" if state == "failed" else "⚠"
            style = "red" if state == "failed" else "yellow"
            output.append(f"{icon} {operation} {state}: ", style=style)
            output.append(path, style="cyan")
            if detail:
                output.append(f" — {' '.join(detail.split())}", style="dim")
        self.query_one(".file-summary", Static).update(output)


class EditCard(EventCard):
    def __init__(self, path: str, old: str, new: str) -> None:
        super().__init__(f"edit:{path}", f"✎ {path}")
        self.old = old
        self.new = new
        self.additions, self.deletions, self.diff = self._diff(old, new)
        self.expanded = True

    @staticmethod
    def _diff(old: str, new: str) -> tuple[int, int, str]:
        import difflib

        lines = list(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        additions = sum(
            line.startswith("+") and not line.startswith("+++")
            for line in lines
        )
        deletions = sum(
            line.startswith("-") and not line.startswith("---")
            for line in lines
        )
        return additions, deletions, "\n".join(lines)

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        diff_lines = self.diff.splitlines()
        visible = diff_lines if self.expanded else diff_lines[:8]
        visible_diff = "\n".join(visible)
        suffix = (
            ""
            if self.expanded or len(diff_lines) <= 8
            else "\n… diff collapsed (Ctrl+T expands)"
        )
        body = (
            f"**{self.label}**  \n"
            f"`+{self.additions} -{self.deletions}`\n\n"
            f"```diff\n{visible_diff}\n```{suffix}"
        )
        self.query_one(Markdown).update(body)


class CommandSafetyIndicator(ActivityIndicator):
    """Safety-check state for a single command invocation."""

    def compose(self) -> ComposeResult:
        yield Static(self.FRAMES[0], classes="activity-spinner")
        yield Static("Running safety check", classes="activity-text")

    def passed(self) -> None:
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("✓")
        self.query_one(".activity-text", Static).update("Safety check passed")

    def failed(self, reason: str | None = None) -> None:
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("⚔️")
        self.query_one(".activity-text", Static).update(
            reason or "Safety check failed"
        )


class RunCommandCard(EventCard):
    """Progressively disclose one command, its safety check, and output."""

    def __init__(self, key: str, command: str) -> None:
        super().__init__(key, "run_command")
        self.command = command
        self.completed = False
        self.multi_command = False
        self.output_expanded = False
        self._full_output = ""
        self.returncode: int | None = None
        self.safety_failed = False
        self.force_compact = False
        self._compact_frame = 0
        self._compact_timer = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="command-compact hidden"):
            yield Static(self.FRAMES[0], classes="command-compact-state")
            yield Static(
                self._collapsed_command(), classes="command-compact-text"
            )
        yield Static(
            self._command_syntax(collapsed=False), classes="command-source"
        )
        yield CommandSafetyIndicator()
        yield Static("", classes="command-output hidden")

    def on_mount(self) -> None:
        self._compact_timer = self.set_interval(
            0.08, self._advance_compact_spinner, pause=True
        )

    @property
    def FRAMES(self) -> tuple[str, ...]:
        return ActivityIndicator.FRAMES

    def _advance_compact_spinner(self) -> None:
        if self.safety_failed:
            self.query_one(".command-compact-state", Static).update("⚔️")
            return
        self.query_one(".command-compact-state", Static).update(
            self.FRAMES[self._compact_frame]
        )
        self._compact_frame = (self._compact_frame + 1) % len(self.FRAMES)

    @staticmethod
    def _preview_command(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= 20:
            return text
        omitted = len(lines) - 16
        return "\n".join([
            *lines[:8],
            f"… {omitted} lines omitted …",
            *lines[-8:],
        ])

    @staticmethod
    def _preview_output(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= 10:
            return text
        omitted = len(lines) - 8
        return "\n".join([
            *lines[:4],
            f"… {omitted} lines omitted …",
            *lines[-4:],
        ])

    def _collapsed_command(self) -> str:
        lines = self.command.splitlines() or [self.command]
        command = lines[0]
        if len(lines) > 1:
            command += " …"
        if len(command) > 120:
            command = command[:119] + "…"
        return command

    def _command_syntax(self, *, collapsed: bool) -> Syntax:
        lines = self.command.splitlines() or [self.command]
        if collapsed:
            command = self._collapsed_command()
        else:
            command = self._preview_command("\n".join(lines))
        return Syntax(command, "bash", word_wrap=True)

    def update_event(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage") or "")
        if isinstance(payload.get("returncode"), int):
            self.returncode = payload["returncode"]
        if stage == "safety_check":
            safety = self.query_one(CommandSafetyIndicator)
            if payload.get("safe") is True:
                safety.passed()
            elif payload.get("safe") is False:
                self.safety_failed = True
                self.force_compact = True
                safety.failed(str(payload.get("reason") or "") or None)

        output = payload.get("result")
        if (
            output is None
            and stage == "execute"
            and payload.get("phase") == "end"
        ):
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, list):
                contents = [
                    str(artifact.get("content"))
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                    and artifact.get("content") not in (None, "")
                ]
                if contents:
                    output = "\n".join(contents)
        if output is not None:
            self.complete(output)

    def complete(self, output: Any) -> None:
        self.completed = True
        if self._compact_timer is not None:
            self._compact_timer.pause()
        self.query_one(".command-compact-state", Static).update(
            self._completion_icon()
        )
        self.query_one(".command-source", Static).update(
            self._command_syntax(collapsed=True)
        )
        self._full_output = self._clean_output(output)
        if not self._full_output:
            self.force_compact = True
        self._render_output()
        self._update_visibility()

    def _completion_icon(self) -> str:
        if self.safety_failed:
            return "⚔️"
        if self.returncode not in (None, 0):
            return "✗"
        return "✓"

    def set_multi_command(self, multi_command: bool) -> None:
        self.multi_command = multi_command
        if self._compact_timer is not None:
            if multi_command and not self.completed and not self.safety_failed:
                self._advance_compact_spinner()
                self._compact_timer.resume()
            else:
                self._compact_timer.pause()
        self._update_visibility()

    def set_output_expanded(self, expanded: bool) -> None:
        self.output_expanded = expanded
        self._render_output()
        self._update_visibility()

    def _render_output(self) -> None:
        if not self.completed:
            return
        output = (
            self._full_output
            if self.output_expanded
            else self._preview_output(self._full_output)
        )
        output = output or "(no output)"
        self.query_one(".command-output", Static).update(
            Syntax(output, "text", word_wrap=True)
        )

    def _update_visibility(self) -> None:
        if not self.is_mounted:
            return
        compact = self.multi_command and not self.output_expanded
        self.query_one(".command-compact").set_class(not compact, "hidden")
        self.query_one(".command-source").set_class(compact, "hidden")
        self.query_one(CommandSafetyIndicator).set_class(compact, "hidden")
        show_output = self.completed and not compact
        self.query_one(".command-output").set_class(not show_output, "hidden")

    @staticmethod
    def _clean_output(output: Any) -> str:
        text = str(output or "")
        if text.startswith("STDOUT:\n") and "\nSTDERR:\n" in text:
            stdout, stderr = text[len("STDOUT:\n") :].split("\nSTDERR:\n", 1)
            if stdout and stderr:
                return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            text = stdout or stderr
        return text.rstrip()

    def set_expanded(self, expanded: bool) -> None:
        self.set_output_expanded(expanded)


class Turn(Static):
    def __init__(self, prompt: str, workspace: Path) -> None:
        super().__init__(classes="turn")
        self.prompt = prompt
        self.workspace = Path(workspace).resolve()
        self.cards: dict[str, EventCard] = {}
        self.transcript: list[str] = []
        self.started_at = monotonic()
        self.token_usage = 0
        self._command_count = 0
        self._commands: list[RunCommandCard] = []
        self._commands_overlapped = False
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
        self.outputs_expanded = False

    def compose(self) -> ComposeResult:
        yield MessageCard("user", self.prompt)
        yield Vertical(classes="events")
        yield Markdown("", classes="transcript hidden")
        yield ActivityIndicator()

    async def event(
        self, payload: dict[str, Any], record_transcript: bool = True
    ) -> None:
        async with self._event_lock:
            await self._event(payload, record_transcript=record_transcript)

    async def _event(
        self, payload: dict[str, Any], *, record_transcript: bool
    ) -> None:
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
        if record_transcript:
            self.record_transcript(payload)

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
        if tool in {"write_code", "write_code_with_repo"} and not isinstance(
            additions, int
        ):
            code = payload.get("code")
            if isinstance(code, str):
                additions = len(code.splitlines())
                deletions = 0
        if (
            tool == "edit_code"
            and path
            and (old is not None or new is not None)
        ):
            edit_card = EditCard(path, str(old or ""), str(new or ""))
            additions = edit_card.additions
            deletions = edit_card.deletions

        if tool in FILE_TOOLS and path:
            operation = FILE_TOOLS[tool]
            outcome = self._file_outcome(payload)
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
            if edit_card is not None:
                edit_key = self._next_summary_key("edit")
                await self._replace_or_mount(edit_key, edit_card)
                self.set_timer(3, lambda: edit_card.set_expanded(False))
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

    def record_transcript(self, payload: Mapping[str, Any]) -> None:
        """Append one raw callback record to the full transcript."""
        self.transcript.append(json.dumps(payload, default=str, sort_keys=True))
        self.query_one(".transcript", Markdown).update(
            "\n".join(f"```json\n{line}\n```" for line in self.transcript)
        )

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
        card = self.cards.get(key) if key else None
        if card is None:
            for candidate in reversed(list(self.cards.values())):
                if (
                    isinstance(candidate, RunCommandCard)
                    and candidate.command == command
                    and not candidate.completed
                ):
                    card = candidate
                    break
        if card is None:
            self._command_count += 1
            key = key or f"command:{self._command_count}"
            card = RunCommandCard(key, command or "(command unavailable)")
            await self._replace_or_mount(key, card)
            self._commands.append(card)
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
            completed = [
                command for command in self._commands if command.completed
            ]
            if completed:
                detailed.add(completed[-1])
        elif self._commands:
            detailed.add(self._commands[-1])

        for command_card in self._commands:
            command_card.set_multi_command(
                command_card not in detailed or command_card.force_compact
            )
            command_card.set_output_expanded(self.outputs_expanded)

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
        await self.query_one(".events", Vertical).mount(card)

    async def add_response(self, response: str) -> None:
        await self.mount(MessageCard("assistant", response))

    def set_transcript(self, enabled: bool) -> None:
        self.query_one(".transcript").set_class(not enabled, "hidden")
        self.query_one(".events").set_class(enabled, "hidden")
        for card in self.cards.values():
            if not isinstance(card, RunCommandCard):
                card.set_expanded(enabled)

    def set_outputs_expanded(self, expanded: bool) -> None:
        self.outputs_expanded = expanded
        for card in self.cards.values():
            if isinstance(card, RunCommandCard):
                card.set_output_expanded(expanded)


class TextualEventHandler(AsyncCallbackHandler):
    """Translate LangChain callbacks into live Textual turn updates."""

    def __init__(self, app: UrsaTextualApp, turn: Turn) -> None:
        self.app = app
        self.turn = turn
        self.tools: dict[Any, dict[str, Any]] = {}

    async def _emit(
        self, data: dict[str, Any], *, record_transcript: bool = True
    ) -> None:
        """Apply callback data on Textual's event-loop thread."""
        data.setdefault("_received_at", monotonic())
        if self.app.is_ui_thread:
            await self.turn.event(data, record_transcript)
        else:
            self.app.call_from_thread(self.turn.event, data, record_transcript)

    async def _record(self, data: Mapping[str, Any]) -> None:
        if self.app.is_ui_thread:
            self.turn.record_transcript(data)
        else:
            self.app.call_from_thread(self.turn.record_transcript, data)

    async def _update_activity(
        self, message: str, *, record_transcript: bool = False
    ) -> None:
        if self.app.is_ui_thread:
            self.turn.update_activity(message)
        else:
            self.app.call_from_thread(self.turn.update_activity, message)
        if record_transcript:
            await self._record({"type": "activity", "message": message})

    async def on_custom_event(self, name: str, data: Any, **_: Any) -> None:
        if name == DEFAULT_EVENT_NAME and isinstance(data, dict):
            tool = str(data.get("tool") or "")
            # The file tools publish their own structured range events while
            # LangChain also emits tool start/end callbacks. The callback is
            # the authoritative timeline event; rendering both produces a
            # second group when the range completes after later activity.
            if tool in FILE_TOOLS:
                phase = str(data.get("phase") or "")
                if phase in {"end", "error"}:
                    result = str(data.get("result") or data.get("error") or "")
                    if phase == "error" or result.casefold().startswith((
                        "failed",
                        "no changes made",
                    )):
                        await self._emit(dict(data))
                    else:
                        await self._record(dict(data))
                    return
                if any(
                    pending.get("tool") == tool
                    for pending in self.tools.values()
                ):
                    await self._record(dict(data))
                    return
            await self._emit(dict(data))

    async def on_llm_start(self, *_: Any, **__: Any) -> None:
        await self._update_activity("Thinking…", record_transcript=True)

    async def on_chat_model_start(self, *_: Any, **__: Any) -> None:
        await self._update_activity("Thinking…", record_transcript=True)

    async def on_llm_new_token(
        self, token: str, *, chunk: Any = None, **_: Any
    ) -> None:
        # Ordinary answer tokens are intentionally ignored. Providers that
        # publish reasoning summaries place them in explicit reasoning or
        # thinking fields on the chunk.
        if trace := _reasoning_trace(chunk):
            await self._update_activity(trace, record_transcript=True)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        inputs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        data = dict(inputs or {})
        if not data and input_str:
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {"input": input_str}
        data["tool"] = serialized.get("name", "tool")
        data["phase"] = "start"
        data["_run_id"] = str(run_id)
        if data["tool"] == "run_command":
            data["_command_id"] = data["_run_id"]
        self.tools[run_id] = data
        await self._emit(data)

    async def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        data = {**data, "phase": "end"}
        if isinstance(output, ToolMessage):
            data["result"] = output.content
            data["status"] = output.status
        else:
            data["result"] = output
        if data.get("tool") in FILE_TOOLS and Turn._file_outcome(data) is None:
            await self._record(data)
            return
        await self._emit(data)

    async def on_tool_error(
        self, error: BaseException, *, run_id: Any, **_: Any
    ) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        if data.get("tool") in FILE_TOOLS:
            await self._emit({
                **data,
                "phase": "error",
                "error": str(error),
                "result": str(error),
            })
            return
        await self._emit({
            **data,
            "phase": "error",
            "error": str(error),
            "result": str(error),
        })

    async def on_llm_end(self, response: Any, **_: Any) -> None:
        count = _token_usage(response)
        await self._record({"type": "llm_end", "total_tokens": count})
        if self.app.is_ui_thread:
            self._record_tokens(count)
        else:
            self.app.call_from_thread(self._record_tokens, count)

    def _record_tokens(self, count: int) -> None:
        self.turn.add_tokens(count)
        self.app.add_tokens(count)


class UrsaTextualApp(App[None]):
    """Prototype full-screen URSA chat application."""

    TITLE = "URSA"
    SUB_TITLE = "Textual HITL"
    BINDINGS: ClassVar = [
        Binding("ctrl+t", "toggle_transcript", "Transcript", show=True),
        Binding("ctrl+o", "toggle_outputs", "Outputs", show=True),
        Binding("ctrl+l", "clear_conversation", "Clear", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding(
            "super+up",
            "previous_turn_marker",
            "Previous turn marker",
            show=False,
            priority=True,
        ),
        Binding(
            "super+down",
            "next_turn_marker",
            "Next turn marker",
            show=False,
            priority=True,
        ),
        Binding(
            "meta+up",
            "previous_turn_marker",
            "Previous turn marker",
            show=False,
            priority=True,
        ),
        Binding(
            "meta+down",
            "next_turn_marker",
            "Next turn marker",
            show=False,
            priority=True,
        ),
    ]
    CSS_PATH = Path(__file__).with_name("app.tcss")

    def __init__(self, hitl: HITL) -> None:
        super().__init__()
        self.hitl = hitl
        self.total_tokens = 0
        self.transcript_mode = False
        self.outputs_expanded = False
        self.current_turn: Turn | None = None
        self._hotlist_open = False
        self._hash_hotlist_origin: tuple[str, tuple[int, int]] | None = None
        self._ui_thread_id: int | None = None
        self._turn_navigation_marker: Widget | None = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(WelcomeBanner(self.hitl), id="conversation")
        yield PromptArea()
        yield Static(id="status")

    def on_mount(self) -> None:
        self._ui_thread_id = threading.get_ident()
        self._update_status("ready")
        self.query_one(PromptArea).focus()

    @property
    def is_ui_thread(self) -> bool:
        return threading.get_ident() == self._ui_thread_id

    def _update_status(self, state: str) -> None:
        self.query_one("#status", Static).update(
            f"{_model_name(self.hitl)} ({_endpoint(self.hitl.model)})  •  "
            f"{self.total_tokens:,} tokens  •  {state}  •  "
            "Ctrl+T transcript  •  Ctrl+O outputs"
        )

    def add_tokens(self, count: int) -> None:
        self.total_tokens += count
        self._update_status("working")

    @on(PromptArea.Submitted)
    async def submit_prompt(self, event: PromptArea.Submitted) -> None:
        prompt_widget = self.query_one(PromptArea)
        prompt_widget.load_text("")
        turn = Turn(event.text, self.hitl.workspace)
        await self.query_one("#conversation", VerticalScroll).mount(turn)
        turn.set_outputs_expanded(self.outputs_expanded)
        self.current_turn = turn
        self._turn_navigation_marker = turn.query_one(".events")
        self.query_one("#conversation", VerticalScroll).scroll_end(
            animate=False
        )
        self._update_status("working")
        prompt_widget.disabled = True
        self.run_worker(
            self._run_agent(turn, event.text), exclusive=True, group="agent"
        )

    async def _run_agent(self, turn: Turn, prompt: str) -> None:
        name, prompt = self._route_prompt(prompt)
        handler = TextualEventHandler(self, turn)
        succeeded = True
        try:
            response = await self.hitl.run_agent(
                name, prompt, callbacks=[handler]
            )
        except Exception as exc:
            succeeded = False
            response = f"**Agent failed:** `{type(exc).__name__}: {exc}`"
        turn.finish_activity(succeeded=succeeded)
        await turn.add_response(response)
        self._turn_navigation_marker = list(turn.query(MessageCard))[-1]
        turn.set_transcript(self.transcript_mode)
        self.query_one("#conversation", VerticalScroll).scroll_end(
            animate=False
        )
        prompt_widget = self.query_one(PromptArea)
        prompt_widget.disabled = False
        prompt_widget.focus()
        self._update_status("ready")

    def _route_prompt(self, prompt: str) -> tuple[str, str]:
        return _route_prompt(self.hitl, prompt)

    @on(TextArea.Changed, "#prompt")
    def prompt_changed(self, event: TextArea.Changed) -> None:
        prompt = event.text_area
        prompt.styles.height = min(10, max(1, len(prompt.document.lines))) + 2
        if self._hotlist_open:
            return
        row, column = prompt.cursor_location
        line = prompt.document.lines[row]
        if column and line[column - 1 : column] in {"@", "#"}:
            trigger = line[column - 1]
            if trigger == "#":
                lines = prompt.text.split("\n")
                lines[row] = lines[row][: column - 1] + lines[row][column:]
                self._hash_hotlist_origin = (
                    "\n".join(lines),
                    (row, column - 1),
                )
            self._hotlist_open = True
            self.call_after_refresh(self._open_hotlist, trigger)
        elif row == 0 and column == 1 and line == "/":
            self._hotlist_open = True
            self.call_after_refresh(self._open_hotlist, "/")

    def _open_hotlist(self, trigger: str) -> None:
        candidates = self._hotlist_candidates(trigger)
        title = {
            "#": "Agents",
            "@": "Workspace paths",
            "/": "Commands",
        }[trigger]
        self.push_screen(
            HotlistScreen(title, candidates),
            callback=lambda choice: self._insert_hotlist_choice(
                trigger, choice
            ),
        )

    def _insert_hotlist_choice(self, trigger: str, choice: str | None) -> None:
        if trigger == "/":
            prompt = self.query_one(PromptArea)
            prompt.load_text("")
            self._hotlist_open = False
            if choice:
                self.call_after_refresh(
                    self._show_command, choice.split(" — ", 1)[0]
                )
            else:
                prompt.focus()
            return
        if trigger == "#":
            self._insert_agent_choice(choice)
            return
        if choice:
            prompt = self.query_one(PromptArea)
            row, column = prompt.cursor_location
            prompt.replace(
                f"{trigger}{choice} ",
                (row, column - 1),
                (row, column),
            )
        self._hotlist_open = False
        self.query_one(PromptArea).focus()

    @staticmethod
    def _cursor_offset(text: str, location: tuple[int, int]) -> int:
        row, column = location
        lines = text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @staticmethod
    def _offset_location(text: str, offset: int) -> tuple[int, int]:
        before = text[:offset]
        return before.count("\n"), len(before.rsplit("\n", 1)[-1])

    def _insert_agent_choice(self, choice: str | None) -> None:
        prompt = self.query_one(PromptArea)
        origin = self._hash_hotlist_origin
        if origin is None:
            original_text = prompt.text
            original_location = prompt.cursor_location
        else:
            original_text, original_location = origin

        if choice is None:
            result = original_text
            result_location = original_location
        else:
            original_offset = self._cursor_offset(
                original_text, original_location
            )
            existing = re.match(r"^#[^\s]+[ \t]*", original_text)
            prefix_end = existing.end() if existing else 0
            body = original_text[prefix_end:]
            body_offset = max(0, original_offset - prefix_end)
            prefix = f"#{choice} "
            result = prefix + body
            result_location = self._offset_location(
                result, len(prefix) + body_offset
            )

        prompt.load_text(result)
        prompt.move_cursor(result_location)
        self._hash_hotlist_origin = None
        self._hotlist_open = False
        prompt.focus()

    def _hotlist_candidates(self, trigger: str) -> list[str]:
        if trigger == "#":
            return sorted(self.hitl.agents)
        if trigger == "/":
            return [
                f"{name} — {description}"
                for name, description in COMMAND_CHOICES.items()
            ]
        workspace = Path(self.hitl.workspace)
        ignored = {".git", ".venv", "__pycache__", "node_modules"}
        paths: Iterable[Path] = (
            workspace.rglob("*") if workspace.exists() else ()
        )
        candidates: list[str] = []
        for path in paths:
            if ignored.intersection(path.parts):
                continue
            relative = str(path.relative_to(workspace))
            if path.is_dir():
                candidates.append(f"{relative}/")
            elif path.is_file():
                candidates.append(relative)
            if len(candidates) == 2000:
                break
        return sorted(candidates)

    def _show_command(self, command: str) -> None:
        content = {
            "agents": self._agents_markdown,
            "status": self._status_markdown,
            "keymap": self._keymap_markdown,
        }.get(command)
        if content is None:
            self.query_one(PromptArea).focus()
            return
        self.push_screen(
            InformationScreen(command.capitalize(), content()),
            callback=lambda _: self.query_one(PromptArea).focus(),
        )

    def _agents_markdown(self) -> str:
        sections: list[str] = []
        for name, agent in self.hitl.agents.items():
            description = str(
                agent.description or "No description available."
            ).strip()
            sections.extend((f"## #{name}", description))
            if agent.config:
                sections.append(
                    "\n".join([
                        "| Option | Value |",
                        "|---|---|",
                        *(
                            f"| `{key}` | `{value}` |"
                            for key, value in agent.config.items()
                        ),
                    ])
                )
        return "\n\n".join(sections)

    def _status_markdown(self) -> str:
        embedding = getattr(self.hitl, "embedding", None)
        rows = [
            ("Tokens", f"{self.total_tokens:,}"),
            ("Workspace", str(Path(self.hitl.workspace).resolve())),
            ("Group", str(getattr(self.hitl, "group", None) or "default")),
            ("LLM model", _model_name(self.hitl)),
            ("LLM endpoint", _endpoint(self.hitl.model)),
            ("Embedding model", _embedding_name(self.hitl)),
            (
                "Embedding endpoint",
                _endpoint(embedding) if embedding is not None else "none",
            ),
        ]
        model_table = "\n".join([
            "| Setting | Value |",
            "|---|---|",
            *(f"| {key} | `{value}` |" for key, value in rows),
        ])
        servers = getattr(getattr(self.hitl, "config", None), "mcp_servers", {})
        if not servers:
            return model_table + "\n\n## MCP servers\n\nNone configured."
        server_rows = []
        for name, server in servers.items():
            if isinstance(server, Mapping):
                transport = str(server.get("transport") or "stdio")
                location = (
                    server.get("url") or server.get("command") or "configured"
                )
            else:
                transport = str(getattr(server, "transport", "stdio"))
                location = (
                    getattr(server, "url", None)
                    or getattr(server, "command", None)
                    or "configured"
                )
            server_rows.append(f"| `{name}` | {transport} | `{location}` |")
        return (
            model_table
            + "\n\n## MCP servers\n\n"
            + "\n".join([
                "| Name | Transport | Location |",
                "|---|---|---|",
                *server_rows,
            ])
        )

    @staticmethod
    def _keymap_markdown() -> str:
        rows = [
            ("Enter", "Submit prompt"),
            ("Shift+Enter", "Insert newline"),
            ("Ctrl+C", "Clear prompt and remember it"),
            ("Up / Down", "Move vertically; prompt history at an edge"),
            ("Left / Right", "Move one character"),
            ("Ctrl/Alt/Option+Left / Right", "Move by word"),
            ("Home / End or Ctrl+A / Ctrl+E", "Start / end of line"),
            ("PageUp / PageDown", "Move one editor page"),
            ("Shift+movement", "Extend selection"),
            ("Backspace / Delete", "Delete left / right"),
            ("Ctrl+W / Ctrl+F", "Delete word left / right"),
            ("Ctrl+U / Ctrl+K", "Delete to line start / end"),
            ("Ctrl+X / Ctrl+V", "Cut / paste"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Tab", "Indent"),
            ("@", "Workspace file or directory picker"),
            ("#", "Agent picker and routing"),
            ("/", "Command picker"),
            ("Picker typing", "Fuzzy-filter choices"),
            ("Picker Up / Down", "Select previous / next choice"),
            ("Picker Enter / Esc", "Choose / cancel"),
            ("Ctrl+T", "Toggle full event transcript"),
            ("Ctrl+O", "Expand or collapse command output"),
            ("Cmd+Up / Cmd+Down", "Previous / next turn marker"),
            ("Ctrl+L", "Clear conversation"),
            ("Ctrl+Q", "Quit"),
            ("Info Up/Down/PageUp/PageDown", "Scroll command details"),
            ("Info Q / Esc", "Close command details"),
        ]
        return "\n".join([
            "| Key | Action |",
            "|---|---|",
            *(f"| `{key}` | {action} |" for key, action in rows),
        ])

    def action_toggle_transcript(self) -> None:
        self.transcript_mode = not self.transcript_mode
        for turn in self.query(Turn):
            turn.set_transcript(self.transcript_mode)
        self._update_status("transcript" if self.transcript_mode else "ready")

    def action_toggle_outputs(self) -> None:
        self.outputs_expanded = not self.outputs_expanded
        for turn in self.query(Turn):
            turn.set_outputs_expanded(self.outputs_expanded)

    def _turn_markers(self) -> list[Widget]:
        markers: list[Widget] = []
        for turn in self.query(Turn):
            messages = list(turn.query(MessageCard))
            if not messages:
                continue
            activity = turn.query_one(
                ".transcript" if self.transcript_mode else ".events"
            )
            markers.extend((messages[0], activity))
            if len(messages) > 1:
                markers.append(messages[-1])
        return markers

    def _navigate_turn_markers(self, offset: int) -> None:
        markers = self._turn_markers()
        if not markers:
            return
        try:
            index = markers.index(self._turn_navigation_marker)
        except ValueError:
            index = len(markers) if offset < 0 else -1
        target_index = max(0, min(len(markers) - 1, index + offset))
        target = markers[target_index]
        self._turn_navigation_marker = target
        conversation = self.query_one("#conversation", VerticalScroll)
        target_y = (
            conversation.scroll_y
            + target.region.y
            - conversation.content_region.y
        )
        if offset < 0 and target_y >= conversation.scroll_y:
            target_y = conversation.scroll_y - max(1, target.region.height)
        elif offset > 0 and target_y <= conversation.scroll_y:
            target_y = conversation.scroll_y + max(1, target.region.height)
        # Deferred scrolling raced the conversation's bottom anchor and could
        # be overwritten after this action returned. Apply the marker scroll
        # in the current refresh and explicitly align it to the top.
        conversation.scroll_to_widget(
            target,
            top=True,
            animate=False,
            immediate=True,
            force=True,
            origin_visible=False,
        )
        # Nested turn children report virtual coordinates relative to their
        # turn, not to the conversation. Convert their current screen region
        # into a conversation scroll offset so every marker visibly moves.
        conversation.scroll_to(
            y=max(0, target_y),
            animate=False,
            immediate=True,
            force=True,
            release_anchor=True,
        )

    def action_previous_turn_marker(self) -> None:
        self._navigate_turn_markers(-1)

    def action_next_turn_marker(self) -> None:
        self._navigate_turn_markers(1)

    async def action_clear_conversation(self) -> None:
        await self.query_one("#conversation", VerticalScroll).remove_children()
        await self.query_one("#conversation", VerticalScroll).mount(
            WelcomeBanner(self.hitl)
        )
        self._turn_navigation_marker = None


def run_textual(hitl: HITL) -> None:
    """Launch the experimental full-screen interface."""
    UrsaTextualApp(hitl).run()


def run_textual_once(hitl: HITL, prompt: str, *, stdout: Any = None) -> str:
    """Run one routed prompt and render its event stream to standard output."""
    output = stdout or sys.stdout
    console = Console(file=output)
    handler = HITLLogEventHandler(console=console, workspace=hitl.workspace)
    agent, routed_prompt = _route_prompt(hitl, prompt)

    async def invoke() -> str:
        return await hitl.run_agent(agent, routed_prompt, callbacks=[handler])

    response = asyncio.run(invoke())
    if handler.emitted_any:
        console.print()
    if console.is_terminal:
        console.print(RichMarkdown(response))
    else:
        print(response, file=output)  # noqa: T201
    return response
