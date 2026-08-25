# ruff: noqa: TID251

"""Default card for tool calls without a specialized presentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import ToolMessage
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Markdown, Static

from ursa.cli.event_cards.base import EventCard
from ursa.cli.widgets import ActivityIndicator


def _json(value: Any, *, compact: bool) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        sort_keys=True,
    )


class ToolCallCard(EventCard):
    """Live JSON input/output inspector for a single tool invocation."""

    def __init__(self, key: str, tool: str, tool_input: Any) -> None:
        super().__init__(key, tool)
        self.tool = tool
        self.tool_input = tool_input
        self.output: Any = None
        self.structured_output: Any = None
        self.has_structured_output = False
        self.completed = False
        self.failed = False
        self._frame = 0
        self._spinner_timer = None

    def compose(self) -> ComposeResult:
        yield Static(Text(f"🛠️ {self.tool}"), classes="tool-call-title")
        with Horizontal(classes="tool-call-summary"):
            yield Static(ActivityIndicator.FRAMES[0], classes="tool-call-state")
            yield Static(
                self._preview(self.tool_input), classes="tool-call-preview"
            )
        with Horizontal(classes="tool-call-details hidden"):
            with Vertical(classes="tool-json-pane tool-input-pane"):
                yield Static("Input", classes="tool-json-title")
                yield Static(classes="tool-json tool-input-json")
            with Vertical(classes="tool-json-pane tool-output-pane hidden"):
                yield Static("Output", classes="tool-json-title")
                yield Static(classes="tool-json tool-output-json")
                yield Markdown(classes="tool-output-markdown hidden")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.08, self._advance_spinner)
        self._render_details()
        self._update_expand_hint()

    @staticmethod
    def _preview(value: Any, limit: int = 120) -> Text:
        preview = _json(value, compact=True)
        if len(preview) > limit:
            preview = preview[: limit - 1] + "…"
        return Text(preview)

    @staticmethod
    def _syntax(value: Any) -> Syntax:
        return Syntax(
            _json(value, compact=False),
            "json",
            word_wrap=True,
            background_color="default",
        )

    def _advance_spinner(self) -> None:
        self.query_one(".tool-call-state", Static).update(
            ActivityIndicator.FRAMES[self._frame]
        )
        self._frame = (self._frame + 1) % len(ActivityIndicator.FRAMES)

    def complete(self, output: Any, *, failed: bool = False) -> None:
        if isinstance(output, ToolMessage):
            failed = failed or output.status == "error"
            self.output = self._text_content(output.content)
            artifact = output.artifact
            if (
                isinstance(artifact, Mapping)
                and "structured_content" in artifact
            ):
                self.structured_output = artifact["structured_content"]
                self.has_structured_output = True
        else:
            self.output = output
        self.completed = True
        self.failed = failed
        self.done = True
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self.query_one(".tool-call-state", Static).update(
            "✗" if failed else "✓"
        )
        self.query_one(".tool-call-preview", Static).update(
            self._preview(
                self.structured_output
                if self.has_structured_output
                else self.output
            )
        )
        self._render_details()

    def update_event(self, payload: Mapping[str, Any]) -> None:
        phase = str(payload.get("phase") or "")
        if phase not in {"end", "error"}:
            return
        failed = phase == "error" or payload.get("status") == "error"
        output = (
            payload.get("error")
            if failed
            else payload.get("tool_message", payload.get("result"))
        )
        self.complete(output, failed=failed)

    @staticmethod
    def _text_content(content: Any) -> Any:
        if not isinstance(content, list):
            return content
        parts = [
            item
            if isinstance(item, str)
            else item.get("text")
            if isinstance(item, Mapping) and item.get("type") == "text"
            else None
            for item in content
        ]
        return (
            "\n".join(parts)
            if all(isinstance(part, str) for part in parts)
            else content
        )

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        if self.is_mounted:
            self._render_details()
            self.query_one(".tool-call-details").set_class(
                not expanded, "hidden"
            )
        self._update_expand_hint()

    def _render_details(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(".tool-input-json", Static).update(
            self._syntax(self.tool_input)
        )
        output_pane = self.query_one(".tool-output-pane")
        output_pane.set_class(not self.completed, "hidden")
        if self.completed:
            json_output = self.query_one(".tool-output-json", Static)
            markdown_output = self.query_one(".tool-output-markdown", Markdown)
            render_as_json = self.has_structured_output or not isinstance(
                self.output, str
            )
            json_output.set_class(not render_as_json, "hidden")
            markdown_output.set_class(render_as_json, "hidden")
            if render_as_json:
                json_output.update(
                    self._syntax(
                        self.structured_output
                        if self.has_structured_output
                        else self.output
                    )
                )
            else:
                markdown_output.update(self.output)
