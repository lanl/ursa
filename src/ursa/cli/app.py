# ruff: noqa: TID251

"""Textual front end for URSA's human-in-the-loop runner."""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, TextArea

from ursa.cli.callbacks import HITLLogEventHandler
from ursa.cli.event_handler import TextualEventHandler
from ursa.cli.helpers import (
    COMMAND_CHOICES,
    _embedding_name,
    _endpoint,
    _model_name,
    _route_prompt,
)
from ursa.cli.runtime import HITL
from ursa.cli.turn import Turn
from ursa.cli.widgets import (
    HotlistScreen,
    InformationScreen,
    MessageCard,
    PromptArea,
    WelcomeBanner,
)


class UrsaTextualApp(App[None]):
    """Full-screen URSA chat application."""

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
