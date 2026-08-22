# ruff: noqa: TID251

"""Textual front end for URSA's human-in-the-loop runner."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Mapping
from math import ceil
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

from ursa.cli.agent_info import load_agent_details
from ursa.cli.callbacks import HITLLogEventHandler
from ursa.cli.event_handler import TextualEventHandler
from ursa.cli.helpers import (
    COMMAND_CHOICES,
    TokenUsage,
    _embedding_name,
    _endpoint,
    _model_name,
    _route_prompt,
)
from ursa.cli.runtime import HITL
from ursa.cli.themes import AVAILABLE_THEMES
from ursa.cli.turn import Turn
from ursa.cli.widgets import (
    AgentsScreen,
    HotlistScreen,
    InformationScreen,
    MessageCard,
    PromptArea,
    ThemeScreen,
    WelcomeBanner,
)
from ursa.util import crossplatform


class ConversationScroll(VerticalScroll):
    """Conversation viewport that reports user-initiated anchor releases."""

    def __init__(
        self,
        *children: Widget,
        on_release: Callable[[], None],
        **kwargs: Any,
    ) -> None:
        self._on_release = on_release
        super().__init__(*children, **kwargs)

    def release_anchor(self) -> None:
        super().release_anchor()
        self._on_release()


class UrsaTextualApp(App[None]):
    """Full-screen URSA chat application."""

    TITLE = "URSA"
    SUB_TITLE = "Textual HITL"
    BINDINGS: ClassVar = [
        Binding(
            "ctrl+c",
            "cancel_agent",
            "Explain active-turn cancellation",
            show=False,
        ),
        Binding(
            "super+c,ctrl+shift+c",
            "copy_text",
            "Copy selected text",
            show=True,
            priority=True,
        ),
        Binding(
            "ctrl+d",
            "hard_quit",
            "Abruptly quit URSA",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+o", "toggle_card_details", "Toggle card details", show=True
        ),
        Binding(
            "ctrl+l", "clear_conversation", "Clear conversation", show=True
        ),
        Binding("ctrl+q", "quit", "Quit gracefully", show=True, priority=True),
        Binding(
            "alt+up",
            "previous_turn_marker",
            "Previous turn marker",
            show=False,
            priority=True,
        ),
        Binding(
            "alt+down",
            "next_turn_marker",
            "Next turn marker",
            show=False,
            priority=True,
        ),
    ]
    CSS_PATH = Path(__file__).with_name("app.tcss")

    def __init__(self, hitl: HITL) -> None:
        super().__init__()
        self.kitty_keyboard_expected = crossplatform.expects_kitty_keyboard()
        for theme in AVAILABLE_THEMES:
            self.register_theme(theme)
        self.theme = AVAILABLE_THEMES[0].name
        self.hitl = hitl
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.card_details_expanded = False
        self.current_turn: Turn | None = None
        self._hotlist_open = False
        self._hotlist_origin: tuple[str, tuple[int, int]] | None = None
        self._ui_thread_id: int | None = None
        self._turn_navigation_marker: Widget | None = None
        self._quit_after_turn = False
        self._conversation_anchor_started = False
        self._conversation_anchor_transition = False
        self._conversation_anchor_generation = 0

    def compose(self) -> ComposeResult:
        yield ConversationScroll(
            WelcomeBanner(self.hitl),
            id="conversation",
            on_release=self._cancel_conversation_anchor_transition,
        )
        yield PromptArea()
        yield Static(id="status")

    def on_mount(self) -> None:
        self._ui_thread_id = threading.get_ident()
        self._update_status("ready")
        self.query_one(PromptArea).focus()

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text using a platform clipboard tool, falling back to OSC52."""
        self._clipboard = text
        if not crossplatform.copy_to_clipboard(text):
            super().copy_to_clipboard(text)

    def action_copy_text(self) -> None:
        """Copy from the focused editor or the current screen selection."""
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            focused.action_copy()
        else:
            self.screen.action_copy_text()

    def on_resize(self) -> None:
        self.call_after_refresh(self._resize_prompt, self.query_one(PromptArea))
        self.call_after_refresh(self._anchor_conversation_if_overflowing)

    async def on_unmount(self) -> None:
        """Release runtime resources on every graceful Textual shutdown."""
        await self.hitl.aclose()

    @property
    def is_ui_thread(self) -> bool:
        return threading.get_ident() == self._ui_thread_id

    @property
    def preferred_newline_key(self) -> str:
        """Return the newline chord most likely to work in this terminal."""
        return "shift+enter" if self.kitty_keyboard_expected else "ctrl+j"

    def _update_status(self, state: str) -> None:
        agent = (
            f"  •  agent {agent_name}"
            if (agent_name := getattr(self.hitl, "agent_name", None))
            else ""
        )
        self.query_one("#status", Static).update(
            f"{_model_name(self.hitl)} ({_endpoint(self.hitl.model)})  •  "
            f"{self.total_tokens:,} tokens{agent}  •  {state}"
        )

    def add_tokens(self, usage: TokenUsage) -> None:
        self.total_tokens += usage.total_tokens
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cached_tokens
        self._update_status("working")

    async def add_turn_event(
        self,
        turn: Turn,
        data: Mapping[str, Any],
    ) -> None:
        """Add an event without disturbing a user who has scrolled up."""
        await turn.event(data)
        self.call_after_refresh(self._anchor_conversation_if_overflowing)

    def _anchor_conversation_if_overflowing(self) -> None:
        """Smoothly reach the bottom, then follow subsequent content."""
        conversation = self.query_one("#conversation", VerticalScroll)
        if (
            not self._conversation_anchor_started
            and conversation.max_scroll_y > 0
        ):
            self._conversation_anchor_started = True
            self._conversation_anchor_transition = True
            self._conversation_anchor_generation += 1
            generation = self._conversation_anchor_generation
            # Start the animation now rather than after the next refresh:
            # a user scroll landing in that gap could not stop an animation
            # that had not begun, and the late-starting animation then
            # overrode the user's position.
            conversation.scroll_end(
                animate=True,
                duration=0.15,
                immediate=True,
                on_complete=lambda: self._finish_conversation_anchor(
                    generation
                ),
            )

    def _cancel_conversation_anchor_transition(self) -> None:
        """Invalidate a pending anchor when scrolling interrupts it."""
        if self._conversation_anchor_transition:
            self._conversation_anchor_transition = False
            self._conversation_anchor_generation += 1

    def _finish_conversation_anchor(self, generation: int) -> None:
        """Anchor only if the initiating transition is still current."""
        if generation != self._conversation_anchor_generation:
            return
        self._conversation_anchor_transition = False
        self.query_one("#conversation", VerticalScroll).anchor()

    def _reset_conversation_auto_follow(
        self, conversation: VerticalScroll
    ) -> None:
        """Invalidate pending work and allow a fresh anchor transition."""
        self._conversation_anchor_generation += 1
        self._conversation_anchor_transition = False
        self._conversation_anchor_started = False
        conversation.anchor(False)

    @on(PromptArea.Submitted)
    async def submit_prompt(self, event: PromptArea.Submitted) -> None:
        prompt_widget = self.query_one(PromptArea)
        prompt_widget.load_text("")
        turn = Turn(event.text, self.hitl.workspace)
        conversation = self.query_one("#conversation", VerticalScroll)
        self._reset_conversation_auto_follow(conversation)
        await conversation.mount(turn)
        self.call_after_refresh(self._anchor_conversation_if_overflowing)
        turn.set_card_details_expanded(self.card_details_expanded)
        self.current_turn = turn
        self._turn_navigation_marker = turn.query_one(".events")
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
            await turn.add_exception(
                exc, "".join(traceback.format_exception(exc))
            )
            response = f"**Agent failed:** `{type(exc).__name__}: {exc}`"
        turn.finish_activity(succeeded=succeeded)
        await turn.add_response(response)
        self.call_after_refresh(self._anchor_conversation_if_overflowing)
        self._turn_navigation_marker = list(turn.query(MessageCard))[-1]
        prompt_widget = self.query_one(PromptArea)
        prompt_widget.disabled = False
        prompt_widget.focus()
        self._update_status("ready")
        if self._quit_after_turn:
            self.exit()

    def _route_prompt(self, prompt: str) -> tuple[str, str]:
        return _route_prompt(self.hitl, prompt)

    def action_cancel_agent(self) -> None:
        """Explain that an active agent cannot be cancelled safely."""
        prompt = self.query_one(PromptArea)
        if prompt.disabled:
            self.notify(
                "Cancelling an active turn is not supported. "
                "Press Ctrl+D to abruptly quit URSA.",
                title="Turn is still running",
                severity="warning",
            )

    def action_hard_quit(self) -> None:
        """Abruptly terminate URSA without waiting for active work."""
        os._exit(130)

    def action_quit(self) -> None:
        """Quit after the active turn, or immediately when idle."""
        if self.query_one(PromptArea).disabled:
            self._quit_after_turn = True
            self.notify(
                "URSA will quit when the active turn finishes. "
                "Press Ctrl+D to quit immediately.",
                title="Waiting for active turn",
                severity="information",
            )
            return
        self.exit()

    @on(TextArea.Changed, "#prompt")
    def prompt_changed(self, event: TextArea.Changed) -> None:
        """Resize for all edits without treating programmatic edits as macros."""
        prompt = event.text_area
        self.call_after_refresh(self._resize_prompt, prompt)

    @on(PromptArea.MacroTyped)
    def macro_typed(self, event: PromptArea.MacroTyped) -> None:
        """Open a picker only for a macro character typed by the user."""
        if self._hotlist_open:
            return
        row, column = event.location
        if event.trigger == "/" and (row, column) != (0, 0):
            return
        self._hotlist_origin = (event.trigger, event.location)
        self._hotlist_open = True
        self.call_after_refresh(self._open_hotlist, event.trigger)

    def _resize_prompt(self, prompt: TextArea) -> None:
        """Fit the prompt to its visual lines within 30% of the terminal."""
        max_content_height = ceil(self.size.height * 0.3)
        content_height = min(
            max_content_height, max(1, prompt.virtual_size.height)
        )
        prompt.styles.height = content_height + 2

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
        prompt = self.query_one(PromptArea)
        origin = self._hotlist_origin
        if trigger == "/":
            self._hotlist_open = False
            self._hotlist_origin = None
            if choice:
                if origin is not None:
                    _, start = origin
                    prompt.replace("", start, (start[0], start[1] + 1))
                self.call_after_refresh(
                    self._show_command, choice.split(" — ", 1)[0]
                )
            else:
                prompt.focus()
            return
        if trigger == "#":
            self._insert_agent_choice(choice)
            return
        if choice and origin is not None:
            _, start = origin
            prompt.replace(
                f"{trigger}{choice} ",
                start,
                (start[0], start[1] + 1),
            )
        self._hotlist_open = False
        self._hotlist_origin = None
        prompt.focus()

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
        origin = self._hotlist_origin

        if choice is not None and origin is not None:
            _, trigger_location = origin
            original_text = prompt.text
            trigger_offset = self._cursor_offset(
                original_text, trigger_location
            )
            text_without_trigger = (
                original_text[:trigger_offset]
                + original_text[trigger_offset + 1 :]
            )
            existing = re.match(r"^#[^\s]+[ \t]*", text_without_trigger)
            prefix_end = existing.end() if existing else 0
            body = text_without_trigger[prefix_end:]
            body_offset = max(0, trigger_offset - prefix_end)
            prefix = f"#{choice} "
            result = prefix + body
            result_location = self._offset_location(
                result, len(prefix) + body_offset
            )
            end = (
                len(prompt.document.lines) - 1,
                len(prompt.document.lines[-1]),
            )
            prompt.replace(
                result,
                (0, 0),
                end,
                maintain_selection_offset=False,
            )
            prompt.move_cursor(result_location)

        self._hotlist_origin = None
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
        # TODO: Traverse asynchronously and remove the arbitrary result cap;
        # large workspaces currently block the UI and stop at 2,000 paths.
        paths: Iterable[Path] = (
            workspace.rglob("*") if workspace.exists() else ()
        )
        candidates: list[str] = []
        for path in paths:
            if ignored.intersection(path.parts):
                continue
            relative = str(path.relative_to(workspace))
            if path.is_dir():
                candidates.append(f"{relative}{os.sep}")
            elif path.is_file():
                candidates.append(relative)
            if len(candidates) == 2000:
                break
        return sorted(candidates)

    async def _show_command(self, command: str) -> None:
        if command == "exit":
            self.action_quit()
            return
        if command == "agents":
            details = await load_agent_details(self.hitl)
            self.push_screen(
                AgentsScreen(details),
                callback=lambda _: self.query_one(PromptArea).focus(),
            )
            return
        if command == "theme":
            choices = [
                self.theme,
                *(
                    theme.name
                    for theme in AVAILABLE_THEMES
                    if theme.name != self.theme
                ),
            ]
            self.push_screen(
                ThemeScreen(choices, initial_theme=self.theme),
                callback=self._select_theme,
            )
            return
        content = {
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

    def _select_theme(self, theme: str | None) -> None:
        if theme is not None:
            self.theme = theme
        self.query_one(PromptArea).focus()

    def _status_markdown(self) -> str:
        embedding = getattr(self.hitl, "embedding", None)
        rows = [
            ("Input tokens", f"{self.input_tokens:,}"),
            ("Output tokens", f"{self.output_tokens:,}"),
            ("Cached tokens", f"{self.cached_tokens:,}"),
            ("Total tokens", f"{self.total_tokens:,}"),
            ("Theme", self.theme),
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
        if agent_name := getattr(self.hitl, "agent_name", None):
            rows.insert(3, ("Agent", str(agent_name)))
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
    def _effective_bindings(owner: type[Any]) -> list[Binding]:
        """Collect Textual bindings with subclass definitions taking priority."""
        bindings: dict[str, Binding] = {}
        for base in reversed(owner.__mro__):
            declared = base.__dict__.get("BINDINGS", ())
            for binding in Binding.make_bindings(declared):
                bindings[binding.key] = binding
        return list(bindings.values())

    def _keymap_markdown(self) -> str:
        sections = (
            ("Application", type(self)),
            ("Prompt editor", PromptArea),
            ("Picker", HotlistScreen),
            ("Information screen", InformationScreen),
        )
        priority_keys = {
            binding.key
            for binding in self._effective_bindings(type(self))
            if binding.priority
        }
        compatibility = (
            "> **Terminal support:** Kitty keyboard support expected for "
            "this terminal."
            if self.kitty_keyboard_expected is True
            else "> **Terminal support:** Kitty keyboard support not "
            "identified; some modified keys may not work."
        )
        output = [
            compatibility,
            "",
        ]
        for title, owner in sections:
            actions: dict[tuple[str, str], list[str]] = {}
            for binding in self._effective_bindings(owner):
                if binding.system or not binding.description:
                    continue
                if owner is not type(self) and binding.key in priority_keys:
                    continue
                identity = (binding.action, binding.description)
                actions.setdefault(identity, []).append(
                    self.get_key_display(binding)
                )
            output.extend([
                f"## {title}",
                "",
                "| Key | Action |",
                "|---|---|",
                *(
                    f"| `{' / '.join(keys)}` | {description} |"
                    for (_, description), keys in actions.items()
                ),
                "",
            ])
        return "\n".join(output).rstrip()

    def action_toggle_card_details(self) -> None:
        self.card_details_expanded = not self.card_details_expanded
        for turn in self.query(Turn):
            turn.set_card_details_expanded(self.card_details_expanded)

    def _turn_markers(self) -> list[Widget]:
        markers: list[Widget] = []
        for turn in self.query(Turn):
            messages = list(turn.query(MessageCard))
            if not messages:
                continue
            activity = turn.query_one(".events")
            markers.extend((messages[0], activity))
            if len(messages) > 1:
                markers.append(messages[-1])
            markers.append(turn.query_one(".turn-end-marker"))
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
        if target_index == len(markers) - 1:
            conversation.anchor()
            return
        target_y = target.virtual_region.y
        ancestor = target.parent
        while ancestor is not None and ancestor is not conversation:
            target_y += ancestor.virtual_region.y
            ancestor = ancestor.parent
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
        if self.query_one(PromptArea).disabled:
            self.notify(
                "Clearing the conversation is not allowed while a turn is "
                "active. Press Ctrl+D to abruptly quit URSA.",
                title="Turn is still running",
                severity="warning",
            )
            return
        conversation = self.query_one("#conversation", VerticalScroll)
        self._reset_conversation_auto_follow(conversation)
        await conversation.remove_children()
        await conversation.mount(WelcomeBanner(self.hitl))
        conversation.scroll_home(animate=False)
        self._turn_navigation_marker = None


def run_textual(hitl: HITL) -> None:
    """Launch the experimental full-screen interface."""
    try:
        UrsaTextualApp(hitl).run()
    finally:
        asyncio.run(hitl.aclose())


def run_textual_once(hitl: HITL, prompt: str, *, stdout: Any = None) -> str:
    """Run one routed prompt and render its event stream to standard output."""
    output = stdout or sys.stdout
    console = Console(file=output)
    handler = HITLLogEventHandler(console=console, workspace=hitl.workspace)
    agent, routed_prompt = _route_prompt(hitl, prompt)

    async def invoke() -> str:
        try:
            return await hitl.run_agent(
                agent, routed_prompt, callbacks=[handler]
            )
        finally:
            await hitl.aclose()

    response = asyncio.run(invoke())
    if handler.emitted_any:
        console.print()
    if console.is_terminal:
        console.print(RichMarkdown(response))
    else:
        print(response, file=output)  # noqa: T201
    return response
