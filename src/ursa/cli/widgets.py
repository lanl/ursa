# ruff: noqa: TID251

"""Reusable widgets and modal screens for the Textual CLI."""

from collections.abc import Sequence
from math import ceil
from pathlib import Path

from rich.cells import cell_len, chop_cells
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Collapsible,
    Input,
    Markdown,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets.option_list import Option

from ursa.agents.base import URSA_VERSION
from ursa.cli.agent_info import AgentDetails, ToolDetails
from ursa.cli.helpers import (
    _embedding_name,
    _endpoint,
    _fuzzy_score,
    _model_name,
)
from ursa.cli.runtime import HITL
from ursa.cli.tips import random_tip


class PromptArea(TextArea):
    """A multiline editor whose bare Enter submits the current prompt."""

    BINDINGS = [
        Binding(
            "enter", "submit_prompt", "Submit prompt", show=False, priority=True
        ),
        Binding(
            "shift+enter",
            "insert_newline",
            "Insert newline",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+c", "clear_prompt", "Clear prompt", show=False, priority=True
        ),
        Binding(
            "up",
            "history_up",
            "Cursor or prompt history up",
            show=False,
            priority=True,
        ),
        Binding(
            "down",
            "history_down",
            "Cursor or prompt history down",
            show=False,
            priority=True,
        ),
        Binding(
            "alt+left,meta+left,alt+b",
            "cursor_word_left",
            "Cursor word left",
            show=False,
            priority=True,
        ),
        Binding(
            "alt+right,meta+right,alt+f",
            "cursor_word_right",
            "Cursor word right",
            show=False,
            priority=True,
        ),
        Binding(
            "@",
            "file_macro",
            "Choose workspace path",
            show=False,
            priority=True,
        ),
        Binding("#", "agent_macro", "Choose agent", show=False, priority=True),
        Binding(
            "/",
            "command_macro",
            "Open command picker",
            show=False,
            priority=True,
        ),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class MacroTyped(Message):
        """A macro trigger inserted by a real keyboard event."""

        def __init__(self, trigger: str, location: tuple[int, int]) -> None:
            super().__init__()
            self.trigger = trigger
            self.location = location

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

    def action_submit_prompt(self) -> None:
        text = self.text.strip()
        if text:
            self._remember(text)
            self.post_message(self.Submitted(text))

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def action_clear_prompt(self) -> None:
        self._remember(self.text)
        self._history_index = len(self.prompt_history)
        self.load_text("")

    def action_history_up(self) -> None:
        if self.prompt_history and (
            not self.text or self.cursor_location[0] == 0
        ):
            index = (
                len(self.prompt_history)
                if self._history_index is None
                else self._history_index
            )
            self._load_history(max(0, index - 1))
            return
        self.action_cursor_up()

    def action_history_down(self) -> None:
        if self._history_index is not None:
            next_index = self._history_index + 1
            if next_index < len(self.prompt_history):
                self._load_history(next_index)
            else:
                self._history_index = None
                self.load_text("")
            return
        self.action_cursor_down()

    def _insert_macro(self, trigger: str) -> None:
        location = self.cursor_location
        self.insert(trigger)
        self.post_message(self.MacroTyped(trigger, location))

    def action_file_macro(self) -> None:
        self._insert_macro("@")

    def action_agent_macro(self) -> None:
        self._insert_macro("#")

    def action_command_macro(self) -> None:
        self._insert_macro("/")


class HotlistScreen(ModalScreen[str | None]):
    """Fuzzy-searchable picker overlaid above the prompt."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel picker", priority=True),
        Binding("up", "previous_choice", "Previous choice", priority=True),
        Binding("down", "next_choice", "Next choice", priority=True),
        Binding("enter", "select_choice", "Select choice", priority=True),
    ]

    def __init__(self, title: str, candidates: Sequence[str]) -> None:
        super().__init__()
        self.picker_title = title
        self.candidates = list(candidates)
        self.matches = list(candidates)

    def compose(self) -> ComposeResult:
        with Vertical(id="hotlist"):
            with Horizontal(id="hotlist-header"):
                yield Static(self.picker_title, id="hotlist-title")
                yield Static("Esc to Exit", id="hotlist-exit-hint")
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

    def action_previous_choice(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_next_choice(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_select_choice(self) -> None:
        options = self.query_one(OptionList)
        if options.option_count:
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


class ThemeScreen(HotlistScreen):
    """Theme picker that previews highlighted themes over the current app."""

    def __init__(
        self,
        candidates: Sequence[str],
        initial_theme: str,
    ) -> None:
        super().__init__("Themes", candidates)
        self.initial_theme = initial_theme

    @on(OptionList.OptionHighlighted)
    def preview_theme(self, event: OptionList.OptionHighlighted) -> None:
        self.app.theme = str(event.option.prompt)

    def action_cancel(self) -> None:
        self.app.theme = self.initial_theme
        self.dismiss(None)


class InformationScreen(ModalScreen[None]):
    """Scrollable command output displayed without leaving the application."""

    BINDINGS = [
        Binding("escape,q", "close", "Close", priority=True),
        Binding("up", "scroll_up", "Scroll up", priority=True),
        Binding("down", "scroll_down", "Scroll down", priority=True),
        Binding("home", "scroll_home", "Scroll to top", priority=True),
        Binding("end", "scroll_end", "Scroll to bottom", priority=True),
        Binding("pageup", "page_up", "Page up", priority=True),
        Binding("pagedown", "page_down", "Page down", priority=True),
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

    def _scroll_view(self) -> VerticalScroll:
        return self.query_one("#information-body", VerticalScroll)

    def action_scroll_up(self) -> None:
        self._scroll_view().action_scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll_view().action_scroll_down()

    def action_scroll_home(self) -> None:
        self._scroll_view().action_scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll_view().action_scroll_end()

    def action_page_up(self) -> None:
        self._scroll_view().action_page_up()

    def action_page_down(self) -> None:
        self._scroll_view().action_page_down()


def _markdown_cell(value: object) -> str:
    """Escape a value for a compact Markdown table cell."""
    return str(value).replace("`", "\\`").replace("|", "\\|").replace("\n", " ")


def _tool_markdown(tool: ToolDetails) -> str:
    """Render expanded tool metadata inspired by the legacy Rich report."""
    return "\n\n".join(
        filter(
            None,
            (
                tool.description,
                "\n".join([
                    "| Setting | Value |",
                    "|---|---|",
                    *(
                        [
                            "| Source | "
                            f"`MCP: {_markdown_cell(tool.mcp_server)}` |"
                        ]
                        if tool.mcp_server
                        else []
                    ),
                    f"| Class | `{_markdown_cell(tool.class_name)}` |",
                    f"| Args schema | `{_markdown_cell(tool.schema_name)}` |",
                    "| Return directly | "
                    f"`{_markdown_cell(tool.return_direct)}` |",
                ]),
                (
                    "\n".join([
                        "### Arguments",
                        "",
                        "| Name | Type | Required | Description |",
                        "|---|---|---|---|",
                        *(
                            "| "
                            f"`{_markdown_cell(argument.name)}` | "
                            f"`{_markdown_cell(argument.type_name)}` | "
                            f"{'yes' if argument.required else 'no'} | "
                            f"{_markdown_cell(argument.description)} |"
                            for argument in tool.arguments
                        ),
                    ])
                    if tool.arguments
                    else ""
                ),
            ),
        )
    )


class AgentsScreen(InformationScreen):
    """Tabbed descriptions and configured-tool details for all agents."""

    def __init__(self, agents: tuple[AgentDetails, ...]) -> None:
        super().__init__("Agents", "")
        self.agents = agents

    def compose(self) -> ComposeResult:
        with Vertical(id="information"):
            yield Static(self.screen_title, id="information-title")
            with TabbedContent(id="agents-tabs"):
                for index, agent in enumerate(self.agents):
                    with TabPane(f"#{agent.name}", id=f"agent-tab-{index}"):
                        with VerticalScroll(classes="agent-details"):
                            yield Markdown(agent.description)
                            if agent.config:
                                yield Markdown(
                                    "\n".join([
                                        "### Configuration",
                                        "",
                                        "| Option | Value |",
                                        "|---|---|",
                                        *(
                                            f"| `{_markdown_cell(key)}` | "
                                            f"`{_markdown_cell(value)}` |"
                                            for key, value in agent.config
                                        ),
                                    ])
                                )
                            yield Static(
                                "Configured tools", classes="agent-tools-title"
                            )
                            if agent.tool_error:
                                yield Static(
                                    "Unable to load tools: " + agent.tool_error,
                                    classes="agent-tools-error",
                                )
                            elif not agent.tools:
                                yield Static(
                                    "No configured tools.",
                                    classes="agent-tools-empty",
                                )
                            for tool in agent.tools:
                                with Collapsible(
                                    title=(
                                        f"{tool.name} (mcp: {tool.mcp_server})"
                                        if tool.mcp_server
                                        else tool.name
                                    ),
                                    collapsed=True,
                                    classes="agent-tool",
                                ):
                                    yield Markdown(_tool_markdown(tool))

    def _scroll_view(self) -> VerticalScroll:
        tabs = self.query_one(TabbedContent)
        return self.query_one(f"#{tabs.active} .agent-details", VerticalScroll)


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
            relative = workspace.relative_to(Path.home())
            self.workspace_text = str(Path("~") / relative)
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
