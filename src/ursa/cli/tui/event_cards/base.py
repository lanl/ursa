# ruff: noqa: TID251

"""Base class for dynamically updated event cards."""

from rich.traceback import Traceback
from textual import events
from textual.app import ComposeResult
from textual.widgets import Markdown, Static


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
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self.refresh_content()
        self._update_expand_hint()

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
        self._update_expand_hint()

    def _update_expand_hint(self) -> None:
        if not self.is_mounted:
            return
        hints = list(self.query(".event-expand-hint"))
        if hints:
            hints[0].update(
                "Click to collapse" if self.expanded else "Click to expand"
            )

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


class ExceptionCard(EventCard):
    """An agent failure whose complete traceback is available on expansion."""

    def __init__(self, key: str, error: BaseException, traceback: str) -> None:
        super().__init__(key, "✖ Exception")
        self.add_class("exception-card")
        self.error = error
        self.rich_traceback = Traceback.from_exception(
            type(error),
            error,
            error.__traceback__,
            width=None,
            code_width=100,
            extra_lines=3,
            word_wrap=True,
            show_locals=False,
            max_frames=1000,
        )
        self.add(f"{type(error).__name__}: {error}", traceback)
        self.mark_done()

    def compose(self) -> ComposeResult:
        yield Markdown("", classes="event-summary")
        yield Static(
            self.rich_traceback,
            classes="exception-traceback hidden",
        )
        yield Static("", classes="event-card-done")
        yield Static("Click to expand", classes="event-expand-hint")

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(Markdown).update(f"**{self.label}**\n- {self.lines[0]}")
        self.query_one(".exception-traceback").set_class(
            not self.expanded, "hidden"
        )
