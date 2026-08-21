# ruff: noqa: TID251

"""File access and editing event cards."""

from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from ursa.cli.event_cards.base import EventCard


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
        yield Static("Click to expand", classes="event-expand-hint")

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
    def __init__(
        self,
        path: str,
        old: str,
        new: str,
        *,
        show_heading: bool = True,
    ) -> None:
        super().__init__(f"edit:{path}", f"✎ {path}")
        self.path = path
        self.show_heading = show_heading
        self.additions, self.deletions, self.diff = self._diff(old, new)
        self.expanded = False

    def compose(self) -> ComposeResult:
        if self.show_heading:
            yield Static("✍️ Editing", classes="edit-group-title")
        with Horizontal(classes="edit-header"):
            yield Static(f"- {Path(self.path).name}", classes="edit-title")
            counts = Text(f"+{self.additions}", style="green")
            counts.append(f" -{self.deletions}", style="red")
            yield Static(counts, classes="edit-counts")
            yield Static("Click to expand", classes="edit-hint")
        yield Static("", classes="edit-outcome hidden")
        yield Static("", classes="event-summary edit-diff hidden")

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
        diff = self.query_one(".edit-diff", Static)
        if not self.expanded:
            diff.add_class("hidden")
            return
        diff.remove_class("hidden")
        diff.update(
            Syntax(
                self.diff,
                "diff",
                word_wrap=True,
                background_color=diff.styles.background.hex,
            )
        )

    def set_outcome(self, state: str, detail: str) -> None:
        outcome = self.query_one(".edit-outcome", Static)
        icon = "✖" if state == "failed" else "⚠"
        style = "red" if state == "failed" else "yellow"
        outcome.update(Text(f"{icon} {detail}", style=style))
        outcome.remove_class("hidden")

    def _update_expand_hint(self) -> None:
        if self.is_mounted:
            self.query_one(".edit-hint", Static).update(
                "Click to collapse" if self.expanded else "Click to expand"
            )
