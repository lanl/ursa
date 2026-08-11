# ruff: noqa: TID251

"""File access and editing event cards."""

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Markdown, Static

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
