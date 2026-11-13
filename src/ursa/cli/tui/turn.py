"""Presentation container for the visible fragment of a persisted turn."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ursa.cli.tui.widgets import ActivityIndicator


class Turn(Static):
    def __init__(self) -> None:
        super().__init__(classes="turn")

    def compose(self) -> ComposeResult:
        yield Vertical(classes="events hidden")
        yield ActivityIndicator()
        yield Static("", classes="turn-end-marker")
