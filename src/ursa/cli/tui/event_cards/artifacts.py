"""Structured artifact event cards."""

from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ursa.cli.tui.event_cards.base import EventCard
from ursa.util.rendering import render_event_artifacts


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
        yield Static("Click to expand", classes="event-expand-hint")

    def refresh_content(self) -> None:
        """Refresh without assuming the summary widget is Markdown."""
        if self.is_mounted:
            self.query_one(".event-summary", Static).update(
                render_event_artifacts(self.artifacts)
            )
