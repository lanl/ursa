"""Search progress event cards."""

from collections.abc import Mapping
from typing import Any

from ursa.cli.tui.event_cards.base import EventCard


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
