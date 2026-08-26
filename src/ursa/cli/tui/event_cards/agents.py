"""Agent progress event cards."""

from collections.abc import Mapping
from typing import Any

from ursa.cli.tui.event_cards.base import EventCard
from ursa.cli.tui.event_cards.files import EditCard
from ursa.cli.tui.helpers import AGENT_LABELS


class AgentEventCard(EventCard):
    """Specialized live summary for non-file agent progress."""

    def __init__(self, key: str, agent: str) -> None:
        icon, label = AGENT_LABELS.get(agent, ("◌", agent or "Agent"))
        super().__init__(key, f"{icon} {label}")

    @staticmethod
    def _stage_icon(agent: str, stage: str, payload: Mapping[str, Any]) -> str:
        if agent in {"PlanningAgent", "planner"}:
            if stage == "reflect_result":
                return "✅" if payload.get("approved") else "🔁"
            return {"generate": "📐", "generate_result": "🗺️"}.get(stage, "📋")
        if agent in {"HypothesizerAgent", "hypothesizer"}:
            return {
                "generate": "✨",
                "generate_result": "💡",
                "critique": "🔬",
                "critique_result": "🧪",
                "competitor": "🧭",
                "competitor_result": "🗣️",
                "finalize": "🛠️",
                "finalize_result": "⭐",
                "summarize": "📝",
                "summarize_result": "📚",
            }.get(stage, "💡")
        if agent in {"LammpsAgent", "lammps"}:
            return {
                "author_input": "📝",
                "choose_potential": "🧲",
                "fix_input": "🛠️",
                "run": "▶",
                "run_result": ("✅" if payload.get("returncode") == 0 else "✖"),
                "summarize_potential": "🔬",
                "summarize_results": "📊",
            }.get(stage, "⚛️")
        return "⚙️"

    def update_event(self, payload: Mapping[str, Any]) -> None:
        agent = str(payload.get("agent") or "")
        message = str(payload.get("message") or payload.get("stage") or "Event")
        stage = str(payload.get("stage") or "")
        detail = payload.get("preview")
        if stage == "reflect_result":
            detail = payload.get("reason")
        elif stage == "choose_potential" and payload.get("phase") == "end":
            detail = "\n".join(
                filter(
                    None,
                    (
                        f"Potential: {payload.get('potential_id')}",
                        f"Index: {payload.get('chosen_index')}",
                        str(payload.get("rationale") or ""),
                    ),
                )
            )
        elif stage == "run" and payload.get("phase") == "error":
            detail = payload.get("error_output") or payload.get("error")
        elif stage == "fix_input" and (
            payload.get("old_code") is not None
            or payload.get("new_code") is not None
        ):
            detail = EditCard._diff(
                str(payload.get("old_code") or ""),
                str(payload.get("new_code") or ""),
            )[2]
        if output_path := payload.get("output_path"):
            output_detail = f"Output: {output_path}"
            detail = f"{detail}\n{output_detail}" if detail else output_detail
        icon = self._stage_icon(agent, stage, payload)
        self.add(f"{icon} {message}", str(detail) if detail else None)
