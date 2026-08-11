# ruff: noqa: TID251

"""Plan generation and review event cards."""

from collections.abc import Sequence
from typing import Any

from textual.app import ComposeResult
from textual.widgets import Markdown, Static

from ursa.cli.event_cards.base import EventCard
from ursa.cli.helpers import _plan_step_text, _truncate_middle


class PlanCard(EventCard):
    """A live draft/review timeline for one plan revision."""

    SPINNER_FRAMES = (".", "..", "...")

    def __init__(self, key: str, revision: int) -> None:
        super().__init__(key, "🗺️ Planning")
        self.revision = revision
        self.steps: list[str] = []
        self.state = "drafting"
        self.review_reason = ""
        self.expanded = False
        self._frame = 0
        self._spinner_timer = None

    def compose(self) -> ComposeResult:
        yield Markdown("", classes="event-summary")
        yield Static("Click to expand", classes="plan-expand-hint")

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.3, self._advance_spinner)
        self.refresh_content()

    def on_resize(self) -> None:
        self.refresh_content()

    def _advance_spinner(self) -> None:
        if self.state not in {"drafting", "reviewing"}:
            if self._spinner_timer is not None:
                self._spinner_timer.pause()
            return
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        self.refresh_content()

    def _resume_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.resume()

    def set_drafting(self) -> None:
        self.state = "drafting"
        self._resume_spinner()
        self.refresh_content()

    def set_plan(self, steps: Sequence[Any]) -> None:
        self.steps = [
            _plan_step_text(index, step) for index, step in enumerate(steps, 1)
        ]
        self.state = "reviewing"
        self._resume_spinner()
        self.refresh_content()

    def set_reviewing(self) -> None:
        self.state = "reviewing"
        self._resume_spinner()
        self.refresh_content()

    def finish_review(self, approved: bool, reason: str = "") -> None:
        self.state = "complete" if approved else "revision_needed"
        self.review_reason = "" if approved else reason.strip()
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self.refresh_content()

    def finish_pending_review(self, *, succeeded: bool) -> None:
        if self.state != "reviewing":
            return
        if succeeded:
            self.finish_review(True)
            return
        self.state = "revision_needed"
        self.review_reason = "Planning stopped before review completed."
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        self.refresh_content()

    def _step_width(self) -> int:
        """Use the complete rendered row, minus Markdown's list indentation."""
        markdown_width = self.query_one(Markdown).content_size.width
        content_width = markdown_width or max(0, self.content_size.width - 4)
        return max(40, content_width - 4)

    def refresh_content(self) -> None:
        if not self.is_mounted:
            return
        if self.expanded or len(self.steps) <= 4:
            visible = self.steps
        else:
            hidden = len(self.steps) - 4
            visible = [
                self.steps[0],
                self.steps[1],
                f"_… {hidden} middle step{'s' if hidden != 1 else ''} hidden …_",
                self.steps[-2],
                self.steps[-1],
            ]
        if not self.expanded:
            width = self._step_width()
            visible = [_truncate_middle(step, width) for step in visible]

        body = [f"**{self.label}**  "]
        status_indent = " "
        if not self.steps:
            spinner = self.SPINNER_FRAMES[self._frame]
            body.append(f"{status_indent}✍️ Drafting Plan{spinner}")
        else:
            plan_label = (
                "Initial Plan" if self.revision == 1 else "Revised Plan"
            )
            body.append(f"{status_indent}📄 **{plan_label}**")
            body.append("")
            for step in visible:
                if "middle step" in step:
                    body.extend(["", f"{status_indent}{step}", ""])
                else:
                    body.append(step)
            body.append("")
            if self.state == "reviewing":
                spinner = self.SPINNER_FRAMES[self._frame]
                body.append(f"{status_indent}📋 Reviewing{spinner}")
            elif self.state == "revision_needed":
                body.append(f"{status_indent}❌ Plan needs another revision")
            elif self.state == "complete":
                body.append(f"{status_indent}✅ 📋 Plan is complete")
            if self.expanded and self.review_reason:
                body.extend([
                    "",
                    f"{status_indent}**Revision feedback**",
                    *(f"> {line}" for line in self.review_reason.splitlines()),
                ])

        expandable = bool(
            self.steps and not self.expanded and len(self.steps) > 4
        )
        self.query_one(".plan-expand-hint").set_class(not expandable, "hidden")
        self.query_one(Markdown).update("\n".join(body))
