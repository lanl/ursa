# ruff: noqa: TID251

"""Versioned JSON presentation state; never serialize Textual widget internals."""

from __future__ import annotations

from typing import Any

from rich.traceback import Frame, Stack, Trace

from ursa.cli.tui.event_cards import (
    AgentEventCard,
    ArtifactCard,
    EditCard,
    EventCard,
    ExceptionCard,
    FileActivityCard,
    PlanCard,
    RunCommandCard,
    SearchEventCard,
    ToolCallCard,
)

COMMON = ("key", "label", "lines", "details", "done", "content_ref")
FIELDS = {
    EventCard: (),
    AgentEventCard: (),
    SearchEventCard: (),
    ArtifactCard: ("artifacts",),
    EditCard: (
        "path",
        "show_heading",
        "additions",
        "deletions",
        "diff",
        "outcome",
    ),
    FileActivityCard: ("files",),
    PlanCard: ("revision", "steps", "state", "review_reason"),
    RunCommandCard: (
        "command",
        "completed",
        "multi_command",
        "_full_output",
        "returncode",
        "execution_failed",
        "safety_failed",
        "force_compact",
        "safety_status",
        "safety_reason",
    ),
    ToolCallCard: (
        "tool",
        "tool_input",
        "output",
        "structured_output",
        "has_structured_output",
        "completed",
        "failed",
    ),
    ExceptionCard: (),
}
TYPES = {cls.__name__: cls for cls in FIELDS}


def encode_card(card: EventCard) -> dict[str, Any]:
    state = {name: getattr(card, name) for name in COMMON + FIELDS[type(card)]}
    if isinstance(card, FileActivityCard):
        state["outcomes"] = [
            [*key, *value] for key, value in card.outcomes.items()
        ]
    if isinstance(card, ExceptionCard):
        state["trace"] = [
            {
                "exc_type": stack.exc_type,
                "exc_value": stack.exc_value,
                "is_cause": stack.is_cause,
                "notes": stack.notes,
                "frames": [
                    {
                        "filename": frame.filename,
                        "lineno": frame.lineno,
                        "name": frame.name,
                        "line": frame.line,
                    }
                    for frame in stack.frames
                ],
            }
            for stack in card.rich_traceback.trace.stacks
        ]
    result = {"version": 1, "type": type(card).__name__, "state": state}
    if card.bundle is not None:
        result["bundle"] = card.bundle
    return result


def decode_card(
    data: dict[str, Any], *, interrupted: bool = False
) -> EventCard:
    """Construct a model or a viewport widget without replaying its events."""
    state = data.get("state", {})
    key = str(state.get("key", "unknown"))
    cls = TYPES.get(data.get("type")) if data.get("version") == 1 else None
    if cls is None:
        card = EventCard(key, "Unsupported card")
        card.add(str(data.get("type", "unknown")))
        return card
    if cls is EditCard:
        card = EditCard(state["path"], "", "")
    elif cls is FileActivityCard:
        card = FileActivityCard(key)
    elif cls is RunCommandCard:
        card = RunCommandCard(key, state["command"])
    elif cls is ToolCallCard:
        card = ToolCallCard(key, state["tool"], state["tool_input"])
    elif cls is PlanCard:
        card = PlanCard(key, state["revision"])
    elif cls is ArtifactCard:
        card = ArtifactCard(key, state["artifacts"])
    elif cls is ExceptionCard:
        # The durable traceback is text, never a pickled exception/object graph.
        card = ExceptionCard(key, RuntimeError(""), "")
        card.rich_traceback.trace = Trace(
            stacks=[
                Stack(**{
                    **stack,
                    "frames": [Frame(**frame) for frame in stack["frames"]],
                })
                for stack in state.get("trace", [])
            ]
        )
    elif cls in (AgentEventCard, SearchEventCard):
        card = cls(key, "")
    else:
        card = EventCard(key, state["label"])
    apply_card(card, data)
    if interrupted:
        finish_card(card)
    return card


def finish_card(card: EventCard, *, succeeded: bool = False) -> None:
    """Close pending work consistently on turn completion and session resume."""
    if isinstance(card, RunCommandCard) and not card.completed:
        card.execution_failed = True
        card.complete(
            "Interrupted before completion", execution_confirmed=False
        )
        card.multi_command = True
    elif isinstance(card, ToolCallCard) and not card.completed:
        card.complete("Interrupted before completion", failed=True)
    elif isinstance(card, PlanCard):
        card.finish_pending_review(succeeded=succeeded)
    elif not card.done:
        card.mark_done()


def apply_card(card: EventCard, data: dict[str, Any]) -> None:
    state = data.get("state", {})
    card.bundle = data.get("bundle")
    for name in COMMON + FIELDS[type(card)]:
        if name in state:
            setattr(card, name, state[name])
    if isinstance(card, FileActivityCard):
        card.files = {
            operation: {path: tuple(counts) for path, counts in files.items()}
            for operation, files in state["files"].items()
        }
        card.outcomes = {
            (operation, path): (status, detail)
            for operation, path, status, detail in state.get("outcomes", [])
        }
    card.set_class(card.done, "summary-done")
    if card.is_mounted:
        card.refresh_content()
        card.set_expanded(card.expanded)
