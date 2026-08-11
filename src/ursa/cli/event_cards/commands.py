# ruff: noqa: TID251

"""Command execution and safety-check event cards."""

from collections.abc import Mapping
from typing import Any

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from ursa.cli.event_cards.base import EventCard
from ursa.cli.widgets import ActivityIndicator


class CommandSafetyIndicator(ActivityIndicator):
    """Safety-check state for a single command invocation."""

    def __init__(self) -> None:
        super().__init__()
        self.status = "pending"

    def compose(self) -> ComposeResult:
        yield Static(self.FRAMES[0], classes="activity-spinner")
        yield Static("Running safety check", classes="activity-text")

    def passed(self) -> None:
        self.status = "passed"
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("✓")
        self.query_one(".activity-text", Static).update("Safety check passed")

    def failed(self, reason: str | None = None) -> None:
        self.status = "failed"
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("⚔️")
        self.query_one(".activity-text", Static).update(
            reason or "Safety check failed"
        )


class RunCommandCard(EventCard):
    """Progressively disclose one command, its safety check, and output."""

    def __init__(self, key: str, command: str) -> None:
        super().__init__(key, "run_command")
        self.command = command
        self.completed = False
        self.multi_command = False
        self.output_expanded = False
        self._full_output = ""
        self.returncode: int | None = None
        self.execution_failed = False
        self.safety_failed = False
        self.force_compact = False
        self._compact_frame = 0
        self._compact_timer = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="command-compact hidden"):
            yield Static(self.FRAMES[0], classes="command-compact-state")
            yield Static(
                self._collapsed_command(), classes="command-compact-text"
            )
        yield Static(
            self._command_syntax(collapsed=False), classes="command-source"
        )
        yield CommandSafetyIndicator()
        yield Static("", classes="command-output hidden")
        yield Static("Click to expand", classes="event-expand-hint")

    def on_mount(self) -> None:
        self._compact_timer = self.set_interval(
            0.08, self._advance_compact_spinner, pause=True
        )
        self._update_expand_hint()

    @property
    def FRAMES(self) -> tuple[str, ...]:
        return ActivityIndicator.FRAMES

    def _advance_compact_spinner(self) -> None:
        if self.safety_failed:
            self.query_one(".command-compact-state", Static).update("⚔️")
            return
        self.query_one(".command-compact-state", Static).update(
            self.FRAMES[self._compact_frame]
        )
        self._compact_frame = (self._compact_frame + 1) % len(self.FRAMES)

    @staticmethod
    def _preview_command(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= 20:
            return text
        omitted = len(lines) - 16
        return "\n".join([
            *lines[:8],
            f"… {omitted} lines omitted …",
            *lines[-8:],
        ])

    @staticmethod
    def _preview_output(text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= 10:
            return text
        omitted = len(lines) - 8
        return "\n".join([
            *lines[:4],
            f"… {omitted} lines omitted …",
            *lines[-4:],
        ])

    def _collapsed_command(self) -> str:
        lines = self.command.splitlines() or [self.command]
        command = lines[0]
        if len(lines) > 1:
            command += " …"
        if len(command) > 120:
            command = command[:119] + "…"
        return command

    def _command_syntax(
        self, *, collapsed: bool, expanded: bool = False
    ) -> Syntax:
        lines = self.command.splitlines() or [self.command]
        if expanded:
            command = "\n".join(lines)
        elif collapsed:
            command = self._collapsed_command()
        else:
            command = self._preview_command("\n".join(lines))
        return Syntax(command, "bash", word_wrap=True)

    def _render_command(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(".command-source", Static).update(
            self._command_syntax(
                collapsed=self.completed and not self.output_expanded,
                expanded=self.output_expanded,
            )
        )

    def update_event(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage") or "")
        phase = str(payload.get("phase") or "")
        if isinstance(payload.get("returncode"), int):
            self.returncode = payload["returncode"]
        if phase == "error" or payload.get("status") == "error":
            self.execution_failed = True
        if stage == "safety_check":
            safety = self.query_one(CommandSafetyIndicator)
            if payload.get("safe") is True:
                safety.passed()
            elif payload.get("safe") is False:
                self.safety_failed = True
                self.force_compact = True
                safety.failed(str(payload.get("reason") or "") or None)

        output = payload.get("result")
        if output is None and stage == "execute" and phase == "end":
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, list):
                contents = [
                    str(artifact.get("content"))
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                    and artifact.get("content") not in (None, "")
                ]
                if contents:
                    output = "\n".join(contents)
        if output is not None or phase == "error":
            self.complete(output)

    def complete(self, output: Any) -> None:
        self.completed = True
        if self._compact_timer is not None:
            self._compact_timer.pause()
        self.query_one(".command-compact-state", Static).update(
            self._completion_icon()
        )
        self._render_command()
        self._full_output = self._clean_output(output)
        if not self._full_output:
            self.force_compact = True
        self._render_output()
        self._update_visibility()

    def _completion_icon(self) -> str:
        if self.safety_failed:
            return "⚔️"
        if self.execution_failed or self.returncode != 0:
            return "✗"
        return "✓"

    def set_multi_command(self, multi_command: bool) -> None:
        self.multi_command = multi_command
        if self._compact_timer is not None:
            if multi_command and not self.completed and not self.safety_failed:
                self._advance_compact_spinner()
                self._compact_timer.resume()
            else:
                self._compact_timer.pause()
        self._update_visibility()

    def set_output_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.output_expanded = expanded
        self._render_command()
        self._render_output()
        self._update_visibility()
        self._update_expand_hint()

    def _render_output(self) -> None:
        if not self.completed:
            return
        output = (
            self._full_output
            if self.output_expanded
            else self._preview_output(self._full_output)
        )
        output = output or "(no output)"
        self.query_one(".command-output", Static).update(
            Syntax(output, "text", word_wrap=True)
        )

    def _update_visibility(self) -> None:
        if not self.is_mounted:
            return
        compact = self.multi_command and not self.output_expanded
        self.query_one(".command-compact").set_class(not compact, "hidden")
        self.query_one(".command-source").set_class(compact, "hidden")
        self.query_one(CommandSafetyIndicator).set_class(compact, "hidden")
        show_output = self.completed and not compact
        self.query_one(".command-output").set_class(not show_output, "hidden")

    @staticmethod
    def _clean_output(output: Any) -> str:
        text = str(output or "")
        if text.startswith("STDOUT:\n") and "\nSTDERR:\n" in text:
            stdout, stderr = text[len("STDOUT:\n") :].split("\nSTDERR:\n", 1)
            if stdout and stderr:
                return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            text = stdout or stderr
        return text.rstrip()

    def set_expanded(self, expanded: bool) -> None:
        self.set_output_expanded(expanded)
