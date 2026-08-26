"""LangChain callback translation for the Textual conversation view."""

from __future__ import annotations

import json
from time import monotonic
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage

from ursa.cli.tui.helpers import (
    FILE_TOOLS,
    TokenUsage,
    _reasoning_trace,
    _token_usage_breakdown,
)
from ursa.cli.tui.turn import Turn
from ursa.util.events import DEFAULT_EVENT_NAME

if TYPE_CHECKING:
    from ursa.cli.tui.app import UrsaTextualApp


class TextualEventHandler(AsyncCallbackHandler):
    """Translate LangChain callbacks into live Textual turn updates."""

    def __init__(self, app: UrsaTextualApp, turn: Turn) -> None:
        self.app = app
        self.turn = turn
        self.tools: dict[Any, dict[str, Any]] = {}

    async def _emit(self, data: dict[str, Any]) -> None:
        """Apply callback data on Textual's event-loop thread."""
        data.setdefault("_received_at", monotonic())
        if self.app.is_ui_thread:
            await self.app.add_turn_event(self.turn, data)
        else:
            self.app.call_from_thread(
                self.app.add_turn_event,
                self.turn,
                data,
            )

    async def _update_activity(self, message: str) -> None:
        if self.app.is_ui_thread:
            self.turn.update_activity(message)
        else:
            self.app.call_from_thread(self.turn.update_activity, message)

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any = None,
        **_: Any,
    ) -> None:
        if name == DEFAULT_EVENT_NAME and isinstance(data, dict):
            data = dict(data)
            if data.get("tool") == "run_command" and run_id is not None:
                # LangChain assigns custom events emitted inside a tool to the
                # tool run. The query text is not a unique identifier when
                # identical commands execute concurrently.
                data["_command_id"] = str(run_id)
            tool = str(data.get("tool") or "")
            # The file tools publish their own structured range events while
            # LangChain also emits tool start/end callbacks. The callback is
            # the authoritative timeline event; rendering both produces a
            # second group when the range completes after later activity.
            if tool in FILE_TOOLS:
                phase = str(data.get("phase") or "")
                if phase in {"end", "error"}:
                    result = str(data.get("result") or data.get("error") or "")
                    if phase == "error" or result.casefold().startswith((
                        "failed",
                        "no changes made",
                    )):
                        await self._emit(data)
                    return
                if any(
                    pending.get("tool") == tool
                    for pending in self.tools.values()
                ):
                    return
            await self._emit(data)

    async def on_llm_start(self, *_: Any, **__: Any) -> None:
        await self._update_activity("Thinking…")

    async def on_chat_model_start(self, *_: Any, **__: Any) -> None:
        await self._update_activity("Thinking…")

    async def on_llm_new_token(
        self, token: str, *, chunk: Any = None, **_: Any
    ) -> None:
        # Ordinary answer tokens are intentionally ignored. Providers that
        # publish reasoning summaries place them in explicit reasoning or
        # thinking fields on the chunk.
        if trace := _reasoning_trace(chunk):
            await self._update_activity(trace)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        inputs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        data = dict(inputs or {})
        if not data and input_str:
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = {"input": input_str}
        data["tool"] = serialized.get("name", "tool")
        data["phase"] = "start"
        data["_run_id"] = str(run_id)
        if data["tool"] == "run_command":
            data["_command_id"] = data["_run_id"]
        self.tools[run_id] = data
        await self._emit(data)

    async def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        data = {**data, "phase": "end"}
        if isinstance(output, ToolMessage):
            data["result"] = output.content
            data["status"] = output.status
            data["tool_message"] = output
        else:
            data["result"] = output
        if data.get("tool") in FILE_TOOLS and Turn._file_outcome(data) is None:
            return
        await self._emit(data)

    async def on_tool_error(
        self, error: BaseException, *, run_id: Any, **_: Any
    ) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        if data.get("tool") in FILE_TOOLS:
            await self._emit({
                **data,
                "phase": "error",
                "error": str(error),
                "result": str(error),
            })
            return
        await self._emit({
            **data,
            "phase": "error",
            "error": str(error),
            "result": str(error),
        })

    async def on_llm_end(self, response: Any, **_: Any) -> None:
        usage = _token_usage_breakdown(response)
        if self.app.is_ui_thread:
            self._record_tokens(usage)
        else:
            self.app.call_from_thread(self._record_tokens, usage)

    def _record_tokens(self, usage: TokenUsage) -> None:
        self.turn.add_tokens(usage.total_tokens)
        self.app.add_tokens(usage)
