"""LangChain callback translation for the Textual conversation view."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from time import monotonic
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage

from ursa.cli.tui.helpers import (
    FILE_TOOLS,
    _reasoning_trace,
    _token_usage_breakdown,
)
from ursa.cli.tui.message_store import COORDINATION_TOOLS
from ursa.cli.tui.turn import Turn
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.util.model_metadata import refresh_ollama_context

if TYPE_CHECKING:
    from ursa.cli.tui.app import UrsaTextualApp


class TextualEventHandler(AsyncCallbackHandler):
    """Translate LangChain callbacks into live Textual turn updates."""

    def __init__(self, app: UrsaTextualApp, turn: Turn) -> None:
        self.app = app
        self.turn = turn
        self.tools: dict[Any, dict[str, Any]] = {}
        self.delegated_turns: dict[str, Turn] = {}

    async def _on_ui(self, method, *args, **kwargs):
        """Dispatch every callback through the same Textual thread boundary."""
        if self.app.is_ui_thread:
            return await method(*args, **kwargs)
        return self.app.call_from_thread(method, *args, **kwargs)

    @staticmethod
    def _agent_path(metadata: Any) -> str | None:
        if not isinstance(metadata, Mapping):
            return None
        path = metadata.get("ursa_message_recipient")
        return str(path) if path else None

    async def _emit(
        self,
        data: dict[str, Any],
        *,
        metadata: Any = None,
    ) -> None:
        """Apply callback data on Textual's event-loop thread."""
        data.setdefault("_received_at", monotonic())
        agent_path = data.get("_agent_path") or self._agent_path(metadata)
        if agent_path:
            data["_agent_path"] = agent_path
        await self._on_ui(self.app.add_agent_event, self.turn, agent_path, data)

    async def _update_activity(
        self,
        message: str,
        *,
        metadata: Any = None,
    ) -> None:
        agent_path = self._agent_path(metadata)
        await self._on_ui(
            self.app.update_agent_activity, self.turn, agent_path, message
        )

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any = None,
        metadata: Any = None,
        **_: Any,
    ) -> None:
        if name == DEFAULT_EVENT_NAME and isinstance(data, dict):
            data = dict(data)
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
                        await self._emit(data, metadata=metadata)
                    return
                if any(
                    pending.get("tool") == tool
                    for pending in self.tools.values()
                ):
                    return
            await self._emit(data, metadata=metadata)

    async def on_llm_start(
        self, *_: Any, metadata: Any = None, **__: Any
    ) -> None:
        await self._update_activity("Thinking…", metadata=metadata)

    on_chat_model_start = on_llm_start

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any = None,
        metadata: Any = None,
        **_: Any,
    ) -> None:
        # Ordinary answer tokens are intentionally ignored. Providers that
        # publish reasoning summaries place them in explicit reasoning or
        # thinking fields on the chunk.
        if trace := _reasoning_trace(chunk):
            await self._update_activity(f"Thinking: {trace}", metadata=metadata)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        inputs: dict[str, Any] | None = None,
        tool_call_id: Any = None,
        metadata: Any = None,
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
            if tool_call_id in (None, ""):
                raise ValueError("run_command callback missing tool_call_id")
            data["tool_call_id"] = str(tool_call_id)
        if agent_path := self._agent_path(metadata):
            data["_agent_path"] = agent_path
        # Coordination inputs can contain large message bodies. They are
        # persisted by the card, so retain only routing keys for the end event.
        self.tools[run_id] = (
            {
                key: value
                for key, value in data.items()
                if key in {"tool", "_run_id", "_agent_path"}
            }
            if data["tool"] in COORDINATION_TOOLS
            else data
        )
        await self._emit(data, metadata=metadata)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        metadata: Any = None,
        **_: Any,
    ) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        if agent_path := self._agent_path(metadata):
            data.setdefault("_agent_path", agent_path)
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
        self,
        error: BaseException,
        *,
        run_id: Any,
        metadata: Any = None,
        **_: Any,
    ) -> None:
        data = self.tools.pop(run_id, {"tool": "tool"})
        if agent_path := self._agent_path(metadata):
            data.setdefault("_agent_path", agent_path)
        await self._emit({
            **data,
            "phase": "error",
            "error": str(error),
            "result": str(error),
        })

    async def on_llm_end(
        self, response: Any, *, metadata: Any = None, **_: Any
    ) -> None:
        model = getattr(self.app.runtime, "model", None)
        if model is not None:
            await asyncio.to_thread(refresh_ollama_context, model)
        usage = _token_usage_breakdown(response)
        await self._on_ui(
            self.app.add_agent_tokens,
            self.turn,
            self._agent_path(metadata),
            usage,
        )

    async def on_agent_message_start(
        self,
        agent_path: str,
        *,
        message_id: str | None = None,
    ) -> None:
        """Start the progress view for a delegated hierarchy message."""
        turn = await self._on_ui(
            self.app.start_agent_activity, agent_path, message_id=message_id
        )
        if message_id is not None:
            self.delegated_turns[message_id] = turn

    async def on_agent_message_end(
        self,
        agent_path: str,
        *,
        succeeded: bool,
        message_id: str | None = None,
    ) -> None:
        """Finish the progress view for a delegated hierarchy message."""
        turn = (
            self.delegated_turns.pop(message_id, None)
            if message_id is not None
            else None
        )
        await self._on_ui(
            self.app.finish_agent_activity,
            self.turn,
            agent_path,
            succeeded=succeeded,
            turn=turn,
        )

    async def on_hierarchy_message(
        self,
        sender: str,
        recipients: list[str] | tuple[str, ...],
        message: str,
    ) -> None:
        """Retain one user/agent message in the session-wide log."""
        await self._on_ui(
            self.app.record_hierarchy_message, sender, recipients, message
        )
