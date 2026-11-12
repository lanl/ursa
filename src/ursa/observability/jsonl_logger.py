"""JSON Lines lifecycle logging for LangChain callbacks."""

from __future__ import annotations

import json
import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TextIO

from langchain_core.callbacks import AsyncCallbackHandler

from ursa.observability.timing import extract_token_usage


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    """Convert callback payloads to values accepted by ``json.dumps``."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseException):
        return {"error_type": type(value).__name__, "message": str(value)}

    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return f"<recursive {type(value).__name__}>"
    seen.add(identity)
    try:
        if isinstance(value, dict):
            return {
                str(key): _json_safe(item, seen) for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item, seen) for item in value]
        if is_dataclass(value):
            return {
                field.name: _json_safe(getattr(value, field.name), seen)
                for field in fields(value)
            }
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="python")
            except TypeError:
                try:
                    dumped = model_dump()
                except Exception:
                    return _safe_repr(value)
            except Exception:
                return _safe_repr(value)
            return _json_safe(dumped, seen)
        return _safe_repr(value)
    except Exception:
        return _safe_repr(value)
    finally:
        seen.remove(identity)


def _record_name(serialized: Any, default: str) -> str:
    if isinstance(serialized, str):
        return serialized
    if not isinstance(serialized, dict):
        return default
    value = serialized.get("name") or serialized.get("id") or default
    if isinstance(value, dict):
        value = value.get("name") or value.get("id") or default
    return (
        "/".join(map(str, value))
        if isinstance(value, (list, tuple))
        else str(value)
    )


def _start_callback(
    record_type: str, default_name: str, *, prefer_model_name: bool = False
):
    """Build a LangChain start callback for named lifecycle records."""

    async def callback(self, serialized, value, **arguments) -> None:
        metadata = arguments.get("metadata") or {}
        name = (
            metadata.get("model")
            if prefer_model_name and isinstance(metadata, dict)
            else arguments.get("name")
        ) or _record_name(serialized, default_name)
        self._lifecycle_start(record_type, str(name), value, arguments)

    return callback


def _terminal_callback(
    record_type: str, event: str, *, include_usage: bool = False
):
    """Build a LangChain end or error callback for lifecycle records."""

    async def callback(self, value, **arguments) -> None:
        fields = {"usage": extract_token_usage(value)} if include_usage else {}
        self._lifecycle_terminal(record_type, event, value, arguments, **fields)

    return callback


def _agent_callback(event: str, field: str):
    """Build a simple agent lifecycle callback."""

    async def callback(self, value, **arguments) -> None:
        self._write(
            "agent",
            event,
            **{field: value},
            **self._lifecycle_context(arguments),
        )

    return callback


class JSONLLogEventHandler(AsyncCallbackHandler):
    """Write LangChain lifecycle events as one JSON object per line."""

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self.stream = stream
        self._names: dict[tuple[str, str], str] = {}

    def _write(
        self,
        record_type: str,
        event: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        name: str | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        record: dict[str, Any] = {
            "type": record_type,
            "event": event,
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "run_id": str(run_id),
            "parent_run_id": (
                str(parent_run_id) if parent_run_id is not None else None
            ),
            "tags": tags or [],
            "metadata": metadata or {},
        }
        if name is not None:
            record["name"] = name
        record.update(fields)

        line = json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        self.stream.write(line)
        self.stream.flush()

    @staticmethod
    def _lifecycle_context(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            key: arguments.get(key)
            for key in ("run_id", "parent_run_id", "tags", "metadata")
        }

    def _lifecycle_start(
        self, record_type: str, name: str, value: Any, arguments: dict[str, Any]
    ) -> None:
        self._names[(record_type, str(arguments["run_id"]))] = name
        self._write(
            record_type,
            "start",
            name=name,
            input=value,
            **self._lifecycle_context(arguments),
        )

    def _lifecycle_terminal(
        self,
        record_type: str,
        event: str,
        value: Any,
        arguments: dict[str, Any],
        **fields: Any,
    ) -> None:
        run_id = arguments["run_id"]
        self._write(
            record_type,
            event,
            name=self._names.pop((record_type, str(run_id)), None),
            **({"output" if event == "end" else "error": value} | fields),
            **self._lifecycle_context(arguments),
        )

    on_chain_start = _start_callback("chain", "chain")
    on_chain_end = _terminal_callback("chain", "end")
    on_chain_error = _terminal_callback("chain", "error")
    on_chat_model_start = _start_callback(
        "chat", "chat", prefer_model_name=True
    )
    on_llm_start = _start_callback("chat", "chat", prefer_model_name=True)
    on_llm_end = _terminal_callback("chat", "end", include_usage=True)
    on_llm_error = _terminal_callback("chat", "error")

    async def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        input_value: Any = inputs
        if inputs is None:
            try:
                input_value = json.loads(input_str) if input_str else None
            except json.JSONDecodeError:
                input_value = {"raw_input": input_str}
        self._lifecycle_start(
            "tool", _record_name(serialized, "tool"), input_value, locals()
        )

    on_tool_end = _terminal_callback("tool", "end")
    on_tool_error = _terminal_callback("tool", "error")
    on_agent_action = _agent_callback("action", "input")
    on_agent_finish = _agent_callback("end", "output")

    async def on_custom_event(
        self,
        name,
        data,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        from ursa.util.events import DEFAULT_EVENT_NAME

        if (
            name != DEFAULT_EVENT_NAME
            or not isinstance(data, dict)
            or not (agent_name := data.get("agent"))
        ):
            return
        self._write(
            "agent",
            str(data.get("phase") or "progress"),
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=str(agent_name),
            tags=tags,
            metadata=metadata,
            data=data,
        )
