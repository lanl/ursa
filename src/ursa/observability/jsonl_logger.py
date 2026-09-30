"""JSON Lines lifecycle logging for LangChain callbacks."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import fields, is_dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

from langchain_core.callbacks import AsyncCallbackHandler


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
        return {
            "error_type": value.__class__.__name__,
            "message": str(value),
        }

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


def _coerce_usage_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)

    for method_name in ("model_dump", "dict", "to_dict", "_asdict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            dumped = method()
        except Exception:
            continue
        if isinstance(dumped, dict):
            return dict(dumped)

    data: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cached_input_tokens",
        "prompt_cache_hits",
        "cache_read_input_tokens",
        "completion_tokens_details",
        "output_tokens_details",
        "output_token_details",
        "prompt_tokens_details",
        "input_tokens_details",
        "input_token_details",
    ):
        if hasattr(value, key):
            data[key] = getattr(value, key)
    if data:
        return data

    if isinstance(value, str):
        return {
            match.group(1): match.group(2)
            for match in re.finditer(r"(\w+)=([0-9]+(?:\.[0-9]+)?)", value)
        }
    return {}


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not number.is_finite() or number < 0 or number != number.to_integral():
        return None
    return int(number)


def _detail_dict(value: Any) -> dict[str, Any]:
    return _coerce_usage_object(value)


def _reasoning_tokens(data: dict[str, Any]) -> int | None:
    candidates = [
        data.get("reasoning_tokens"),
        _detail_dict(data.get("completion_tokens_details")).get(
            "reasoning_tokens"
        ),
        _detail_dict(data.get("output_tokens_details")).get("reasoning_tokens"),
        _detail_dict(data.get("output_tokens_details")).get("thinking_tokens"),
    ]
    for key, value in _detail_dict(data.get("output_token_details")).items():
        if key == "reasoning" or key.endswith("_reasoning"):
            candidates.append(value)
    values = [
        value for item in candidates if (value := _to_int(item)) is not None
    ]
    return max(values, default=None)


def _cached_tokens(data: dict[str, Any]) -> int | None:
    candidates = [
        data.get("cached_tokens"),
        data.get("cached_input_tokens"),
        _detail_dict(data.get("prompt_tokens_details")).get("cached_tokens"),
        data.get("prompt_cache_hits"),
        data.get("cache_read_input_tokens"),
        _detail_dict(data.get("input_tokens_details")).get("cached_tokens"),
    ]
    for key, value in _detail_dict(data.get("input_token_details")).items():
        if key == "cache_read" or key.endswith("_cache_read"):
            candidates.append(value)
    values = [
        value for item in candidates if (value := _to_int(item)) is not None
    ]
    return max(values, default=None)


def _first_token_value(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _to_int(data.get(key))
        if value is not None:
            return value
    return None


def _normalize_usage(value: Any) -> dict[str, int]:
    data = _coerce_usage_object(value)
    values = {
        "input_tokens": _first_token_value(
            data, "input_tokens", "prompt_tokens"
        ),
        "output_tokens": _first_token_value(
            data, "output_tokens", "completion_tokens"
        ),
        "reasoning_tokens": _reasoning_tokens(data),
        "cached_tokens": _cached_tokens(data),
    }
    return {key: value for key, value in values.items() if value is not None}


def _sum_usage(sources: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for source in sources:
        usage = _normalize_usage(source)
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return total


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract usage once, preferring LangChain's normalized metadata."""
    selected_usage: list[dict[str, Any]] = []
    usage_metadata: list[dict[str, Any]] = []
    response_metadata_usage: list[dict[str, Any]] = []
    llm_output_usage: list[dict[str, Any]] = []

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        raw = llm_output.get("token_usage") or llm_output.get("usage")
        if usage := _coerce_usage_object(raw):
            llm_output_usage.append(usage)

    for generation_group in getattr(response, "generations", None) or []:
        generations = (
            generation_group
            if isinstance(generation_group, (list, tuple))
            else [generation_group]
        )
        for generation in generations:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            normalized = _coerce_usage_object(
                getattr(message, "usage_metadata", None)
            )
            if normalized:
                usage_metadata.append(normalized)
            response_metadata = getattr(message, "response_metadata", None)
            raw_usage: dict[str, Any] = {}
            if isinstance(response_metadata, dict):
                raw = response_metadata.get(
                    "token_usage"
                ) or response_metadata.get("usage")
                raw_usage = _coerce_usage_object(raw)
                if raw_usage:
                    response_metadata_usage.append(raw_usage)
            if normalized or raw_usage:
                selected_usage.append(normalized or raw_usage)

    selected = selected_usage or llm_output_usage
    total = _sum_usage(selected)

    # Providers sometimes put normalized totals on the message and cache or
    # reasoning details in a lower-priority raw metadata carrier. Enrich those
    # fields without adding duplicate token totals.
    for sources in (
        usage_metadata,
        response_metadata_usage,
        llm_output_usage,
    ):
        extras = _sum_usage(sources)
        for key in ("reasoning_tokens", "cached_tokens"):
            if key in extras:
                total[key] = max(total.get(key, 0), extras[key])
    return total


def _record_name(serialized: Any, default: str) -> str:
    if isinstance(serialized, str):
        return serialized
    if not isinstance(serialized, dict):
        return default
    name = serialized.get("name")
    if name:
        return str(name)
    identifier = serialized.get("id")
    if isinstance(identifier, dict):
        return str(identifier.get("name") or identifier.get("id") or default)
    if isinstance(identifier, (list, tuple)):
        return "/".join(map(str, identifier))
    return str(identifier or default)


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

    def _start_name(self, record_type: str, run_id: Any, name: str) -> str:
        self._names[(record_type, str(run_id))] = name
        return name

    def _end_name(self, record_type: str, run_id: Any) -> str | None:
        return self._names.pop((record_type, str(run_id)), None)

    async def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        callback_name = kwargs.get("name")
        name = self._start_name(
            "chain",
            run_id,
            str(callback_name or _record_name(serialized, "chain")),
        )
        self._write(
            "chain",
            "start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            tags=tags,
            metadata=metadata,
            input=inputs,
        )

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain",
            "end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("chain", run_id),
            tags=tags,
            metadata=metadata,
            output=outputs,
        )

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain",
            "error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("chain", run_id),
            tags=tags,
            metadata=metadata,
            error=error,
        )

    async def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (metadata or {}).get("model") or _record_name(serialized, "chat")
        self._start_name("chat", run_id, str(name))
        self._write(
            "chat",
            "start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=str(name),
            tags=tags,
            metadata=metadata,
            input=messages,
        )

    async def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        await self.on_chat_model_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chat",
            "end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("chat", run_id),
            tags=tags,
            metadata=metadata,
            output=response,
            usage=_extract_usage(response),
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chat",
            "error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("chat", run_id),
            tags=tags,
            metadata=metadata,
            error=error,
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = self._start_name(
            "tool", run_id, _record_name(serialized, "tool")
        )
        input_value: Any = inputs
        if inputs is None:
            try:
                input_value = json.loads(input_str) if input_str else None
            except json.JSONDecodeError:
                input_value = {"raw_input": input_str}
        self._write(
            "tool",
            "start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            tags=tags,
            metadata=metadata,
            input=input_value,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool",
            "end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("tool", run_id),
            tags=tags,
            metadata=metadata,
            output=output,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "tool",
            "error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=self._end_name("tool", run_id),
            tags=tags,
            metadata=metadata,
            error=error,
        )

    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "agent",
            "action",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            input=action,
        )

    async def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "agent",
            "end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            output=finish,
        )

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from ursa.util.events import DEFAULT_EVENT_NAME

        if name != DEFAULT_EVENT_NAME or not isinstance(data, dict):
            return
        agent_name = data.get("agent")
        if not agent_name:
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
