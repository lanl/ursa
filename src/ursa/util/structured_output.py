from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

_MISSING = object()


@dataclass
class StructuredOutputAttempt:
    """Diagnostic information for one structured-output attempt."""

    method: str
    repair_index: int | None
    error: str
    raw: Any = None
    parsing_error: Any = None


@dataclass
class StructuredOutputResult:
    """Normalized structured-output result."""

    parsed: Any
    raw: Any = None
    parsing_error: Any = None
    attempts: list[StructuredOutputAttempt] = field(default_factory=list)


class StructuredOutputError(RuntimeError):
    """Raised when a model cannot produce usable structured output."""

    def __init__(
        self,
        message: str,
        *,
        context: str = "structured output",
        attempts: Sequence[StructuredOutputAttempt] | None = None,
    ):
        super().__init__(message)
        self.context = context
        self.attempts = list(attempts or [])


def invoke_structured(
    llm: Any,
    schema: type[Any],
    input: Any,
    *,
    config: Any | None = None,
    include_raw: bool = False,
    methods: Iterable[str | None] = (None, "function_calling"),
    validator: Callable[[Any], Any] | None = None,
    fallback: Any = _MISSING,
    context: str = "structured output",
    repair: bool | int = False,
) -> Any | StructuredOutputResult:
    """Invoke an LLM with structured output and validate the parsed result.

    The helper treats both a direct ``None`` result and an ``include_raw=True``
    payload with ``parsed is None`` as failures. It tries each requested method,
    optionally performs repair retries, and then either returns a caller-provided
    fallback or raises ``StructuredOutputError``.

    Args:
        llm: Chat model exposing ``with_structured_output``.
        schema: Structured-output schema, preferably a Pydantic ``BaseModel``.
        input: Prompt/messages passed to ``invoke``.
        config: Optional LangChain runnable config.
        include_raw: If true, return ``StructuredOutputResult`` with raw payload.
        methods: Structured-output methods to try. ``None`` means provider default.
        validator: Optional callable for extra validation/normalization. It should
            return the validated parsed value or raise an exception.
        fallback: Optional fallback parsed value. If provided, it is validated and
            returned after all attempts fail.
        context: Human-readable context for diagnostics.
        repair: Optional repair attempts. ``True`` means 1, ``False`` means 0;
            integers request that many repair rounds.

    Returns:
        Parsed value when ``include_raw`` is false, otherwise ``StructuredOutputResult``.
    """

    normalized_methods = tuple(methods) or (None,)
    repair_attempts = _repair_count(repair)
    attempts: list[StructuredOutputAttempt] = []

    result = _try_methods(
        llm,
        schema,
        input,
        config=config,
        include_raw=include_raw,
        methods=normalized_methods,
        validator=validator,
        context=context,
        attempts=attempts,
        repair_index=None,
    )
    if result is not None:
        return result

    last_error = (
        attempts[-1].error if attempts else "unknown structured-output failure"
    )
    last_raw = attempts[-1].raw if attempts else None
    for repair_index in range(1, repair_attempts + 1):
        repair_input = _build_repair_input(
            input,
            schema,
            context=context,
            error=last_error,
            raw=last_raw,
            repair_index=repair_index,
            repair_attempts=repair_attempts,
        )
        result = _try_methods(
            llm,
            schema,
            repair_input,
            config=config,
            include_raw=include_raw,
            methods=normalized_methods,
            validator=validator,
            context=context,
            attempts=attempts,
            repair_index=repair_index,
        )
        if result is not None:
            return result
        last_error = attempts[-1].error if attempts else last_error
        last_raw = attempts[-1].raw if attempts else last_raw

    if fallback is not _MISSING:
        try:
            parsed = _validate_parsed(schema, fallback, validator=validator)
        except Exception as exc:
            attempts.append(
                StructuredOutputAttempt(
                    method="fallback",
                    repair_index=None,
                    error=f"Fallback did not validate: {exc}",
                    raw=fallback,
                )
            )
        else:
            fallback_result = StructuredOutputResult(
                parsed=parsed,
                raw=None,
                parsing_error=None,
                attempts=attempts,
            )
            return fallback_result if include_raw else parsed

    message = _format_error_message(context, attempts)
    raise StructuredOutputError(message, context=context, attempts=attempts)


def _try_methods(
    llm: Any,
    schema: type[Any],
    input: Any,
    *,
    config: Any | None,
    include_raw: bool,
    methods: Sequence[str | None],
    validator: Callable[[Any], Any] | None,
    context: str,
    attempts: list[StructuredOutputAttempt],
    repair_index: int | None,
) -> Any | StructuredOutputResult | None:
    for method in methods:
        method_name = method or "default"
        output = _MISSING
        try:
            kwargs: dict[str, Any] = {}
            if include_raw:
                kwargs["include_raw"] = True
            if method is not None:
                kwargs["method"] = method
            structured_llm = llm.with_structured_output(schema, **kwargs)
            invoke_kwargs = {"config": config} if config is not None else {}
            output = structured_llm.invoke(input, **invoke_kwargs)
            parsed, raw, parsing_error = _extract_output(output, include_raw)
            parsed = _validate_parsed(schema, parsed, validator=validator)
            result = StructuredOutputResult(
                parsed=parsed,
                raw=raw,
                parsing_error=parsing_error,
                attempts=attempts,
            )
            return result if include_raw else parsed
        except Exception as exc:
            raw = None
            parsing_error = None
            if output is not _MISSING:
                raw, parsing_error = _diagnostics_from_output(output)
            attempts.append(
                StructuredOutputAttempt(
                    method=method_name,
                    repair_index=repair_index,
                    error=f"{type(exc).__name__}: {exc}",
                    raw=raw,
                    parsing_error=parsing_error,
                )
            )
    return None


def _extract_output(output: Any, include_raw: bool) -> tuple[Any, Any, Any]:
    if include_raw:
        if not isinstance(output, dict):
            raise ValueError(
                "include_raw=True structured output did not return a mapping."
            )
        parsed = output.get("parsed")
        raw = output.get("raw")
        parsing_error = output.get("parsing_error")
        if parsed is None:
            detail = f" Parsing error: {parsing_error}" if parsing_error else ""
            raise ValueError(
                "Structured-output parser returned parsed=None." + detail
            )
        return parsed, raw, parsing_error

    if output is None:
        raise ValueError("Structured-output invocation returned None.")
    return output, None, None


def _diagnostics_from_output(output: Any) -> tuple[Any, Any]:
    if isinstance(output, dict):
        return output.get("raw", output), output.get("parsing_error")
    return output, None


def _validate_parsed(
    schema: type[Any],
    parsed: Any,
    *,
    validator: Callable[[Any], Any] | None,
) -> Any:
    if parsed is None:
        raise ValueError("Structured-output parsed value is None.")

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if isinstance(parsed, schema):
            value = parsed
        else:
            try:
                value = schema.model_validate(parsed)
            except ValidationError:
                raise
            except Exception as exc:
                raise ValueError(
                    f"Could not validate parsed output as {schema.__name__}: {exc}"
                ) from exc
    else:
        # TypedDict and other schema objects generally do not provide runtime
        # validation.  At minimum, reject None; call-site validators can enforce
        # field-level requirements when needed.
        value = parsed

    if validator is not None:
        value = validator(value)
        if value is None:
            raise ValueError("Structured-output validator returned None.")
    return value


def _repair_count(repair: bool | int) -> int:
    if isinstance(repair, bool):
        return 1 if repair else 0
    return max(0, int(repair))


def _build_repair_input(
    original_input: Any,
    schema: type[Any],
    *,
    context: str,
    error: str,
    raw: Any,
    repair_index: int,
    repair_attempts: int,
) -> list[BaseMessage]:
    schema_description = _schema_description(schema)
    raw_text = _safe_text(raw)
    original_text = _safe_text(original_input)
    return [
        SystemMessage(
            content=(
                "You are repairing a failed structured-output response. "
                "Return only a response that satisfies the requested schema."
            )
        ),
        HumanMessage(
            content=(
                f"Context: {context}\n"
                f"Repair attempt: {repair_index} of {repair_attempts}\n\n"
                f"Required schema:\n{schema_description}\n\n"
                f"Previous error:\n{error}\n\n"
                f"Previous invalid/raw response:\n{raw_text}\n\n"
                f"Original prompt/messages:\n{original_text}\n\n"
                "Produce a valid structured response for the schema."
            )
        ),
    ]


def _schema_description(schema: type[Any]) -> str:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return json.dumps(schema.model_json_schema(), indent=2, default=str)
        except Exception:
            return schema.__name__
    return getattr(schema, "__name__", repr(schema))


def _safe_text(value: Any, *, max_chars: int = 8000) -> str:
    try:
        if isinstance(value, list):
            text = "\n".join(_message_or_value_text(item) for item in value)
        else:
            text = _message_or_value_text(value)
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def _message_or_value_text(value: Any) -> str:
    if isinstance(value, BaseMessage):
        return f"{value.__class__.__name__}: {value.content}"
    return str(value)


def _format_error_message(
    context: str, attempts: Sequence[StructuredOutputAttempt]
) -> str:
    if not attempts:
        return f"Failed to obtain structured output for {context}."
    details = "; ".join(
        f"{attempt.method}"
        f"{f' repair#{attempt.repair_index}' if attempt.repair_index else ''}"
        f" -> {attempt.error}"
        for attempt in attempts
    )
    return f"Failed to obtain structured output for {context}: {details}"
