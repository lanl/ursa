"""Responses API -> Chat Completions fallback for OpenAI-compatible endpoints.

URSA defaults ``openai``-provider models to the Responses API, which most
gateway models (e.g. AI Portal) now serve. Some OpenAI-compatible endpoints only
partially implement it. For example, a LiteLLM model group in front of vLLM may
accept text turns but reject an ``input_image`` inside a ``function_call_output``.

:class:`ResponsesFallbackChatOpenAI` keeps Responses as the default and, when a
request fails with an error that points at the Responses request itself,
retries that request once over Chat Completions. If the retry succeeds, the
endpoint + model pair is pinned to Chat Completions for the rest of the process
(shared across ``bind_tools`` / ``model_copy`` copies), so the failed attempt
is paid once. If the retry also fails, nothing is pinned and the *original*
error is raised: the diagnosis was wrong, so we must not mask a real error.

On the Chat Completions path, images in tool results are moved into a user
message that follows the tool results. The Chat Completions spec only allows
text parts in ``role: "tool"`` messages, and many servers reject images there.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import openai
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

__all__ = ["ResponsesFallbackChatOpenAI", "with_responses_fallback"]

# (base_url, model) pairs that fell back successfully. Process-wide on purpose:
# agents call ``model_copy()`` / ``bind_tools()``, which must share the pin.
_CHAT_COMPLETIONS_PINNED: set[tuple[str, str]] = set()

# Text that ties a 400/422 to the shape of a *Responses* request (as opposed to
# e.g. a bad tool schema or invalid key). Matched case-insensitively.
_RESPONSES_MARKERS = (
    "/responses",
    "responses api",
    "input_image",
    "input_text",
    "input_file",
    "function_call_output",
    "output_text",
    "'input[",
    '"input[',
)
_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})
# ChatOpenAI puts these in every request when set, but only the Responses API
# accepts them: Chat Completions raises ``TypeError ... unexpected keyword
# argument 'reasoning'`` before any request is sent.
_RESPONSES_ONLY_PARAMS = (
    "reasoning",
    "include",
    "context_management",
    "truncation",
    "previous_response_id",
)


def _chat_completions_params(payload: dict) -> dict:
    """Make a Chat Completions payload valid when Responses-only settings are set.

    ``reasoning={"effort": ...}`` becomes the Chat Completions equivalent
    ``reasoning_effort`` (an explicit ``reasoning_effort`` wins); the rest of
    the Responses-only settings are dropped.
    """
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
        payload.setdefault("reasoning_effort", reasoning["effort"])
    for key in _RESPONSES_ONLY_PARAMS:
        payload.pop(key, None)
    return payload


def _is_responses_incompatibility(exc: BaseException) -> bool:
    """Whether ``exc`` plausibly means "this endpoint can't serve this
    Responses request". Deliberately narrow: auth, rate-limit, server, and
    context-length errors are never treated as incompatibility."""
    if isinstance(exc, ContextOverflowError):
        return False
    if not isinstance(exc, openai.APIStatusError):
        return False
    status = exc.status_code
    if status in (404, 405, 501):
        # Route missing. A 404 for an unknown model fails on the retry too, so
        # it is never pinned and the original error still surfaces.
        return True
    if status not in (400, 422):
        return False
    param = getattr(exc, "param", None)
    if isinstance(param, str) and param.startswith("input"):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RESPONSES_MARKERS)


def _relocate_tool_images(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Move image blocks out of tool results into a following user message.

    The user message is inserted after the whole run of consecutive tool
    results, because Chat Completions requires every tool result to directly
    follow the assistant turn that requested it.
    """
    out: list[BaseMessage] = []
    pending: list[dict] = []

    def flush() -> None:
        if pending:
            out.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Image output from the tool call(s) above:",
                        },
                        *pending,
                    ]
                )
            )
            pending.clear()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            flush()
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            images = [
                b
                for b in msg.content
                if isinstance(b, dict) and b.get("type") in _IMAGE_TYPES
            ]
            if images:
                pending.extend(images)
                rest = [b for b in msg.content if b not in images]
                text = "".join(
                    b if isinstance(b, str) else str(b.get("text") or "")
                    for b in rest
                    if isinstance(b, str) or b.get("type") == "text"
                )
                note = "[image output attached in the next message]"
                msg = msg.model_copy(
                    update={"content": f"{text}\n{note}" if text else note}
                )
        out.append(msg)
    flush()
    return out


class ResponsesFallbackChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that falls back from Responses to Chat Completions."""

    def _fallback_key(self) -> tuple[str, str]:
        return (self.openai_api_base or "", self.model_name or "")

    @property
    def _pinned(self) -> bool:
        return self._fallback_key() in _CHAT_COMPLETIONS_PINNED

    def _use_responses_api(self, payload: dict) -> bool:
        if self._pinned:
            return False
        return super()._use_responses_api(payload)

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        if self._use_responses_api({**self._default_params, **kwargs}):
            return super()._get_request_payload(input_, stop=stop, **kwargs)
        input_ = _relocate_tool_images(
            self._convert_input(input_).to_messages()
        )
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _chat_completions_params(payload)

    def _should_fall_back(self, exc: BaseException, kwargs: dict) -> bool:
        return (
            not self._pinned
            and super()._use_responses_api({**self._default_params, **kwargs})
            and _is_responses_incompatibility(exc)
        )

    def _pin(self, exc: BaseException) -> None:
        _CHAT_COMPLETIONS_PINNED.add(self._fallback_key())
        logger.warning(
            "Responses API request to %s (%s) failed (%s); using Chat "
            "Completions for this endpoint/model for the rest of the session. "
            "Set `use_responses_api: false` for this model to skip the probe.",
            self.model_name,
            self.openai_api_base or "default endpoint",
            str(exc)[:300],
        )

    def _unpin(self) -> None:
        _CHAT_COMPLETIONS_PINNED.discard(self._fallback_key())

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        except Exception as exc:
            if not self._should_fall_back(exc, kwargs):
                raise
            original = exc
        self._pin(original)
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        except Exception as retry_exc:
            self._unpin()
            logger.warning(
                "Chat Completions fallback also failed: %s", retry_exc
            )
            raise original from None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return await super()._agenerate(
                messages, stop, run_manager, **kwargs
            )
        except Exception as exc:
            if not self._should_fall_back(exc, kwargs):
                raise
            original = exc
        self._pin(original)
        try:
            return await super()._agenerate(
                messages, stop, run_manager, **kwargs
            )
        except Exception as retry_exc:
            self._unpin()
            logger.warning(
                "Chat Completions fallback also failed: %s", retry_exc
            )
            raise original from None

    def _stream(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        # Only an error before the first chunk can be retried transparently.
        started = False
        try:
            for chunk in super()._stream(*args, **kwargs):
                started = True
                yield chunk
            return
        except Exception as exc:
            if started or not self._should_fall_back(exc, kwargs):
                raise
            original = exc
        self._pin(original)
        try:
            yield from super()._stream(*args, **kwargs)
        except Exception as retry_exc:
            self._unpin()
            logger.warning(
                "Chat Completions fallback also failed: %s", retry_exc
            )
            raise original from None

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        started = False
        try:
            async for chunk in super()._astream(*args, **kwargs):
                started = True
                yield chunk
            return
        except Exception as exc:
            if started or not self._should_fall_back(exc, kwargs):
                raise
            original = exc
        self._pin(original)
        try:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk
        except Exception as retry_exc:
            self._unpin()
            logger.warning(
                "Chat Completions fallback also failed: %s", retry_exc
            )
            raise original from None


def with_responses_fallback(llm: Any) -> Any:
    """Enable Responses -> Chat Completions fallback on ``llm`` in place.

    Only plain ``ChatOpenAI`` instances are upgraded; other models, existing
    subclasses (e.g. ``TracedChatOpenAI``), and Azure are returned unchanged.
    The upgrade swaps the instance's class rather than re-creating it: the
    subclass declares no fields or private attributes, only method overrides,
    so the object keeps its identity, SDK clients (custom httpx/SSL settings),
    and any attributes callers already attached to it.
    """
    if type(llm) is ChatOpenAI:
        llm.__class__ = ResponsesFallbackChatOpenAI
    return llm
