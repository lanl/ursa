"""Make message histories safe to replay to a different chat model.

URSA keeps one canonical message history per agent thread, but users switch
models mid-conversation (TUI model picker, persisted agents reloaded with a new
config, gateways such as AIPortal that serve Claude, GPT, and others behind one
OpenAI-compatible endpoint). An ``AIMessage`` stores its producer's *native*
wire items: Responses-API ``reasoning`` items with ``encrypted_content``,
``function_call`` items with ``status``, Anthropic ``thinking`` signatures,
Gemini thought signatures, provider item ids, and so on.

Integrations differ in whether they filter foreign items (verified against
installed langchain-openai 1.2 / langchain-anthropic 1.4 / langchain-google-genai
4.2):

* ``ChatAnthropic`` drops ``reasoning``/``function_call`` blocks unless the
  message's ``model_provider`` is ``"anthropic"``.
* ``ChatOpenAI`` (Chat Completions) flattens history to text + tool calls.
* ``ChatOpenAI`` (Responses API) replays every ``reasoning``/``function_call``
  item verbatim, regardless of which model produced it -> 400s such as
  ``Unknown parameter: 'input[N].status'``. Behind a gateway every model reports
  ``model_provider="openai"``, so provider checks cannot catch this.
* ``ChatGoogleGenerativeAI`` raises ``KeyError: 'reasoning'`` on
  Responses-shaped reasoning blocks.

So we cannot rely on the integration. :func:`make_portable` rebuilds every AI
message produced by a *different model* from provider-neutral parts only (an
allowlist: visible text + ``tool_calls``). New provider fields or block types
therefore cannot leak into a foreign request. Messages from the current model,
and messages of unknown origin, are left untouched, so single-model behavior
is unchanged.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

__all__ = ["is_foreign", "make_portable", "model_name"]

_TEXT_TYPES = frozenset({"text", "output_text"})
# Anthropic requires ^[A-Za-z0-9_-]+$; OpenAI Chat Completions caps ids at 40.
_SAFE_TOOL_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
# Version decorations that differ between the requested and reported name,
# e.g. "gpt-5.4" vs "gpt-5.4-2026-03-05", "claude-sonnet-4-5" vs
# "claude-sonnet-4-5-20250929", Bedrock "...-v1:0", Gemini "models/...".
_VERSION_SUFFIX = re.compile(
    r"(?:[-@](?:\d{4}-?\d{2}-?\d{2}|latest))?(?:-v\d+(?::\d+)?)?$"
)


def _unwrap(llm: Any) -> Any:
    """Follow ``bind()``/``bind_tools()`` wrappers down to the chat model."""
    for _ in range(10):
        bound = getattr(llm, "bound", None)
        if bound is None:
            break
        llm = bound
    return llm


def model_name(llm: Any) -> str | None:
    """Best-effort configured model name of a (possibly bound) chat model."""
    base = _unwrap(llm)
    for attr in ("model_name", "model", "model_id"):
        value = getattr(base, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize(name: str) -> str:
    name = name.strip().lower().removeprefix("models/")
    return _VERSION_SUFFIX.sub("", name)


def _origin(msg: AIMessage) -> str | None:
    meta = msg.response_metadata or {}
    for key in ("model_name", "model", "model_id"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def is_foreign(msg: BaseMessage, target: str | None) -> bool:
    """True when ``msg`` is an AI message produced by a model other than
    ``target``. Unknown origin on either side counts as native (no change)."""
    if not isinstance(msg, AIMessage) or not target:
        return False
    origin = _origin(msg)
    return origin is not None and _normalize(origin) != _normalize(target)


def _visible_text(content: Any) -> str:
    """Visible text only: reasoning, signatures and item metadata never leak."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in _TEXT_TYPES:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _portable_id(tool_call_id: Any) -> str:
    """Keep provider-safe ids; map others to a stable hash (deterministic, so
    the rewritten history is reproducible and prompt caches still hit)."""
    tid = str(tool_call_id)
    if _SAFE_TOOL_ID.fullmatch(tid):
        return tid
    return "call_" + hashlib.sha256(tid.encode()).hexdigest()[:32]


def _portable_ai(msg: AIMessage, id_map: dict[str, str]) -> AIMessage | None:
    calls = []
    for call in msg.tool_calls:
        new_id = _portable_id(call.get("id"))
        if call.get("id") is not None:
            id_map[str(call["id"])] = new_id
        calls.append({
            "name": call["name"],
            "args": call["args"],
            "id": new_id,
            "type": "tool_call",
        })
    text = _visible_text(msg.content)
    if not text.strip() and not calls:
        # Reasoning-only turn: nothing portable remains, and empty assistant
        # turns are rejected by some providers (e.g. Anthropic).
        return None
    return AIMessage(
        content=text,
        tool_calls=calls,
        name=msg.name,
        id=msg.id,
        # No model_name here on purpose: the rebuilt message is safe for every
        # provider, so it must read as "unknown origin" and never be rebuilt again.
        response_metadata={"ursa_portable_from": _origin(msg)},
    )


def make_portable(
    messages: Sequence[BaseMessage], llm: Any
) -> tuple[list[BaseMessage], bool]:
    """Rebuild AI messages produced by a model other than ``llm``.

    Returns ``(messages, changed)``. Text and tool calls are kept; reasoning,
    signatures, encrypted content, item ids/status, and provider metadata of
    foreign turns are dropped. Unsafe foreign tool-call ids are rewritten
    consistently in the matching ``ToolMessage``\\ s. Idempotent.
    """
    target = model_name(llm)
    out: list[BaseMessage] = []
    id_map: dict[str, str] = {}
    changed = False
    for msg in messages:
        if is_foreign(msg, target):
            changed = True
            rebuilt = _portable_ai(msg, id_map)
            if rebuilt is not None:
                out.append(rebuilt)
        elif (
            isinstance(msg, ToolMessage)
            and id_map.get(msg.tool_call_id, msg.tool_call_id)
            != msg.tool_call_id
        ):
            out.append(
                msg.model_copy(
                    update={"tool_call_id": id_map[msg.tool_call_id]}
                )
            )
        else:
            out.append(msg)
    return out, changed
