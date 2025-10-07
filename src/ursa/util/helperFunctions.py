from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, Iterable, List, Union

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable


# --- if you already have your own versions, reuse them ---
def _parse_args(v: Any) -> Dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {"_raw": v}
    return {"_raw": v}


def extract_tool_calls(msg: AIMessage) -> List[Dict[str, Any]]:
    # Prefer normalized field
    if msg.tool_calls:
        out = []
        for tc in msg.tool_calls:
            name = getattr(tc, "name", None) or tc.get("name")
            args = getattr(tc, "args", None) or tc.get("args")
            call_id = getattr(tc, "id", None) or tc.get("id")
            out.append({"name": name, "args": _parse_args(args), "id": call_id})
        return out

    # Fallbacks (OpenAI raw payloads)
    ak = msg.additional_kwargs or {}
    if ak.get("tool_calls"):
        out = []
        for tc in ak["tool_calls"]:
            fn = tc.get("function", {}) or {}
            out.append({
                "name": fn.get("name"),
                "args": _parse_args(fn.get("arguments")),
                "id": tc.get("id"),
            })
        return out

    if ak.get("function_call"):
        fn = ak["function_call"]
        return [
            {
                "name": fn.get("name"),
                "args": _parse_args(fn.get("arguments")),
                "id": None,
            }
        ]
    return []


# -----------------------------------------------------------------------------


ToolRegistry = Dict[str, Union[Runnable, Callable[..., Any]]]


def _stringify_output(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _invoke_tool(
    tool: Union[Runnable, Callable[..., Any]], args: Dict[str, Any]
) -> Any:
    # Runnable (LangChain tools & chains)
    if isinstance(tool, Runnable):
        return tool.invoke(args)
    # Plain callable
    try:
        return tool(**args)
    except TypeError:
        # Some tools expect a single positional payload
        return tool(args)


def run_tool_calls(
    ai_msg: AIMessage,
    tools: Union[ToolRegistry, Iterable[Union[Runnable, Callable[..., Any]]]],
) -> List[BaseMessage]:
    """
    Args:
        ai_msg: The LLM's AIMessage containing tool calls.
        tools: Either a dict {name: tool} or an iterable of tools (must have `.name`
               for mapping). Each tool can be a Runnable or a plain callable.

    Returns:
        out: list[BaseMessage] to feed back to the model
    """
    # Build a name->tool map
    if isinstance(tools, dict):
        registry: ToolRegistry = tools  # type: ignore
    else:
        registry = {}
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if not name:
                raise ValueError(f"Tool {t!r} has no discoverable name.")
            registry[name] = t  # type: ignore

    calls = extract_tool_calls(ai_msg)

    if not calls:
        return []

    out: List[BaseMessage] = []
    for call in calls:
        name = call.get("name")
        args = call.get("args", {}) or {}
        call_id = call.get("id") or f"call_{uuid.uuid4().hex}"

        # 1) the AIMessage that generated the call
        out.append(ai_msg)

        # 2) the ToolMessage with the execution result (or error)
        if name not in registry:
            content = f"ERROR: unknown tool '{name}'."
        else:
            try:
                result = _invoke_tool(registry[name], args)
                content = _stringify_output(result)
            except Exception as e:
                content = f"ERROR: {type(e).__name__}: {e}"

        out.append(
            ToolMessage(content=content, tool_call_id=call_id, name=name)
        )

    return out
