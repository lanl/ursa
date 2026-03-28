from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


def nested_agent_kwargs(
    config: RunnableConfig | None, *, checkpoint_ns: str | None = None
) -> dict[str, Any]:
    """Forward supported runtime config keys to nested agent calls."""
    runtime_config = config or {}
    invoke_kwargs: dict[str, Any] = {}
    for key in (
        "callbacks",
        "metadata",
        "tags",
        "recursion_limit",
        "configurable",
    ):
        if key in runtime_config:
            invoke_kwargs[key] = runtime_config[key]
    if checkpoint_ns is not None:
        configurable = dict(invoke_kwargs.get("configurable") or {})
        configurable["checkpoint_ns"] = checkpoint_ns
        invoke_kwargs["configurable"] = configurable
    return invoke_kwargs


def message_text(message: Any) -> str:
    """Return plain text content from message-like objects."""
    if isinstance(message, BaseMessage):
        return message.text
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content or "")
