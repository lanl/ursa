"""Small, reusable probes for configured external services."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel


@tool
def _ursa_health_echo(value: str) -> str:
    """Return a value unchanged for an URSA health check."""
    return value


class _HealthResponse(BaseModel):
    status: str


def probe_chat_model(model: Any) -> str:
    """Raise unless a chat model accepts a minimal request."""
    if model.invoke("URSA health check: reply briefly.") is None:
        raise ValueError("chat invocation returned no response")
    return "Model returned a response"


def probe_chat_tool_use(model: Any) -> str:
    """Raise unless a chat model honors an explicit tool choice."""
    response = model.bind_tools(
        [_ursa_health_echo], tool_choice=_ursa_health_echo.name
    ).invoke("Call ursa_health_echo with the value 'ursa'.")
    calls = getattr(response, "tool_calls", None) or ()
    if not any(
        call.get("name") == _ursa_health_echo.name
        and call.get("args", {}).get("value") == "ursa"
        for call in calls
        if isinstance(call, Mapping)
    ):
        raise ValueError("model did not return the requested tool call")
    return "Model returned the required tool call"


def probe_chat_structured_output(model: Any) -> str:
    """Raise unless a chat model returns the requested typed response."""
    from ursa.util.structured_output import invoke_structured

    response = invoke_structured(
        model,
        _HealthResponse,
        "Return a structured response whose status is exactly 'ok'.",
        context="URSA health check",
    )
    if not isinstance(response, _HealthResponse) or response.status != "ok":
        raise ValueError("model did not return the expected status='ok'")
    return "Model returned a valid typed response"


def probe_embedding_model(config: Any) -> str:
    """Raise unless an embedding configuration returns a non-empty vector."""
    vector = config.init_embedding().embed_query("URSA health check")
    if not isinstance(vector, Sequence) or not vector:
        raise ValueError("embedding invocation returned no vector")
    return f"Embedding invocation returned {len(vector)} dimensions"


async def probe_mcp_server(name: str, server: Any) -> str:
    """Raise unless an MCP server connects and advertises its tools."""
    from ursa.util.mcp import start_mcp_client

    tools = await start_mcp_client({name: server}).get_tools(server_name=name)
    return f"Connected; server advertised {len(tools)} tool(s)"
