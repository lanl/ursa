from pathlib import Path
from typing import Optional

from langchain.tools import ToolRuntime
from langgraph.store.base import BaseStore

from ursa.agents.base import AgentContext


def make_runtime(
    workspace: Path,
    *,
    tool_call_id: str = "tool-call",
    thread_id: str = "thread",
    limit: int = 3000,
    store: Optional[BaseStore] = None,
) -> ToolRuntime[AgentContext]:
    """Construct a minimal ToolRuntime populated with AgentContext."""
    return ToolRuntime(
        state={},
        context=AgentContext(workspace=workspace, tool_character_limit=limit),
        config={"metadata": {"thread_id": thread_id}},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=store,
    )
