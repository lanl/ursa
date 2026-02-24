from pathlib import Path

from langchain.chat_models import BaseChatModel
from langchain.tools import ToolRuntime
from langgraph.store.base import BaseStore

from ursa.agents.base import AgentContext


def make_runtime(
    workspace: Path,
    llm: BaseChatModel,
    *,
    tool_call_id: str = "tool-call",
    thread_id: str = "thread",
    limit: int = 3000,
    store: BaseStore | None = None,
) -> ToolRuntime[AgentContext]:
    """Construct a minimal ToolRuntime populated with AgentContext."""
    return ToolRuntime(
        state={},
        context=AgentContext(
            llm=llm,
            workspace=workspace,
            tool_character_limit=limit,
        ),
        config={"metadata": {"thread_id": thread_id}},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=store,
    )
