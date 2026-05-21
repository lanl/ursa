from pathlib import Path
from typing import Any

from langchain.chat_models import BaseChatModel
from langchain.tools import ToolRuntime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
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
    config: dict[str, Any] | None = None,
) -> ToolRuntime[AgentContext]:
    """Construct a minimal ToolRuntime populated with AgentContext."""
    return ToolRuntime(
        state={},
        context=AgentContext(
            llm=llm,
            workspace=workspace,
            tool_character_limit=limit,
        ),
        config=config or {"metadata": {"thread_id": thread_id}},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=store,
    )


class CustomEventRecorder(BaseCallbackHandler):
    """Record custom events emitted from a Runnable parent run."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_custom_event(
        self,
        name: str,
        data: Any,
        **kwargs: Any,
    ) -> None:
        self.events.append((name, data))


def invoke_with_parent_run(call):
    """Run ``call(config)`` inside a LangChain parent run."""
    return RunnableLambda(lambda _, config: call(config)).invoke(None)


def _inject_runtime_config(config: dict[str, Any], value: Any) -> None:
    if isinstance(value, ToolRuntime):
        value.config = config


def invoke_with_event_recorder(
    f,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, CustomEventRecorder]:
    """Run ``f(*args, **kwargs)`` inside a parent run and capture events."""
    recorder = CustomEventRecorder()

    def call(_, config):
        for arg in args:
            _inject_runtime_config(config, arg)
        for value in kwargs.values():
            _inject_runtime_config(config, value)
        return f(*args, **kwargs)

    result = RunnableLambda(call).invoke(
        None,
        config={"callbacks": [recorder]},
    )
    return result, recorder
