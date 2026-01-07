from typing import Annotated, TypedDict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from ursa.prompt_library.chatter_prompts import get_chatter_system_prompt

from .base import BaseAgent


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str


class ChatAgent(BaseAgent[ChatState]):
    """Chat Agent"""

    state_type = ChatState

    def _response_node(self, state: ChatState) -> ChatState:
        res = self.llm.invoke(state["messages"])
        return {"messages": [res]}

    def format_query(self, prompt: str, state: ChatState | None = None):
        if state is None:
            state = ChatState(
                messages=[SystemMessage(content=get_chatter_system_prompt())]
            )
        state["messages"].append(HumanMessage(content=prompt))

        return state

    def format_result(self, result: ChatState) -> str:
        return result["messages"][-1].content

    def _build_graph(self):
        self.add_node(self._response_node)
        self.graph.set_entry_point("_response_node")
        self.graph.set_finish_point("_response_node")
