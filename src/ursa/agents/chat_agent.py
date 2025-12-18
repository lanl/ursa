from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from .base import BaseAgent


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str


class ChatAgent(BaseAgent[ChatState]):
    state_type = ChatState

    def _response_node(self, state: ChatState) -> ChatState:
        res = self.llm.invoke(
            state["messages"], {"configurable": {"thread_id": self.thread_id}}
        )
        return {"messages": [res]}

    def _build_graph(self):
        self.add_node(self._response_node)
        self.graph.set_entry_point("_response_node")
        self.graph.set_finish_point("_response_node")
