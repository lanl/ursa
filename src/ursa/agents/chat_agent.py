from typing import Annotated, Any, Mapping

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .base import BaseAgent


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str


class ChatAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self._build_graph()

    def _response_node(self, state: ChatState) -> ChatState:
        res = self.llm.invoke(
            state["messages"], {"configurable": {"thread_id": self.thread_id}}
        )
        return {"messages": [res]}

    def _build_graph(self):
        graph = StateGraph(ChatState)
        self.add_node(graph, self._response_node)
        graph.set_entry_point("_response_node")
        graph.set_finish_point("_response_node")
        self._action = graph.compile(checkpointer=self.checkpointer)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 1000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        return self._action.invoke(inputs, config)


def main():
    model = ChatOpenAI(
        model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2
    )
    websearcher = ChatAgent(llm=model)
    problem_string = "What is your name?"
    print("Prompt: ", problem_string)
    result = websearcher.invoke(problem_string)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Response: ", main())
