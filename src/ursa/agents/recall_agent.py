from typing import Any, Mapping, TypedDict

from langchain.chat_models import BaseChatModel
from langgraph.graph import StateGraph

from .base import BaseAgent


class RecallState(TypedDict):
    query: str
    memory: str


class RecallAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, memory, **kwargs):
        super().__init__(llm, **kwargs)
        self.memorydb = memory
        self._action = self._build_graph()

    def _remember(self, state: RecallState) -> str:
        memories = self.memorydb.retrieve(state["query"])
        summarize_query = f"""
        You are being given the critical task of generating a detailed description of logged information 
        to an important official to make a decision. Summarize the following memories that are related to 
        the statement. Ensure that any specific details that are important are retained in the summary.

        Query: {state["query"]}

        """

        for memory in memories:
            summarize_query += f"Memory: {memory} \n\n"
        state["memory"] = self.llm.invoke(summarize_query).content
        return state

    def _build_graph(self):
        graph = StateGraph(RecallState)

        self.add_node(graph, self._remember)
        graph.set_entry_point("_remember")
        graph.set_finish_point("_remember")
        return graph.compile(checkpointer=self.checkpointer)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 100000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        if "query" not in inputs:
            raise ("'query' is a required argument")

        output = self._action.invoke(inputs, config)
        return output["memory"]
