from typing import TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .base import BaseAgent


class RecallState(TypedDict):
    query: str
    memory: str


class RecallAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, memory, **kwargs):
        super().__init__(llm, **kwargs)

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
        state["memory"] = StrOutputParser().invoke(
            self.llm.invoke(summarize_query)
        )
        return state

    def _build_graph(self):
        self.add_node(self._remember)
        self.graph.set_entry_point("_remember")
        self.graph.set_finish_point("_remember")
