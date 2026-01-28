from typing import TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from ursa.util.memory_logger import AgentMemory

from .base import BaseAgent


class RecallState(TypedDict, total=False):
    query: str
    memory: str


class RecallAgent(BaseAgent):
    def __init__(self, llm: BaseChatModel, memory: AgentMemory, **kwargs):
        super().__init__(llm, **kwargs)
        self.memory = memory

    def format_query(self, prompt: str, state: RecallState | None = None):
        return RecallState(query=prompt)

    def format_result(self, state: RecallState) -> str:
        return state["memory"]

    async def _remember(self, state: RecallState) -> str:
        memories = self.memory.retrieve(state["query"])
        summarize_query = f"""
        You are being given the critical task of generating a detailed description of logged information
        to an important official to make a decision. Summarize the following memories that are related to
        the statement. Ensure that any specific details that are important are retained in the summary.

        Query: {state["query"]}

        """

        for memory in memories:
            summarize_query += f"Memory: {memory} \n\n"
        state["memory"] = StrOutputParser().invoke(
            await self.llm.ainvoke(summarize_query)
        )
        return state

    def _build_graph(self):
        self.add_node(self._remember)
        self.graph.set_entry_point("_remember")
        self.graph.set_finish_point("_remember")
