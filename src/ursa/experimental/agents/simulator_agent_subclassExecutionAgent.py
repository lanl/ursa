import sqlite3
from pathlib import Path
from typing import (
    Optional,
)

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.agents import ExecutionAgent, RAGAgent
from ursa.prompt_library.simulator_prompts import simulation_coordinator_prompt

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class SimulatorAgent(ExecutionAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding=None,
        use_web=False,
        tokens_before_summarize: Optional[int] = 50000,
        messages_to_keep: Optional[int] = 20,
        safe_codes: Optional[list[str]] = None,
        workspace: Optional[str] = "ursa_simulation_workspace",
        **kwargs,
    ):
        super().__init__(
            llm,
            tokens_before_summarize=tokens_before_summarize,
            messages_to_keep=messages_to_keep,
            safe_codes=safe_codes,
            workspace=workspace,
            **kwargs,
        )
        self.embedding = embedding
        self.use_web = use_web
        # Replacing the execution agent system prompt with the simulator coordination prompt.
        #     Otherwise we should keep the same structure as the execution agent.
        self.executor_prompt = simulation_coordinator_prompt

        if self.embedding:
            self.rag_agent = RAGAgent(
                llm=self.llm,
                workspace=self.workspace,
                thread_id="simulation_documentation_rag",
                embedding=self.embedding,
                database_path="../simulator_docs",
            )

            @tool
            def documentation_rag(query: str) -> str:
                """
                Query a RAG database for information from documentation on the scientific computing model.

                Arguments:
                    query: String query to send to the RAG database to obtain information from documentation

                Returns:
                    summary: string summary of the information in the RAG database relevant to the query
                """
                return self.doc_rag(query)

            self.add_tool(documentation_rag)

        if not self.use_web:
            self.remove_tool([
                "run_web_search",
                "run_osti_search",
                "run_arxiv_search",
            ])

    def doc_rag(self, query: str) -> str:
        """
        Query a RAG database for information from documentation on the scientific computing model.

        Arguments:
            query: String query to send to the RAG database to obtain information from documentation

        Returns:
            summary: string summary of the information in the RAG database relevant to the query
        """
        print(f"[RAG QUERY]: {query}")
        if self.embedding:
            result = self.rag_agent.invoke(
                context=query,
            )
            return result["summary"]
        else:
            return "Tool Failed: No RAG database available."


def main():
    problem = (
        "Your task is to perform a parameter sweep of dcopf using an open source "
        "code for optimizing power systems, PowerModels.jl. "
        "The parameter sweep will be performed on the load parameters 10 times by choosing "
        "a random number between 0.8 and 1.2 and multiplying the load by this factor."
        "I require that each parameter configuration be stored in its own input file, ieee14."
        "I require that the code used to perform the task be stored."
        "I require that the code be executed and output saved to a csv file. "
        "Produce a plot with opf objective value on the x axis and load factor on the y axis."
    )
    workspace = "ursa_simulator_test"

    db_path = Path(workspace) / "checkpoint.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    embedding = init_embeddings("openai:text-embedding-3-small")
    model = init_chat_model(
        model="openai:gpt-5.2",
    )

    simulator = SimulatorAgent(
        llm=model,
        workspace=workspace,
        embedding=embedding,
        checkpointer=checkpointer,
        use_web=True,
    )
    simulator.thread_id = "dcopf_test_executor"

    result = simulator.invoke(problem)

    print("\n\n".join([x.content for x in result["messages"]]))
