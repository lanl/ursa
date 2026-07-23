from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.tools import ToolRuntime, tool
from langgraph.graph.message import add_messages

from ursa.agents import ChatAgent, RAGAgent
from ursa.agents.base import BaseAgent
from ursa.prompt_library.execution_prompts import recap_prompt
from ursa.util import Checkpointer
from ursa.util.events import AgentEvents, ToolEvents

documenter_prompt = """
You are a responsible and efficient agent tasked ensuring adequate documentation to pass 
to the team who will run simulations. 

Your responsibilities are as follows:

1. Ensure that a custom guide for setting up and running simulations in your workspace. 
   - For consistency with downstream users, the file should be called user_guide.md
   - The guide should be a comprehensive guide for a new user to get started, with all detail
       necessary to get going. Users will also be given access to other documentation as needed
       but should be able to carry out most tasks based on your guide alone.
   - The file may exist from previous efforts, but if not, create it. 
2. Assess completeness of the guide for the tasks at hand and research further needed information
    - You may have access to tools like: web/research literature search or a RAG database with 
       detailed code documentation. Use all available tools to find information you need. 

Remember, your documentation will be central to downstream efforts to run simlations or write
code scaffolding to carry out complex simulation campaigns. 

Stay in your lane! Only work on developing and writing documentation.
"""

runner_prompt = """
You are a responsible and efficient agent tasked with coordinating agentic execution to perform
simulation-based science by setting up, executing, and analyzing the results of scientific computation

Your responsibilities are as follows:

1. Carry out the requested simulation run or simulation campaign. 
   - Your first resource should be a file called user_guide.md which is a custom guide created
       to help you get started. Beyond this, you may have access to web/literature search tools or RAG
       databases with relevant information that can be queried.
2. Carry out the simulation or set up and execute a successful simulation campaign based on the user request.
    A) It is best practice to perform a smoke test (minimal size, short duration, coarse mesh, etc.) first 
        to ensure proper set up before scaling to the full problem execution
        - If smoke test fails, debug before scaling up.

    B) Monitor for failure modes
        - nonzero exit codes, NaNs/divergence, unstable timestep, memory errors, missing outputs
        - If failures occur:
            - attempt the smallest change consistent with the documentation to resolve
            - Avoid many random changes; Prefer a structured hypothesis → test cycle

    C) Analyze the outputs of the simulation
        - Produce analysis artifacts , such as:
            - reproducible analysis scripts
                - Keep scripts in a top-level analysis/ folder.
                - Ensure scripts are runnable and reference relative paths.
                - Prefer clear, minimal dependencies; document how to run the analysis.
            - summary tables (CSV/TSV) of key metrics
            - plots (PNG/PDF)
            - derived datasets (e.g., processed fields, reduced-order metrics)

    D) Update Documentation
        - If you learned anything new (correct flags, pitfalls, missing steps, interpretation details),
            update user_guide.md to communicate all pertinant information to future users

You are responsible for carrying out the users request accurately, safely, and transparently.
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]
    code_files: list[str]
    goal: str


class SimulatorAgent(BaseAgent):
    state_type = State

    def __init__(
        self,
        llm: BaseChatModel,
        embedding=None,
        log_state: bool = False,
        use_web: bool = False,
        workspace: Optional[Path | str] = None,
        checkpointer=Checkpointer,
        thread_id=str,
        **kwargs,
    ):
        super().__init__(llm, workspace=workspace, **kwargs)
        self.documenter_prompt = documenter_prompt
        self.runner_prompt = runner_prompt
        self.recap_prompt = recap_prompt
        self.log_state = log_state
        self.thread_id = thread_id
        self.embedding = embedding
        self.use_web = use_web
        self.llm = llm
        self.documenter = ChatAgent(
            llm=self.llm,
            workspace=workspace,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_documenter",
        )
        self.runner = ChatAgent(
            llm=self.llm,
            workspace=workspace,
            checkpointer=checkpointer,
            thread_id=self.thread_id + "_runner",
        )

        self.documenter.executor_prompt = self.documenter_prompt
        self.runner.executor_prompt = self.runner_prompt

        self.runner.safe_codes = [
            "All standard programming languages and tools for compiling and running codes and the associated files available in your workspace."
        ]

        self.documenter.remove_tool(["run_command", "write_code"])

        if not self.use_web:
            self.documenter.remove_tool([
                "run_web_search",
                "run_osti_search",
                "run_arxiv_search",
            ])
        if not self.use_web:
            self.runner.remove_tool([
                "run_web_search",
                "run_osti_search",
                "run_arxiv_search",
            ])

        if self.embedding:
            self.rag_agent = RAGAgent(
                llm=self.llm,
                workspace=self.workspace,
                embedding=self.embedding,
                thread_id="simulation_documentation_rag",
                database_path="../simulator_docs",
            )

            @tool
            def documentation_rag(
                query: str,
                runtime: ToolRuntime,
            ) -> str:
                """
                Query a RAG database for information from documentation on the scientific computing model.

                Arguments:
                    query: String query to send to the RAG database to obtain information from documentation

                Returns:
                    summary: string summary of the information in the RAG database relevant to the query
                """
                ToolEvents.from_runtime("documentation_rag", runtime).emit(
                    "Querying simulation documentation",
                    stage="query",
                    query=query,
                )
                return self.doc_rag(query)

            self.documenter.add_tool(documentation_rag)
            self.runner.add_tool(documentation_rag)

    def doc_rag(self, query: str) -> str:
        """
        Query a RAG database for information from documentation on the scientific computing model.

        Arguments:
            query: String query to send to the RAG database to obtain information from documentation

        Returns:
            summary: string summary of the information in the RAG database relevant to the query
        """
        if self.embedding:
            result = self.rag_agent.invoke(
                context=query,
            )
            return result["summary"]
        else:
            return "Tool Failed: No RAG database available."

    # Define the function that calls the model
    def _documenter(self, state: State) -> State:
        goal = state["messages"][-1].text
        AgentEvents(agent=self.name, config=self.build_config()).emit(
            "Beginning documentation assessment",
            stage="document",
        )
        response = self.documenter.invoke(state)
        return {"messages": response["messages"], "goal": goal}

    # Define the function that calls the model
    def _runner(self, state: State) -> State:
        AgentEvents(agent=self.name, config=self.build_config()).emit(
            "Beginning simulation",
            stage="simulate",
        )
        response = self.runner.invoke(state["goal"])
        return {"messages": response["messages"]}

    # Define the function that calls the model
    def _summarize(self, state: State) -> State:
        messages = state["messages"] + [HumanMessage(content=recap_prompt)]
        # 2) Invoke the LLM to generate a recap; capture content even on failure.
        try:
            response = self.llm.invoke(
                messages, self.build_config(tags=["recap"])
            )
            response_content = response.text
        except Exception as ee:
            try:
                response = self.runner.llm.invoke(
                    input=messages,
                    config=self.build_config(tags=["recap"]),
                )
                response_content = response.text
            except Exception as e:
                response_content = f"Response errors {ee} and {e}"
                response = AIMessage(content=response_content)
                events = AgentEvents(
                    agent=self.name, config=self.build_config()
                )
                events.emit(
                    "Primary recap failed",
                    stage="recap",
                    phase="error",
                    error_type=type(ee).__name__,
                    error=str(ee),
                )
                events.emit(
                    "Fallback recap failed",
                    stage="recap",
                    phase="error",
                    error_type=type(e).__name__,
                    error=str(e),
                )

        updated_messages = [*state["messages"], response]

        if self.log_state:
            save_state = state.copy()
            save_state["messages"] = updated_messages
            self.write_state(self.workspace / "combined_agent.json", save_state)
        return {"messages": updated_messages}

    def _build_graph(self):
        self.add_node(self._documenter)
        self.add_node(self._runner)
        self.add_node(self._summarize)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.set_entry_point("_documenter")
        self.graph.add_edge("_documenter", "_runner")
        self.graph.add_edge("_runner", "_summarize")
        self.graph.set_finish_point("_summarize")


def main():
    from langchain.chat_models import init_chat_model

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

    model = init_chat_model(
        model="openai:gpt-5.2",
    )

    agent = SimulatorAgent(llm=model, log_state=True, workspace=workspace)
    result = agent.invoke(problem)
    print(result["messages"][-1].text)  # noqa: T201


if __name__ == "__main__":
    main()
