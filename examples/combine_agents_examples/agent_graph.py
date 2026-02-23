import asyncio
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langchain.embeddings import init_embeddings
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState

from ursa.agents import (
    ArxivAgent,
    ExecutionAgent,
    RecallAgent,
)
from ursa.agents.base import AgentWithTools, BaseAgent
from ursa.agents.execution_agent import ExecutionState
from ursa.prompt_library.execution_prompts import recap_prompt
from ursa.util.memory_logger import AgentMemory

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


workspace = "combined_workspace"

runner_prompt = """
You are a responsible and efficient agent tasked with coordinating agentic execution to solve a specific problem.

Your responsibilities are as follows:

1. Carefully review the request, ensuring you fully understand its purpose and requirements before execution.
2. Use the appropriate tools available to execute each step effectively, including (and possibly combining multiple tools as needed):
   - Make requests to an execution agent who can write and run code to solve your request.
   - Make requests to an arxiv agent that can query and summarize recent research papers on the ArXiv on a topic.
   - Utilize a rememberer agent that can query its memory for similar tasks so that you can remember if anything similar was done before.
       - You should consider using this agent anytime you want to check if you have taken a relevant past action.
3. Clearly document each action you take, including:
   - The tools or methods you used.
   - Any code written, commands executed, or searches performed by the agents you are working with.
   - Outcomes, results, or errors encountered during execution.
4. Immediately highlight and clearly communicate any steps that appear unclear, unsafe, or impractical before proceeding.

Your goal is to carry out the provided plan accurately, safely, and transparently, maintaining accountability at each step.
"""


model = ChatOpenAI(
    model="gpt-5-nano",
    # max_completion_tokens=50000,
)
embedding = init_embeddings("openai:text-embedding-3-large")
memory = AgentMemory(embedding_model=embedding)

arxiver = ArxivAgent(
    llm=model,
    workspace=workspace,
    summarize=True,
    process_images=False,
    max_results=3,
    rag_embedding=embedding,
    database_path="database_neutron_star",
    summaries_path="database_summaries_neutron_star",
    vectorstore_path="vectorstores_neutron_star",
    download=True,
)

executor = ExecutionAgent(llm=model, workspace=workspace)

rememberer = RecallAgent(llm=model, memory=memory, workspace=workspace)


@tool
async def query_arxiver(search_query: str, context: str) -> str:
    """
    Use the Arxiv agent to search for research papers on Arxiv and summarize them using the specified context.

    Args:
        search_query: Between 1 and 8 words search query for the arxiv search api
        context: Contexual information to be extracted from each paper

    """
    print(f"{GREEN}[Arxiver Search] - {search_query}{RESET}")
    print(f"{GREEN}[Arxiver Context] - {context}{RESET}")
    return await arxiver.ainvoke(query=search_query, context=context)


@tool
async def query_executor(
    request: str, state: Annotated[dict, InjectedState]
) -> ExecutionState:
    """
    Use the Execution agent to write and run code to solve a task.

    Args:
        request: Text request of the task to be carried out. Be clear about the goals and any important details

    """
    print(f"{RED}[Executor Request] - {request}{RESET}")
    return await executor.ainvoke(request)


@tool
async def query_rememberer(request: str) -> str:
    """
    Check logs of past tasks to see if you have a memory of doing something similar

    Args:
        request: Short string to be used as a RAG query to identify similar previous messages

    """
    print(f"{BLUE}[Rememberer Request] - {request}{RESET}")
    return await rememberer.ainvoke(query=request)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    code_files: list[str]


class CombinedAgent(AgentWithTools, BaseAgent):
    state_type = State

    def __init__(
        self,
        llm: BaseChatModel,
        log_state: bool = False,
        **kwargs,
    ):
        tools = [query_arxiver, query_executor, query_rememberer]
        super().__init__(llm, tools=tools, **kwargs)
        self.runner_prompt = runner_prompt
        self.recap_prompt = recap_prompt
        self.log_state = log_state

    # Define the function that calls the model
    async def _runner(self, state: State) -> State:
        existing_messages = state.get("messages", [])
        if existing_messages and isinstance(
            existing_messages[0], SystemMessage
        ):
            prepared_messages = existing_messages.copy()
            prepared_messages[0] = SystemMessage(content=self.runner_prompt)
        else:
            prepared_messages = [
                SystemMessage(content=self.runner_prompt),
                *existing_messages,
            ]

        response = await self.llm.ainvoke(
            prepared_messages,
            {"configurable": {"thread_id": self.thread_id}},
        )
        updated_messages = [*prepared_messages, response]

        if self.log_state:
            saved_state = state.copy()
            saved_state["messages"] = updated_messages
            self.write_state("combined_agent.json", saved_state)

        return {"messages": updated_messages}

    # Define the function that calls the model
    async def _summarize(self, state: ExecutionState) -> ExecutionState:
        messages = [SystemMessage(content=recap_prompt)] + state["messages"]
        response = await self.llm.ainvoke(
            messages, {"configurable": {"thread_id": self.thread_id}}
        )
        memories: list[str] = []
        # Handle looping through the messages
        for x in state["messages"]:
            if not isinstance(x, AIMessage) or not x.tool_calls:
                memories.append(x.text)
            else:
                tool_strings = []
                for tool in x.tool_calls:
                    tool_name = "Tool Name: " + tool["name"]
                    tool_strings.append(tool_name)
                    tool_strings.extend(
                        f"Arg: {y!s}\nValue: {tool['args'][y]!s}"
                        for y in tool["args"]
                    )
                memories.append("\n".join(tool_strings))
        memories.append(response.text)
        memory.add_memories(memories)
        updated_messages = [*state["messages"], response]

        if self.log_state:
            save_state = state.copy()
            save_state["messages"] = updated_messages
            self.write_state(self.workspace / "combined_agent.json", save_state)
        return {"messages": updated_messages}

    def _build_graph(self):
        self.llm = self.llm.bind_tools(self.tools.values())
        self.add_node(self._runner)
        self.add_node(self.tool_node, "_tool_node")
        self.add_node(self._summarize)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.set_entry_point("_runner")

        self.graph.add_conditional_edges(
            "_runner",
            should_continue,
            {
                "continue": "_tool_node",
                "summarize": "_summarize",
            },
        )

        self.graph.add_edge("_tool_node", "_runner")
        self.graph.set_finish_point("_summarize")


# Define the function that determines whether to continue or not
def should_continue(state: ExecutionState) -> Literal["summarize", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarize"
    # Otherwise if there is, we continue
    else:
        return "continue"


async def main():
    agent = CombinedAgent(llm=model, log_state=True, workspace=workspace)
    result = await agent.ainvoke("""What are the constraints on the neutron star radius and what uncertainties are there on the constraints?
                Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This
                will be reviewed by experts in the field so technical accuracy and clarity is critical.""")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
