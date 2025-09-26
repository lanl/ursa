import os
from typing import Annotated, Literal

import randomname
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from typing_extensions import TypedDict

from ursa.agents import (
    ArxivAgent,
    BaseAgent,
    BaseChatModel,
    ExecutionAgent,
    ExecutionState,
    RecallAgent,
)
from ursa.prompt_library.execution_prompts import summarize_prompt
from ursa.util.memory_logger import AgentMemory

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

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


model = ChatLiteLLM(
    model="openai/o3",
    max_tokens=50000,
)
embedding = OpenAIEmbeddings()
memory = AgentMemory(embedding_model=embedding)

arxiver = ArxivAgent(
    llm=model,
    summarize=True,
    process_images=False,
    max_results=5,
    rag_embedding=embedding,
    database_path="database_neutron_star",
    summaries_path="database_summaries_neutron_star",
    vectorstore_path="vectorstores_neutron_star",
    download_papers=True,
)

executor = ExecutionAgent(llm=model)

rememberer = RecallAgent(llm=model, memory=memory)


@tool
def query_arxiver(search_query: str, context: str) -> str:
    """
    Use the Arxiv agent to search for research papers on Arxiv and summarize them using the specified context.

    Args:
        search_query: Between 1 and 8 words search query for the arxiv search api
        context: Contexual information to be extracted from each paper

    """
    print(f"{GREEN}[Arxiver Search] - {search_query}{RESET}")
    print(f"{GREEN}[Arxiver Context] - {context}{RESET}")
    return arxiver.run(arxiv_search_query=search_query, context=context)


@tool
def query_executor(
    request: str, state: Annotated[dict, InjectedState]
) -> ExecutionState:
    """
    Use the Execution agent to write and run code to solve a task.

    Args:
        request: Text request of the task to be carried out. Be clear about the goals and any important details

    """
    init = {
        "messages": [HumanMessage(content=request)],
        "workspace": state["workspace"],
    }
    print(f"{RED}[Executor Request] - {request}{RESET}")
    return executor.action.invoke(init)


@tool
def query_rememberer(request: str) -> str:
    """
    Check logs of past tasks to see if you have a memory of doing something similar

    Args:
        request: Short string to be used as a RAG query to identify similar previous messages

    """
    print(f"{BLUE}[Rememberer Request] - {request}{RESET}")
    return rememberer.remember(query=request)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_progress: str
    code_files: list[str]
    workspace: str


class CombinedAgent(BaseAgent):
    def __init__(
        self,
        llm: str | BaseChatModel = "openai/gpt-4o-mini",
        log_state: bool = False,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.runner_prompt = runner_prompt
        self.summarize_prompt = summarize_prompt
        self.tools = [query_arxiver, query_executor, query_rememberer]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)
        self.log_state = log_state

        self._initialize_agent()

    # Define the function that calls the model
    def runner(self, state: State) -> State:
        new_state = state.copy()
        if "workspace" not in new_state.keys():
            new_state["workspace"] = randomname.get_name()
            print(
                f"{RED}Creating the folder {BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} for this project.{RESET}"
            )
        os.makedirs(new_state["workspace"], exist_ok=True)

        if isinstance(new_state["messages"][0], SystemMessage):
            new_state["messages"][0] = SystemMessage(content=self.runner_prompt)
        else:
            new_state["messages"] = [
                SystemMessage(content=self.runner_prompt)
            ] + state["messages"]
        response = self.llm.invoke(
            new_state["messages"],
            {"configurable": {"thread_id": self.thread_id}},
        )
        if self.log_state:
            self.write_state("combined_agent.json", new_state)
        return {"messages": [response], "workspace": new_state["workspace"]}

    # Define the function that calls the model
    def summarize(self, state: ExecutionState) -> ExecutionState:
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
        response = self.llm.invoke(
            messages, {"configurable": {"thread_id": self.thread_id}}
        )
        memories = []
        # Handle looping through the messages
        for x in state["messages"]:
            if not isinstance(x, AIMessage):
                memories.append(x.content)
            elif not x.tool_calls:
                memories.append(x.content)
            else:
                tool_strings = []
                for tool in x.tool_calls:
                    tool_name = "Tool Name: " + tool["name"]
                    tool_strings.append(tool_name)
                    for y in tool["args"]:
                        tool_strings.append(
                            f"Arg: {str(y)}\nValue: {str(tool['args'][y])}"
                        )
                memories.append("\n".join(tool_strings))
        memories.append(response.content)
        memory.add_memories(memories)
        save_state = state.copy()
        save_state["messages"].append(response)
        if self.log_state:
            self.write_state("combined_agent.json", save_state)
        return {"messages": [response.content]}

    def _initialize_agent(self):
        self.graph = StateGraph(State)

        self.graph.add_node("agent", self.runner)
        self.graph.add_node("action", self.tool_node)
        self.graph.add_node("summarize", self.summarize)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "agent")

        self.graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "summarize": "summarize",
            },
        )

        self.graph.add_edge("action", "agent")
        self.graph.add_edge("summarize", END)

        self.action = self.graph.compile(checkpointer=self.checkpointer)

    def run(self, prompt, recursion_limit=1000):
        inputs = {"messages": [HumanMessage(content=prompt)]}
        return self.action.invoke(
            inputs,
            {
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": self.thread_id},
            },
        )


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


def main():
    agent = CombinedAgent(llm=model, log_state=True)
    result = agent.run(
        prompt="""What are the constraints on the neutron star radius and what uncertainties are there on the constraints? 
                Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
                will be reviewed by experts in the field so technical accuracy and clarity is critical."""
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
