from deepagents import CompiledSubAgent, create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from ursa.agents import ExecutionAgent

llm = init_chat_model("openai:gpt-5-nano")
exe_graph = ExecutionAgent(llm=llm)._action

# # Create a custom agent graph
# custom_graph = create_agent(
#     model=exe_graph,
#     # tools=specialized_tools,
#     system_prompt="You are a specialized agent for data analysis...",
# )

# Use it as a custom subagent
exe_subagent = CompiledSubAgent(
    name="executor",
    description="Specialized agent for writing/executing code",
    runnable=exe_graph,
)

subagents = [exe_subagent]

agent = create_deep_agent(
    model=llm,
    # tools=[internet_search],
    system_prompt="You are a data scientist. When asked to write code, use the executor agent.",
    subagents=[exe_subagent],
)


results = []


def run(query: str):
    results.append(result := agent.invoke({"messages": [HumanMessage(query)]}))
    return result


run("Write a very minimal python script to compute Pi using Monte Carlo.")
