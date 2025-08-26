from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langchain_openai import OpenAIEmbeddings

from ursa.agents import ExecutionAgent
from ursa.util.memory_logger import AgentMemory

### Run a simple example of an Execution Agent.

# Define a simple problem
problem = """ 
Optimize the six-hump camel function. 
    Start by evaluating that function at 10 locations.
    Then utilize Bayesian optimization to build a surrogate model 
        and sequentially select points until the function is optimized. 
    Carry out the optimization and report the results.
"""

model = ChatLiteLLM(
    model="openai/o3",
    max_tokens=30000,
)

embedding_kwargs = None
embedding_model = OpenAIEmbeddings(**(embedding_kwargs or {}))
memory = AgentMemory(embedding_model=embedding_model)

# Initialize the agent
executor = ExecutionAgent(agent_memory=memory, llm=model)

set_workspace = False

if set_workspace:
    # Syntax if you want to explicitly set the directory to work in
    init = {
        "messages": [HumanMessage(content=problem)],
        "workspace": "workspace_BO",
    }

    print(f"\nSolving problem: {problem}\n")

    # Solve the problem
    final_results = executor.action.invoke(init)
else:
    final_results = executor.run(problem)

for x in final_results["messages"]:
    print(x.content)
