from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

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

model = ChatOllama(model="gpt-oss:20b")

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

memory = AgentMemory(embedding_model=embedding_model, path=".")


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
    final_results = executor.invoke(init)
else:
    final_results = executor.invoke(problem)

for x in final_results["messages"]:
    print(x.content)
