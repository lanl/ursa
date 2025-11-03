from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary
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

model = init_chat_model(
    model="openai:gpt-5-mini",
    max_completion_tokens=30000,
)

embedding_kwargs = None
embedding_model = OpenAIEmbeddings(**(embedding_kwargs or {}))
memory = AgentMemory(embedding_model=embedding_model)

tid = "run-" + __import__("uuid").uuid4().hex[:8]

# Initialize the agent
executor = ExecutionAgent(
    agent_memory=memory, llm=model, enable_metrics=True
)  # , enable_metrics=False if you don't want metrics
executor.thread_id = tid

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

render_session_summary(tid)

for x in final_results["messages"]:
    print(x.content)
