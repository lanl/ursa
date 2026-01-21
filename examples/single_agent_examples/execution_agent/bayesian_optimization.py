from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary

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


# Initialize the agent
executor = ExecutionAgent(
    llm=model,
    enable_metrics=True,
)  # , enable_metrics=False if you don't want metrics

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

render_session_summary(executor.thread_id)
