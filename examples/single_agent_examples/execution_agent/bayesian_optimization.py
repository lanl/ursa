from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer

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

workspace = Path("./workspace_BO")
checkpointer = Checkpointer.from_workspace(workspace)

# Initialize the agent
executor = ExecutionAgent(
    llm=model,
    enable_metrics=True,
    thread_id="BO_test",
    workspace=workspace,
    checkpointer=checkpointer,
)


final_results = executor.invoke(problem)

render_session_summary(executor.thread_id)
