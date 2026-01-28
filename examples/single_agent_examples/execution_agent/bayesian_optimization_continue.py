from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer

### Run a simple example of continuing the Execution Agent from a checkpoint.

problem = """
Make a plot of the evaluations of the target function with the running minimum overlaid to show convergence. 
Make a second plot highlighting the important inputs of the function.
"""

model = init_chat_model(
    model="openai:gpt-5-mini",
    max_completion_tokens=30000,
)

workspace = Path(
    "./workspace_BO"
)  # Point at the same workspace as the original run.
checkpointer = Checkpointer.from_workspace(workspace)

# Initialize the agent
executor = ExecutionAgent(
    llm=model,
    enable_metrics=True,
    thread_id="BO_test",  # Set the thread_id to the same as the previous result
    workspace=workspace,
    checkpointer=checkpointer,
)

final_results = executor.invoke(problem)


render_session_summary(executor.thread_id)
