"""
Demo of `PlanningAgent` + `ExecutionAgent`.

Plans, implements, and benchmarks several techniques to compute the N-th
Fibonacci number, then explains which approach is the best.
"""

import sqlite3
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from rich import get_console

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.workflows import PlanningExecutorWorkflow

console = get_console()

# Define the workspace
workspace = "example_fibonacci_finder"

# Define a simple problem
index_to_find = 35
problem = (
    f"Create a single python script to compute the Fibonacci \n"
    f"number at position {index_to_find} in the sequence.\n\n"
    f"Compute the answer through more than one distinct technique, \n"
    f"benchmark and compare the approaches then explain which one is the best."
)


# Init the model
model = init_chat_model(model="openai:gpt-5-mini")

# Setup checkpointing
db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Init the agents with the model and checkpointer
executor = ExecutionAgent(llm=model, checkpointer=checkpointer)
executor_config = {
    "recursion_limit": 999_999,
    "configurable": {"thread_id": executor.thread_id},
}

planner = PlanningAgent(llm=model, checkpointer=checkpointer)
planner_config = {
    "recursion_limit": 999_999,
    "configurable": {"thread_id": planner.thread_id},
}

workflow = PlanningExecutorWorkflow(
    planner=planner,
    executor=executor,
    workspace=workspace,
    enable_metrics=True,
)

workflow.invoke(problem)

render_session_summary(workflow.thread_id)
