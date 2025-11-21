"""
Demo of `PlanningAgent` + `ExecutionAgent`.

Plans, implements, and benchmarks several techniques to compute the N-th
Fibonacci number, then explains which approach is the best.
"""

from uuid import uuid4

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.workflows import PlanningExecutorWorkflow

tid = "run-" + uuid4().hex[:8]

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

# Setup Planning Agent
planner_model = init_chat_model(model="openai:o4-mini")
planner = PlanningAgent(llm=planner_model, enable_metrics=True, thread_id=tid)

# Setup Execution Agent
executor_model = init_chat_model(model="openai:o4-mini")
executor = ExecutionAgent(
    llm=executor_model, enable_metrics=True, thread_id=tid
)

# Initialize workflow
workflow = PlanningExecutorWorkflow(
    planner=planner, executor=executor, workspace=workspace
)

# Run problem through the workflow
workflow(problem)

# Print agent telemetry data
render_session_summary(tid)
