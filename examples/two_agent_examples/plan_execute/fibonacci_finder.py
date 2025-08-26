"""
Demo of `PlanningAgent` + `ExecutionAgent`.

Plans, implements, and benchmarks several techniques to compute the N-th
Fibonacci number, then explains which approach is the best.
"""

import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite import SqliteSaver
from rich import get_console
from rich.panel import Panel

from ursa.agents import ExecutionAgent, PlanningAgent

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
model = ChatLiteLLM(model="openai/o3")

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
    "configurable": {"thread_id": executor.thread_id},
}

# Create a plan
with console.status(
    "[bold deep_pink1]Planning overarching steps . . .",
    spinner="point",
    spinner_style="deep_pink1",
):
    planner_prompt = f"Break this down into one step per technique:\n{problem}"

    planning_output = planner.action.invoke(
        {"messages": [HumanMessage(content=planner_prompt)]},
        planner_config,
    )

    console.print(
        Panel(
            planning_output["messages"][-1].content,
            title="[bold yellow1]:clipboard: Plan",
            border_style="yellow1",
        )
    )

# Execution loop
last_step_summary = "No previous step."
for i, step in enumerate(planning_output["plan_steps"]):
    step_prompt = (
        f"You are contributing to the larger solution:\n"
        f"{problem}\n\n"
        f"Previous-step summary:\n"
        f"{last_step_summary}\n\n"
        f"Current step:\n"
        f"{step}"
    )

    console.print(
        f"[bold orange3]Solving Step {step['id']}:[/]\n[orange3]{step_prompt}[/]"
    )

    # Invoke the agent
    result = executor.action.invoke(
        {
            "messages": [HumanMessage(content=step_prompt)],
            "workspace": workspace,
        },
        executor_config,
    )

    last_step_summary = result["messages"][-1].content
    console.print(
        Panel(
            last_step_summary,
            title=f"Step {i + 1} Final Response",
            border_style="orange3",
        )
    )
