"""
Demo of `ExecutionAgent`.

Iteratively builds three solutions for summing the first N integers,
benchmarks them against each other and verifies the results mach.
"""

import sqlite3
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite import SqliteSaver
from rich import get_console
from rich.panel import Panel

from ursa.agents import ExecutionAgent

console = get_console()  # always returns the same instance

# Define the workspace
workspace = "example_integer_sum"

# Define a simple problem
problem = [
    """
Create a python function that finds the sum of the first N positive integers with a for loop.
Time how long it takes to sum the first 10,000 and print the results to the console.
""",
    """
Add a new function that computes the same value using the built-in sum function, no loops.
Compare the timing for these two methods on the first 100,000 integers, and check the results match.
""",
    """
Add a third function that uses a static formula the compute the same value.
Compare the timing for all three methods on the first million integers, and check the results match.
""",
]

# Init the model
model = ChatLiteLLM(model="openai/o3")

# Setup checkpointing
db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Init the execution agent with the model and checkpointer
executor = ExecutionAgent(llm=model, checkpointer=checkpointer)
executor_config = {
    "recursion_limit": 999_999,
    "configurable": {"thread_id": executor.thread_id},
}

# Execution loop
for i, step_prompt in enumerate(problem):
    console.print(
        f"[bold orange3]Solving Step {i + 1}:[/]\n[orange3]{step_prompt}[/]"
    )

    # Invoke the agent
    result = executor.action.invoke(
        {
            "messages": [HumanMessage(content=step_prompt)],
            "workspace": workspace,
        },
        executor_config,
    )

    console.print(
        Panel(
            result["messages"][-1].content,
            title=f"Step {i + 1} Final Response",
            border_style="orange3",
        )
    )
