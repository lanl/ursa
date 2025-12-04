"""
Demo of `PlanningAgent` + `ExecutionAgent`.

Plans, implements, and benchmarks several techniques to compute the N-th
Fibonacci number, then explains which approach is the best.
"""

from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from rich import get_console
from rich.panel import Panel

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer
from ursa.util.plan_renderer import render_plan_steps_rich

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
checkpointer = Checkpointer.from_workspace(Path(workspace))

# Init the agents with the model and checkpointer
thread_id = uuid4().hex
executor = ExecutionAgent(
    llm=model, checkpointer=checkpointer, thread_id=thread_id
)
planner = PlanningAgent(
    llm=model, checkpointer=checkpointer, thread_id=thread_id
)

# Create a plan
with console.status(
    "[bold deep_pink1]Planning overarching steps . . .",
    spinner="point",
    spinner_style="deep_pink1",
):
    planner_prompt = f"Break this down into one step per technique:\n{problem}"

    planning_output = planner.invoke({
        "messages": [HumanMessage(content=planner_prompt)],
        "reflection_steps": 0,
    })

    render_plan_steps_rich(planning_output["plan_steps"])


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
        f"[bold orange3]Solving Step {i}:[/]\n[orange3]{step_prompt}[/]"
    )

    # Invoke the agent
    result = executor.invoke({
        "messages": [HumanMessage(content=step_prompt)],
        "workspace": workspace,
    })

    last_step_summary = result["messages"][-1].content
    console.print(
        Panel(
            last_step_summary,
            title=f"Step {i + 1} Final Response",
            border_style="orange3",
        )
    )

render_session_summary(thread_id)
