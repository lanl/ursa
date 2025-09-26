"""
Demo of `PlanningAgent` + `ExecutionAgent`.

Plans, implements, and benchmarks several techniques to compute the N-th
Fibonacci number, then explains which approach is the best.
"""


from pathlib import Path
import sqlite3
import coolname

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.agents import ExecutionAgent, PlanningAgent

from ursa.prompt_library.computation_tool_prompts import (
    code_schema_prompt
)


from rich import get_console
from rich.panel import Panel

console = get_console()

# Define the workspace
workspace = coolname.generate_slug(2)

# Define a simple problem
index_to_find = 35





code_instance = """
{
  "code": {
    "name": "PowerModels.jl",
    "options": {
      "description": "An open source code for optimizing power systems",
    }
  },
  "inputs": [
    {
      "name": "ieee14",
      "description": "Input data file" 
    },
    {
      "name": "dcopf",
      "description": "computation to run" 
    },
  ],
  "outputs": [
    {
      "name": "csv",
      "description": "The output is a julia dictionary.  For each run, 
      extract the MW output of each generator"
    },
  ],
}
"""



problem = (
    "Your task is to perform a parameter sweep of a complex computational model. "
    "The parameter sweep will be performed on the load parameters 10 times by choosing "
    "a random number between 0.8 and 1.2 and multiplying the load by this factor"
    "I require that each parameter configuration be stored in its own input file. "
    "I require that the code used to perform the task be stored."
    "I require that the code be executed and saved to a file. "
    "Produce a plot with opf objective value on the x axis and load factor on the y axis."
    
    f"Here is the schema used to describe the computational model:\n{code_schema_prompt}\n\n"

    f"Here is the description of what I want to run using this schema:\n{code_instance}"
)

# Init the model
model = ChatLiteLLM(
    model="openai/o3"
)

# Setup checkpointing
db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Init the agents with the model and checkpointer
executor = ExecutionAgent(llm=model, checkpointer=checkpointer)
executor_config = {
    "recursion_limit": 999_999,
    "configurable": { "thread_id": executor.thread_id }
}

planner = PlanningAgent(llm=model, checkpointer=checkpointer)
planner_config = {
    "recursion_limit": 999_999,
    "configurable": { "thread_id": executor.thread_id }
}

# Create a plan
with console.status("[bold deep_pink1]Planning overarching steps . . .",
                    spinner="point", spinner_style="deep_pink1"):
    planner_prompt = f"Break this down into one step per technique:\n{problem}"

    planning_output = planner.action.invoke({
        "messages": [HumanMessage(content=planner_prompt)]
        }, 
        planner_config,
    )
    
    console.print(
        Panel(planning_output["messages"][-1].content,
            title="[bold yellow1]:clipboard: Plan",
            border_style="yellow1")
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

    console.print(f"[bold orange3]Solving Step {step['id']}:[/]\n[orange3]{step_prompt}[/]")

    # Invoke the agent
    result = executor.action.invoke(
        {
            "messages": [HumanMessage(content=step_prompt)],
            "workspace": workspace,
        }, 
        executor_config
    )

    last_step_summary = result["messages"][-1].content
    console.print(Panel(last_step_summary, title=f"Step {i+1} Final Response", border_style="orange3"))
