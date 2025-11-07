import sqlite3
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.workflows import SimulationUseWorkflow

# Define the workspace
workspace = "example_dcopf_use"

executor_model = init_chat_model(model="openai:o3")
planner_model = init_chat_model(model="openai:o3")

# Setup checkpointing
db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Init the agents with the model and checkpointer
executor = ExecutionAgent(
    llm=executor_model, checkpointer=checkpointer, enable_metrics=True
)
planner = PlanningAgent(
    llm=planner_model, checkpointer=checkpointer, enable_metrics=True
)

problem = (
    "Your task is to perform a parameter sweep of a complex computational model. "
    "The parameter sweep will be performed on the load parameters 10 times by choosing "
    "a random number between 0.8 and 1.2 and multiplying the load by this factor"
    "I require that each parameter configuration be stored in its own input file. "
    "I require that the code used to perform the task be stored."
    "I require that the code be executed and saved to a file. "
    "Produce a plot with opf objective value on the x axis and load factor on the y axis."
)

tool_description = """
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

workflow = SimulationUseWorkflow(
    planner=planner,
    executor=executor,
    workspace=workspace,
    tool_description=tool_description,
    enable_metrics=True,
)

workflow.invoke(problem)  # raw_debug=True doesn't seem to trigger anything.

render_session_summary(workflow.thread_id)
