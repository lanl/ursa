"""
Demo of Computation Agent to perform a parameter sweep on the DC OPF problem

"""


from pathlib import Path
import sqlite3
import coolname

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.agents import ComputationToolAgent

from ursa.prompt_library.computation_tool_prompts import (
    code_schema_prompt
)


from rich import get_console
from rich.panel import Panel

console = get_console()

# Define the workspace
workspace = coolname.generate_slug(2)


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

problem = (
    "Your task is to perform a parameter sweep of a complex computational model. "
    "The parameter sweep will be performed on the load parameters 10 times by choosing "
    "a random number between 0.8 and 1.2 and multiplying the load by this factor"
    "I require that each parameter configuration be stored in its own input file. "
    "I require that the code used to perform the task be stored."
    "I require that the code be executed and saved to a file. "
    "Produce a plot with opf objective value on the x axis and load factor on the y axis."
)


# Init the model
model = ChatLiteLLM(
    model="openai/o3"
)

# Initialize the agent
computation_tool = ComputationToolAgent(llm=model)

final_results = computation_tool.run(problem, tool_description)

for x in final_results["messages"]:
    print(x.content)

