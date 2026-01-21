# planning_executor.py
from rich import get_console
from rich.panel import Panel

from ursa.util.plan_renderer import render_plan_steps_rich
from ursa.workflows.base_workflow import BaseWorkflow

console = get_console()

code_schema_prompt = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CodeExecutionDescriptor",
  "type": "object",
  "properties": {
    "code": {
      "type": "object",
      "description": "Details about the code to run.",
      "properties": {
        "name": {
          "type": "string",
          "description": "The name or identifier of the code/script to run."
        },
        "options": {
          "type": "object",
          "description": "A set of key-value options or parameters for code execution.",
          "additionalProperties": {
            "type": ["string", "number", "boolean"]
          }
        }
      },
      "required": ["name"]
    },
    "inputs": {
      "type": "array",
      "description": "List of input parameters with names and descriptions.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the input parameter."
          },
          "description": {
            "type": "string",
            "description": "Description of the input parameter."
          }
        },
        "required": ["name", "description"]
      }
    },
    "outputs": {
      "type": "array",
      "description": "List of expected outputs with names and descriptions.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the output value."
          },
          "description": {
            "type": "string",
            "description": "Description of what the output represents."
          }
        },
        "required": ["name", "description"]
      }
    },
  },
  "required": ["code", "inputs", "outputs"]
}
"""


class SimulationUseWorkflow(BaseWorkflow):
    def __init__(
        self, planner, executor, workspace, tool_description, **kwargs
    ):
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace
        self.tool_schema = code_schema_prompt
        self.tool_description = tool_description

    def _invoke(self, task: str, **kw):
        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            planner_prompt = (
                f"Break this down into one step per technique:\n{task}"
                f"Here is the schema used to describe the computational model:\n{self.tool_schema}"
                f"Here is the description of what to run using this schema:\n{self.tool_description}"
            )

            planning_output = self.planner.invoke(planner_prompt)

            render_plan_steps_rich(planning_output["plan"].steps)

        # Execution loop
        last_step_summary = "No previous step."
        for i, step in enumerate(planning_output["plan"].steps):
            step_prompt = (
                f"You are contributing to the larger solution:\n"
                f"{task}\n\n"
                f"Here is the schema used to describe a relevant computational model:\n"
                f"{self.tool_schema}\n\n"
                f"Here is the description of what to run using this schema:\n"
                f"{self.tool_description}\n\n"
                f"Previous-step summary:\n"
                f"{last_step_summary}\n\n"
                f"Current step:\n"
                f"{step}\n\n"
                "Execute this step and report results for the executor of the next step."
                "Do not use placeholders."
                "Run commands to execute code generated for the step if applicable."
                "Only address the current step. Stay in your lane."
            )

            console.print(
                Panel(
                    step_prompt,
                    title=f"[bold orange3 on black]Solving Step {i + 1}",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )

            # Invoke the agent
            result = self.executor.invoke(step_prompt)

            last_step_summary = result["messages"][-1].content

            console.print(
                Panel(
                    last_step_summary,
                    title=f"Step {i + 1} Final Response",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )
        return last_step_summary


def main():
    import sqlite3
    from pathlib import Path
    from uuid import uuid4

    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.sqlite import SqliteSaver

    from ursa.agents import ExecutionAgent, PlanningAgent
    from ursa.observability.timing import render_session_summary

    tid = "run-" + uuid4().hex[:8]

    # Define the workspace
    workspace = "example_dcopf_use"

    executor_model = init_chat_model(model="openai:o4-mini")
    planner_model = init_chat_model(model="openai:o4-mini")

    # Setup checkpointing
    db_path = Path(workspace) / "checkpoint.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Init the agents with the model and checkpointer
    executor = ExecutionAgent(
        llm=executor_model,
        checkpointer=checkpointer,
        enable_metrics=True,
        thread_id=tid + "_executor",
    )

    planner = PlanningAgent(
        llm=planner_model,
        checkpointer=checkpointer,
        enable_metrics=True,
        thread_id=tid + "_planner",
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
        tool_schema=code_schema_prompt,
        enable_metrics=True,
    )

    workflow(problem)  # raw_debug=True doesn't seem to trigger anything.

    render_session_summary(tid + "_executor")


if __name__ == "__main__":
    main()
