# ruff: noqa: TID251
"""Deprecated planner/executor workflow specialized for simulator schemas."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

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
    }
  },
  "required": ["code", "inputs", "outputs"]
}
"""


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content or "")


class SimulationUseWorkflow(BaseWorkflow):
    """Deprecated simulator-schema-aware planning/execution loop."""

    def __init__(
        self,
        planner: Any,
        executor: Any,
        workspace: Any,
        tool_description: str,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "SimulationUseWorkflow is deprecated; use "
            "PlanningExecutionAgent with simulator tools instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace
        self.tool_schema = code_schema_prompt
        self.tool_description = tool_description

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        *,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        task = str(inputs.get("task", "") or "").strip()
        if not task:
            raise ValueError("SimulationUseWorkflow requires a task.")

        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            planner_prompt = (
                f"Break this down into one step per technique:\n{task}"
                "Here is the schema used to describe the computational model:\n"
                f"{self.tool_schema}"
                "Here is the description of what to run using this schema:\n"
                f"{self.tool_description}"
            )
            planning_output = self.planner.invoke(planner_prompt)
            render_plan_steps_rich(planning_output["plan"].steps)

        last_step_summary = "No previous step."
        for i, step in enumerate(planning_output["plan"].steps):
            step_prompt = (
                f"You are contributing to the larger solution:\n{task}\n\n"
                "Here is the schema used to describe a relevant computational model:\n"
                f"{self.tool_schema}\n\n"
                "Here is the description of what to run using this schema:\n"
                f"{self.tool_description}\n\n"
                f"Previous-step summary:\n{last_step_summary}\n\n"
                f"Current step:\n{step}\n\n"
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
            result = self.executor.invoke(step_prompt)
            last_step_summary = _message_text(result["messages"][-1])
            console.print(
                Panel(
                    last_step_summary,
                    title=f"Step {i + 1} Final Response",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )
        return last_step_summary
