import sqlite3
from pathlib import Path
from typing import Any, Mapping

from langgraph.checkpoint.sqlite import SqliteSaver
from rich import get_console
from rich.panel import Panel

from ursa.util.plan_renderer import render_plan_steps_rich
from ursa.workflows.base_workflow import BaseWorkflow

console = get_console()


class PlanningExecutorWorkflow(BaseWorkflow):
    """
    The Planning-Executor workflow is a workflow that composes two agents in a for-loop:
        - The planning agent takes the user input, develops a step-by-step plan as a list
        - The list is passed, entry by entry to an execution agent to carry out the plan.
    """

    def __init__(self, planner, executor, workspace, **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace

        # Setup checkpointing
        db_path = Path(workspace) / "checkpoint.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        self.planner.checkpointer = checkpointer
        self.executor.checkpointer = checkpointer

    def _invoke(self, inputs: Mapping[str, Any], **kw):
        task = str(inputs["task"])
        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            planner_prompt = (
                f"Break this down into one step per technique:\n{task}"
            )
            planning_output = self.planner.invoke(planner_prompt)

            render_plan_steps_rich(planning_output["plan"].steps)

        # Execution loop
        last_step_summary = "No previous step."
        for i, step in enumerate(planning_output["plan"].steps):
            step_prompt = (
                f"You are contributing to the larger solution:\n"
                f"{task}\n\n"
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

            result = self.executor.invoke(step_prompt)

            last_step_summary = result["messages"][-1].text

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
    from uuid import uuid4

    from langchain.chat_models import init_chat_model

    from ursa.agents import ExecutionAgent, PlanningAgent
    from ursa.observability.timing import render_session_summary

    tid = "run-" + uuid4().hex[:8]

    # Define the workspace
    workspace = "example_fibonacci_finder"

    # Define a simple problem
    index_to_find = 35

    problem = (
        f"Create a single python script to compute the Fibonacci \n"
        f"number at position {index_to_find} in the sequence.\n\n"
        # f"Compute the answer through more than one distinct technique, \n"
        # f"benchmark and compare the approaches then explain which one is the best."
    )

    # Setup Planning Agent
    planner_model = init_chat_model(model="openai:o4-mini")
    planner = PlanningAgent(
        llm=planner_model, enable_metrics=True, thread_id=tid
    )

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


if __name__ == "__main__":
    main()
