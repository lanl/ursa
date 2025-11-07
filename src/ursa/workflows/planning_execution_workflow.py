# planning_executor.py
from langchain_core.messages import HumanMessage
from rich import get_console
from rich.panel import Panel

from ursa.workflows.base_workflow import BaseWorkflow

"""
The Planning-Executor workflow is a workflow that composes two agents in a for-loop:
  - The planning agent takes the user input, develops a step-by-step plan as a list
  - The list is passed, entry by entry to an execution agent to carry out the plan.
"""

console = get_console()


class PlanningExecutorWorkflow(BaseWorkflow):
    def __init__(self, planner, executor, workspace, **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.workspace = workspace
        self._adopt(self.planner)
        self._adopt(self.executor)

    def _invoke(self, task: str, **kw):
        with console.status(
            "[bold deep_pink1]Planning overarching steps . . .",
            spinner="point",
            spinner_style="deep_pink1",
        ):
            planner_prompt = (
                f"Break this down into one step per technique:\n{task}"
            )
            planning_output = self.planner.invoke({
                "messages": [HumanMessage(content=planner_prompt)]
            })

            console.print(
                Panel(
                    planning_output["messages"][-1].content,
                    title="[bold yellow1 on black]:clipboard: Plan",
                    border_style="yellow1 on black",
                    style="yellow1 on black",
                )
            )

        # Execution loop
        last_step_summary = "No previous step."
        for i, step in enumerate(planning_output["plan_steps"]):
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
                    title=f"[bold orange3 on black]Solving Step {step['id']}",
                    border_style="orange3 on black",
                    style="orange3 on black",
                )
            )

            result = self.executor.invoke(
                {
                    "messages": [HumanMessage(content=step_prompt)],
                    "workspace": self.workspace,
                },
            )

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

    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.sqlite import SqliteSaver

    from ursa.agents import ExecutionAgent, PlanningAgent
    from ursa.observability.timing import render_session_summary

    tid = "run-" + __import__("uuid").uuid4().hex[:8]

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

    executor_model = init_chat_model(model="openai:o4-mini")
    planner_model = init_chat_model(model="openai:o4-mini")

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

    agent = PlanningExecutorWorkflow(
        planner=planner,
        executor=executor,
        workspace=workspace,
        enable_metrics=True,
    )
    agent.thread_id = tid

    agent.invoke(problem)

    render_session_summary(tid)
