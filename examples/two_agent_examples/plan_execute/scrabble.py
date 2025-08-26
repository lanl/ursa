import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.sqlite import SqliteSaver

# rich console stuff for beautification
from rich import get_console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ursa.agents import ExecutionAgent, PlanningAgent

console = get_console()  # always returns the same instance


def main(mode: str):
    """Run a simple example of an agent."""
    try:
        # 1. problem statement

        min_score = 10
        workspace = "scrabble"
        problem = (
            f"Find an English word whose letters appear in strictly alphabetical order "
            f"and whose Scrabble score is at least {min_score}.  Write a Python program using normal "
            f"Scrabble letter scores / rules to evaluate proposed words.  Generate at least 10 "
            f"words that have at least this minimum score of {min_score}, sort them highest to "
            f"lowest, and report the words with their scores.  "
        )

        problem = problem + (
            "I am on a corporate VPN.  If you need to access the internet, my corporate root CA "
            "certificate is at ~/zscaler_root.pem"
        )

        # print the problem we're solving in a nice little box / panel
        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold cyan]Solving problem:[/] {problem}",
                    justify="center",
                ),
                border_style="cyan",
            )
        )

        # 2. LLM & agents
        model = ChatLiteLLM(
            model="openai/o3"
            # model="openai/o1"
            if mode == "prod"
            else "ollama_chat/llama3.1:8b",
            max_tokens=10000 if mode == "prod" else 4000,
            max_retries=2,
        )

        db_path = Path(workspace) / "checkpoint.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        # Initialize the agents
        planner = PlanningAgent(llm=model, checkpointer=checkpointer)
        executor = ExecutionAgent(llm=model, checkpointer=checkpointer)

        # 3. top level planning
        # planning agent . . .
        with console.status(
            "[bold green]Planning overarching steps . . .", spinner="point"
        ):
            planning_output = planner.action.invoke(
                {"messages": [HumanMessage(content=problem)]},
                {
                    "recursion_limit": 999_999,
                    "configurable": {"thread_id": planner.thread_id},
                },
            )

        console.print(
            Panel(
                planning_output["messages"][-1].content, title="[yellow]📋 Plan"
            )
        )

        last_step_summary = "Beginning to break down step 1 of the plan."
        detail_planner_prompt = "Flesh out the details of this step and generate substeps to handle the details."

        # ── OUTER progress bar over main plan steps ─────────────────────────────
        with Progress(
            SpinnerColumn(spinner_name="point"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            transient=True,
        ) as progress:
            # planning_output is the main planning agent's plans - so they're high level
            # steps that need to be carried out

            main_task = progress.add_task(
                "Main plan steps", total=len(planning_output["plan_steps"])
            )

            # for each of the overarching planning steps . . .
            main_step_number = 1
            for main_step in planning_output["plan_steps"]:
                # ---- detail planning -------------------------------------------------
                step_prompt = (
                    f"You are contributing to the larger solution:\n{problem}\n\n"
                    f"Previous-step summary: {last_step_summary}\n"
                    f"Current step: {main_step}\n\n"
                    f"{detail_planner_prompt}"
                )
                console.print(
                    Panel.fit(
                        Text.from_markup(
                            f"[bold cyan]STEP {main_step_number} - LLM Prompt:[/] {step_prompt}",
                            justify="center",
                        ),
                        border_style="cyan",
                    )
                )

                detail_output = planner.action.invoke(
                    {"messages": [HumanMessage(content=step_prompt)]},
                    {
                        "recursion_limit": 999_999,
                        "configurable": {"thread_id": planner.thread_id},
                    },
                )

                # ---- sub-steps execution --------------------------------------------
                sub_task = progress.add_task(
                    f"Sub-steps for: {str(main_step)[:40]}…",
                    total=len(detail_output["plan_steps"]),
                )

                last_sub_summary = "Start sub-steps."
                sub_step_number = 1
                for sub in detail_output["plan_steps"]:
                    sub_prompt = (
                        f"You are contributing to the larger solution:\n{problem}\n\n"
                        f"Previous-substep summary: {last_sub_summary}\n"
                        f"Current step: {sub}\n\n"
                        "Execute this step and report the results fully—no placeholders."
                    )
                    console.print(
                        Panel.fit(
                            Text.from_markup(
                                f"[bold red]Sub-STEP {sub_step_number} - LLM Prompt:[/] {sub_prompt}",
                                justify="center",
                            ),
                            border_style="red",
                        )
                    )

                    final_results = executor.action.invoke(
                        {
                            "messages": [HumanMessage(content=sub_prompt)],
                            "workspace": workspace,
                        },
                        {
                            "recursion_limit": 999_999,
                            "configurable": {"thread_id": executor.thread_id},
                        },
                    )

                    last_sub_summary = final_results["messages"][-1].content
                    progress.console.log(last_sub_summary)  # live streaming log
                    progress.advance(sub_task)

                    sub_step_number += 1

                progress.remove_task(sub_task)  # collapse bar
                last_step_summary = last_sub_summary
                progress.advance(main_task)
                main_step_number += 1

        # ── 5 · Wrap-up ──────────────────────────────────────────
        answer = last_step_summary
        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold white on green] ✔  Answer:[/] {answer}"
                ),
                border_style="green",
            )
        )
        return answer

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    mode = "dev" if sys.argv[-1] == "dev" else "prod"
    final_output = main(mode=mode)  # dev or prod
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)

    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold white on green] ✔  Answer:[/] {final_output}"
            ),
            border_style="green",
        )
    )

    console.rule("[bold cyan]Run complete")
