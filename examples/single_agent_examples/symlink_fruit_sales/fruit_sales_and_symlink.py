import sys

import coolname
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

# rich console stuff for beautification
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ursa.agents import ExecutionAgent

console = Console()  # global console object

# notice the symlink example demonstrate a source and dest dir not having the same
# name - simple, but just want to draw attention to that
symlinkdict = {"source": "./data_dir", "dest": "data"}

problem = (
    # notice below we refer to the destination dir 'data' where we expect the working
    # dir will have a symlink to.  Notice also we're not specifically saying what the
    # filename is, so we're leaving it to the agentic framework to figure that out.
    """
Your task is to use the tabular data in the 'data' dir as input.  You are to produce
two plots:
1. a barplot of total fruit sales grouped by fruit.
2. a time-series with a 3-day rolling average of sales with lines for each
   of the fruits.
All plots are to be saved to disk (*DO NOT TRY AND DISPLAY TO THE USER'S SCREEN*).
Use seaborn for the plotting and make them look visually appealing such as black
lines around bars and any other additions you think are interesting.
"""
)

workspace = f"fruit_sales_{coolname.generate_slug(2)}"
workspace_header = f"[cyan] (- [bold cyan]{workspace}[reset][cyan] -) [reset]"


def main(model_name: str):
    """Run a simple example of an agent."""
    try:
        model = ChatLiteLLM(
            model=model_name,
            max_tokens=10000,
            max_retries=2,
        )

        # 4. Choose a fun emoji based on the model family (swap / extend as you add more)
        if model_name.startswith("openai"):
            model_emoji = "🤖"  # OpenAI
        elif "llama" in model_name.lower():
            model_emoji = "🦙"  # Llama
        else:
            model_emoji = "🧠"  # Fallback / generic LLM

        # 5. Print the panel, now with model info
        console.print(
            Panel.fit(
                f":rocket:  [bold bright_blue]{workspace}[/bold bright_blue]  :rocket:\n"
                f"{model_emoji}  [bold cyan]{model_name}[/bold cyan]",
                title="[bold green]ACTIVE WORKSPACE[/bold green]",
                border_style="bright_magenta",
                padding=(1, 4),
            )
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

        # Initialize the agent
        # no planning agent for this one - let's YOLO and go risk it
        executor = ExecutionAgent(llm=model)

        final_results = executor.action.invoke(
            {
                "messages": [HumanMessage(content=problem)],
                "workspace": workspace,
                "symlinkdir": symlinkdict,
            },
            {
                "recursion_limit": 999_999,
                "configurable": {"thread_id": executor.thread_id},
            },
        )

        last_step_summary = final_results["messages"][-1].content

        return last_step_summary, workspace

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    # ── interactive model picker ───────────────────────────────────────
    DEFAULT_MODELS = (
        "openai/o3",
        "openai/o3-mini",
    )

    try:
        print("\nChoose the model to run with:")
        for i, m in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {m}")
        print("Or type your own model string (Ctrl-C to quit):")

        while True:
            choice = input("> ").strip()

            # User chose one of the default numbers
            if choice in {"1", "2"}:
                model = DEFAULT_MODELS[int(choice) - 1]
                break

            # User typed a non-empty custom string
            if choice:
                model = choice
                break

            # Empty input → prompt again
            print("Please enter 1, 2, or a custom model name.")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

    # ── continue exactly as before ─────────────────────────────────────
    final_output, workspace = main(model_name=model)

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
    console.print(
        Panel.fit(
            f":rocket:  [bold bright_blue]{workspace}[/bold bright_blue]  :rocket:",
            title="[bold green]WORKSPACE RESULTS IN[/bold green]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )
