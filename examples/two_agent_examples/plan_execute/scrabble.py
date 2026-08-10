import sys
from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model
from rich import get_console
from rich.panel import Panel
from rich.text import Text

from ursa.agents import PlanningExecutionAgent
from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer
from ursa.util.events import configure_event_logging

configure_event_logging()
console = get_console()


def main(mode: str) -> None:
    """Find high-scoring alphabetically ordered words with one parent agent."""
    min_score = 10
    workspace = Path("scrabble")
    task = (
        "Find English words whose letters appear in strictly alphabetical "
        f"order and whose Scrabble score is at least {min_score}. Write and "
        "execute a Python program using standard Scrabble scores to evaluate "
        "candidates. Generate at least 10 qualifying words, sort them from "
        "highest to lowest score, and report each word and score. If internet "
        "access is needed, the corporate root CA is at ~/zscaler_root.pem."
    )

    console.print(
        Panel.fit(
            Text.from_markup(f"[bold cyan]Solving problem:[/] {task}"),
            border_style="cyan",
        )
    )

    model = init_chat_model(
        model="openai:gpt-5.4-mini" if mode == "prod" else "ollama:llama3.1:8b",
        max_tokens=10000 if mode == "prod" else 4000,
        max_retries=2,
    )
    workspace.mkdir(parents=True, exist_ok=True)
    checkpointer = Checkpointer.from_workspace(workspace)

    agent = PlanningExecutionAgent(
        llm=model,
        workspace=workspace,
        checkpointer=checkpointer,
        thread_id="run-" + uuid4().hex[:8],
        enable_metrics=True,
    )
    try:
        with console.status(
            "[bold green]Planning and executing . . .", spinner="point"
        ):
            result = agent.invoke(task, config={"recursion_limit": 999_999})
        console.print(
            Panel(
                result["messages"][-1].text,
                title="[yellow]Final result",
            )
        )
        render_session_summary(agent.thread_id)
    finally:
        agent.close()


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "local"
    if mode not in {"local", "prod"}:
        raise SystemExit("Usage: scrabble.py [local|prod]")
    main(mode)
