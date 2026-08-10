"""Run the persistent planning/execution agent from a YAML configuration.

Unlike the former example, this program does not create independent planner and
executor agents or databases. One parent owns the model, workspace, thread, and
SQLite checkpointer; its native subgraphs persist under ``planner`` and
``executor`` checkpoint namespaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from rich import get_console
from rich.panel import Panel
from rich.text import Text

from ursa.agents import PlanningExecutionAgent
from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer
from ursa.util.events import configure_event_logging
from ursa.util.plan_execute_utils import (
    load_yaml_config,
    setup_llm,
    setup_workspace,
    timed_input_with_countdown,
)

configure_event_logging()
console = get_console()


def _choose_model(
    choices: tuple[str, ...], default: str | None, timeout: int
) -> str:
    """Choose one model for every phase of the parent graph."""
    if default and (timeout <= 0 or not sys.stdin.isatty()):
        return default
    if timeout <= 0 or not sys.stdin.isatty():
        return default or choices[0]

    print("\nChoose the model to use for planning and execution:")
    for index, model in enumerate(choices, 1):
        print(f"  {index}. {model}")
    print(f"Press Enter for: {default or choices[0]}")
    choice = timed_input_with_countdown("> ", timeout)
    if not choice or not choice.strip():
        return default or choices[0]
    value = choice.strip()
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1]
    return value


def run(
    model_name: str,
    config: Any,
    *,
    workspace_override: str | None = None,
    planning_mode: str = "single",
) -> tuple[str, str]:
    """Construct and invoke one persistent planning/execution parent."""
    problem = str(getattr(config, "problem", "") or "").strip()
    if not problem:
        raise ValueError(
            "The YAML configuration must define a non-empty problem."
        )

    project = str(getattr(config, "project", "run") or "run")
    models_cfg = getattr(config, "models", {}) or {}
    workspace = setup_workspace(workspace_override, project, model_name)
    workspace_path = Path(workspace)

    model = setup_llm(
        model_choice=model_name,
        models_cfg=models_cfg,
        agent_name="planning_execution",
    )
    checkpointer = Checkpointer.from_workspace(workspace_path)
    agent = PlanningExecutionAgent(
        llm=model,
        workspace=workspace_path,
        checkpointer=checkpointer,
        thread_id=workspace_path.name,
        enable_metrics=True,
        metrics_dir="ursa_metrics",
    )

    if planning_mode == "hierarchical":
        problem = (
            f"{problem}\n\nCreate a sufficiently detailed plan so each step is "
            "directly executable. Keep all planning and execution in this one "
            "persistent parent graph."
        )

    agent_input: dict[str, Any] = {
        "messages": [HumanMessage(content=problem)],
        "symlinkdir": getattr(config, "symlink", {}) or {},
    }
    try:
        with console.status(
            "[bold green]Planning and executing . . .", spinner="point"
        ):
            state = agent.invoke(
                agent_input,
                config={"recursion_limit": 999_999},
            )
        final_output = state["messages"][-1].text
        render_session_summary(agent.thread_id)
        return final_output, workspace
    finally:
        agent.close()


def parse_args() -> tuple[argparse.Namespace, Any, str, str]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--workspace", help="Persistent parent workspace")
    parser.add_argument(
        "--planning-mode",
        choices=["hierarchical", "single"],
        help="Plan detail level; both modes use the same native parent graph.",
    )
    parser.add_argument(
        "--interactive-timeout",
        type=int,
        default=60,
        help="Seconds before model selection uses the configured default.",
    )
    parser.add_argument(
        "--stepwise-exit",
        action="store_true",
        help="Removed: use LangGraph Command interrupts for approval/resume.",
    )
    parser.add_argument(
        "--resume-from",
        help="Removed: the parent resumes its own SQLite checkpoint namespaces.",
    )
    args = parser.parse_args()
    if args.stepwise_exit or args.resume_from:
        parser.error(
            "--stepwise-exit and --resume-from belonged to the old two-agent "
            "example. Use the parent's persisted thread and Command-based "
            "interrupt/resume instead."
        )

    config = load_yaml_config(args.config)
    models_cfg = getattr(config, "models", {}) or {}
    choices = tuple(
        models_cfg.get("choices")
        or (
            "openai:gpt-5",
            "openai:gpt-5.4-mini",
            "openai:o3",
            "openai:o3-mini",
        )
    )
    model_name = _choose_model(
        choices,
        models_cfg.get("default"),
        args.interactive_timeout,
    )
    configured_mode = getattr(config, "planning_mode", None)
    planning_cfg = getattr(config, "planning", {}) or {}
    if isinstance(planning_cfg, dict):
        configured_mode = planning_cfg.get("mode", configured_mode)
    planning_mode = args.planning_mode or configured_mode or "single"
    return args, config, model_name, planning_mode


def display_result(final_output: str, workspace: str) -> None:
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold white on green] ✔  Final Output:[/] {final_output}"
            ),
            border_style="green",
        )
    )
    console.rule("[bold cyan]Run complete")
    console.print(
        Panel.fit(
            f"[bold bright_blue]{workspace}[/bold bright_blue]",
            title="[bold green]WORKSPACE RESULTS IN[/bold green]",
            border_style="bright_magenta",
        )
    )


if __name__ == "__main__":
    cli_args, yaml_config, selected_model, selected_mode = parse_args()
    output, active_workspace = run(
        selected_model,
        yaml_config,
        workspace_override=cli_args.workspace,
        planning_mode=selected_mode,
    )
    display_result(output, active_workspace)
