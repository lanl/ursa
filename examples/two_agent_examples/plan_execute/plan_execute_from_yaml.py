import argparse
import sys
from types import SimpleNamespace as NS
from typing import Any
from uuid import uuid4

import randomname
import yaml
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# rich console stuff for beautification
from rich import get_console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary

console = get_console()  # always returns the same instance


def main(model_name: str, config: Any, planning_mode: str = "hierarchical"):
    """
    Use:
      - config.project  (e.g., run dir names, titles - "sortnet" for example)
      - config.problem  (the whole long problem statement)
      - config.symlink  (None or a mapping source to dest) {"source": "...", "dest": "..."} or None
    """
    try:
        problem = getattr(config, "problem", "")
        project = getattr(config, "project", "run")
        workspace = f"{project}_{randomname.get_name()}"
        # workspace_header = f"[cyan] (- [bold cyan]{workspace}[reset][cyan] -) [reset]"
        symlinkdict = getattr(config, "symlink", {}) or None

        model = init_chat_model(
            model=model_name,
            max_completion_tokens=10000,
            max_retries=2,
            model_kwargs={
                # "reasoning": {"effort": "high"},
            },
            # temperature=0.2,
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
                    # justify="center",
                ),
                border_style="cyan",
            )
        )

        # Initialize the agents
        tid = uuid4().hex
        planner = PlanningAgent(llm=model, thread_id=tid)
        executor = ExecutionAgent(llm=model, thread_id=tid)

        # 3. top level planning
        # planning agent . . .
        with console.status(
            "[bold green]Planning overarching steps . . .", spinner="point"
        ):
            planning_output = planner.invoke(
                {"messages": [HumanMessage(content=problem)]},
                config={
                    "recursion_limit": 999_999,
                    "configurable": {"thread_id": planner.thread_id},
                },
            )
        console.print(
            Panel(
                planning_output["messages"][-1].content, title="[yellow]📋 Plan"
            )
        )

        last_step_summary = (
            "Beginning hierarchical execution."
            if planning_mode == "hierarchical"
            else "Beginning single-pass execution of top-level steps."
        )
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
                console.print(
                    Panel.fit(
                        Text.from_markup(
                            f"[bold cyan]STEP {main_step_number} - Current Step:[/] {main_step}"
                        ),
                        border_style="cyan",
                    )
                )

                if planning_mode == "hierarchical":
                    # ---- detail planning -------------------------------------------------
                    step_prompt = (
                        f"You are contributing to the larger solution:\n{problem}\n\n"
                        f"Previous-step summary: {last_step_summary}\n"
                        f"Current step: {main_step}\n\n"
                        f"{detail_planner_prompt}"
                    )
                    detail_output = planner.invoke(
                        {"messages": [HumanMessage(content=step_prompt)]},
                        config={
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
                                    f"[bold red]Sub-STEP {sub_step_number} - Current Sub-Step:[/] {sub}"
                                ),
                                border_style="red",
                            )
                        )

                        final_results = executor.invoke(
                            {
                                "messages": [HumanMessage(content=sub_prompt)],
                                "workspace": workspace,
                                "symlinkdir": symlinkdict,
                            },
                            config={
                                "recursion_limit": 999_999,
                                "configurable": {
                                    "thread_id": executor.thread_id
                                },
                            },
                        )

                        last_sub_summary = final_results["messages"][-1].content
                        progress.console.log(
                            last_sub_summary
                        )  # live streaming log
                        progress.advance(sub_task)
                        sub_step_number += 1

                    progress.remove_task(sub_task)  # collapse bar
                    last_step_summary = last_sub_summary

                else:
                    # ---- single-pass execution (no re-planning) -------------------------
                    sub_task = progress.add_task(
                        f"Execute: {str(main_step)[:40]}…", total=1
                    )
                    sub_prompt = (
                        f"You are contributing to the larger solution:\n{problem}\n\n"
                        f"Previous-step summary: {last_step_summary}\n"
                        f"Current step: {main_step}\n\n"
                        "Execute this step and report the results fully—no placeholders."
                    )
                    final_results = executor.invoke(
                        {
                            "messages": [HumanMessage(content=sub_prompt)],
                            "workspace": workspace,
                            "symlinkdir": symlinkdict,
                        },
                        config={
                            "recursion_limit": 999_999,
                            "configurable": {"thread_id": executor.thread_id},
                        },
                    )
                    last_step_summary = final_results["messages"][-1].content
                    progress.console.log(last_step_summary)
                    progress.advance(sub_task)
                    progress.remove_task(sub_task)

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

        render_session_summary(tid)

        return answer, workspace

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with YAML config.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--planning-mode",
        choices=["hierarchical", "single"],
        help="Choose 'hierarchical' (plan -> re-plan each step -> execute) or 'single' (plan once -> execute each step).",
    )
    args = parser.parse_args()

    # --- load YAML -> dict -> shallow namespace (top-level keys only) ---
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
            if not isinstance(raw_cfg, dict):
                raise ValueError("Top-level YAML must be a mapping/object.")
            cfg = NS(**raw_cfg)  # top-level attrs; nested remain dicts
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        sys.exit(2)

    # Optional fields we can use later:
    #   cfg.get("project")   -> e.g. "sortnet"
    #   cfg.get("problem")   -> long text block
    #   cfg.get("symlink")   -> {"source": "...", "dest": "..."} or None

    # ── interactive model picker (from config if provided) ────────────
    models_cfg = getattr(cfg, "models", {}) or {}
    DEFAULT_MODELS = tuple(
        models_cfg.get("choices")
        or (
            "openai:gpt-5",
            "openai:o3",
            "openai:o3-mini",
        )
    )
    DEFAULT_MODEL = models_cfg.get("default")  # may be None

    def _choose_planning_mode_interactive(default_mode: str) -> str:
        print("\nSelect planning mode:")
        print(
            "  1. hierarchical  (Plan -> re-plan each step -> execute sub-steps)"
        )
        print(
            "  2. single        (Plan once -> execute each top-level step directly)"
        )
        if default_mode:
            print(f"(Press Enter for default: {default_mode})")
        while True:
            choice = input("> ").strip().lower()
            if not choice and default_mode:
                return default_mode
            if choice.isdigit():
                if choice == "1":
                    return "hierarchical"
                if choice == "2":
                    return "single"
            if choice in ("hierarchical", "single"):
                return choice
            print("Please enter 1, 2, 'hierarchical', or 'single'.")

    try:
        print("\nChoose the model to run with:")
        for i, m in enumerate(DEFAULT_MODELS, 1):
            print(f"  {i}. {m}")
        if DEFAULT_MODEL:
            print(f"(Press Enter for default: {DEFAULT_MODEL})")
        print("Or type your own model string (Ctrl-C to quit):")

        while True:
            choice = input("> ").strip()

            if not choice and DEFAULT_MODEL:
                model = DEFAULT_MODEL
                break

            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(DEFAULT_MODELS):
                    model = DEFAULT_MODELS[idx - 1]
                    break

            if choice:  # custom model string
                model = choice
                break

            valid_nums = ", ".join(
                str(i) for i in range(1, len(DEFAULT_MODELS) + 1)
            )
            if DEFAULT_MODEL:
                print(
                    f"Please enter {valid_nums}, a custom model, or press Enter for default."
                )
            else:
                print(f"Please enter {valid_nums} or a custom model.")

        # -- planning-mode resolution: CLI > config > interactive > default --
        config_mode = None
        # support either cfg.planning.mode or cfg.planning_mode
        planning_cfg = getattr(cfg, "planning", None)
        if isinstance(planning_cfg, dict):
            config_mode = planning_cfg.get("mode")
        if not config_mode:
            config_mode = getattr(cfg, "planning_mode", None)

        planning_mode = (
            args.planning_mode
            or config_mode
            or _choose_planning_mode_interactive("hierarchical")
        )

    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

    final_output, workspace = main(
        model_name=model, config=cfg, planning_mode=planning_mode
    )

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
