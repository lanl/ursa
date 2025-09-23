import argparse
import ssl
import sys
from types import SimpleNamespace as NS
from typing import Any

import coolname
import httpx
import litellm
import truststore
import yaml
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

# rich console stuff for beautification
from rich import get_console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ursa.agents import ExecutionAgent, PlanningAgent

ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # use macOS Keychain
litellm.client_session = httpx.Client(verify=ctx, timeout=30)


console = get_console()  # always returns the same instance


def main(model_name: str, config: Any):
    """
    Use:
      - config.project  (e.g., run dir names, titles - "sortnet" for example)
      - config.problem  (the whole long problem statement)
      - config.symlink  (None or a mapping source to dest) {"source": "...", "dest": "..."} or None
    """
    try:
        problem = config.problem
        workspace = f"{config.project}_{coolname.generate_slug(2)}"
        # workspace_header = f"[cyan] (- [bold cyan]{workspace}[reset][cyan] -) [reset]"
        symlinkdict = getattr(cfg, "symlink", {}) or None

        model = ChatLiteLLM(
            model=model_name,
            max_tokens=10000,
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
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

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
                            f"[bold cyan]STEP {main_step_number} - Current Step:[/] {main_step}",
                            # justify="center",
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
                                f"[bold red]Sub-STEP {sub_step_number} - Current Sub-Step:[/] {sub}",
                                # justify="center",
                            ),
                            border_style="red",
                        )
                    )

                    final_results = executor.action.invoke(
                        {
                            "messages": [HumanMessage(content=sub_prompt)],
                            "workspace": workspace,
                            "symlinkdir": symlinkdict,
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
        return answer, workspace

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with YAML config.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
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
            "openai/gpt-5",
            "openai/o3",
            "openai/o3-mini",
        )
    )
    DEFAULT_MODEL = models_cfg.get("default")  # may be None

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
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

    final_output, workspace = main(model_name=model, config=cfg)

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
