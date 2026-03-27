#!/usr/bin/env python3
"""
Run the Hypothesizer agent on the YAML config/problem + local input docs.

Usage:
    python doc_critique_writer.py --config path/to/config.yaml [--workspace my_ws]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

# LLM + agent helpers from your refactor
from rich import get_console
from rich.panel import Panel

# Hypothesizer agent class (your codebase)
from ursa.agents import HypothesizerAgent
from ursa.cli.config import UrsaConfig, deep_interp_env
from ursa.util.llm_factory import setup_llm
from ursa.util.run_meta import save_run_meta
from ursa.util.workspace import ensure_symlink, setup_workspace

console = get_console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Hypothesizer using YAML config"
    )
    p.add_argument(
        "--config",
        required=True,
        help="YAML config file (same format as runner)",
    )
    p.add_argument(
        "--workspace", required=False, help="Workspace path (optional)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        raw_cfg = deep_interp_env(raw_cfg)
        cfg = UrsaConfig.model_validate(raw_cfg, extra="ignore")
    except FileNotFoundError:
        console.print("[red]Config file not found[/]")
        sys.exit(2)

    model_choice = cfg.llm_model.model

    # workspace
    project = raw_cfg.get("project", "run")
    workspace = setup_workspace(
        user_specified_workspace=args.workspace,
        project=project,
        model_name=model_choice,
        console=console,
    )
    # --- apply symlink into the NEW workspace (so any input docs exists there) ---
    symlink_cfg = raw_cfg.get("symlink")
    if (
        isinstance(symlink_cfg, dict)
        and symlink_cfg.get("source")
        and symlink_cfg.get("dest")
    ):
        ensure_symlink(workspace=workspace, symlink_cfg=symlink_cfg)

    # --- prepare Hypothesizer agent ---
    llm = setup_llm(
        model_config=cfg.llm_model,
        agent_name="hypothesizer",
        console=console,
    )

    # max_iterations: look in YAML under a top-level 'hypothesizer' key or fallback
    hs_cfg = raw_cfg.get("hypothesizer", {}) or {}
    max_iters = hs_cfg
    max_iterations = (
        max_iters.get("max_iterations") if isinstance(max_iters, dict) else None
    )
    if max_iterations is None:
        max_iterations = raw_cfg.get("max_iterations", 2)
    try:
        max_iterations = int(max_iterations)
    except Exception:
        max_iterations = 2

    # whether to use web search (default False for privacy/local-docs workflows)
    use_search = False
    if isinstance(hs_cfg, dict):
        use_search = bool(hs_cfg.get("use_search", False))

    # instantiate agent
    agent = HypothesizerAgent(
        llm=llm,
        max_iterations=max_iterations,
        workspace=workspace,
        enable_web_search=use_search,
    )

    prompts_cfg = hs_cfg.get("prompts", {}) if isinstance(hs_cfg, dict) else {}

    if isinstance(prompts_cfg, dict):
        p1 = prompts_cfg.get("agent1_hypothesizer")
        p2 = prompts_cfg.get("agent2_critic")
        p3 = prompts_cfg.get("agent3_competitor")

        # Override only if provided (otherwise keep defaults from prompt_library)
        if p1:
            agent.hypothesizer_prompt = p1
        if p2:
            agent.critic_prompt = p2
        if p3:
            agent.competitor_prompt = p3

        if any([p1, p2, p3]):
            console.print("[dim]Loaded prompt overrides from YAML.[/]")

    # save run meta fingerprint so other tools can find this run
    save_run_meta(workspace, tool="hypothesizer_agent", model=model_choice)

    # --- problem text from YAML ---
    problem_text = raw_cfg.get("problem")
    if not problem_text:
        console.print(
            "[yellow]No 'problem' found in config; using default test question.[/]"
        )
        problem_text = "Why did the test server intermittently fail health checks after the 02:00 deploy?"

    # --- derive local docs directory from YAML symlink.dest if present ---
    symlink_cfg = raw_cfg.get("symlink", {}) or {}
    # --- derive local docs directory from YAML symlink.dest (INSIDE workspace) ---
    input_docs_dir = None
    if isinstance(symlink_cfg, dict):
        dest = symlink_cfg.get("dest")
        if dest:
            input_docs_dir = str((Path(workspace) / dest).resolve())

    # If there is no symlink, allow explicit hypothesizer.input_docs_dir (also treat as inside workspace if relative)
    if isinstance(hs_cfg, dict) and not input_docs_dir:
        explicit = hs_cfg.get("input_docs_dir")
        if explicit:
            p = Path(explicit).expanduser()
            input_docs_dir = (
                str((Path(workspace) / p).resolve())
                if not p.is_absolute()
                else str(p)
            )

    # sanity: if still None, leave it None (agent will behave without docs)
    console.print(
        Panel.fit(
            f"[bold]Problem:[/] {problem_text}",
            title="[bold green]Hypothesizer Input[/]",
        )
    )
    if input_docs_dir:
        console.print(f"[dim]Using local docs dir: {input_docs_dir}[/]")

    # build the initial state for the agent
    initial_state = {
        "question": problem_text,
        "question_search_query": "",
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "agent1_solution": [],
        "agent2_critiques": [],
        "agent3_perspectives": [],
        "solution": "",
        "summary_report": "",
        "visited_sites": set(),
        # custom
        "input_docs_dir": input_docs_dir,
    }

    console.print(
        f"[blue]Invoking Hypothesizer (max_iterations={max_iterations}, use_search={use_search})...[/]"
    )

    result = asyncio.run(
        agent.ainvoke(
            initial_state,
            config={
                "recursion_limit": 999999,
                "configurable": {"thread_id": agent.thread_id},
            },
        )
    )

    # Print results
    solution = result.get("solution", "")
    summary = result.get("summary_report", "")

    console.print(
        Panel.fit(
            f"[bold green]Hypothesizer Final Solution[/]\n\n{solution[:400]}",
            title="[bold green]Result Preview[/]",
        )
    )
    # If model returned JSON-ish content, try to pretty print it
    try:
        parsed = json.loads(solution)
        console.print(Panel.fit("[bold]Structured output (JSON)[/]\n"))
        console.print_json(data=parsed)
    except Exception:
        # not JSON - show full solution in truncated form and write to disk
        out_path = Path(workspace) / "hypothesizer_solution.txt"
        out_path.write_text(solution, encoding="utf-8")
        console.print(f"[dim]Full solution written to {out_path}[/]")

    # write summary/latex preview if present
    if summary:
        summary_path = Path(workspace) / "hypothesizer_summary.tex"
        summary_path.write_text(summary, encoding="utf-8")
        console.print(
            f"[dim]LaTeX summary written to {summary_path} (preview saved)[/]"
        )

    console.print("\nDone.")


if __name__ == "__main__":
    main()
