import argparse

# needed for checkpoint / restart
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# rich console stuff for beautification
from rich import get_console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.util.checkpoint_fs import (
    ckpt_dir,
    hash_plan,
    load_exec_progress,
    parse_snapshot_indices,
    resolve_resume_checkpoint,
    restore_executor_from_snapshot,
    save_exec_progress,
    snapshot_sqlite_db,
    sync_progress_for_snapshot_single,
)

#########################################################################
# BEGIN: URSA utilities imports
#########################################################################
from ursa.util.config_loader import (
    get_config_planning_mode,
    get_default_model,
    get_default_models,
    get_models_cfg,
    load_yaml_config,
)
from ursa.util.interactive import timed_input_with_countdown
from ursa.util.llm_factory import resolve_model_choice, setup_llm
from ursa.util.logo_generator import kickoff_logo
from ursa.util.plan_renderer import render_plan_steps_rich
from ursa.util.run_meta import (
    load_run_meta,
    lock_or_warn_planning_mode,
    save_run_meta,
)
from ursa.util.workspace import setup_workspace

#########################################################################
# END: URSA utilities imports
#########################################################################


console = get_console()  # always returns the same instance


#########################################################################
# BEGIN: Helpers related to checkpoint/restart
#########################################################################
def _state_values(snapshot):
    """Normalize get_state(...) result to a plain dict of values."""
    try:
        return snapshot.values  # newer LangGraph
    except AttributeError:
        return snapshot  # older wrapper / already a dict


def _extract_values_from_checkpoint_tuple(cp_tuple) -> dict | None:
    """
    Works across saver serde variants.
    Expected shape: cp_tuple.checkpoint with either 'channel_values' or 'values'.
    """
    if not cp_tuple:
        return None
    ckpt = getattr(cp_tuple, "checkpoint", None) or {}
    values = ckpt.get("channel_values") or ckpt.get("values") or {}
    return values or None


def load_latest_planner_state_from_sqlite(saver, thread_id: str):
    """
    Try to fetch the latest checkpoint for this thread from the SqliteSaver.
    Returns (values_dict | None, resumed_cfg | None, debug_msg)
    """
    base_cfg = {"configurable": {"thread_id": thread_id}}
    # 1) Fast path: get_tuple (latest for thread)
    try:
        tup = saver.get_tuple(base_cfg)
        vals = _extract_values_from_checkpoint_tuple(tup)
        if vals:
            # include the resolved checkpoint_id so you can pin to it if needed
            cp_id = tup.config["configurable"].get("checkpoint_id")
            cfg = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}
            }
            return (
                vals,
                cfg,
                f"[dbg] resumed via get_tuple checkpoint_id={cp_id}",
            )
    except Exception as e:
        print("[dbg] get_tuple error:", e)

    # 2) Fallback: iterate history (ordered newest→oldest)
    try:
        latest = None
        for t in saver.list(base_cfg):  # generator, newest first
            latest = t
            break
        if latest:
            vals = _extract_values_from_checkpoint_tuple(latest)
            if vals:
                cp_id = latest.config["configurable"].get("checkpoint_id")
                cfg = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": cp_id,
                    }
                }
                return (
                    vals,
                    cfg,
                    f"[dbg] resumed via list checkpoint_id={cp_id}",
                )
    except Exception as e:
        print("[dbg] saver.list error:", e)

    return None, None, "[dbg] no existing planner checkpoint found"


#########################################################################
# END: Helpers related to checkpoint/restart
#########################################################################


#########################################################################
# BEGIN: Helpers for viewing the plan
#########################################################################
# --- nice rendering for planner output ---
def _last_message_text(messages) -> str:
    if not messages:
        return "<no messages>"
    last = messages[-1]
    return last.text


#########################################################################
# END: Helpers for viewing the plan
#########################################################################


def step_to_text(step) -> str:
    if isinstance(step, dict):
        name = (
            step.get("name")
            or step.get("title")
            or step.get("id")
            or "Unnamed step"
        )
        desc = step.get("description") or ""
        return f"{name}\n{desc}" if desc else name
    return str(step)


#########################################################################
# BEGIN: Helpers for hierarchical planning / progress
#########################################################################
# ========= Hierarchical progress helpers =========
def _hier_progress_file(workspace: str) -> Path:
    return Path(workspace) / "hier_progress.json"


def _read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def load_hier_progress(workspace: str) -> dict:
    # shape: {"main": {"next_index": int, "plan_hash": str}, "subs": {"<main_idx>": {"next_index": int, "plan_hash": str, "last_summary": str}}}
    return _read_json(
        _hier_progress_file(workspace),
        {"main": {"next_index": 0, "plan_hash": None}, "subs": {}},
    )


def save_hier_progress(workspace: str, data: dict) -> None:
    _hier_progress_file(workspace).write_text(json.dumps(data, indent=2))


def save_hier_main_progress(
    workspace: str, next_index: int, plan_hash: str, data: dict | None = None
) -> dict:
    d = data or load_hier_progress(workspace)
    d["main"] = {"next_index": int(next_index), "plan_hash": plan_hash}
    save_hier_progress(workspace, d)
    return d


def load_hier_sub_progress(workspace: str, main_idx: int) -> dict:
    d = load_hier_progress(workspace)
    return d.get("subs", {}).get(
        str(main_idx),
        {
            "next_index": 0,
            "plan_hash": None,
            "last_summary": "Start sub-steps.",
        },
    )


def save_hier_sub_progress(
    workspace: str,
    main_idx: int,
    next_index: int,
    plan_hash: str,
    last_summary: str | None = None,
    data: dict | None = None,
) -> dict:
    d = data or load_hier_progress(workspace)
    subs = d.setdefault("subs", {})
    entry = {"next_index": int(next_index), "plan_hash": plan_hash}
    if last_summary is not None:
        entry["last_summary"] = last_summary
    subs[str(main_idx)] = entry
    save_hier_progress(workspace, d)
    return d


#########################################################################
# END: Helpers for hierarchical planning / progress
#########################################################################


#########################################################################
# BEGIN: Assorted other helpers
#########################################################################
def _print_next_step(prefix: str, next_zero: int, total: int, workspace: str):
    """
    Pretty checkpoint message. next_zero is the 0-based index you'll resume at.
    Prints a 'done' message if next_zero >= total, else prints 'next step X of N' (1-based).
    """
    print("\n[checkpoint] executor progress saved")
    print(f"[checkpoint] workspace: {workspace}")
    if total <= 0:
        print("[checkpoint] no steps in plan.")
    elif next_zero >= total:
        print(f"[checkpoint] all {total} steps complete.")
    else:
        print(f"[checkpoint] next step: {next_zero + 1} of {total}")
    if prefix:
        print(prefix)


#########################################################################
# END: Assorted other helpers
#########################################################################


def setup_agents(
    workspace: str,
    model_choice: str,
    models_cfg: dict | None,
) -> tuple[str, tuple, tuple]:
    # --- checkpoint plumbing (unchanged) ---
    edb_path = ckpt_dir(workspace) / "executor_checkpoint.db"
    econn = sqlite3.connect(str(edb_path), check_same_thread=False)
    executor_checkpointer = SqliteSaver(econn)

    pdb_path = ckpt_dir(workspace) / "planner_checkpoint.db"
    pconn = sqlite3.connect(str(pdb_path), check_same_thread=False)
    planner_checkpointer = SqliteSaver(pconn)

    planner_llm = setup_llm(
        model_choice=model_choice,
        models_cfg=models_cfg or {},
        agent_name="planner",
        console=console,
    )
    executor_llm = setup_llm(
        model_choice=model_choice,
        models_cfg=models_cfg or {},
        agent_name="executor",
        console=console,
    )

    # Initialize the agents
    thread_id = Path(workspace).name

    planner = PlanningAgent(
        llm=planner_llm,
        checkpointer=planner_checkpointer,
        enable_metrics=True,
        metrics_dir="ursa_metrics",
        thread_id=thread_id,
        workspace=workspace,
    )  # include checkpointer

    executor = ExecutionAgent(
        llm=executor_llm,
        checkpointer=executor_checkpointer,
        enable_metrics=True,
        metrics_dir="ursa_metrics",
        thread_id=thread_id,
        workspace=workspace,
    )  # include checkpointer

    print(f"[dbg] planner_db_abs: {Path(pdb_path).resolve()}")
    print(f"[dbg] cwd: {Path.cwd().resolve()}")
    print(f"[dbg] thread_id = {thread_id}")

    return (
        thread_id,
        (planner, planner_checkpointer, pdb_path),
        (executor, executor_checkpointer, edb_path),
    )


def main_plan_load_or_perform(
    planner,
    planner_checkpointer,
    pdb_path,
    thread_id,
    problem,
    stepwise_exit,
    workspace,
):
    # -- Resume-or-plan: planner checkpoint aware ----------------
    with console.status(
        "[bold green]Checking planner checkpoint . . .", spinner="point"
    ):
        values, _, dbg = load_latest_planner_state_from_sqlite(
            planner_checkpointer, thread_id
        )
        print(dbg)

    # if values, then we are resuming from a checkpoint!  YAY!
    if values:
        # title = "[yellow]📋 Resumed Plan" if values else "[yellow]📋 Plan"
        # choose the dict that has messages/plan_steps
        plan_dict = values["__root__"]

        plan_dict["plan_steps"] = [
            {
                "name": plan_step.name,
                "description": plan_step.description,
                "expected_outputs": plan_step.expected_outputs,
                "success_criteria": plan_step.success_criteria,
                "requires_code": plan_step.requires_code,
            }
            for plan_step in plan_dict["plan"].steps
        ]
        # NOTE: This path is where we've recovered from a checkpoint
    else:
        # Fresh plan - need to do a plan
        with console.status(
            "[bold green]Planning overarching steps . . .", spinner="point"
        ):
            planning_output = planner.invoke(problem)

        planning_output["plan_steps"] = [
            {
                "name": plan_step.name,
                "description": plan_step.description,
                "expected_outputs": plan_step.expected_outputs,
                "success_criteria": plan_step.success_criteria,
                "requires_code": plan_step.requires_code,
            }
            for plan_step in planning_output["plan"].steps
        ]

        plan_dict = planning_output

        # pretty table with steps
        render_plan_steps_rich(planning_output["plan"].steps, highlight_index=0)

        # This is where we force an exit to demonstrate checkpointing - the following isn't normal,
        # it's specifically to demonstrate this functionality!
        # remind user how to resume
        if stepwise_exit:
            print(f"\n[checkpoint] Saved planning checkpoint to: {pdb_path}")
            print(f"[checkpoint] workspace: {workspace}")
            print(f"[checkpoint] (thread_id = workspace basename): {thread_id}")
            print(
                "\nRe-run this program with the SAME --workspace to resume the plan.\n"
            )
            print("Planning done, exiting")
            exit()

    # NOTE:
    # This is where we figure out where we are in the execution of the plan, what step
    # we are on
    # unify the plan dict for both fresh and resumed paths
    plan_steps = plan_dict.get("plan_steps") or []
    plan_sig = hash_plan(plan_steps)
    save_run_meta(
        workspace, plan_sig=plan_sig, plan_steps_count=len(plan_steps)
    )

    return plan_dict, plan_steps, plan_sig


def get_or_create_subplan(
    planner,
    planner_checkpointer,
    thread_id: str,
    workspace: str,
    problem: str,
    main_step,
    m_idx: int,
    stepwise_exit: bool,
    hierarchical: bool,
):
    if not hierarchical:
        # Single mode: 1-item synthetic sub-plan
        return {"plan_steps": [main_step]}, hash_plan([main_step]), None, None

    sub_tid = f"{thread_id}::detail::{m_idx}"
    sub_values, _, dbg = load_latest_planner_state_from_sqlite(
        planner_checkpointer, sub_tid
    )
    print(dbg)

    if sub_values:
        sub_steps = sub_values.get("plan_steps") or []
        return sub_values, hash_plan(sub_steps), sub_tid, None

    # Need to plan sub-steps
    detail_planner_prompt = "Flesh out this main step into concrete sub-steps to fully accomplish it."
    step_prompt = (
        f"You are contributing to the larger solution:\n{problem}\n\n"
        f"Current main step:\n{step_to_text(main_step)}\n\n"
        f"{detail_planner_prompt}"
    )

    _old_tid = planner.thread_id
    planner.thread_id = sub_tid
    try:
        sub_output = planner.invoke(
            {"messages": [HumanMessage(content=step_prompt)]},
            config={
                "recursion_limit": 999_999,
                "configurable": {"thread_id": sub_tid},
            },
        )
    finally:
        planner.thread_id = _old_tid

    sub_output["plan_steps"] = [
        {
            "name": plan_step.name,
            "description": plan_step.description,
            "expected_outputs": plan_step.expected_outputs,
            "success_criteria": plan_step.success_criteria,
            "requires_code": plan_step.requires_code,
        }
        for plan_step in sub_output["plan"].steps
    ]

    sub_steps = sub_output.get("plan_steps") or []
    sub_sig = hash_plan(sub_steps)

    # persist initial sub-progress (index=0)
    save_hier_sub_progress(
        workspace,
        m_idx,
        next_index=0,
        plan_hash=sub_sig,
        last_summary="Start sub-steps.",
    )

    if stepwise_exit:
        print(f"\n[checkpoint] sub-plan saved for MAIN STEP {m_idx + 1}")
        print(f"[checkpoint] workspace: {workspace}")
        print(f"[checkpoint] sub_thread_id: {sub_tid}")
        print(
            "Re-run with the SAME --workspace to execute the first sub-step.\n"
        )
        exit()

    return {"plan_steps": sub_steps}, sub_sig, sub_tid, sub_output


def run_substeps(
    executor,
    problem: str,
    main_step,
    sub_steps: list,
    load_progress,
    save_progress,
    m_idx: int,
    total_main: int,
    workspace: str,
    symlinkdict,
    stepwise_exit: bool,
):
    sub_prog = load_progress(m_idx)
    sub_start_idx = int(sub_prog.get("next_index", 0))
    prev_sub_summary = sub_prog.get("last_summary", "Start sub-steps.")
    total_sub = len(sub_steps)

    render_plan_steps_rich(sub_steps, highlight_index=sub_start_idx)

    last_ran_summary = prev_sub_summary  # <— track the latest one we executed

    while sub_start_idx < total_sub:
        current_sub = sub_steps[sub_start_idx]
        with Progress(
            SpinnerColumn(spinner_name="point"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("Step {task.fields[current]}/{task.total}"),
            console=console,
            transient=True,
        ) as sub_progress:
            sub_task = sub_progress.add_task(
                f"Execute sub-step: {step_to_text(current_sub)[:60]} . . .",
                total=1,
                completed=0,
                current=1,
            )

            sub_exec_prompt = (
                f"You are contributing to the larger solution:\n{problem}\n\n"
                f"Main step:\n{step_to_text(main_step)}\n\n"
                f"Previous sub-step summary: {prev_sub_summary}\n"
                f"Current sub-step:\n{step_to_text(current_sub)}\n\n"
                "Execute this sub-step and report the results fully—no placeholders."
            )

            sub_result = executor.invoke(
                {
                    "messages": [HumanMessage(content=sub_exec_prompt)],
                    "workspace": executor.workspace,
                    "symlinkdir": symlinkdict,
                },
                config={
                    "recursion_limit": 999_999,
                    "configurable": {"thread_id": executor.thread_id},
                },
            )

            last_sub_summary = sub_result["messages"][-1].text
            last_ran_summary = last_sub_summary  # <—
            sub_progress.console.log(last_sub_summary)
            sub_progress.advance(sub_task)
            sub_progress.update(sub_task, current=1, completed=1)
            sub_progress.remove_task(sub_task)

        next_sub_zero = sub_start_idx + 1
        save_progress(m_idx, next_sub_zero, last_sub_summary)

        if stepwise_exit:
            print("\n[checkpoint] hierarchical executor progress saved")
            print(f"[checkpoint] workspace: {workspace}")
            print(f"[checkpoint] main step: {m_idx + 1} of {total_main}")
            _print_next_step(
                prefix="Re-run with the SAME --workspace to continue.\n",
                next_zero=next_sub_zero,
                total=total_sub,
                workspace=workspace,
            )
            exit()

        prev_sub_summary = last_sub_summary
        sub_start_idx = next_sub_zero

    return last_ran_summary


def main(
    model_name: str,
    config: Any,
    planning_mode: str = "single",
    user_specified_workspace: str = None,
    stepwise_exit: bool = False,
    resume_from: str | None = None,
    interactive_timeout: int = 60,
):
    try:
        problem = getattr(config, "problem", "")
        project = getattr(config, "project", "run")
        symlinkdict = getattr(config, "symlink", {}) or None

        # sets up the workspace, run config json, etc.

        workspace = setup_workspace(
            user_specified_workspace=user_specified_workspace,
            project=project,
            model_name=model_name,
            console=console,
        )

        print(workspace)
        print(user_specified_workspace)

        # --- decide which checkpoint to start from ---
        try:
            chosen_ckpt = resolve_resume_checkpoint(
                workspace=workspace,
                resume_from=resume_from,
                timeout=interactive_timeout,
                choose_fn=None,  # leave None to use default choose_checkpoint
                input_fn=timed_input_with_countdown,
            )
            if chosen_ckpt:
                restore_executor_from_snapshot(workspace, chosen_ckpt)
                # (Optional) also print a gentle reminder:
                print(
                    "Press Ctrl-C now to abort and rerun with a different --resume-from, if this was unintentional."
                )
        except KeyboardInterrupt:
            print("\nAborted by user before opening databases.")
            sys.exit(1)

        # lock planning_mode per workspace
        planning_mode, mode_locked = lock_or_warn_planning_mode(
            workspace, planning_mode, console=console
        )
        console.print(
            Panel.fit(
                f"Mode: [bold]{planning_mode}[/] (locked per workspace)",
                border_style="blue",
            )
        )

        # ---- One-time project logo kickoff (per workspace) -----------------
        # Use run_meta.json to ensure we do this only once for this workspace.
        meta = load_run_meta(workspace)
        # MINIMAL CONFIG
        logo_cfg = getattr(config, "logo", {}) or {}
        logo_enabled = bool(logo_cfg.get("enabled", True))

        if logo_enabled and not meta.get("logo_created"):
            scene_style = logo_cfg.get(
                "scene", "random"
            )  # noir/sci-fi/etc or "random"
            stickers_enabled = bool(logo_cfg.get("stickers", True))
            n_variants = int(logo_cfg.get("n", 4))
            logo_model_choice = logo_cfg.get("model", "openai:gpt-image-1")

            # pick the model string & a providers map
            providers = (getattr(config, "models", {}) or {}).get(
                "providers", {}
            ) or {}

            v_provider, v_model, v_extra = resolve_model_choice(
                logo_model_choice, {"providers": providers}
            )
            scene_dir = Path(workspace) / "logo_art" / "scenes"
            sticker_dir = Path(workspace) / "logo_art" / "stickers"
            try:
                # SCENE artwork — uses existing defaults for everything else
                _ = kickoff_logo(
                    problem_text=problem,
                    workspace=workspace,
                    out_dir=scene_dir,
                    # size omitted (auto with aspect)
                    background="opaque",
                    quality="high",
                    n=n_variants,
                    style=scene_style,  # <- only knob exposed via YAML
                    mode="scene",
                    aspect="wide",
                    style_intensity="overt",
                    image_model=v_model,
                    image_model_provider=v_provider,
                    image_provider_kwargs=v_extra,
                    console=console,
                    on_done=lambda p: console.print(
                        Panel.fit(
                            f"[bold yellow]Project art saved:[/] {p}",
                            border_style="yellow",
                        )
                    ),
                    on_error=lambda e: console.print(
                        Panel.fit(
                            f"[bold red]Art generation failed:[/] {e}",
                            border_style="red",
                        )
                    ),
                )

                # STICKER artwork — optional
                if stickers_enabled:
                    _ = kickoff_logo(
                        problem_text=problem,
                        workspace=workspace,
                        out_dir=sticker_dir,
                        size="1024x1024",
                        background="opaque",
                        quality="high",
                        n=n_variants,
                        style="sticker",
                        image_model=v_model,
                        image_model_provider=v_provider,
                        image_provider_kwargs=v_extra,
                        console=console,
                        on_done=lambda p: console.print(
                            Panel.fit(
                                f"[bold yellow]Project sticker art saved:[/] {p}",
                                border_style="yellow",
                            )
                        ),
                        on_error=lambda e: console.print(
                            Panel.fit(
                                f"[bold red]Art sticker generation failed:[/] {e}",
                                border_style="red",
                            )
                        ),
                    )

            finally:
                # Even if kickoff_logo fails, mark that we attempted it so we don't spam runs.
                # Remove this flag manually if you want to re-generate art for this workspace.
                save_run_meta(workspace, logo_created=True)
        # --------------------------------------------------------------------

        models_cfg = getattr(config, "models", {}) or {}

        # gets the agents we'll use for this example including their checkpointer handles and database
        thread_id, planner_tuple, executor_tuple = setup_agents(
            workspace=workspace,
            model_choice=model_name,
            models_cfg=models_cfg,
        )
        planner, planner_checkpointer, pdb_path = planner_tuple
        executor, _, edb_path = executor_tuple

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

        save_run_meta(workspace, thread_id=thread_id, model_name=model_name)

        # do the main planning step, or load it from checkpoint
        plan_dict, plan_steps, plan_sig = main_plan_load_or_perform(
            planner,
            planner_checkpointer,
            pdb_path,
            thread_id,
            problem,
            stepwise_exit,
            workspace,
        )

        # If we restored from a numbered snapshot in SINGLE mode, align executor_progress.json
        if planning_mode == "single" and chosen_ckpt is not None:
            try:
                sync_progress_for_snapshot_single(
                    workspace, chosen_ckpt, plan_sig
                )
            except Exception as e:
                print(
                    f"[warn] failed to sync progress JSON for {chosen_ckpt.name}: {e}"
                )

        # --- compute resume indices from chosen snapshot (used for logs + hier priming) ---
        resume_main_0 = resume_sub_next = None
        if chosen_ckpt is not None:
            _m1, _s1 = parse_snapshot_indices(chosen_ckpt)
            if _m1:
                resume_main_0 = max(0, _m1 - 1)
                resume_sub_next = int(_s1) if _s1 is not None else 0

        # Simple resume summary
        if planning_mode == "single" and chosen_ckpt is not None:
            k, _ = parse_snapshot_indices(chosen_ckpt)
            if k is not None:
                print(
                    f"[resume] Single mode: next_index={k} (will start at top-level step {k + 1})."
                )
        elif (
            planning_mode == "hierarchical"
            and chosen_ckpt is not None
            and resume_main_0 is not None
        ):
            human_main = resume_main_0 + 1
            human_sub = (resume_sub_next + 1) if resume_sub_next else 1
            print(
                f"[resume] Hierarchical: MAIN next_index={resume_main_0} (human {human_main}), "
                f"SUB next_index={resume_sub_next} (human {human_sub})."
            )

        if planning_mode == "hierarchical":
            # ----- MAIN PLAN PROGRESS -----
            hprog = load_hier_progress(workspace)
            if hprog.get("main", {}).get("plan_hash") != plan_sig:
                console.print(
                    Panel.fit(
                        "[bold yellow]Top-level plan changed — resetting hierarchical progress to step 0.[/]",
                        border_style="yellow",
                    )
                )
                hprog = {
                    "main": {"next_index": 0, "plan_hash": plan_sig},
                    "subs": {},
                }
                save_hier_progress(workspace, hprog)

            if resume_main_0 is not None:
                hprog = save_hier_main_progress(
                    workspace,
                    next_index=resume_main_0,
                    plan_hash=plan_sig,
                    data=hprog,
                )

            # we could be coming back into this plan at someplace in the middle, so this our start
            # index for THIS instantiation
            main_start_idx = int(hprog["main"]["next_index"])
            total_main = len(plan_steps)

            # Show main plan with highlight on the next main step to work on
            render_plan_steps_rich(plan_steps, highlight_index=main_start_idx)

            # If all main steps done, you're finished
            if main_start_idx >= total_main:
                console.print(
                    Panel.fit(
                        "[bold green]All main steps complete.[/]",
                        border_style="green",
                    )
                )
                return "All hierarchical steps completed.", workspace

            # ----- LOOP OVER MAIN STEPS (resuming at main_start_idx) -----
            for m_idx in range(main_start_idx, total_main):
                main_step = plan_steps[m_idx]
                step_num = m_idx + 1
                console.print(
                    Panel.fit(
                        Text.from_markup(
                            f"[bold cyan]MAIN STEP {step_num} of {total_main}[/]\n\n{step_to_text(main_step)}"
                        ),
                        border_style="cyan",
                    )
                )

                # get or create the subplan (hierarchical=True)
                sub_values, sub_sig, _sub_tid, _ = get_or_create_subplan(
                    planner,
                    planner_checkpointer,
                    thread_id,
                    workspace,
                    problem,
                    main_step,
                    m_idx,
                    stepwise_exit,
                    hierarchical=True,
                )

                sub_steps = sub_values.get("plan_steps") or []

                # If we resumed into this main step, set its sub "next_index" before running
                if (resume_main_0 is not None) and (m_idx == resume_main_0):
                    target_next = min(resume_sub_next, len(sub_steps))
                    save_hier_sub_progress(
                        workspace,
                        m_idx,
                        next_index=target_next,
                        plan_hash=sub_sig,
                        last_summary=f"Resumed from snapshot {chosen_ckpt.name}",
                    )

                # closures for reading/writing sub-progress with plan-hash guard
                def load_progress_h(m):
                    prog = load_hier_sub_progress(workspace, m)
                    if prog.get("plan_hash") != sub_sig:
                        save_hier_sub_progress(
                            workspace,
                            m,
                            next_index=0,
                            plan_hash=sub_sig,
                            last_summary="Start sub-steps.",
                        )
                        return {
                            "next_index": 0,
                            "last_summary": "Start sub-steps.",
                        }
                    return prog

                def save_progress_h(m, next_idx, last_summary):
                    # persist hierarchical sub-step progress
                    save_hier_sub_progress(
                        workspace, m, next_idx, sub_sig, last_summary
                    )

                    # snapshot after each completed sub-step:
                    # m is 0-based main step index; next_idx is the 1-based count of completed sub-steps
                    try:
                        main_no = m + 1
                        sub_no = int(
                            next_idx
                        )  # just-finished sub-step (1-based)
                        snapshot_path = (
                            ckpt_dir(workspace)
                            / f"executor_checkpoint_{main_no}_{sub_no}.db"
                        )
                        snapshot_sqlite_db(edb_path, snapshot_path)
                        print(
                            f"[checkpoint] saved sub-step snapshot: {snapshot_path.name}"
                        )
                    except Exception as e:
                        print(
                            f"[warn] failed to snapshot executor DB for main {m + 1} sub {next_idx}: {e}"
                        )

                _ = run_substeps(
                    executor,
                    problem,
                    main_step,
                    sub_steps,
                    load_progress_h,
                    save_progress_h,
                    m_idx,
                    total_main,
                    workspace,
                    symlinkdict,
                    stepwise_exit,
                )

                # advance main progress and continue
                hprog = save_hier_main_progress(
                    workspace,
                    next_index=m_idx + 1,
                    plan_hash=plan_sig,
                    data=hprog,
                )
                console.print(
                    Panel.fit(
                        f"[bold green]All sub-steps complete for MAIN STEP {step_num}. Advancing to next main step.[/]",
                        border_style="green",
                    )
                )

            return "All hierarchical steps completed.", workspace

        # this is the single planning step pathway, not hierarchical
        elif planning_mode == "single":
            # figure out where to resume execution
            exec_prog = load_exec_progress(workspace)

            if exec_prog.get("plan_hash") != plan_sig:
                start_idx = 0
                prev_summary = (
                    "Beginning single-pass execution of top-level steps."
                )
            else:
                start_idx = int(exec_prog.get("next_index", 0))
                prev_summary = exec_prog.get(
                    "last_summary",
                    "Beginning single-pass execution of top-level steps.",
                )

            render_plan_steps_rich(plan_steps, highlight_index=start_idx)
            total_steps = len(plan_steps)

            if start_idx >= total_steps:
                console.print(
                    Panel.fit(
                        "[bold green]All steps already executed according to progress file.[/]\n"
                        "Delete executor_progress.json to re-run, or change the plan.",
                        border_style="green",
                    )
                )
            else:
                for idx in range(start_idx, total_steps):
                    main_step = plan_steps[idx]
                    console.print(
                        Panel.fit(
                            Text.from_markup(
                                f"[bold cyan]STEP {idx + 1} of {total_steps}[/]\n\n{step_to_text(main_step)}"
                            ),
                            border_style="cyan",
                        )
                    )

                    # synthetic 1-item sub-plan (hierarchical=False)
                    sub_values, sub_sig, _tid, _ = get_or_create_subplan(
                        planner,
                        None,
                        thread_id,
                        workspace,
                        problem,
                        main_step,
                        idx,
                        stepwise_exit,
                        hierarchical=False,
                    )
                    sub_steps = sub_values["plan_steps"]

                    def load_progress_single(_m):
                        # always run the single sub-step for this main step
                        return {"next_index": 0, "last_summary": prev_summary}

                    def save_progress_single(m, _next_idx, last_summary):
                        # advance main-step pointer
                        save_exec_progress(
                            workspace,
                            next_index=m + 1,
                            plan_hash=plan_sig,
                            last_summary=last_summary,
                        )
                        # snapshot the executor checkpoint after completing top-level step m (1-based)
                        try:
                            step_no = m + 1
                            snapshot_path = (
                                ckpt_dir(workspace)
                                / f"executor_checkpoint_{step_no}.db"
                            )
                            snapshot_sqlite_db(edb_path, snapshot_path)
                            print(
                                f"[checkpoint] saved step snapshot: {snapshot_path.name}"
                            )
                        except Exception as e:
                            print(
                                f"[warn] failed to snapshot executor DB for step {m + 1}: {e}"
                            )

                    prev_summary = run_substeps(
                        executor,
                        problem,
                        main_step,
                        sub_steps,
                        load_progress_single,
                        save_progress_single,
                        idx,
                        total_steps,
                        workspace,
                        symlinkdict,
                        stepwise_exit,
                    )

        # Wrap-up
        answer = prev_summary or "Plan completed."
        render_session_summary(thread_id)
        return answer, workspace

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def parse_args_and_user_inputs():
    parser = argparse.ArgumentParser(description="Run with YAML config.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--workspace",
        required=False,
        help="Path to workspace, where checkpoint exists, else it will be created.",
    )
    parser.add_argument(
        "--planning-mode",
        choices=["hierarchical", "single"],
        help="Choose 'hierarchical' (plan -> re-plan each step -> execute) or 'single' (plan once -> execute each step).",
    )
    parser.add_argument(
        "--stepwise-exit",
        action="store_true",
        help="Exit after each plan/sub-plan/step checkpoint (demo mode). Default: continue without exiting.",
    )
    parser.add_argument(
        "--resume-from",
        required=False,
        help="Checkpoint file to restore executor state from (e.g. executor_checkpoint_5.db or executor_checkpoint_3_2.db). "
        "If omitted, you'll be prompted (with timeout).",
    )
    parser.add_argument(
        "--interactive-timeout",
        type=int,
        default=60,
        help="Seconds to wait at interactive prompts (model/mode/checkpoint) before defaulting. "
        "Set 0 to default immediately (useful for headless/HPC).",
    )
    args = parser.parse_args()

    # --- load YAML -> shallow namespace (top-level keys only) ---
    try:
        cfg = load_yaml_config(args.config)
    except FileNotFoundError:
        sys.exit(2)
    except Exception:
        sys.exit(2)

    # ── config-driven model choices ────────────
    models_cfg = get_models_cfg(cfg)
    DEFAULT_MODELS = get_default_models(models_cfg)
    DEFAULT_MODEL = get_default_model(models_cfg)

    # ── timeout-aware interactive helpers ────────────
    def _choose_model_interactive(
        default_models: tuple[str, ...],
        default_model: str | None,
        timeout_sec: int,
    ) -> str:
        print("\nChoose the model to run with:")
        for i, m in enumerate(default_models, 1):
            print(f"  {i}. {m}")
        if default_model:
            print(f"(Press Enter for default: {default_model})")
        else:
            print(
                f"(No configured default; Enter will pick: {default_models[0]})"
            )
        print("Or type your own model string (Ctrl-C to quit).")
        print(
            f"(No response in {timeout_sec}s → default will be selected automatically.)"
        )

        while True:
            choice = timed_input_with_countdown("> ", timeout_sec)
            if choice is None or choice.strip() == "":
                return default_model or default_models[0]
            s = choice.strip()
            if s.isdigit():
                idx = int(s)
                if 1 <= idx <= len(default_models):
                    return default_models[idx - 1]
                print(
                    f"Please enter a number 1..{len(default_models)}, a custom model, or press Enter."
                )
                continue
            return s

    def _choose_planning_mode_interactive(
        default_mode: str, timeout_sec: int
    ) -> str:
        print("\nSelect planning mode:")
        print(
            "  1. hierarchical  (Plan -> re-plan each step -> execute sub-steps)"
        )
        print(
            "  2. single        (Plan once -> execute each top-level step directly)"
        )
        if default_mode:
            print(f"(Press Enter for default: {default_mode})")
        print(
            f"(No response in {timeout_sec}s → default will be selected automatically.)"
        )

        while True:
            choice = timed_input_with_countdown("> ", timeout_sec)
            if choice is None or choice.strip() == "":
                return default_mode
            s = choice.strip().lower()
            if s.isdigit():
                if s == "1":
                    return "hierarchical"
                if s == "2":
                    return "single"
            if s in ("hierarchical", "single"):
                return s
            print("Please enter 1, 2, 'hierarchical', or 'single'.")

    try:
        # Model (CLI didn’t provide a model string variable elsewhere, so pick interactively)
        model = _choose_model_interactive(
            DEFAULT_MODELS, DEFAULT_MODEL, args.interactive_timeout
        )

        # Planning mode resolution: CLI > config > interactive > default('single')
        config_mode = get_config_planning_mode(cfg)

        planning_mode = (
            args.planning_mode
            or config_mode
            or _choose_planning_mode_interactive(
                "single", args.interactive_timeout
            )
        )

    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)

    return args, cfg, model, planning_mode


def display_final_output(final_output):
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold white on green] ✔  Final Output:[/] {final_output}"
            ),
            border_style="green",
        )
    )


def display_end_of_run_info(workspace: str):
    console.rule("[bold cyan]Run complete")
    console.print(
        Panel.fit(
            f":rocket:  [bold bright_blue]{workspace}[/bold bright_blue]  :rocket:",
            title="[bold green]WORKSPACE RESULTS IN[/bold green]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )


# command line invocation pathway
if __name__ == "__main__":
    # first, parse the command line args and then get any special user
    # inputs we're looking for
    args, cfg, model, planning_mode = parse_args_and_user_inputs()

    # then, run the agentic workflow
    final_output, workspace = main(
        model_name=model,
        config=cfg,
        planning_mode=planning_mode,
        user_specified_workspace=args.workspace,
        stepwise_exit=args.stepwise_exit,
        resume_from=args.resume_from,
        interactive_timeout=args.interactive_timeout,
    )

    display_final_output(final_output)
    display_end_of_run_info(workspace)
