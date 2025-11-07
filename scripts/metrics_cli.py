from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

from ursa.observability.metrics_charts import (
    compute_attribution,
    compute_llm_wall_seconds,  # used in single-file tokens-rate
    extract_llm_token_stats,
    extract_time_breakdown,
    plot_lollipop_time,
    plot_time_breakdown,
    plot_token_kde,
    plot_token_rates_bar,
    plot_token_rates_by_model,
    plot_token_totals_bar,
    plot_tokens_bar_by_model,
    plot_tokens_by_agent_stacked,
    plot_tps_by_agent_grouped,
)
from ursa.observability.metrics_io import load_metrics
from ursa.observability.metrics_session import (
    aggregate_super_token_stats_by_agent,
    aggregate_super_tokens_by_model,
    compute_thread_time_bases,
    extract_thread_time_breakdown,
    extract_thread_token_stats,
    extract_thread_token_stats_by_agent,
    list_threads_summary,
    plot_thread_timeline_interactive,
    scan_directory_for_threads,
)

# -----------------------
#      Helpers
# -----------------------


def _default_out_path(in_path: str, chart: str) -> str:
    base, _ = os.path.splitext(in_path)
    return f"{base}_breakdown_{chart}.png"


def _ensure_dir(p: str) -> None:
    os.makedirs(p or ".", exist_ok=True)


def _dt(x: str) -> datetime:
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


# -----------------------
#   Bulk "ALL" runner
# -----------------------


def run_all(
    *,
    dir_path: str,
    group_llm: bool,
    group_by: str,
    log_x: bool,
    min_label_pct: float,
    verbose: bool = True,
) -> int:
    """
    Generate:
      • Thread-level: lollipop, tokens-bar, tokens-kde, tokens-rate, timeline-html
      • Per-run (each JSON): lollipop, tokens-bar, tokens-kde
    for every thread discovered under dir_path.
    """
    sessions: Dict[str, list] = scan_directory_for_threads(dir_path)
    if not sessions:
        print(f"No thread_ids found in directory: {dir_path!r}")
        return 0

    print(f"Found {len(sessions)} thread_id(s) under {dir_path!r}.\n")

    for thread_id, runs in sessions.items():
        print(f"== Thread: {thread_id}  (runs: {len(runs)}) ==")

        # --- Thread-level charts ---
        try:
            # 1) Time breakdown (lollipop) — aggregated
            total, parts, ctx = extract_thread_time_breakdown(
                runs, group_llm=group_llm
            )
            out = os.path.join(dir_path, f"thread_{thread_id}_lollipop.png")
            _ensure_dir(os.path.dirname(out))
            plot_lollipop_time(
                total,
                parts,
                out,
                title="Time (seconds) by component — thread total",
                log_x=log_x,
                min_label_pct=min_label_pct,
                context=ctx,
            )
            if verbose:
                print(f"  ✓ thread lollipop   -> {out}")

            # 2) Thread token totals (bar) + KDE
            totals, samples, ctx = extract_thread_token_stats(runs)

            out_bar = os.path.join(
                dir_path, f"thread_{thread_id}_tokens_bar.png"
            )
            plot_token_totals_bar(
                totals,
                out_bar,
                title="LLM Token Totals by Category — thread",
                context=ctx,
            )
            if verbose:
                print(f"  ✓ thread tokens-bar -> {out_bar}")

            out_kde = os.path.join(
                dir_path, f"thread_{thread_id}_tokens_kde.png"
            )
            plot_token_kde(
                samples,
                out_kde,
                title="LLM Token Usage — KDE (thread)",
                context=ctx,
                log_x=log_x,
            )
            if verbose:
                print(f"  ✓ thread tokens-kde -> {out_kde}")

            # 3) Thread tokens-rate (two baselines)
            llm_seconds, elapsed_seconds = compute_thread_time_bases(runs)
            out_rate = os.path.join(
                dir_path, f"thread_{thread_id}_tokens_rate.png"
            )
            plot_token_rates_bar(
                totals,
                llm_seconds,
                elapsed_seconds,
                out_rate,
                title="tokens per second — thread",
                context=ctx,
            )
            if verbose:
                print(f"  ✓ thread tokens-rate-> {out_rate}")

            # 4) Interactive Timeline HTML
            out_html = os.path.join(
                dir_path, f"thread_{thread_id}_timeline.html"
            )
            plot_thread_timeline_interactive(runs, out_html, group_by=group_by)
            if verbose:
                print(f"  ✓ thread timeline   -> {out_html}")

            # 5) Thread-level: aggregate by AGENT (tokens + TPS)
            try:
                totals_by_agent, llm_secs_by_agent, thread_secs = (
                    extract_thread_token_stats_by_agent(runs)
                )

                footer = [
                    f"thread: {thread_id} • runs: {len(runs)}",
                    f"thread window: {thread_secs:,.3f}s",
                ]

                out_agents_tokens = os.path.join(
                    dir_path, f"thread_{thread_id}_agents_tokens.png"
                )
                plot_tokens_by_agent_stacked(
                    totals_by_agent,
                    out_agents_tokens,
                    title="LLM Token Totals by Agent (thread)",
                    footer_lines=footer,
                )
                if verbose:
                    print(f"  ✓ thread agents-tokens -> {out_agents_tokens}")

                out_agents_tps = os.path.join(
                    dir_path, f"thread_{thread_id}_agents_tps.png"
                )
                plot_tps_by_agent_grouped(
                    totals_by_agent,
                    llm_secs_by_agent,
                    thread_secs,
                    out_agents_tps,
                    title="Tokens per second by Agent (thread)",
                    footer_lines=footer,
                )
                if verbose:
                    print(f"  ✓ thread agents-tps    -> {out_agents_tps}")
            except Exception as e:
                print(
                    f"  ! error generating agent-level charts for thread {thread_id!r}: {e}"
                )

        except Exception as e:
            print(
                f"  ! error generating thread-level charts for {thread_id!r}: {e}"
            )

        # --- Per-run charts (each JSON) ---
        # --- Per-run charts (each JSON) ---
        for r in runs:
            try:
                payload = load_metrics(r.path)
                ctx_run = payload.get("context") or {}

                # A) time breakdown (lollipop)
                total, parts = extract_time_breakdown(
                    payload, group_llm=group_llm
                )
                out = _default_out_path(r.path, "lollipop")
                _ensure_dir(os.path.dirname(out))
                plot_lollipop_time(
                    total,
                    parts,
                    out,
                    title="Time (seconds) by component",
                    log_x=log_x,
                    min_label_pct=min_label_pct,
                    context=ctx_run,
                )
                if verbose:
                    print(f"    - run lollipop    -> {out}")

                # B) token totals (bar) + KDE
                totals_run, samples_run = extract_llm_token_stats(payload)

                out_bar = _default_out_path(r.path, "tokens_bar")
                plot_token_totals_bar(
                    totals_run,
                    out_bar,
                    title="LLM Token Totals by Category",
                    context=ctx_run,
                )
                if verbose:
                    print(f"    - run tokens-bar  -> {out_bar}")

                out_kde = _default_out_path(r.path, "tokens_kde")
                plot_token_kde(
                    samples_run,
                    out_kde,
                    title="LLM Token Usage — KDE",
                    context=ctx_run,
                    log_x=log_x,
                )
                if verbose:
                    print(f"    - run tokens-kde  -> {out_kde}")

                # C) tokens-rate (per-run)  ← ADD THIS BLOCK
                att = compute_attribution(payload)
                llm_wall = compute_llm_wall_seconds(payload)
                llm_seconds = (
                    llm_wall
                    if llm_wall > 0
                    else float(att.get("llm_total_s", 0.0) or 0.0)
                )

                s = ctx_run.get("started_at") and _dt(ctx_run["started_at"])
                e = ctx_run.get("ended_at") and _dt(ctx_run["ended_at"])
                window_seconds = (
                    max(0.0, (e - s).total_seconds())
                    if (s and e)
                    else float(att.get("total_s", 0.0) or 0.0)
                )

                out_rate = _default_out_path(r.path, "tokens_rate")
                plot_token_rates_bar(
                    totals_run,
                    llm_seconds,
                    window_seconds,
                    out_rate,
                    title="Tokens per second",
                    context=ctx_run,
                )
                if verbose:
                    print(f"    - run tokens-rate -> {out_rate}")

            except Exception as e:
                print(
                    f"    ! error on run {getattr(r, 'run_id', None) or r.path}: {e}"
                )

        print("")  # blank line between threads

    print("All charts generated.")
    return 0


# -----------------------
#  Recursive + SUPER
# -----------------------


def _find_thread_dirs_recursive(root: str) -> List[str]:
    """Walk the tree and return subdirs that contain at least one thread_id (JSONs)."""
    found: List[str] = []
    for cur, dirs, files in os.walk(root):
        sessions = scan_directory_for_threads(cur)
        if sessions:
            found.append(cur)
    return sorted(set(found))


def _aggregate_super_across_dirs(
    thread_dirs: List[str],
    *,
    group_llm: bool,
) -> Tuple[float, float, float, dict, dict, dict]:
    """
    Returns:
      total_all, llm_total_all, tool_total_all,
      token_totals_all (dict),
      token_samples_all (dict of lists),
      context_all (dict with started_at/ended_at over the entire span)
    """
    total_all = 0.0
    llm_total_all = 0.0
    tool_total_all = 0.0

    token_totals_all: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    # IMPORTANT: dict-of-lists, not a flat list
    token_samples_all: Dict[str, List[float]] = {
        "input_tokens": [],
        "output_tokens": [],
        "reasoning_tokens": [],
        "cached_tokens": [],
        "total_tokens": [],
    }

    # context bounds
    min_start: datetime | None = None
    max_end: datetime | None = None

    for d in thread_dirs:
        sessions = scan_directory_for_threads(d)
        for _tid, runs in (sessions or {}).items():
            # time breakdown per thread
            t_total, parts, _ = extract_thread_time_breakdown(
                runs, group_llm=group_llm
            )
            total_all += float(t_total or 0.0)

            # sum llm/tool parts (other will be derived)
            for name, secs in parts:
                if str(name).startswith("llm:"):
                    llm_total_all += float(secs or 0.0)
                elif str(name).startswith("tool:"):
                    tool_total_all += float(secs or 0.0)

            # tokens per thread
            t_totals, t_samples, t_ctx = extract_thread_token_stats(runs)
            for k in token_totals_all.keys():
                token_totals_all[k] += int(t_totals.get(k, 0) or 0)

            # MERGE samples category-by-category
            for k in token_samples_all.keys():
                token_samples_all[k].extend(list(t_samples.get(k, []) or []))

            # widen context window
            s = t_ctx.get("started_at")
            e = t_ctx.get("ended_at")
            if s:
                ds = _dt(s)
                min_start = (
                    ds if (min_start is None or ds < min_start) else min_start
                )
            if e:
                de = _dt(e)
                max_end = de if (max_end is None or de > max_end) else max_end

    context_all = {
        "agent": "SUPER",
        "thread_id": f"{len(thread_dirs)} thread dirs",
        "run_id": "",
        "started_at": min_start.isoformat() if min_start else "",
        "ended_at": max_end.isoformat() if max_end else "",
    }
    return (
        total_all,
        llm_total_all,
        tool_total_all,
        token_totals_all,
        token_samples_all,
        context_all,
    )


def run_all_recursive_with_super(
    *,
    root_dir: str,
    group_llm: bool,
    group_by: str,
    log_x: bool,
    min_label_pct: float,
) -> int:
    thread_dirs = _find_thread_dirs_recursive(root_dir)
    if not thread_dirs:
        print(f"No thread directories found under: {root_dir!r}")
        return 0

    print(
        f"Discovered {len(thread_dirs)} thread directories under {root_dir!r}.\n"
    )

    # 1) Run the per-thread/per-run pipeline inside each discovered thread dir
    for d in thread_dirs:
        print(f"--- Processing thread dir: {d}")
        run_all(
            dir_path=d,
            group_llm=group_llm,
            group_by=group_by,
            log_x=log_x,
            min_label_pct=min_label_pct,
            verbose=True,
        )

    # 2) SUPER aggregation across all discovered threads
    print("\n--- Building SUPER report (across all discovered threads) ---")
    total_all, llm_all, tool_all, tok_totals, tok_samples, ctx_all = (
        _aggregate_super_across_dirs(thread_dirs, group_llm=group_llm)
    )

    # parts: llm, tool, other
    other_all = max(0.0, total_all - (llm_all + tool_all))
    parts_all = []
    if llm_all > 0:
        parts_all.append(("llm:total", llm_all))
    if tool_all > 0:
        parts_all.append(("tool:total", tool_all))
    parts_all.append(("other", other_all))

    # Where to save SUPER artifacts: at root
    _ensure_dir(root_dir)

    # lollipop (time)
    super_lolli = os.path.join(root_dir, "super_lollipop.png")
    plot_lollipop_time(
        total_all,
        parts_all,
        super_lolli,
        title="Time (seconds) by component — ALL threads",
        log_x=log_x,
        min_label_pct=min_label_pct,
        context=ctx_all,
    )
    print(f"  ✓ SUPER lollipop   -> {super_lolli}")

    # token totals (bar)
    super_bar = os.path.join(root_dir, "super_tokens_bar.png")
    plot_token_totals_bar(
        tok_totals,
        super_bar,
        title="LLM Token Totals by Category — ALL threads",
        context=ctx_all,
    )
    print(f"  ✓ SUPER tokens-bar -> {super_bar}")

    # token KDE (all samples merged)
    super_kde = os.path.join(root_dir, "super_tokens_kde.png")
    plot_token_kde(
        tok_samples,
        super_kde,
        title="LLM Token Usage — KDE (ALL threads)",
        context=ctx_all,
        log_x=log_x,
    )
    print(f"  ✓ SUPER tokens-kde -> {super_kde}")

    # token rates (two baselines): sum per-thread LLM-active seconds and elapsed window
    # Reuse compute_thread_time_bases per thread dir
    sum_llm_sec = 0.0
    sum_win_sec = 0.0
    for d in thread_dirs:
        sessions = scan_directory_for_threads(d)
        for _tid, runs in (sessions or {}).items():
            llm_s, win_s = compute_thread_time_bases(runs)
            sum_llm_sec += float(llm_s or 0.0)
            sum_win_sec += float(win_s or 0.0)

    super_rate = os.path.join(root_dir, "super_tokens_rate.png")
    plot_token_rates_bar(
        tok_totals,
        sum_llm_sec,
        sum_win_sec,
        super_rate,
        title="Tokens per second — ALL threads",
        context=ctx_all,
    )
    print(f"  ✓ SUPER tokens-rate-> {super_rate}")

    # --- SUPER by-LLM breakdowns ---
    totals_by_model, llm_secs_by_model, window_super, ctx_super = (
        aggregate_super_tokens_by_model(thread_dirs)
    )

    super_bar_models = os.path.join(root_dir, "super_tokens_bar_by_model.png")
    plot_tokens_bar_by_model(
        totals_by_model,
        super_bar_models,
        title="LLM Token Totals by Category — by model",
        context=ctx_super,
    )
    print(f"  ✓ SUPER tokens-bar (by model) -> {super_bar_models}")

    super_rate_models = os.path.join(root_dir, "super_tokens_rate_by_model.png")
    plot_token_rates_by_model(
        totals_by_model,
        llm_secs_by_model,
        window_super,
        super_rate_models,
        title="Tokens per second — by model",
        context=ctx_super,
    )
    print(f"  ✓ SUPER tokens-rate (by model) -> {super_rate_models}")

    # --- SUPER by-AGENT breakdowns (across all discovered thread dirs) ---
    # Build a combined sessions dict that includes runs from every discovered thread dir.
    # Use a composite key to avoid collisions if thread_ids repeat across different subdirs.
    combined_sessions: Dict[str, List] = {}
    for d in thread_dirs:
        sess = scan_directory_for_threads(d) or {}
        for tid, runs in sess.items():
            key = (
                f"{os.path.basename(d)}::{tid}"
                if tid in combined_sessions
                else tid
            )
            combined_sessions[key] = runs

    totals_by_agent, llm_secs_by_agent, sum_thread_secs, summary = (
        aggregate_super_token_stats_by_agent(combined_sessions)
    )

    footer = [
        f"threads: {summary['n_threads']} • runs: {summary['n_runs']}",
        f"Σ thread windows: {summary['sum_thread_secs']:,.3f}s",
    ]

    super_agents_tokens = os.path.join(root_dir, "super_agents_tokens.png")
    plot_tokens_by_agent_stacked(
        totals_by_agent,
        super_agents_tokens,
        title="LLM Token Totals by Agent — ALL threads",
        footer_lines=footer,
    )
    print(f"  ✓ SUPER agents-tokens -> {super_agents_tokens}")

    super_agents_tps = os.path.join(root_dir, "super_agents_tps.png")
    plot_tps_by_agent_grouped(
        totals_by_agent,
        llm_secs_by_agent,
        sum_thread_secs,
        super_agents_tps,
        title="Tokens per second by Agent — ALL threads",
        footer_lines=footer,
    )
    print(f"  ✓ SUPER agents-tps    -> {super_agents_tps}")

    print("\nSUPER report complete.")
    return 0


# -----------------------
#         CLI
# -----------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Telemetry metrics plotting (single-file, thread-level, bulk 'all', or recursive 'all-recursive' with SUPER)."
    )
    ap.add_argument(
        "json_path",
        nargs="?",
        help="Path to a single telemetry metrics JSON (needed for single-file charts).",
    )
    ap.add_argument(
        "--chart",
        choices=[
            "all",  # default: thread-level & per-run inside --dir
            "all-recursive",  # NEW: walk subdirs, run 'all' in each, then SUPER aggregate at root
            # single-file
            "pie",
            "bar",
            "lollipop",
            "tokens-bar",
            "tokens-kde",
            "tokens-rate",
            # thread-level
            "timeline-html",
            "thread-lollipop",
            "thread-tokens-bar",
            "thread-tokens-kde",
            "thread-tokens-rate",
            "thread-agents-tokens",
            "thread-agents-tps",
            "super-agents-tokens",
            "super-agents-tps",
        ],
        default="all",
        help="Which chart(s) to generate. Default: all (thread-level for all threads in --dir + three per-run charts).",
    )
    ap.add_argument(
        "--dir",
        help="Directory containing metrics JSON files (default: json_path's directory or current dir)",
    )
    ap.add_argument(
        "--list-threads",
        action="store_true",
        help="List thread_ids found in --dir and exit",
    )
    ap.add_argument(
        "--thread", help="Limit to this thread_id for thread-level charts"
    )

    # Visual / behavior options
    ap.add_argument(
        "--group-llm",
        action="store_true",
        help="Group all LLM rows into a single 'llm:total' slice",
    )
    ap.add_argument(
        "--group-by",
        choices=["agent", "run"],
        default="run",
        help="timeline-html y-axis grouping (default: run)",
    )
    ap.add_argument(
        "--log-x",
        action="store_true",
        help="Use log x-axis (lollipop and tokens-kde)",
    )
    ap.add_argument(
        "--min-label-pct",
        type=float,
        default=0.0,
        help="Hide labels below this %% in lollipop bars",
    )  # <- escape %
    ap.add_argument(
        "--title",
        default="",
        help="Custom chart title (used in targeted modes)",
    )
    ap.add_argument(
        "--out", help="Output path for targeted modes (single chart)"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Print totals (graph/llm/tool/unattributed) and exit (single JSON only)",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.050,
        help="Tolerance for attribution check (seconds)",
    )

    args = ap.parse_args(argv)

    # Resolve a directory for 'all' and scans
    dir_path = args.dir or (
        os.path.dirname(args.json_path) if args.json_path else "."
    )
    dir_path = dir_path or "."

    # --- list threads only ---
    if args.list_threads:
        sessions = scan_directory_for_threads(dir_path)
        if not sessions:
            print("No thread_ids found.")
            return 0
        print("Thread_IDs found:")
        for i, (tid, count) in enumerate(
            list_threads_summary(sessions), start=1
        ):
            print(f"{i}: {tid!r} [{count} agent JSONs]")
        return 0

    # --- default: ALL (non-recursive) ---
    if args.chart == "all":
        return run_all(
            dir_path=dir_path,
            group_llm=bool(args.group_llm),
            group_by=str(args.group_by),
            log_x=bool(args.log_x),
            min_label_pct=float(args.min_label_pct),
        )

    # --- ALL-RECURSIVE (with SUPER) ---
    if args.chart == "all-recursive":
        return run_all_recursive_with_super(
            root_dir=dir_path,
            group_llm=bool(args.group_llm),
            group_by=str(args.group_by),
            log_x=bool(args.log_x),
            min_label_pct=float(args.min_label_pct),
        )

    # ---------------- Specialized modes below ----------------

    # --- thread-level interactive timeline ---
    if args.chart == "timeline-html":
        sessions = scan_directory_for_threads(dir_path)
        if not sessions:
            print("No thread_ids found in directory.")
            return 1
        if not args.thread:
            print("Thread_IDs found:")
            for i, (tid, count) in enumerate(
                list_threads_summary(sessions), start=1
            ):
                print(f"{i}: {tid!r} [{count} agent JSONs]")
            print("\nRe-run with: --chart timeline-html --thread <thread_id>")
            return 0
        runs = sessions.get(args.thread)
        if not runs:
            print(f"No JSONs found for thread_id: {args.thread!r}")
            return 1
        out_path = args.out or os.path.join(
            dir_path, f"thread_{args.thread}_timeline.html"
        )
        saved = plot_thread_timeline_interactive(
            runs, out_path, group_by=args.group_by
        )
        print(f"Saved interactive timeline to: {saved}")
        return 0

    # --- thread-level single chart routes ---
    if args.chart in ("thread-agents-tokens", "thread-agents-tps"):
        if not args.dir:
            print(f"{args.chart} requires --dir and --thread")
            return 2
        sessions = scan_directory_for_threads(args.dir)
        if not sessions:
            print("No thread_ids found in directory.")
            return 1
        if not args.thread:
            print("Thread_IDs found:")
            for i, (tid, count) in enumerate(
                list_threads_summary(sessions), start=1
            ):
                print(f"{i}: {tid!r} [{count} agent JSONs]")
            print(f"\nRe-run with: --chart {args.chart} --thread <thread_id>")
            return 0

        runs = sessions.get(args.thread)
        if not runs:
            print(f"No JSONs found for thread_id: {args.thread!r}")
            return 1

        totals_by_agent, llm_secs_by_agent, thread_secs = (
            extract_thread_token_stats_by_agent(runs)
        )

        footer = [
            f"thread: {args.thread} • runs: {len(runs)}",
            f"thread window: {thread_secs:,.3f}s",
        ]

        if args.chart == "thread-agents-tokens":
            out_path = args.out or os.path.join(
                args.dir, f"thread_{args.thread}_agents_tokens.png"
            )
            saved = plot_tokens_by_agent_stacked(
                totals_by_agent,
                out_path,
                title=args.title or "LLM Token Totals by Agent (thread)",
                footer_lines=footer,
            )
        else:
            out_path = args.out or os.path.join(
                args.dir, f"thread_{args.thread}_agents_tps.png"
            )
            saved = plot_tps_by_agent_grouped(
                totals_by_agent,
                llm_secs_by_agent,
                thread_secs,
                out_path,
                title=args.title or "Tokens per second by Agent (thread)",
                footer_lines=footer,
            )
        print(f"Saved chart to: {saved}")
        return 0

    if args.chart in (
        "thread-lollipop",
        "thread-tokens-bar",
        "thread-tokens-kde",
        "thread-tokens-rate",
    ):
        sessions = scan_directory_for_threads(dir_path)
        if not sessions:
            print("No thread_ids found in directory.")
            return 1
        if not args.thread:
            print("Thread_IDs found:")
            for i, (tid, count) in enumerate(
                list_threads_summary(sessions), start=1
            ):
                print(f"{i}: {tid!r} [{count} agent JSONs]")
            print(f"\nRe-run with: --chart {args.chart} --thread <thread_id>")
            return 0
        runs = sessions.get(args.thread)
        if not runs:
            print(f"No JSONs found for thread_id: {args.thread!r}")
            return 1

        os.makedirs(dir_path, exist_ok=True)

        if args.chart == "thread-lollipop":
            total, parts, ctx = extract_thread_time_breakdown(
                runs, group_llm=args.group_llm
            )
            out_path = args.out or os.path.join(
                dir_path, f"thread_{args.thread}_lollipop.png"
            )
            saved = plot_lollipop_time(
                total,
                parts,
                out_path,
                title=args.title
                or "Time (seconds) by component — thread total",
                log_x=args.log_x,
                min_label_pct=args.min_label_pct,
                context=ctx,
            )
            print(f"Saved chart to: {saved}")
            return 0

        totals, samples, ctx = extract_thread_token_stats(runs)

        if args.chart == "thread-tokens-bar":
            out_path = args.out or os.path.join(
                dir_path, f"thread_{args.thread}_tokens_bar.png"
            )
            saved = plot_token_totals_bar(
                totals,
                out_path,
                title=args.title or "LLM Token Totals by Category — thread",
                context=ctx,
            )
            print(f"Saved chart to: {saved}")
            return 0

        if args.chart == "thread-tokens-kde":
            out_path = args.out or os.path.join(
                dir_path, f"thread_{args.thread}_tokens_kde.png"
            )
            saved = plot_token_kde(
                samples,
                out_path,
                title=args.title or "LLM Token Usage — KDE (thread)",
                context=ctx,
                log_x=args.log_x,
            )
            print(f"Saved chart to: {saved}")
            return 0

        # thread-tokens-rate
        llm_seconds, elapsed_seconds = compute_thread_time_bases(runs)
        out_path = args.out or os.path.join(
            dir_path, f"thread_{args.thread}_tokens_rate.png"
        )
        saved = plot_token_rates_bar(
            totals,
            llm_seconds,
            elapsed_seconds,
            out_path,
            title=args.title or "tokens per second — thread",
            context=ctx,
        )
        print(f"Saved chart to: {saved}")
        return 0

    # --- SUPER (directory) agent-level charts ---
    if args.chart in ("super-agents-tokens", "super-agents-tps"):
        sessions = scan_directory_for_threads(dir_path)
        if not sessions:
            print("No thread_ids found in directory.")
            return 1

        totals_by_agent, llm_secs_by_agent, sum_thread_secs, summary = (
            aggregate_super_token_stats_by_agent(sessions)
        )

        footer = [
            f"threads: {summary['n_threads']} • runs: {summary['n_runs']}",
            f"Σ thread windows: {summary['sum_thread_secs']:,.3f}s",
        ]

        if args.chart == "super-agents-tokens":
            out_path = args.out or os.path.join(
                dir_path, "super_agents_tokens.png"
            )
            saved = plot_tokens_by_agent_stacked(
                totals_by_agent,
                out_path,
                title=args.title
                or f"SUPER: {summary['n_threads']} thread dirs — tokens by agent",
                footer_lines=footer,
            )
        else:
            out_path = args.out or os.path.join(
                dir_path, "super_agents_tps.png"
            )
            saved = plot_tps_by_agent_grouped(
                totals_by_agent,
                llm_secs_by_agent,
                sum_thread_secs,
                out_path,
                title=args.title
                or f"SUPER: {summary['n_threads']} thread dirs — tokens/sec by agent",
                footer_lines=footer,
            )

        print(f"Saved chart to: {saved}")
        return 0

    # --- single-file routes ---
    if not args.json_path:
        print("error: json_path is required for this chart.")
        return 2

    payload = load_metrics(args.json_path)
    ctx = payload.get("context") or {}

    # quick check
    if args.check:
        att = compute_attribution(payload)
        total = att["total_s"]
        llm_total = att["llm_total_s"]
        tool_total = att["tool_total_s"]
        unattributed = att["unattributed_s"]
        overage = att["overage_s"]
        print(f"graph:graph total_s : {total:.6f}s")
        print(f"  LLM total_s       : {llm_total:.6f}s")
        print(f"  Tool total_s      : {tool_total:.6f}s")
        print(f"  Unattributed      : {unattributed:.6f}s")
        if overage > 0:
            print(f"  Overage (llm+tool - total): {overage:.6f}s")
        delta = abs((llm_total + tool_total + unattributed) - total)
        ok = delta <= args.epsilon
        print(
            f"Δ (recon - total)   : {delta:.6f}s  -> {'OK' if ok else 'WARN'} (ε={args.epsilon:.3f}s)"
        )
        return 0 if ok else 1

    # single-file tokens-rate
    if args.chart == "tokens-rate":
        totals, _ = extract_llm_token_stats(payload)
        att = compute_attribution(payload)
        llm_wall = compute_llm_wall_seconds(payload)
        llm_seconds = (
            llm_wall
            if llm_wall > 0
            else float(att.get("llm_total_s", 0.0) or 0.0)
        )

        s = ctx.get("started_at") and _dt(ctx.get("started_at"))
        e = ctx.get("ended_at") and _dt(ctx.get("ended_at"))
        window_seconds = (
            max(0.0, (e - s).total_seconds())
            if (s and e)
            else float(att.get("total_s", 0.0) or 0.0)
        )

        out_path = args.out or _default_out_path(args.json_path, "tokens_rate")
        saved = plot_token_rates_bar(
            totals,
            llm_seconds,
            window_seconds,
            out_path,
            title=args.title or "Tokens per second",
            context=ctx,
        )
        print(f"Saved chart to: {saved}")
        return 0

    # single-file lollipop / pie / bar
    total, parts = extract_time_breakdown(payload, group_llm=args.group_llm)
    out_path = args.out or _default_out_path(args.json_path, args.chart)
    _ensure_dir(os.path.dirname(out_path))

    if args.chart in ("tokens-bar", "tokens-kde"):
        totals, samples = extract_llm_token_stats(payload)
        out_path = args.out or _default_out_path(args.json_path, args.chart)
        if args.chart == "tokens-bar":
            saved = plot_token_totals_bar(
                totals,
                out_path,
                title=args.title,
                context=ctx,
            )
        else:
            saved = plot_token_kde(
                samples,
                out_path,
                title=args.title,
                context=ctx,
                log_x=args.log_x,
            )
        print(f"Saved chart to: {saved}")
        return 0

    if args.chart == "lollipop":
        saved = plot_lollipop_time(
            total,
            parts,
            out_path,
            title=args.title,
            log_x=args.log_x,
            min_label_pct=args.min_label_pct
            if args.min_label_pct is not None
            else 0.0,
            context=ctx,
        )
    else:
        saved = plot_time_breakdown(
            total,
            parts,
            out_path,
            title=args.title,
            chart=args.chart,
            min_label_pct=args.min_label_pct
            if args.min_label_pct is not None
            else (1.0 if args.chart == "bar" else 1.0),
            context=ctx,
        )
    print(f"Saved chart to: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
