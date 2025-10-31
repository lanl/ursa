# metrics_session.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ursa.observability.metrics_charts import (
    compute_attribution,
    extract_llm_token_stats,  # reuse per-file tokens
    extract_time_breakdown,  # reuse per-file extraction
)
from ursa.observability.metrics_io import load_metrics


def _dt(x: str) -> datetime:
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


# Compact, predictable layout
_LAYOUT = dict(
    header_y=0.965,
    subtitle_y=0.915,
    legend_y=0.885,  # legend sits just under the subtitle
    ax_rect=(0.10, 0.28, 0.86, 0.58),  # [left, bottom, width, height]
    footer1_y=0.105,
    footer2_y=0.070,
)

ISO_RE = re.compile(r"Z$")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ISO_RE.sub("+00:00", ts)).astimezone(
            timezone.utc
        )
    except Exception:
        return None


def _fmt_iso_pretty(ts: str | None) -> str:
    dt = _parse_iso(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt else (ts or "")


@dataclass
class RunRecord:
    path: str
    agent: str
    thread_id: str
    run_id: str
    started_at: datetime
    ended_at: datetime

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


# -------------------------------
#         Directory scan
# -------------------------------
def scan_directory_for_threads(dir_path: str) -> Dict[str, List[RunRecord]]:
    """
    Scan a directory for metrics JSONs and group them by thread_id.
    Returns: {thread_id: [RunRecord, ...]}
    """
    sessions: Dict[str, List[RunRecord]] = {}
    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith(".json"):
            continue
        fp = os.path.join(dir_path, name)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ctx = payload.get("context") or {}
            agent = str(ctx.get("agent") or "")
            thread_id = str(ctx.get("thread_id") or "")
            run_id = str(ctx.get("run_id") or "")
            s = _parse_iso(ctx.get("started_at"))
            e = _parse_iso(ctx.get("ended_at"))
            if not (thread_id and agent and run_id and s and e):
                continue
            rec = RunRecord(
                path=fp,
                agent=agent,
                thread_id=thread_id,
                run_id=run_id,
                started_at=s,
                ended_at=e,
            )
            sessions.setdefault(thread_id, []).append(rec)
        except Exception:
            continue

    for tid, runs in sessions.items():
        runs.sort(key=lambda r: r.started_at)
    return sessions


def list_threads_summary(
    sessions: Dict[str, List[RunRecord]],
) -> List[Tuple[str, int]]:
    """Return [(thread_id, count)] sorted by count desc, then thread_id."""
    out = [(tid, len(runs)) for tid, runs in sessions.items()]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


# -------------------------------
#         Timeline plot
# -------------------------------


def _abbrev_agent(agent: str) -> str:
    # Common cleanup: drop trailing 'Agent', trim to a tidy length
    base = agent[:-5] if agent.endswith("Agent") else agent
    base = base.strip()
    return base if len(base) <= 10 else (base[:9] + "…")


def _label_for_run(r: RunRecord, seconds_precision: int = 1) -> str:
    return f"{_abbrev_agent(r.agent)}:{r.run_id[:6]} ({r.duration_s:.{seconds_precision}f}s)"


def plot_thread_timeline(
    runs: List[RunRecord],
    out_path: str,
    *,
    title: str = "Agent runs timeline",
    stacked_rows: bool = False,  # False = single lane (“flow”); True = one row per run
    min_label_sec: float = 0.50,  # skip labels for bars shorter than this
) -> str:
    """
    Draw a time timeline for a single thread_id across multiple agent runs.

    Fixes:
      • Smarter labels: shortened text + alternating vertical offset to avoid overlap
      • Exact bounds: xlim is precisely first start → last end
      • Legend outside bars: compact legend drawn above the axes
    """
    if not runs:
        raise ValueError("No runs provided for timeline")

    # timeline bounds (exact)
    t0 = min(r.started_at for r in runs)
    t1 = max(r.ended_at for r in runs)

    thread_id = runs[0].thread_id
    header = f"Thread: {thread_id}"
    subtitle = title
    footer1 = f"runs: {len(runs)}"
    footer2 = f"{t0.strftime('%Y-%m-%d %H:%M:%S UTC')} \u2192 {t1.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    # Map agents to stable colors
    agents = [r.agent for r in runs]
    uniq_agents: List[str] = []
    for a in agents:
        if a not in uniq_agents:
            uniq_agents.append(a)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        a: color_cycle[i % max(1, len(color_cycle))]
        for i, a in enumerate(uniq_agents)
    }

    if stacked_rows:
        fig_h = max(2.5, 0.35 * len(runs))
        fig = plt.figure(figsize=(10, fig_h))
        ax = fig.add_axes(list(_LAYOUT["ax_rect"]))
        bar_h = 0.8
        y_positions = list(range(len(runs)))
        for yi, r in zip(y_positions, runs):
            xs = mdates.date2num(r.started_at)
            w = mdates.date2num(r.ended_at) - xs
            ax.barh(
                yi,
                w,
                left=xs,
                height=bar_h,
                edgecolor="black",
                color=color_map.get(r.agent),
            )
            if r.duration_s >= min_label_sec:
                ax.text(
                    xs + w / 2,
                    yi,
                    _label_for_run(r, 1),
                    ha="center",
                    va="center",
                    fontsize="small",
                )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([
            f"{_abbrev_agent(r.agent)}:{r.run_id[:6]}" for r in runs
        ])
        ax.invert_yaxis()
    else:
        # Single lane “flow” with alternating label bands
        fig = plt.figure(figsize=(10, 2.4))
        ax = fig.add_axes(list(_LAYOUT["ax_rect"]))
        lane_y, lane_h = 0.0, 1.0

        segs, colors = [], []
        for r in runs:
            xs = mdates.date2num(r.started_at)
            w = mdates.date2num(r.ended_at) - xs
            segs.append((xs, w))
            colors.append(color_map.get(r.agent))
        ax.broken_barh(
            segs, (lane_y, lane_h), facecolors=colors, edgecolor="black"
        )

        # Alternate label positions: upper/lower halves of the lane
        upper_y = lane_y + lane_h * 0.72
        lower_y = lane_y + lane_h * 0.28
        for idx, ((xs, w), r) in enumerate(zip(segs, runs)):
            if r.duration_s < min_label_sec:
                continue
            y = upper_y if (idx % 2 == 0) else lower_y
            ax.text(
                xs + w / 2,
                y,
                _label_for_run(r, 1),
                ha="center",
                va="center",
                fontsize="small",
            )

        ax.set_yticks([])

    # x axis formatting (exact bounds, no margins)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))
    ax.margins(x=0)  # ensure bars touch the exact edges
    ax.set_xlabel("Time (UTC)", fontsize="small")
    ax.tick_params(labelsize="small")

    # Header / subtitle
    fig.text(0.5, _LAYOUT["header_y"], header, ha="center", va="top")
    fig.text(
        0.5,
        _LAYOUT["subtitle_y"],
        subtitle,
        ha="center",
        va="top",
        fontsize="small",
    )

    # Legend (outside the axes, above the chart)
    if len(uniq_agents) > 0:
        handles = [
            plt.Line2D([0], [0], color=color_map[a], lw=6) for a in uniq_agents
        ]
        fig.legend(
            handles,
            [_abbrev_agent(a) for a in uniq_agents],
            loc="center",
            bbox_to_anchor=(0.5, _LAYOUT["legend_y"]),
            ncol=max(1, min(4, len(uniq_agents))),
            frameon=False,
            fontsize="x-small",
        )

    # Footers
    fig.text(
        0.5,
        _LAYOUT["footer1_y"],
        footer1,
        ha="center",
        va="center",
        fontsize="x-small",
    )
    fig.text(
        0.5,
        _LAYOUT["footer2_y"],
        footer2,
        ha="center",
        va="center",
        fontsize="x-small",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def runs_to_dataframe(runs: list[RunRecord]) -> pd.DataFrame:
    rows = [
        {
            "thread_id": r.thread_id,
            "agent": r.agent,
            "run_id": r.run_id,
            "label": f"{r.agent}:{r.run_id[:8]}",
            "start": r.started_at,
            "end": r.ended_at,
            "duration_s": r.duration_s,
            "started_at": r.started_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "ended_at": r.ended_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        for r in runs
    ]
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


def plot_thread_timeline_interactive(
    runs: list[RunRecord],
    out_html: str,
    *,
    group_by: str = "agent",  # "agent" (few lanes) or "run" (one lane per run)
) -> str:
    import plotly.express as px

    if not runs:
        raise ValueError("No runs provided")
    df = runs_to_dataframe(runs)
    thread_id = runs[0].thread_id

    ycol = "agent" if group_by == "agent" else "label"
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y=ycol,
        color="agent",
        hover_data={
            "agent": True,
            "run_id": True,
            "duration_s": ":.2f",
            "started_at": True,
            "ended_at": True,
            "label": False,
            "start": False,
            "end": False,
        },
        title=f"Thread: {thread_id} — Agent runs timeline (interactive)",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        legend_title_text="Agent",
        margin=dict(l=40, r=20, t=60, b=40),
        hoverlabel=dict(namelength=-1),
    )
    # lock bounds to exact workflow window
    fig.update_xaxes(range=[df["start"].min(), df["end"].max()])
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    return out_html


def export_thread_csv(runs: list[RunRecord], out_csv: str) -> str:
    df = runs_to_dataframe(runs)
    df.to_csv(out_csv, index=False)
    return out_csv


def aggregate_thread_context(runs: list[RunRecord]) -> dict:
    """Build a context dict for charts at the thread level."""
    if not runs:
        return {}
    t0 = min(r.started_at for r in runs).astimezone(timezone.utc)
    t1 = max(r.ended_at for r in runs).astimezone(timezone.utc)
    thread_id = runs[0].thread_id
    # We intentionally set agent="Thread" so chart headers read "Thread : <id>"
    return {
        "agent": "Thread",
        "thread_id": thread_id,
        "run_id": "",
        "started_at": t0.isoformat(),
        "ended_at": t1.isoformat(),
    }


def extract_thread_time_breakdown(
    runs: list[RunRecord],
    *,
    group_llm: bool = False,
) -> tuple[float, list[tuple[str, float]], dict]:
    """
    Sum graph totals and parts (llm/tools/other) across all runs of a thread.
    Returns (total_seconds, parts_list, context).
    """
    total_sum = 0.0
    parts_acc: dict[str, float] = {}
    for r in runs:
        payload = load_metrics(r.path)
        total_i, parts_i = extract_time_breakdown(payload, group_llm=group_llm)
        total_sum += float(total_i or 0.0)
        for label, sec in parts_i:
            parts_acc[label] = parts_acc.get(label, 0.0) + float(sec or 0.0)

    # stable order: by seconds desc
    parts = sorted(parts_acc.items(), key=lambda kv: kv[1], reverse=True)
    ctx = aggregate_thread_context(runs)
    return total_sum, parts, ctx


def extract_thread_token_stats(
    runs: list[RunRecord],
) -> tuple[dict[str, int], dict[str, list[int]], dict]:
    """
    Merge token totals and concatenate per-call samples across all runs of a thread.
    Returns (totals, samples, context).
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    samples = {k: [] for k in totals.keys()}

    for r in runs:
        payload = load_metrics(r.path)
        t_i, s_i = extract_llm_token_stats(payload)
        # sum totals
        for k in totals:
            totals[k] += int(t_i.get(k, 0) or 0)
        # concat samples
        for k in samples:
            samples[k].extend(list(s_i.get(k, []) or []))

    ctx = aggregate_thread_context(runs)
    return totals, samples, ctx


def compute_thread_time_bases(runs: list[RunRecord]) -> tuple[float, float]:
    """
    Return (llm_active_seconds, thread_elapsed_seconds) for a thread.
    - llm_active_seconds: sum of LLM total_s across all runs
    - thread_elapsed_seconds: (max ended_at - min started_at)
    """
    if not runs:
        return (0.0, 0.0)
    llm = 0.0
    for r in runs:
        payload = load_metrics(r.path)
        att = compute_attribution(payload)
        llm += float(att.get("llm_total_s", 0.0) or 0.0)
    start = min(r.started_at for r in runs).astimezone(timezone.utc)
    end = max(r.ended_at for r in runs).astimezone(timezone.utc)
    elapsed = max(0.0, (end - start).total_seconds())
    return (llm, elapsed)


def extract_run_tokens_by_model(
    payload: dict,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    """
    Returns:
      tokens_by_model: {model: {input_tokens,...,total_tokens}}
      seconds_by_model: {model: llm_active_seconds_sum}
    """
    tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seconds_by_model: Dict[str, float] = defaultdict(float)

    # 1) tokens by model from llm_events
    for ev in payload.get("llm_events") or []:
        name = ev.get("name") or ""
        model = (
            name.split("llm:", 1)[-1]
            if name.startswith("llm:")
            else (name or "unknown")
        )
        roll = ((ev.get("metrics") or {}).get("usage_rollup")) or {}
        for k in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        ):
            try:
                tokens_by_model[model][k] += int(float(roll.get(k, 0) or 0))
            except Exception:
                pass

    # 2) llm-active seconds by model from tables.llm
    for row in (payload.get("tables") or {}).get("llm") or []:
        n = row.get("name") or ""
        model = (
            n.split("llm:", 1)[-1] if n.startswith("llm:") else (n or "unknown")
        )
        try:
            seconds_by_model[model] += float(row.get("total_s") or 0.0)
        except Exception:
            pass

    # Cast out of defaultdicts
    return {m: dict(d) for m, d in tokens_by_model.items()}, dict(
        seconds_by_model
    )


def extract_thread_tokens_by_model(
    runs,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float], float, dict]:
    """
    Aggregate by LLM across all runs in a single thread.
    Returns:
      tokens_by_model, seconds_by_model, window_seconds, context
    """
    tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seconds_by_model: Dict[str, float] = defaultdict(float)

    min_start, max_end = None, None
    for r in runs:
        payload = load_metrics(r.path)
        t_by_model, s_by_model = extract_run_tokens_by_model(payload)

        for m, d in t_by_model.items():
            for k, v in d.items():
                tokens_by_model[m][k] += int(v or 0)
        for m, secs in s_by_model.items():
            seconds_by_model[m] += float(secs or 0.0)

        ctx = payload.get("context") or {}
        s, e = ctx.get("started_at"), ctx.get("ended_at")
        if s:
            ds = _dt(s)
            min_start = (
                ds if (min_start is None or ds < min_start) else min_start
            )
        if e:
            de = _dt(e)
            max_end = de if (max_end is None or de > max_end) else max_end

    window_seconds = (
        (max_end - min_start).total_seconds()
        if (min_start and max_end)
        else 0.0
    )
    ctx_out = {
        "agent": "Thread",
        "thread_id": runs[0].thread_id if runs else "",
        "run_id": "",
        "started_at": min_start.isoformat() if min_start else "",
        "ended_at": max_end.isoformat() if max_end else "",
    }
    return (
        {m: dict(tokens_by_model[m]) for m in tokens_by_model},
        dict(seconds_by_model),
        window_seconds,
        ctx_out,
    )


def aggregate_super_tokens_by_model(
    thread_dirs: List[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float], float, dict]:
    """
    Walk all thread subdirs and aggregate per-LLM totals and LLM-active seconds.
    Returns:
      tokens_by_model, seconds_by_model, window_seconds, context
    """
    tokens_by_model: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seconds_by_model: Dict[str, float] = defaultdict(float)

    min_start, max_end = None, None

    for d in thread_dirs:
        sessions = scan_directory_for_threads(d)
        for _tid, runs in (sessions or {}).items():
            for r in runs:
                payload = load_metrics(r.path)
                t_by_model, s_by_model = extract_run_tokens_by_model(payload)

                for m, dct in t_by_model.items():
                    for k, v in dct.items():
                        tokens_by_model[m][k] += int(v or 0)
                for m, secs in s_by_model.items():
                    seconds_by_model[m] += float(secs or 0.0)

                ctx = payload.get("context") or {}
                s, e = ctx.get("started_at"), ctx.get("ended_at")
                if s:
                    ds = _dt(s)
                    min_start = (
                        ds
                        if (min_start is None or ds < min_start)
                        else min_start
                    )
                if e:
                    de = _dt(e)
                    max_end = (
                        de if (max_end is None or de > max_end) else max_end
                    )

    window_seconds = (
        (max_end - min_start).total_seconds()
        if (min_start and max_end)
        else 0.0
    )
    ctx_out = {
        "agent": "SUPER",
        "thread_id": f"{len(thread_dirs)} thread dirs",
        "run_id": "",
        "started_at": min_start.isoformat() if min_start else "",
        "ended_at": max_end.isoformat() if max_end else "",
    }
    return (
        {m: dict(tokens_by_model[m]) for m in tokens_by_model},
        dict(seconds_by_model),
        window_seconds,
        ctx_out,
    )


def extract_thread_token_stats_by_agent(
    runs: List[RunRecord],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float], float]:
    """
    Aggregate token totals and LLM-active seconds by AGENT for a single thread.
    Returns:
      (totals_by_agent, llm_secs_by_agent, thread_window_seconds)
        - totals_by_agent: {agent: {input_tokens, output_tokens, reasoning_tokens, cached_tokens, total_tokens}}
        - llm_secs_by_agent: {agent: seconds}
        - thread_window_seconds: (max end - min start) across all runs in this thread
    """
    totals_by_agent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    llm_secs_by_agent: Dict[str, float] = defaultdict(float)

    if not runs:
        return {}, {}, 0.0

    # compute thread window
    min_start = min(r.started_at for r in runs)
    max_end = max(r.ended_at for r in runs)
    thread_window_seconds = max(0.0, (max_end - min_start).total_seconds())

    for r in runs:
        payload = load_metrics(r.path)

        # tokens for this run -> bucket by r.agent
        run_totals, _ = extract_llm_token_stats(payload)
        for k, v in run_totals.items():
            try:
                totals_by_agent[r.agent][k] += int(v or 0)
            except Exception:
                pass

        # llm-active seconds for this run -> bucket by r.agent
        try:
            att = compute_attribution(payload)
            llm_secs_by_agent[r.agent] += float(
                att.get("llm_total_s", 0.0) or 0.0
            )
        except Exception:
            pass

    # cast out of defaultdicts
    return (
        {a: dict(d) for a, d in totals_by_agent.items()},
        dict(llm_secs_by_agent),
        float(thread_window_seconds),
    )


def aggregate_super_token_stats_by_agent(
    sessions: Dict[str, List[RunRecord]],
) -> Tuple[
    Dict[str, Dict[str, int]], Dict[str, float], float, Dict[str, float]
]:
    """
    Aggregate across ALL threads in a directory (non-recursive).
    Input:
      sessions: {thread_id: [RunRecord, ...]}  (from scan_directory_for_threads)
    Returns:
      totals_by_agent, llm_secs_by_agent, sum_thread_secs, summary
        - totals_by_agent: {agent: {...token categories...}}
        - llm_secs_by_agent: {agent: seconds}
        - sum_thread_secs: sum of each thread's (max end - min start)
        - summary: {"n_threads": ..., "n_runs": ..., "sum_thread_secs": ...}
    """
    totals_by_agent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    llm_secs_by_agent: Dict[str, float] = defaultdict(float)
    sum_thread_secs = 0.0
    n_runs = 0

    for _tid, runs in (sessions or {}).items():
        if not runs:
            continue
        n_runs += len(runs)

        # thread window for this thread
        t_min = min(r.started_at for r in runs)
        t_max = max(r.ended_at for r in runs)
        sum_thread_secs += max(0.0, (t_max - t_min).total_seconds())

        # per-run aggregation by agent
        for r in runs:
            payload = load_metrics(r.path)

            # tokens
            run_totals, _ = extract_llm_token_stats(payload)
            for k, v in run_totals.items():
                try:
                    totals_by_agent[r.agent][k] += int(v or 0)
                except Exception:
                    pass

            # llm seconds
            try:
                att = compute_attribution(payload)
                llm_secs_by_agent[r.agent] += float(
                    att.get("llm_total_s", 0.0) or 0.0
                )
            except Exception:
                pass

    summary = {
        "n_threads": float(len(sessions)),
        "n_runs": float(n_runs),
        "sum_thread_secs": float(sum_thread_secs),
    }

    return (
        {a: dict(d) for a, d in totals_by_agent.items()},
        dict(llm_secs_by_agent),
        float(sum_thread_secs),
        summary,
    )
