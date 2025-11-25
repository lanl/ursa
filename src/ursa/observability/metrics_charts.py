# charts.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # safe for headless environments
import datetime as _dt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde  # type: ignore

# Layout spec for compact charts (fractions of figure size)
_LAYOUT = dict(
    header_y=1.1,  # big header (agent : thread_id)
    subtitle_y=1.0,  # smaller subtitle (title/total)
    ax_rect=(0.12, 0.310, 0.84, 0.58),  # [left, bottom, width, height]
    footer1_y=0.105,  # run_id
    footer2_y=0.050,  # started → ended
)


def compute_llm_wall_seconds(payload: dict) -> float:
    evs = payload.get("llm_events") or []
    intervals = []
    for ev in evs:
        t0, t1 = ev.get("t_start"), ev.get("t_end")
        if (
            isinstance(t0, (int, float))
            and isinstance(t1, (int, float))
            and t1 >= t0
        ):
            intervals.append((t0, t1))
    if not intervals:
        return 0.0
    intervals.sort()
    wall = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            wall += cur_e - cur_s
            cur_s, cur_e = s, e
    wall += cur_e - cur_s
    return float(wall)


def compute_attribution(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns a dict with totals useful for validation/printing:
      total_s        = graph:graph (or any graph:* fallback)
      llm_total_s    = sum of llm rows
      tool_total_s   = sum of tool rows
      unattributed_s = max(0, total_s - (llm_total_s + tool_total_s))
      overage_s      = max(0, (llm_total_s + tool_total_s) - total_s)
    """
    tables = payload.get("tables") or {}
    # total via our updated finder
    total_s, _ = extract_time_breakdown(payload)  # reuse finder logic
    # but extract_time_breakdown returns (total, parts); we only need total
    total_s = _find_graph_total_seconds(payload)

    llm_total_s = sum(
        float(r.get("total_s") or 0.0) for r in (tables.get("llm") or [])
    )
    tool_total_s = sum(
        float(r.get("total_s") or 0.0) for r in (tables.get("tool") or [])
    )
    unattributed_s = max(0.0, total_s - (llm_total_s + tool_total_s))
    overage_s = max(0.0, (llm_total_s + tool_total_s) - total_s)
    return {
        "total_s": total_s,
        "llm_total_s": llm_total_s,
        "tool_total_s": tool_total_s,
        "unattributed_s": unattributed_s,
        "overage_s": overage_s,
    }


def _extract_context(payload: Dict[str, Any]) -> Dict[str, str]:
    """Return a normalized context dict with agent/thread/run_id/started/ended."""
    ctx = payload.get("context") or {}
    return {
        "agent": str(ctx.get("agent") or ""),
        "thread_id": str(ctx.get("thread_id") or ""),
        "run_id": str(ctx.get("run_id") or ""),
        "started_at": str(ctx.get("started_at") or ""),
        "ended_at": str(ctx.get("ended_at") or ""),
    }


def _fmt_iso_pretty(ts: str) -> str:
    """ISO8601 -> 'YYYY-MM-DD HH:MM:SS UTC' (best-effort; falls back to the original)."""
    if not ts:
        return ""
    try:
        dt = _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = dt.astimezone(_dt.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts


def _find_graph_total_seconds(payload: Dict[str, Any]) -> float:
    """
    Total time = tables.runnable row where name == 'graph:graph'
    Fallback to totals.graph_total_s if needed.
    """
    tables = payload.get("tables") or {}
    runnable_rows = tables.get("runnable") or []
    for row in runnable_rows:
        if str(row.get("name", "")).startswith("graph:"):
            return float(row.get("total_s") or 0.0)
    totals = payload.get("totals") or {}
    try:
        return float(totals.get("graph_total_s") or 0.0)
    except Exception:
        return 0.0


def _aggregate_tools_seconds(
    payload: Dict[str, Any],
) -> List[Tuple[str, float]]:
    tables = payload.get("tables") or {}
    tool_rows = tables.get("tool") or []
    out: List[Tuple[str, float]] = []
    for row in tool_rows:
        name = str(row.get("name") or "tool:unknown")
        try:
            total_s = float(row.get("total_s") or 0.0)
        except Exception:
            total_s = 0.0
        out.append((f"tool:{name}", total_s))
    return out


def _aggregate_llm_seconds(
    payload: Dict[str, Any], *, group_llm: bool
) -> List[Tuple[str, float]]:
    tables = payload.get("tables") or {}
    llm_rows = tables.get("llm") or []
    if group_llm:
        total = 0.0
        for r in llm_rows:
            try:
                total += float(r.get("total_s") or 0.0)
            except Exception:
                pass
        return [("llm:total", total)] if total > 0 else []
    else:
        out: List[Tuple[str, float]] = []
        for row in llm_rows:
            name = str(row.get("name") or "llm:unknown")
            try:
                total_s = float(row.get("total_s") or 0.0)
            except Exception:
                total_s = 0.0
            out.append((name, total_s))
        return out


def extract_time_breakdown(
    payload: Dict[str, Any], *, group_llm: bool = False
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Returns (total_seconds, parts), where parts is a list of (label, seconds).
    parts = [each tool, each llm (or grouped), "other"] with "other" >= 0.
    """
    total = _find_graph_total_seconds(payload)
    parts: List[Tuple[str, float]] = []
    parts.extend(_aggregate_tools_seconds(payload))
    parts.extend(_aggregate_llm_seconds(payload, group_llm=group_llm))

    used = sum(v for _, v in parts)
    other = max(0.0, total - used)
    parts.append(("other", other))

    # drop zero entries to keep charts tidy
    parts = [(k, v) for k, v in parts if v > 0.0]
    return total, parts


def plot_time_breakdown(
    total: float,
    parts: List[Tuple[str, float]],
    out_path: str,
    *,
    title: str = "",
    chart: str = "pie",  # "pie" or "bar"
    min_label_pct: float = 1.0,
    context: Dict[str, Any] | None = None,  # NEW
) -> str:
    labels = [k for k, _ in parts]
    values = [v for _, v in parts]
    overall = sum(values) or 1.0

    # ----- build header/footer text from context -----
    ctx = context or {}
    agent = str(ctx.get("agent") or "")
    thread_id = str(ctx.get("thread_id") or "")
    run_id = str(ctx.get("run_id") or "")
    started = _fmt_iso_pretty(str(ctx.get("started_at") or ""))
    ended = _fmt_iso_pretty(str(ctx.get("ended_at") or ""))
    header = " : ".join([p for p in [agent, thread_id] if p]) or ""
    subtitle = title or f"Time Breakdown (total = {total:.3f}s)"

    if chart == "bar":
        fig = plt.figure(figsize=(8, 1.8))
        ax = fig.add_axes([0.12, 0.30, 0.84, 0.56])

        left = 0.0
        for label, val in parts:
            width = val / overall
            ax.barh([0], [width], left=left, edgecolor="black")
            pct = width * 100.0
            if pct >= min_label_pct:
                ax.text(
                    left + width / 2.0,
                    0,
                    f"{label} ({pct:.1f}%)",
                    ha="center",
                    va="center",
                )
            left += width

        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Share of graph:graph wall time")

        if header:
            fig.text(0.5, 0.965, header, ha="center", va="top")
            fig.text(
                0.5, 0.915, subtitle, ha="center", va="top", fontsize="small"
            )
        else:
            fig.text(0.5, 0.945, subtitle, ha="center", va="top")

        if run_id:
            fig.text(
                0.5,
                0.10,
                f"run_id: {run_id}",
                ha="center",
                va="center",
                fontsize="x-small",
            )
        if started or ended:
            fig.text(
                0.5,
                0.06,
                f"{started} \u2192 {ended}",
                ha="center",
                va="center",
                fontsize="x-small",
            )

        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out_path

    # --- pie ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.08, 0.22, 0.84, 0.70])  # tighter

    def _fmt(pct):
        abs_val = (pct / 100.0) * overall
        return f"{pct:.1f}%\n{abs_val:.3f}s"

    ax.pie(values, labels=labels, autopct=_fmt, startangle=90)
    ax.axis("equal")

    if header:
        fig.text(0.5, 0.965, header, ha="center", va="top")
        fig.text(
            0.5,
            0.915,
            (title or f"Time Breakdown (total = {total:.3f}s)"),
            ha="center",
            va="top",
            fontsize="small",
        )
    else:
        fig.text(
            0.5,
            0.945,
            (title or f"Time Breakdown (total = {total:.3f}s)"),
            ha="center",
            va="top",
        )

    if run_id:
        fig.text(
            0.5,
            0.10,
            f"run_id: {run_id}",
            ha="center",
            va="center",
            fontsize="x-small",
        )
    if started or ended:
        fig.text(
            0.5,
            0.06,
            f"{started} \u2192 {ended}",
            ha="center",
            va="center",
            fontsize="x-small",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def plot_lollipop_time(
    total: float,
    parts: List[Tuple[str, float]],
    out_path: str,
    *,
    title: str = "",
    log_x: bool = True,
    min_label_pct: float = 0.0,  # you said you set default to 0.0
    show_seconds: bool = True,
    show_percent: bool = True,
    exclude_zero: bool = True,
    context: Dict[str, Any] | None = None,  # NEW
) -> str:
    data = [(k, v) for (k, v) in parts if (v > 0 if exclude_zero else True)]
    data.sort(key=lambda kv: kv[1])
    labels = [k for k, _ in data]
    values = [v for _, v in data]

    # ----- context header/footer -----
    ctx = context or {}
    agent = str(ctx.get("agent") or "")
    thread_id = str(ctx.get("thread_id") or "")
    run_id = str(ctx.get("run_id") or "")
    started = _fmt_iso_pretty(str(ctx.get("started_at") or ""))
    ended = _fmt_iso_pretty(str(ctx.get("ended_at") or ""))
    header = " : ".join([p for p in [agent, thread_id] if p]) or ""
    subtitle = title or f"Time (seconds) by component (total = {total:.3f}s)"

    # --- explicit layout: one Axes, exact placement ---
    fig_h = max(2.2, 0.35 * max(1, len(values)))
    fig = plt.figure(figsize=(8, fig_h))
    ax = fig.add_axes(list(_LAYOUT["ax_rect"]))

    # plot
    y = range(len(values))
    ax.hlines(y, xmin=0, xmax=values, linewidth=1)
    ax.plot(values, y, "o")

    if log_x:
        vmin = min(values) if values else 0.0
        if vmin <= 0:
            vmin = min([v for v in values if v > 0] + [1e-6])
        ax.set_xscale("log")
        ax.set_xlim(
            left=vmin * 0.8, right=(max(values) * 1.1 if values else 1.0)
        )

    # axes cosmetics
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Seconds (log scale)" if log_x else "Seconds", fontsize="small"
    )
    ax.tick_params(labelsize="small")

    # header/subtitle (figure text so they don't push the axes)
    if header:
        fig.text(0.5, _LAYOUT["header_y"], header, ha="center", va="top")
        fig.text(
            0.5,
            _LAYOUT["subtitle_y"],
            subtitle,
            ha="center",
            va="top",
            fontsize="small",
        )
    else:
        fig.text(
            0.5, (_LAYOUT["header_y"] - 0.02), subtitle, ha="center", va="top"
        )

    # annotate dots
    for yi, val in zip(y, values):
        pct = (val / total * 100.0) if total > 0 else 0.0
        if pct >= min_label_pct:
            bits = []
            if show_percent:
                bits.append(f"{pct:.2f}%")
            if show_seconds:
                bits.append(f"{val:.3f}s")
            ax.text(
                val,
                yi,
                "  " + " ".join(bits),
                va="center",
                ha="left",
                fontsize="small",
            )

    # compact footers
    if run_id:
        fig.text(
            0.5,
            _LAYOUT["footer1_y"],
            f"run_id: {run_id}",
            ha="center",
            va="center",
            fontsize="x-small",
        )
    if started or ended:
        fig.text(
            0.5,
            _LAYOUT["footer2_y"],
            f"{started} \u2192 {ended}",
            ha="center",
            va="center",
            fontsize="x-small",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def extract_llm_token_stats(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    """
    Return (totals, samples) for LLM token usage from Telemetry payload.
    - totals: sum across all LLM calls
    - samples: list per call for KDE (one list per category)
    Categories: input_tokens, output_tokens, reasoning_tokens, cached_tokens, total_tokens
    """
    events = payload.get("llm_events") or []
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    samples = {k: [] for k in totals.keys()}

    for ev in events:
        m = (ev.get("metrics") or {}).get("usage_rollup") or {}

        # Normalize with safe int coercion
        def _gi(key: str) -> int:
            try:
                v = m.get(key)
                if v is None:
                    return 0
                return int(float(v))
            except Exception:
                return 0

        # Prefer explicit input/output; fall back to prompt/completion mirrors
        it = _gi("input_tokens") or _gi("prompt_tokens")
        ot = _gi("output_tokens") or _gi("completion_tokens")
        rt = _gi("reasoning_tokens")
        ct = _gi("cached_tokens")
        tt = _gi("total_tokens")
        # Ensure total is at least input+output (providers sometimes omit)
        tt = max(tt, it + ot)

        # Update
        totals["input_tokens"] += it
        totals["output_tokens"] += ot
        totals["reasoning_tokens"] += rt
        totals["cached_tokens"] += ct
        totals["total_tokens"] += tt

        samples["input_tokens"].append(it)
        samples["output_tokens"].append(ot)
        if rt:
            samples["reasoning_tokens"].append(rt)
        if ct:
            samples["cached_tokens"].append(ct)
        samples["total_tokens"].append(tt)

    return totals, samples


def plot_token_totals_bar(
    totals: Dict[str, int],
    out_path: str,
    *,
    title: str = "",
    context: Dict[str, Any] | None = None,
) -> str:
    """
    Horizontal bar chart of token totals by category.
    """

    order = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    ]
    labels = [lbl.replace("_", " ") for lbl in order]
    values = [int(totals.get(k, 0)) for k in order]

    ctx = context or {}
    agent = str(ctx.get("agent") or "")
    thread = str(ctx.get("thread_id") or "")
    run_id = str(ctx.get("run_id") or "")
    started = _fmt_iso_pretty(str(ctx.get("started_at") or ""))
    ended = _fmt_iso_pretty(str(ctx.get("ended_at") or ""))

    header = " : ".join([p for p in [agent, thread] if p]) or ""
    subtitle = title or "LLM Token Totals by Category"

    fig_h = max(2.2, 0.35 * len(labels))
    fig = plt.figure(figsize=(8, fig_h))
    ax = fig.add_axes(list(_LAYOUT["ax_rect"]))

    y = list(range(len(labels)))
    ax.barh(y, values, edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # put input_tokens on top
    ax.set_xlabel("Tokens", fontsize="small")
    ax.tick_params(labelsize="small")

    # annotate counts on bars
    for yi, val in zip(y, values):
        ax.text(val, yi, f"  {val:,}", va="center", ha="left", fontsize="small")

    if header:
        fig.text(0.5, _LAYOUT["header_y"], header, ha="center", va="top")
        fig.text(
            0.5,
            _LAYOUT["subtitle_y"],
            subtitle,
            ha="center",
            va="top",
            fontsize="small",
        )
    else:
        fig.text(
            0.5, (_LAYOUT["header_y"] - 0.02), subtitle, ha="center", va="top"
        )

    if run_id:
        fig.text(
            0.5,
            _LAYOUT["footer1_y"],
            f"run_id: {run_id}",
            ha="center",
            va="center",
            fontsize="x-small",
        )
    if started or ended:
        fig.text(
            0.5,
            _LAYOUT["footer2_y"],
            f"{started} \u2192 {ended}",
            ha="center",
            va="center",
            fontsize="x-small",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def plot_token_kde(
    samples: Dict[str, List[int]],
    out_path: str,
    *,
    title: str = "",
    context: Dict[str, Any] | None = None,
    log_x: bool = False,
    bandwidth: float | None = None,
    fill_alpha: float = 0.15,
    line_alpha: float = 0.85,
) -> str:
    """
    Overlay KDEs for input/output/reasoning/cached/total tokens.
    - Uses scipy.stats.gaussian_kde if available; falls back to a Gaussian-smoothed histogram.
    - No seaborn.
    """

    # categories & pretty labels (skip empty series automatically)
    order = [
        ("input_tokens", "input tokens"),
        ("output_tokens", "output tokens"),
        ("reasoning_tokens", "reasoning tokens"),
        ("cached_tokens", "cached tokens"),
        ("total_tokens", "total tokens"),
    ]

    # Gather non-empty arrays
    series = []
    for key, label in order:
        arr = np.asarray(samples.get(key, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:  # need at least 2 for KDE
            series.append((key, label, arr))

    if not series:
        # Nothing to plot; create an empty figure with a note
        fig = plt.figure(figsize=(8, 2.0))
        fig.text(0.5, 0.5, "No token samples to plot", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out_path

    # Context
    ctx = context or {}
    agent = str(ctx.get("agent") or "")
    thread = str(ctx.get("thread_id") or "")
    run_id = str(ctx.get("run_id") or "")
    started = _fmt_iso_pretty(str(ctx.get("started_at") or ""))
    ended = _fmt_iso_pretty(str(ctx.get("ended_at") or ""))

    header = " : ".join([p for p in [agent, thread] if p]) or ""
    subtitle = title or "LLM Token Usage — KDE"

    # Build x-grid across all series
    all_max = max(float(np.max(a)) for _, _, a in series)
    x_min = 0.0
    x_max = max(1.0, all_max * 1.05)
    x = np.linspace(x_min, x_max, 600)

    # Try scipy KDE; else fallback to simple Gaussian smoothing of hist density
    def _kde(arr: np.ndarray) -> np.ndarray:
        kde = gaussian_kde(arr, bw_method=bandwidth)
        return kde.evaluate(x)

    # Plot
    fig = plt.figure(figsize=(8, 2.8))
    ax = fig.add_axes(list(_LAYOUT["ax_rect"]))

    for _, label, arr in series:
        y = _kde(arr)
        ax.plot(x, y, alpha=line_alpha, label=label)
        ax.fill_between(x, 0, y, alpha=fill_alpha)

    if log_x:
        # Avoid log(0)
        ax.set_xscale("log")
        ax.set_xlim(left=max(1e-6, x_min + 1e-6), right=x_max)

    ax.set_xlabel(
        "Tokens" + (" (log scale)" if log_x else ""), fontsize="small"
    )
    ax.set_ylabel("Density", fontsize="small")
    ax.tick_params(labelsize="small")
    ax.legend(loc="upper right", fontsize="x-small", frameon=False)

    if header:
        fig.text(0.5, _LAYOUT["header_y"], header, ha="center", va="top")
        fig.text(
            0.5,
            _LAYOUT["subtitle_y"],
            subtitle,
            ha="center",
            va="top",
            fontsize="small",
        )
    else:
        fig.text(
            0.5, (_LAYOUT["header_y"] - 0.02), subtitle, ha="center", va="top"
        )

    if run_id:
        fig.text(
            0.5,
            _LAYOUT["footer1_y"],
            f"run_id: {run_id}",
            ha="center",
            va="center",
            fontsize="x-small",
        )
    if started or ended:
        fig.text(
            0.5,
            _LAYOUT["footer2_y"],
            f"{started} \u2192 {ended}",
            ha="center",
            va="center",
            fontsize="x-small",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def plot_token_rates_bar(
    totals: dict[str, int],
    llm_seconds: float,
    window_seconds: float,
    out_path: str,
    *,
    title: str = "Tokens per second (two baselines)",
    context: dict | None = None,
    categories: list[str] | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    cats = categories or [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    ]
    labels = [c.replace("_", " ") for c in cats]
    vals = [int(totals.get(c, 0) or 0) for c in cats]

    def _rate(x: int, denom: float) -> float:
        return (float(x) / denom) if denom and denom > 0 else 0.0

    rates_llm = [_rate(x, llm_seconds) for x in vals]
    rates_win = [_rate(x, window_seconds) for x in vals]

    ctx = context or {}
    agent = str(ctx.get("agent") or "")
    thread = str(ctx.get("thread_id") or "")
    run_id = str(ctx.get("run_id") or "")
    # shorten very long run_ids in the footer to keep things readable
    run_id_short = f"{run_id[:8]}" if run_id else ""
    started = _fmt_iso_pretty(str(ctx.get("started_at") or ""))
    ended = _fmt_iso_pretty(str(ctx.get("ended_at") or ""))

    header = " : ".join([p for p in [agent, thread] if p]) or ""

    # ---------------- layout: reserve dynamic space for 2–3 footer lines -------
    show_warn = bool(window_seconds > 0 and llm_seconds > window_seconds)
    footer_lines = [
        f"LLM-active (sum): {llm_seconds:.3f}s • window: {window_seconds:.3f}s"
    ]
    if show_warn:
        pf = llm_seconds / window_seconds
        footer_lines.append(
            f"Note: LLM sum exceeds window → parallel LLM work (~{pf:.2f}× overlap)."
        )
    if run_id_short or started or ended:
        right = "  ".join(
            s
            for s in [
                f"run_id: {run_id_short}" if run_id_short else "",
                f"{started} \u2192 {ended}" if (started or ended) else "",
            ]
            if s
        )
        footer_lines.append(right)

    n_footer = len(footer_lines)
    base_h = 2.8
    h = base_h + 0.35 * max(0, n_footer - 2)

    # widen the figure a bit to make room for an outside legend
    fig = plt.figure(figsize=(10.0, h))

    # Leave a right margin (legend will sit outside on the right)
    ax_left, ax_width = 0.07, 0.75  # was 0.08, 0.84
    ax_bottom = 0.24 + 0.05 * max(0, n_footer - 2)
    ax_height = 0.58
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    x = np.arange(len(labels))
    width = 0.38

    bars1 = ax.bar(
        x - width / 2,
        rates_llm,
        width,
        label="per LLM-sec (sum)",
        color="tab:purple",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        rates_win,
        width,
        label="per thread-sec",
        color="tab:gray",
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("tokens / second", fontsize="small")
    ax.tick_params(labelsize="small")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),  # just outside the axes, top-right corner
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
    )

    # annotate bars (top center)
    def _annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize="x-small",
            )

    _annotate(bars1)
    _annotate(bars2)

    # header + subtitle
    header_y = 0.985
    subtitle_y = 0.955
    if header:
        fig.text(0.5, header_y, header, ha="center", va="top")
        fig.text(
            0.5, subtitle_y, title, ha="center", va="top", fontsize="small"
        )
    else:
        fig.text(0.5, (header_y - 0.015), title, ha="center", va="top")

    # footer stack (unchanged)
    y0, step = 0.09, 0.035
    for i, line in enumerate(footer_lines):
        fig.text(
            0.5,
            y0 + i * step,
            line,
            ha="center",
            va="center",
            fontsize="x-small",
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_path


def plot_tokens_bar_by_model(
    totals_by_model: Dict[str, Dict[str, int]],
    out_path: str,
    *,
    title: str = "LLM Token Totals by Category — by model",
    context: dict | None = None,
) -> str:
    import matplotlib.pyplot as plt

    cats = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    ]
    labels = [c.replace("_", " ") for c in cats]
    models = sorted(totals_by_model.keys())
    if not models:
        raise ValueError("No models to plot.")

    rows = len(models)
    fig_h = 2.2 + 2.2 * rows
    fig, axes = plt.subplots(rows, 1, figsize=(12, fig_h), sharex=False)
    if rows == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        vals = [
            int(totals_by_model.get(model, {}).get(k, 0) or 0) for k in cats
        ]
        bars = ax.barh(labels, vals, edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_width(),
                b.get_y() + b.get_height() / 2,
                f" {v:,}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_title(model, loc="left", fontsize=11)
        ax.set_xlabel("Tokens")

    # Header/subtitle/footer
    if context:
        header = (
            f"{context.get('agent', '')}: {context.get('thread_id', '')}".strip(
                " :"
            )
        )
        if header:
            fig.suptitle(header, fontsize=14, y=0.98)
    if title:
        fig.text(0.5, 0.94, title, ha="center", fontsize=12)
    if context:
        # Hide global start→end for SUPER aggregates (not meaningful across overlapping threads)
        if (context.get("agent") or "").upper() != "SUPER":
            s, e = context.get("started_at"), context.get("ended_at")
            if s and e:
                fig.text(0.5, 0.02, f"{s} → {e}", ha="center", fontsize=9)

    fig.tight_layout(rect=(0.02, 0.06, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_token_rates_by_model(
    totals_by_model: Dict[str, Dict[str, int]],
    llm_seconds_by_model: Dict[str, float],
    window_seconds: float,
    out_path: str,
    *,
    title: str = "Tokens per second — by model",
    context: dict | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    cats = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
    ]
    xlabels = [c.replace("_", " ") for c in cats]
    models = sorted(totals_by_model.keys())
    if not models:
        raise ValueError("No models to plot.")

    rows = len(models)
    fig_h = 2.4 + 2.4 * rows
    fig, axes = plt.subplots(rows, 1, figsize=(12, fig_h), sharex=True)
    if rows == 1:
        axes = [axes]

    x = np.arange(len(cats))
    w = 0.38
    denom_thread = max(1e-9, float(window_seconds or 0.0))

    for i, model in enumerate(models):
        ax = axes[i]
        totals = [
            int(totals_by_model.get(model, {}).get(k, 0) or 0) for k in cats
        ]
        denom_llm = max(
            1e-9, float(llm_seconds_by_model.get(model, 0.0) or 0.0)
        )

        per_llm = [v / denom_llm for v in totals]
        per_thread = [v / denom_thread for v in totals]

        b1 = ax.bar(
            x - w / 2,
            per_llm,
            width=w,
            label="per LLM-sec (sum)",
            edgecolor="black",
        )
        b2 = ax.bar(
            x + w / 2,
            per_thread,
            width=w,
            label="per thread-sec",
            edgecolor="black",
        )

        for bx in (b1, b2):
            for rect in bx:
                h = rect.get_height()
                if h > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        h,
                        f"{h:,.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x, xlabels)
        ax.set_ylabel("tokens / second")
        ax.set_title(model, loc="left", fontsize=11)
        if i == 0:
            ax.legend(loc="upper right")

    # Header/subtitle/footer
    if context:
        header = (
            f"{context.get('agent', '')}: {context.get('thread_id', '')}".strip(
                " :"
            )
        )
        if header:
            fig.suptitle(header, fontsize=14, y=0.98)
    if title:
        fig.text(0.5, 0.94, title, ha="center", fontsize=12)

    if context:
        is_super = str(context.get("agent") or "") == "SUPER"

        s, e = context.get("started_at"), context.get("ended_at")
        llm_sum = sum(
            float(llm_seconds_by_model.get(m, 0.0) or 0.0) for m in models
        )

        # Only show overlap note when we are actually showing a single window denominator
        overlap_note = ""
        if (
            (not is_super)
            and window_seconds > 0
            and llm_sum > window_seconds * 1.05
        ):
            overlap_note = f"  Note: LLM sum exceeds window → parallel LLM work (~{llm_sum / window_seconds:.2f}× overlap)."

        # Hide the global start→end line for SUPER aggregates
        if (not is_super) and s and e:
            fig.text(0.5, 0.04, f"{s} → {e}", ha="center", fontsize=9)

        # For SUPER, drop the “• window: …” part entirely
        if is_super:
            footer = f"LLM-active (sum across models): {llm_sum:.3f}s"
        else:
            footer = f"LLM-active (sum across models): {llm_sum:.3f}s • window: {window_seconds:.3f}s{overlap_note}"

        fig.text(0.5, 0.02, footer, ha="center", fontsize=9)

    fig.tight_layout(rect=(0.02, 0.08, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tokens_by_agent_stacked(
    totals_by_agent: Dict[str, Dict[str, int]],
    out_path: str,
    *,
    title: str = "LLM Token Totals by Agent (thread)",
    footer_lines: List[str] | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    cats = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ]
    pretty = {
        "input_tokens": "input tokens",
        "output_tokens": "output tokens",
        "reasoning_tokens": "reasoning tokens",
        "cached_tokens": "cached tokens",
    }

    if not totals_by_agent:
        fig = plt.figure(figsize=(10, 2.0))
        fig.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    agents = sorted(
        totals_by_agent.keys(),
        key=lambda a: sum(int(totals_by_agent[a].get(k, 0) or 0) for k in cats),
        reverse=True,
    )
    x = np.arange(len(agents))
    width = 0.65

    # Build stacked series
    series = []
    for k in cats:
        series.append([
            int(totals_by_agent.get(a, {}).get(k, 0) or 0) for a in agents
        ])

    totals_per_agent = [sum(vals) for vals in zip(*series)]
    max_total = max(totals_per_agent) if totals_per_agent else 0

    fig_h = 3.0 + 0.18 * max(0, len(agents) - 6)
    fig = plt.figure(figsize=(max(10, 1.2 * len(agents)), fig_h))
    ax = fig.add_axes([0.08, 0.28, 0.78, 0.60])  # leave right margin for legend

    bottoms = np.zeros(len(agents), dtype=float)
    bars_all = []
    for k, vals in zip(cats, series):
        b = ax.bar(
            x,
            vals,
            width=width,
            bottom=bottoms,
            edgecolor="black",
            label=pretty[k],
        )
        bars_all.append(b)
        bottoms = bottoms + np.array(vals, dtype=float)

    # Axis & labels
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=28, ha="right")
    ax.set_ylabel("Tokens", fontsize="small")
    ax.tick_params(labelsize="small")
    ax.set_ylim(0, max_total * 1.08 if max_total > 0 else 1.0)

    # Annotate totals on top of each stack (only if > 0)
    for xi, tot in enumerate(totals_per_agent):
        if tot > 0:
            ax.text(
                xi,
                tot,
                f"{tot:,}",
                ha="center",
                va="bottom",
                fontsize="x-small",
            )

    # Legend outside
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        fontsize="small",
    )

    # Title / footer
    fig.suptitle(title, y=0.97)
    y0 = 0.08
    if footer_lines:
        for i, line in enumerate(footer_lines):
            fig.text(
                0.5,
                y0 + i * 0.035,
                line,
                ha="center",
                va="center",
                fontsize="x-small",
            )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tps_by_agent_grouped(
    totals_by_agent: Dict[str, Dict[str, int]],
    llm_secs_by_agent: Dict[str, float],
    thread_window_seconds: float,
    out_path: str,
    *,
    title: str = "Tokens per second by Agent (thread)",
    footer_lines: List[str] | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    cats = [
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
    ]

    if not totals_by_agent:
        fig = plt.figure(figsize=(10, 2.0))
        fig.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Order agents by total tokens desc
    agents = sorted(
        totals_by_agent.keys(),
        key=lambda a: sum(int(totals_by_agent[a].get(k, 0) or 0) for k in cats),
        reverse=True,
    )

    def _rate_sum(agent: str, denom: float) -> List[float]:
        denom = float(denom or 0.0)
        vals = [
            int(totals_by_agent.get(agent, {}).get(k, 0) or 0) for k in cats
        ]
        if denom <= 0:
            return [0.0 for _ in vals]
        return [v / denom for v in vals]

    per_llm = [_rate_sum(a, llm_secs_by_agent.get(a, 0.0)) for a in agents]
    per_thr = [_rate_sum(a, thread_window_seconds) for a in agents]

    # Prepare bars
    n_agents = len(agents)
    x = np.arange(n_agents)
    width = 0.38

    fig_h = 3.0 + 0.18 * max(0, n_agents - 6)
    fig = plt.figure(figsize=(max(12, 1.3 * n_agents), fig_h))
    ax = fig.add_axes([0.08, 0.28, 0.78, 0.60])

    # For grouped by category, build sums per agent for each baseline
    # We’ll draw two bars per agent per category by offsetting x positions.
    # But for readability, we instead draw two *stacks* per agent: one for LLM-sec and one for thread-sec.
    # Build totals per agent for each baseline across categories.
    # Here we keep the same layout as your other TPS chart: two bars per category,
    # but along the agent axis it’s clearer to sum categories, so we’ll show TOTAL only.
    # If you want per-category grouped bars per agent, ping me and I’ll switch layout.

    # -> Simpler: show TOTAL tokens/sec per agent (two baselines).
    per_llm_total = [sum(vals) for vals in per_llm]
    per_thr_total = [sum(vals) for vals in per_thr]

    b1 = ax.bar(
        x - width / 2,
        per_llm_total,
        width=width,
        label="per LLM-sec (sum)",
        edgecolor="black",
    )
    b2 = ax.bar(
        x + width / 2,
        per_thr_total,
        width=width,
        label="per thread-sec",
        edgecolor="black",
    )

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h,
                    f"{h:,.2f}",
                    ha="center",
                    va="bottom",
                    fontsize="x-small",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=28, ha="right")
    ax.set_ylabel("tokens / second", fontsize="small")
    ax.tick_params(labelsize="small")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        fontsize="small",
    )

    fig.suptitle(title, y=0.97)
    y0 = 0.08
    if footer_lines:
        for i, line in enumerate(footer_lines):
            fig.text(
                0.5,
                y0 + i * 0.035,
                line,
                ha="center",
                va="center",
                fontsize="x-small",
            )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
