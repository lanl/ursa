# ursa/observability/timing.py
from __future__ import annotations

import collections
import datetime
import importlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Tuple

from langchain_core.callbacks import BaseCallbackHandler
from rich import get_console
from rich.box import HEAVY
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

NAME_W, COUNT_W, TOTAL_W, AVG_W, MAX_W = 30, 7, 12, 12, 12
COL_PAD = (0, 1)  # top/bottom, left/right padding in the Rich table cells


def _get_pricing_module():
    candidates = (
        "ursa.observability.pricing",
        "ursa.observability.llm_pricing",
        "pricing",
        "llm_pricing",
    )
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "load_registry") and hasattr(mod, "price_payload"):
                return mod
        except Exception:
            continue
    return None


def _to_snake(s: str) -> str:
    s = str(s)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)  # CamelCase -> snake_case
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


_SESSIONS: dict[str, "SessionRollup"] = {}


@dataclass
class _Bucket:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, count: int, total_s: float, max_ms: float):
        self.count += int(count or 0)
        self.total_ms += float(total_s or 0.0) * 1000.0
        self.max_ms = max(self.max_ms, float(max_ms or 0.0))

    def as_row(self, name: str):
        avg_ms = (self.total_ms / self.count) if self.count else 0.0
        return (name, self.count, self.total_ms / 1000.0, avg_ms, self.max_ms)


@dataclass
class SessionRollup:
    thread_id: str
    runs: int = 0
    agents: set = field(default_factory=set)

    # times/costs
    wall_sum_s: float = 0.0  # sum of each run's wall time
    llm_total_s: float = 0.0  # derived from llm buckets
    tool_total_s: float = 0.0  # derived from tool buckets
    cost_total_usd: float = 0.0

    # breakdowns
    runnable_by_name: dict[str, _Bucket] = field(
        default_factory=lambda: defaultdict(_Bucket)
    )
    tool_by_name: dict[str, _Bucket] = field(
        default_factory=lambda: defaultdict(_Bucket)
    )
    llm_by_name: dict[str, _Bucket] = field(
        default_factory=lambda: defaultdict(_Bucket)
    )
    cost_by_model_usd: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    # temporal bounds
    started_at: str | None = None
    ended_at: str | None = None

    def ingest(self, payload: dict) -> None:
        from datetime import datetime

        def p(ts):
            try:
                return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
            except Exception:
                return None

        ctx = payload.get("context") or {}
        agent = ctx.get("agent") or "agent"
        s_iso, e_iso = ctx.get("started_at"), ctx.get("ended_at")
        s_dt, e_dt = p(s_iso), p(e_iso)

        self.runs += 1
        self.agents.add(agent)

        # wall time: sum of each run; keep overall min/max for elapsed
        if s_dt and e_dt:
            self.wall_sum_s += max(0.0, (e_dt - s_dt).total_seconds())
            if not self.started_at or (
                p(self.started_at) and s_dt < p(self.started_at)
            ):
                self.started_at = s_iso
            if not self.ended_at or (
                p(self.ended_at) and e_dt > p(self.ended_at)
            ):
                self.ended_at = e_iso

        # aggregate tables
        tables = payload.get("tables") or {}
        for row in tables.get("runnable") or []:
            self.runnable_by_name[row["name"]].add(
                row["count"], row["total_s"], row["max_ms"]
            )
        for row in tables.get("tool") or []:
            self.tool_by_name[row["name"]].add(
                row["count"], row["total_s"], row["max_ms"]
            )
        for row in tables.get("llm") or []:
            self.llm_by_name[row["name"]].add(
                row["count"], row["total_s"], row["max_ms"]
            )

        # recompute llm/tool totals from buckets
        self.llm_total_s = (
            sum(b.total_ms for b in self.llm_by_name.values()) / 1000.0
        )
        self.tool_total_s = (
            sum(b.total_ms for b in self.tool_by_name.values()) / 1000.0
        )

        # costs (if priced)
        costs = payload.get("costs") or {}
        self.cost_total_usd += float(costs.get("total_usd") or 0.0)
        for model, amt in (costs.get("by_model_usd") or {}).items():
            try:
                self.cost_by_model_usd[model] += float(amt)
            except Exception:
                pass


def _session_ingest(payload: dict) -> None:
    tid = (payload.get("context") or {}).get("thread_id")
    if not tid:
        return
    _SESSIONS.setdefault(tid, SessionRollup(thread_id=tid)).ingest(payload)


def _rows_from_bucket_map(
    d: dict[str, _Bucket],
) -> list[tuple[str, int, float, float, float]]:
    rows = [b.as_row(name) for name, b in d.items()]
    rows.sort(key=lambda r: r[2], reverse=True)  # sort by total(s)
    return rows


def render_session_summary(thread_id: str):
    roll = _SESSIONS.get(thread_id)
    console = get_console()
    if not roll:
        msg = f"No session data for thread_id '{thread_id}'."
        console.print(
            Panel(msg, title="[bold]Session Summary[/]", border_style="red")
        )
        return msg

    # header
    header_lines = []
    agents_list = ", ".join(sorted(roll.agents)) or "—"
    header_lines.append(
        f"[bold magenta]Session[/] • thread [bold]{thread_id}[/] [dim]• runs {roll.runs} • agents {len(roll.agents)}[/]"
    )
    # both elapsed window and sum of runs
    elapsed = None
    if roll.started_at and roll.ended_at:
        header_lines.append(f"[dim]{roll.started_at} → {roll.ended_at}[/dim]")
        # display elapsed in panel footer text (computing here)
        try:
            from datetime import datetime

            s, e = (
                datetime.fromisoformat(roll.started_at.replace("Z", "+00:00")),
                datetime.fromisoformat(roll.ended_at.replace("Z", "+00:00")),
            )
            elapsed = max(0.0, (e - s).total_seconds())
        except Exception:
            elapsed = None
    if elapsed is not None:
        header_lines[-1] += (
            f"   [bold]wall (elapsed)[/]: {elapsed:,.2f}s   [bold]wall (sum)[/]: {roll.wall_sum_s:,.2f}s"
        )
    else:
        header_lines.append(f"[bold]wall (sum)[/]: {roll.wall_sum_s:,.2f}s")

    # combined tables (aligned widths)
    t_nodes = _mk_table(
        "Per-Node / Runnable Timing (session)",
        _rows_from_bucket_map(roll.runnable_by_name),
    )
    t_tools = _mk_table(
        "Per-Tool Timing (session)", _rows_from_bucket_map(roll.tool_by_name)
    )
    t_llms = _mk_table(
        "Per-LLM Timing (session)", _rows_from_bucket_map(roll.llm_by_name)
    )

    # cost-by-model table (aligned with a smaller schema)
    from rich.table import Table

    t_cost = Table(
        title="Cost by Model (USD)",
        title_style="bold white",
        box=HEAVY,
        expand=False,
        pad_edge=False,
        header_style="bold",
        padding=COL_PAD,
    )
    t_cost.add_column(
        "Model",
        style="cyan",
        no_wrap=True,
        width=NAME_W,
        min_width=NAME_W,
        max_width=NAME_W,
    )
    t_cost.add_column(
        "Cost",
        justify="right",
        width=TOTAL_W,
        min_width=TOTAL_W,
        max_width=TOTAL_W,
    )
    if roll.cost_by_model_usd:
        for model, amt in sorted(
            roll.cost_by_model_usd.items(), key=lambda kv: kv[1], reverse=True
        ):
            t_cost.add_row(model, f"${amt:,.6f}")
    else:
        t_cost.add_row("—", "$0.000000")

    # attribution block
    attrib = [
        "[bold]Session Totals[/]",
        f"  LLM total:   {roll.llm_total_s:,.2f}s",
        f"  Tool total:  {roll.tool_total_s:,.2f}s",
        (f"  Wall (elapsed): {elapsed:,.2f}s" if elapsed is not None else None),
        f"  Wall (sum):  {roll.wall_sum_s:,.2f}s",
        f"[bold]Cost total:[/] [bold green]${roll.cost_total_usd:,.6f}[/]",
        f"[dim]Agents:[/] {agents_list}",
    ]
    attrib = [a for a in attrib if a is not None]

    renderables = [
        Text.from_markup("\n".join(header_lines)),
        Rule(),
        t_nodes,
        t_tools,
        t_llms,
        Rule(),
        t_cost,
        Rule(),
        Text.from_markup("\n".join(attrib)),
    ]
    panel = Panel.fit(
        Group(*renderables),
        title=f"[bold white]Session Summary[/] • [cyan]{thread_id}[/]",
        border_style="bright_magenta",
        padding=(1, 2),
        box=HEAVY,
    )
    console.print(panel)


# ---------------------------
#               Aggregators
# ---------------------------


@dataclass
class _Agg:
    # list of (name, elapsed_ms, ok)
    records: List[Tuple[str, float, bool]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, name: str, elapsed_ms: float, ok: bool) -> None:
        with self._lock:
            self.records.append((name, elapsed_ms, ok))

    def buckets(self) -> List[Tuple[str, int, float, float, float]]:
        # -> [(name, count, total_secs, avg_ms, max_ms)]
        by_name: Dict[str, List[float]] = defaultdict(list)
        with self._lock:
            for name, ms, _ok in self.records:
                by_name[name].append(ms)
        rows = []
        for name, times in by_name.items():
            total_ms = sum(times)
            rows.append((
                name,
                len(times),
                total_ms / 1000.0,
                total_ms / len(times),
                max(times),
            ))
        rows.sort(key=lambda r: r[2], reverse=True)  # by total seconds
        return rows


# ---------------------------------
#         Callback Handlers
# ---------------------------------


class PerToolTimer(BaseCallbackHandler):
    """Times each tool call via callbacks; robust to decorator order."""

    def __init__(self, agg: _Agg | None = None):
        self.agg = agg or _Agg()
        self._starts: Dict[Any, Tuple[str, float]] = {}

    def _name(self, serialized) -> str:
        # serialized can be None, dict, str, or contain nested ids
        if isinstance(serialized, dict):
            name = serialized.get("name")
            if not name:
                sid = serialized.get("id")
                if isinstance(sid, dict):
                    name = sid.get("name") or sid.get("id")
                elif isinstance(sid, (list, tuple)):
                    name = "/".join(map(str, sid))
                elif sid is not None:
                    name = str(sid)
            return name or "unknown_tool"
        if isinstance(serialized, str):
            return serialized
        return "unknown_tool"

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = self._name(serialized)
        self._starts[run_id] = (name, time.perf_counter())

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, t0 = self._starts.pop(
            run_id, ("unknown_tool", time.perf_counter())
        )
        self.agg.add(name, (time.perf_counter() - t0) * 1000.0, True)

    def on_tool_error(self, error, *, run_id, **kwargs):
        name, t0 = self._starts.pop(
            run_id, ("unknown_tool", time.perf_counter())
        )
        self.agg.add(name, (time.perf_counter() - t0) * 1000.0, False)


class PerRunnableTimer(BaseCallbackHandler):
    """
    Times LangChain/LangGraph runnables (chains, graphs, nodes). You’ll usually
    see node names in `serialized.get('name')` or `serialized.get('id')`.
    """

    def __init__(self, agg: _Agg | None = None):
        self.agg = agg or _Agg()
        self._starts: Dict[Any, Tuple[str, float]] = {}

    def _name(self, serialized) -> str:
        # serialized can be None, dict, str, or contain nested ids
        if isinstance(serialized, dict):
            name = serialized.get("name")
            if not name:
                sid = serialized.get("id")
                if isinstance(sid, dict):
                    name = sid.get("name") or sid.get("id")
                elif isinstance(sid, (list, tuple)):
                    name = "/".join(map(str, sid))
                elif sid is not None:
                    name = str(sid)
            return name or "runnable"
        if isinstance(serialized, str):
            return serialized
        return "runnable"

    # Chains/graphs/nodes map onto these events:
    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        base_name = self._name(serialized)

        # Root span (keep)
        if parent_run_id is None:
            name = base_name
            if name == "runnable" and tags:
                name = tags[-1]  # e.g., "graph"
            name = f"graph:{name}"
            self._starts[run_id] = (name, time.perf_counter())
            return

        # ---- Child span (graph node) ----
        md = metadata if isinstance(metadata, dict) else {}

        # Only keep spans that our wrapper marked with a namespace.
        # This filters out internal 'graph:step:N:<node>' duplicates.
        ns = md.get("ursa_ns")
        if not ns:
            return  # ignore un-namespaced child spans

        # node base name (prefer explicit metadata)
        node_base = (
            md.get("langgraph_node")
            or md.get("node_name")
            or md.get("langgraph:node")
            or base_name
        )

        # canonicalize "graph:step:N:<node>" → "<node>"
        if isinstance(node_base, str) and node_base.startswith("graph:step:"):
            # split on last colon so "graph:step:N:<node>" → "<node>"
            parts = node_base.split(":", 3)
            if len(parts) == 4:
                node_base = parts[3]

        # namespace + snake casing for safety
        def _to_snake(s: str) -> str:
            import re

            s = str(s)
            s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)
            s = s.replace("-", "_").replace(" ", "_")
            return s.lower()

        ns = _to_snake(ns)
        qualified = f"{ns}:{node_base}"
        name = f"node:{qualified}"

        self._starts[run_id] = (name, time.perf_counter())

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        name, t0 = self._starts.pop(run_id, ("runnable", time.perf_counter()))
        self.agg.add(name, (time.perf_counter() - t0) * 1000.0, True)

    def on_chain_error(self, error, *, run_id, **kwargs):
        name, t0 = self._starts.pop(run_id, ("runnable", time.perf_counter()))
        self.agg.add(name, (time.perf_counter() - t0) * 1000.0, False)


def _to_int(x, default=0):
    try:
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            # handles "340" or "340.0"
            return int(float(x))
    except Exception:
        pass
    return default


def _acc_from(d: dict, roll: dict):
    # Map whatever keys exist into our canonical fields
    it = _to_int(d.get("input_tokens", d.get("prompt_tokens")))
    ot = _to_int(d.get("output_tokens", d.get("completion_tokens")))
    tt = _to_int(d.get("total_tokens", it + ot))

    roll["input_tokens"] += it
    roll["output_tokens"] += ot
    roll["total_tokens"] += tt

    # Keep prompt/completion mirrors too
    roll["prompt_tokens"] += _to_int(d.get("prompt_tokens", it))
    roll["completion_tokens"] += _to_int(d.get("completion_tokens", ot))

    # extras / synonyms
    # reasoning
    roll["reasoning_tokens"] += _to_int(
        d.get("reasoning_tokens")
        or (d.get("completion_tokens_details") or {}).get("reasoning_tokens")
    )
    # cached
    cached = (
        d.get("cached_tokens")
        or d.get("cached_input_tokens")
        or (d.get("prompt_tokens_details") or {}).get("cached_tokens")
        or d.get("prompt_cache_hits")
    )
    roll["cached_tokens"] += _to_int(cached)

    # costs if exposed (keep as floats)
    for k in ("input_cost", "output_cost", "total_cost"):
        v = d.get(k)
        if v is not None:
            try:
                roll[k] += float(v)
            except Exception:
                pass


def _maybe_add_extras(d: dict, roll: dict):
    if not isinstance(d, dict):
        return
    # reasoning
    rt = d.get("reasoning_tokens") or (
        d.get("completion_tokens_details") or {}
    ).get("reasoning_tokens")
    roll["reasoning_tokens"] += _to_int(rt)
    # cached
    cached = (
        d.get("cached_tokens")
        or d.get("cached_input_tokens")
        or (d.get("prompt_tokens_details") or {}).get("cached_tokens")
        or d.get("prompt_cache_hits")
    )
    roll["cached_tokens"] += _to_int(cached)


class PerLLMTimer(BaseCallbackHandler):
    """Times LLM calls (chat/completions) and captures usage/metrics."""

    def __init__(self, agg: _Agg | None = None, keep_max: int = 1000):
        self.agg = agg or _Agg()
        self._starts: Dict[Any, Tuple[str, float, list, dict]] = {}
        self.samples: collections.deque = collections.deque(maxlen=keep_max)

    def _name(self, serialized, metadata, tags) -> str:
        model = (metadata or {}).get("model")
        if model:
            return f"llm:{model}"
        if isinstance(serialized, dict):
            name = serialized.get("name")
            if not name:
                sid = serialized.get("id")
                if isinstance(sid, dict):
                    name = sid.get("name") or sid.get("id")
                elif isinstance(sid, (list, tuple)):
                    name = "/".join(map(str, sid))
                elif sid is not None:
                    name = str(sid)
            return f"llm:{name or 'unknown'}"
        if isinstance(serialized, str):
            return f"llm:{serialized}"
        if tags:
            return f"llm:{tags[-1]}"
        return "llm:unknown"

    def on_llm_start(
        self, serialized, prompts, *, run_id, tags=None, metadata=None, **kwargs
    ):
        name = self._name(serialized, metadata, tags)
        self._starts[run_id] = (
            name,
            time.perf_counter(),
            tags or [],
            metadata or {},
            time.time(),  # wall-clock start (epoch seconds)
        )

    def _extract_metrics(self, response) -> dict:
        """
        Aggregate usage/metadata from multiple providers into a consistent shape.
        Priority for rollup: usage_metadata (if any) > response_metadata.token_usage > llm_output.{token_usage|usage}
        We still include all raw sources alongside the normalized rollup.
        """
        out = {}
        sources_token_usage = []  # raw token_usage dicts from response_metadata
        sources_usage_meta = []  # raw usage_metadata dicts
        roll = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }

        try:
            # 1) llm_output
            llm_output = getattr(response, "llm_output", None)
            if isinstance(llm_output, dict):
                out["llm_output"] = llm_output
                tu = llm_output.get("token_usage") or llm_output.get("usage")
                coerced_tu = _coerce_usage(tu)
                if coerced_tu:
                    out["llm_output_token_usage"] = coerced_tu  # clean copy

            # 2) generations -> response_metadata / usage_metadata
            gens = getattr(response, "generations", None)
            resp_meta_list, usage_meta_list = [], []
            if gens:
                for gen_list in gens:
                    for gen in (
                        gen_list
                        if isinstance(gen_list, (list, tuple))
                        else [gen_list]
                    ):
                        msg = getattr(gen, "message", None)
                        if msg is None:
                            continue
                        rm = getattr(msg, "response_metadata", None)
                        if isinstance(rm, dict):
                            resp_meta_list.append(rm)
                            tu = rm.get("token_usage") or rm.get("usage")
                            coerced = _coerce_usage(tu)
                            if coerced:
                                sources_token_usage.append(coerced)
                        um = getattr(msg, "usage_metadata", None)
                        if isinstance(um, dict):
                            usage_meta_list.append(um)
                            sources_usage_meta.append(dict(um))

            if resp_meta_list:
                out["response_metadata"] = resp_meta_list
            if usage_meta_list:
                out["usage_metadata"] = usage_meta_list

            # 3) Build the normalized rollup with priority
            if sources_usage_meta:
                for d in sources_usage_meta:
                    _acc_from(d, roll)
                out["usage_source"] = "usage_metadata"
            elif sources_token_usage:
                for d in sources_token_usage:
                    _acc_from(d, roll)
                out["usage_source"] = "response_metadata.token_usage"
            else:
                # fall back to llm_output if we coerced anything
                coerced = out.get("llm_output_token_usage") or {}
                if coerced:
                    _acc_from(coerced, roll)
                    out["usage_source"] = "llm_output.token_usage"

            def _extract_extras(d: dict) -> dict:
                if not isinstance(d, dict):
                    return {"reasoning_tokens": 0, "cached_tokens": 0}
                # reasoning
                rt = d.get("reasoning_tokens") or (
                    d.get("completion_tokens_details") or {}
                ).get("reasoning_tokens")
                # cached
                cached = (
                    d.get("cached_tokens")
                    or d.get("cached_input_tokens")
                    or (d.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    )
                    or d.get("prompt_cache_hits")
                )

                def _to_int(x):
                    try:
                        return int(float(x))
                    except Exception:
                        return 0

                return {
                    "reasoning_tokens": _to_int(rt),
                    "cached_tokens": _to_int(cached),
                }

            # Enrich from non-selected sources only (avoid double-counting the same info)
            src = out.get("usage_source")
            extras_candidates = []
            if src != "llm_output.token_usage":
                extras_candidates.append(
                    _extract_extras(out.get("llm_output_token_usage") or {})
                )
            if src != "response_metadata.token_usage":
                for d in sources_token_usage:
                    extras_candidates.append(_extract_extras(d or {}))

            if extras_candidates:
                # choose the strongest signal present rather than summing duplicates
                roll["reasoning_tokens"] += max(
                    e["reasoning_tokens"] for e in extras_candidates
                )
                roll["cached_tokens"] += max(
                    e["cached_tokens"] for e in extras_candidates
                )

            # Final consistency guards
            if roll["prompt_tokens"] == 0 and roll["input_tokens"] > 0:
                roll["prompt_tokens"] = roll["input_tokens"]
            if roll["completion_tokens"] == 0 and roll["output_tokens"] > 0:
                roll["completion_tokens"] = roll["output_tokens"]
            # Ensure total is at least input+output (some providers omit total)
            roll["total_tokens"] = max(
                roll["total_tokens"],
                roll["input_tokens"] + roll["output_tokens"],
                roll["prompt_tokens"] + roll["completion_tokens"],
            )

            if any(v for v in roll.values()):
                out["usage_rollup"] = roll

        except Exception as e:
            out["parse_error"] = repr(e)

        return out

    def on_llm_end(self, response, *, run_id, **kwargs):
        name, t0, tags, metadata, wall_t0 = self._starts.pop(
            run_id, ("llm:unknown", time.perf_counter(), [], {}, time.time())
        )
        ms = (time.perf_counter() - t0) * 1000.0
        wall_t1 = time.time()
        self.agg.add(name, ms, True)
        metrics = self._extract_metrics(response)
        self.samples.append({
            "name": name,
            "ms": ms,
            "ok": True,
            "tags": tags,
            "metadata": metadata,
            "metrics": metrics,
            "t_start": wall_t0,
            "t_end": wall_t1,  # <— add these
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        name, t0, tags, metadata = self._starts.pop(
            run_id, ("llm:unknown", time.perf_counter(), [], {})
        )
        ms = (time.perf_counter() - t0) * 1000.0
        self.agg.add(name, ms, False)
        self.samples.append({
            "name": name,
            "ms": ms,
            "ok": False,
            "tags": tags,
            "metadata": metadata,
            "metrics": {"error": repr(error)},
        })


def _coerce_usage(obj) -> dict:
    """
    Best-effort normalize provider token-usage objects into a dict.
    Handles dicts, pydantic-ish objects with .dict()/.model_dump(), plain objects
    with attributes, and string reprs like 'Usage(prompt_tokens=..., ...)'.
    Returns a (possibly empty) dict.
    """
    if obj is None:
        return {}

    # Already a dict
    if isinstance(obj, dict):
        return dict(obj)

    # Objects that can dump themselves
    for meth in ("dict", "model_dump", "to_dict", "_asdict"):
        if hasattr(obj, meth):
            try:
                return dict(getattr(obj, meth)())
            except Exception:
                pass

    # Objects with attributes
    attrs = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    )
    if any(hasattr(obj, a) for a in attrs):
        d = {}
        for a in attrs:
            v = getattr(obj, a, None)
            if v is not None:
                try:
                    d[a] = int(v)
                except Exception:
                    pass

        # Common nested details
        try:
            ctd = getattr(obj, "completion_tokens_details", None)
            if ctd is not None:
                dd = {}
                for k in (
                    "reasoning_tokens",
                    "accepted_prediction_tokens",
                    "rejected_prediction_tokens",
                    "audio_tokens",
                    "text_tokens",
                ):
                    val = getattr(ctd, k, None)
                    if isinstance(val, (int, float)):
                        dd[k] = int(val)
                if dd:
                    d["completion_tokens_details"] = dd
            ptd = getattr(obj, "prompt_tokens_details", None)
            if ptd is not None:
                dd = {}
                for k in (
                    "cached_tokens",
                    "audio_tokens",
                    "image_tokens",
                    "text_tokens",
                ):
                    val = getattr(ptd, k, None)
                    if isinstance(val, (int, float)):
                        dd[k] = int(val)
                if dd:
                    d["prompt_tokens_details"] = dd
        except Exception:
            pass

        return d

    # String repr like "Usage(completion_tokens=340, prompt_tokens=328, total_tokens=668, ...)"
    if isinstance(obj, str):
        pairs = {k: int(v) for k, v in re.findall(r"(\w+)=([0-9]+)", obj)}
        # pull some nested detail hints if present
        for probe in ("reasoning_tokens", "cached_tokens"):
            if probe not in pairs:
                m = re.search(rf"{probe}=([0-9]+)", obj)
                if m:
                    pairs[probe] = int(m.group(1))
        return pairs

    return {}


# ---------------------------------
#        Decorator
# ---------------------------------


# Keep the decorator, but move it out of base.py to avoid bloat.
def timed_tool(tool_name: str, sink: _Agg | None = None):
    """
    Simple timing decorator for tools; complements PerToolTimer callbacks.
    If you're already using the callback, this adds a local measurement too.
    """
    sink = sink or _Agg()

    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            ok = True
            try:
                return fn(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                sink.add(tool_name, (time.perf_counter() - t0) * 1000.0, ok)

        return wrapper

    return deco


# ---------------------------------
#         Rendering helpers
# ---------------------------------


def render_table(
    title: str, rows: Iterable[Tuple[str, int, float, float, float]]
) -> str:
    # rows: (name, count, total_s, avg_ms, max_ms)
    out = []
    out.append(f"\n{title}")
    out.append(
        "┏{0}┳{1}┳{2}┳{3}┳{4}┓".format(
            "━" * 30, "━" * 7, "━" * 11, "━" * 10, "━" * 10
        )
    )
    out.append(
        "┃ {0:<28} ┃ {1:>5} ┃ {2:>9} ┃ {3:>8} ┃ {4:>8} ┃".format(
            "Name", "Count", "Total(s)", "Avg(ms)", "Max(ms)"
        )
    )
    out.append(
        "┡{0}╇{1}╇{2}╇{3}╇{4}┩".format(
            "━" * 30, "━" * 7, "━" * 11, "━" * 10, "━" * 10
        )
    )
    for name, cnt, tot_s, avg_ms, max_ms in rows:
        out.append(
            "│ {0:<28} │ {1:>5} │ {2:>9.2f} │ {3:>8.0f} │ {4:>8.0f} │".format(
                name[:28], cnt, tot_s, avg_ms, max_ms
            )
        )
    out.append(
        "└{0}┴{1}┴{2}┴{3}┴{4}┘".format(
            "─" * 30, "─" * 7, "─" * 11, "─" * 10, "─" * 10
        )
    )
    return "\n".join(out)


def _parse_iso(ts: str | None):
    if not ts:
        return None
    # handle both "...Z" and "+00:00"
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _mk_table(
    title: str, rows: list[tuple[str, int, float, float, float]]
) -> Table:
    t = Table(
        title=title,
        title_style="bold white",
        box=HEAVY,
        show_lines=False,
        expand=False,  # <- important: don’t stretch columns differently per table
        pad_edge=False,
        padding=COL_PAD,
        header_style="bold",
    )

    # lock all column widths so every table renders identically
    t.add_column(
        "Name",
        style="cyan",
        no_wrap=True,
        overflow="ellipsis",
        width=NAME_W,
        min_width=NAME_W,
        max_width=NAME_W,
    )
    t.add_column(
        "Count",
        justify="right",
        width=COUNT_W,
        min_width=COUNT_W,
        max_width=COUNT_W,
    )
    t.add_column(
        "Total(s)",
        justify="right",
        width=TOTAL_W,
        min_width=TOTAL_W,
        max_width=TOTAL_W,
    )
    t.add_column(
        "Avg(ms)",
        justify="right",
        width=AVG_W,
        min_width=AVG_W,
        max_width=AVG_W,
    )
    t.add_column(
        "Max(ms)",
        justify="right",
        width=MAX_W,
        min_width=MAX_W,
        max_width=MAX_W,
    )

    if not rows:
        t.add_row("—", "0", f"{0.00:,.2f}", f"{0:,.0f}", f"{0:,.0f}")
        return t

    for name, count, total_s, avg_ms, max_ms in rows:
        # keep your color hint for graph rows
        name_cell = (
            f"[bright_magenta]{name}[/]"
            if str(name).startswith("graph:")
            else name
        )
        t.add_row(
            name_cell,
            f"{count:,}",  # right-justified by column, with thousands separator
            f"{total_s:,.2f}",
            f"{avg_ms:,.0f}",
            f"{max_ms:,.0f}",
        )
    return t


def _truncate_pad(s: str, width: int) -> str:
    s = str(s)
    if len(s) <= width:
        return s.ljust(width)
    if width <= 3:
        return s[:width]
    return s[: width - 3] + "..."


def _plain_table(rows):
    header = (
        f"{'Name':<{NAME_W}} | "
        f"{'Count':>{COUNT_W}} | "
        f"{'Total(s)':>{TOTAL_W}} | "
        f"{'Avg(ms)':>{AVG_W}} | "
        f"{'Max(ms)':>{MAX_W}}"
    )
    lines = [header]

    if not rows:
        lines.append(
            f"{'—':<{NAME_W}} | "
            f"{0:>{COUNT_W}d} | "
            f"{0.00:>{TOTAL_W},.2f} | "
            f"{0:>{AVG_W},.0f} | "
            f"{0:>{MAX_W},.0f}"
        )
        return "\n".join(lines)

    for n, c, ts, am, mm in rows:
        name = _truncate_pad(n, NAME_W)
        lines.append(
            f"{name} | "
            f"{c:>{COUNT_W}d} | "
            f"{ts:>{TOTAL_W},.2f} | "
            f"{am:>{AVG_W},.0f} | "
            f"{mm:>{MAX_W},.0f}"
        )

    return "\n".join(lines)


# ---------------------------------
#          Facade to use
# ---------------------------------
@dataclass
class Telemetry:
    enable: bool = True
    debug_raw: bool = False  # toggle raw dump
    output_dir: str = "metrics"  # where to save JSON
    save_json_default: bool = True  # opt-in autosave

    tool: PerToolTimer = field(default_factory=PerToolTimer)
    runnable: PerRunnableTimer = field(default_factory=PerRunnableTimer)
    llm: PerLLMTimer = field(default_factory=PerLLMTimer)

    # Run-scoped context we’ll embed in the JSON filename/body
    context: Dict[str, Any] = field(default_factory=dict)

    # ---------- JSON/export helpers ----------
    def begin_run(self, *, agent: str, thread_id: str) -> None:
        """Call at the start of BaseAgent.invoke()."""

        # --- reset per-run aggregators so each run is isolated ---
        self.tool = PerToolTimer()
        self.runnable = PerRunnableTimer()
        self.llm = PerLLMTimer()
        # --------------------------------------------------------------

        self.context.clear()
        self.context.update({
            "agent": agent,
            "thread_id": thread_id,
            "run_id": uuid.uuid4().hex,
            "started_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        })

    @property
    def callbacks(self) -> List[BaseCallbackHandler]:
        return [] if not self.enable else [self.tool, self.runnable, self.llm]

    def _snapshot(self) -> dict:
        """Collect everything we might want to inspect."""

        def _as_dict(obj):
            try:
                return dict(vars(obj))
            except Exception:
                return repr(obj)

        # Keys like run_id can be UUIDs; stringify to be safe
        def _stringify_keys(d):
            try:
                return {str(k): v for k, v in d.items()}
            except Exception:
                return repr(d)

        return {
            "runnable": {
                "_starts": _stringify_keys(
                    getattr(self.runnable, "_starts", {})
                ),
                "agg": _as_dict(getattr(self.runnable, "agg", {})),
                "buckets": list(
                    getattr(self.runnable.agg, "buckets", lambda: [])()
                ),
            },
            "tool": {
                "_starts": _stringify_keys(getattr(self.tool, "_starts", {})),
                "agg": _as_dict(getattr(self.tool, "agg", {})),
                "buckets": list(
                    getattr(self.tool.agg, "buckets", lambda: [])()
                ),
            },
            "llm": {
                "_starts": _stringify_keys(getattr(self.llm, "_starts", {})),
                "agg": _as_dict(getattr(self.llm, "agg", {})),
                "buckets": list(getattr(self.llm.agg, "buckets", lambda: [])()),
            },
        }

    def _records_struct(self) -> dict:
        def _normalize(rec_list):
            # aggregator stores tuples like (name, ms, ok)
            out = []
            for r in rec_list:
                try:
                    name, ms, ok = r
                except Exception:
                    # fallback if shape changed
                    name, ms, ok = (str(r), None, None)
                out.append({"name": name, "ms": ms, "ok": bool(ok)})
            return out

        return {
            "runnable": _normalize(getattr(self.runnable.agg, "records", [])),
            "tool": _normalize(getattr(self.tool.agg, "records", [])),
            "llm": _normalize(getattr(self.llm.agg, "records", [])),
        }

    def _tables_struct(self) -> dict:
        """Structured tables ready for JSON."""

        def _rows(rows):
            # rows are (name, count, total_s, avg_ms, max_ms)
            return [
                {"name": n, "count": c, "total_s": ts, "avg_ms": a, "max_ms": m}
                for (n, c, ts, a, m) in rows
            ]

        return {
            "runnable": _rows(self.runnable.agg.buckets()),
            "tool": _rows(self.tool.agg.buckets()),
            "llm": _rows(self.llm.agg.buckets()),
        }

    def _totals(self, tables: dict) -> dict:
        runnable_rows = tables.get("runnable") or []

        # choose the single root graph row (fallback to 0.0 if missing)
        graph_rows = [
            r
            for r in runnable_rows
            if str(r.get("name", "")).startswith("graph:")
        ]
        graph_total = max(
            (float(r.get("total_s") or 0.0) for r in graph_rows),
            default=0.0,
        )

        llm_total = sum(
            float(r.get("total_s") or 0.0) for r in (tables.get("llm") or [])
        )
        tool_total = sum(
            float(r.get("total_s") or 0.0) for r in (tables.get("tool") or [])
        )

        return {
            "graph_total_s": graph_total,
            "llm_total_s": llm_total,
            "tool_total_s": tool_total,
            "unattributed_s": max(0.0, graph_total - (llm_total + tool_total)),
        }

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def _default_filepath(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        agent = (self.context.get("agent") or "agent").replace(" ", "_")
        thread_id = self.context.get("thread_id") or "thread"
        run_id = (self.context.get("run_id") or "run")[:8]
        fname = f"{ts}_{agent}_{thread_id}_{run_id}.json"
        return os.path.join(self.output_dir, fname)

    def _json_default(self, o):
        # dataclasses --> dict
        try:
            import dataclasses

            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
        except Exception:
            pass
        # Everything else (locks, functions, callbacks, etc.) --> repr string
        return repr(o)

    def _save_json(self, payload: dict, filepath: str | None = None) -> str:
        path = filepath or self._default_filepath()
        self._ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            )
        return path

    def to_json(
        self, *, include_raw_snapshot: bool, include_raw_records: bool
    ) -> dict:
        tables = self._tables_struct()
        out = {
            "context": {
                **self.context,
                "ended_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
            "tables": tables,
            "totals": self._totals(tables),
            "llm_events": list(getattr(self.llm, "samples", [])),
        }
        if include_raw_snapshot:
            out["raw_snapshot"] = self._snapshot()
        if include_raw_records:
            out["raw_records"] = self._records_struct()
        return out

    def render(
        self,
        raw: bool | None = None,
        save_json: bool | None = None,
        filepath: str | None = None,
        save_raw_snapshot: bool | None = None,
        save_raw_records: bool | None = None,
    ):
        if not self.enable:
            return ""

        # --- Gather tables ---
        r_rows = self.runnable.agg.buckets()
        t_rows = self.tool.agg.buckets()
        l_rows = self.llm.agg.buckets() if hasattr(self, "llm") else []

        # --- Build priceable payload early (also gives us context) ---
        inc_snapshot = True if save_raw_snapshot is None else save_raw_snapshot
        inc_records = True if save_raw_records is None else save_raw_records
        payload = self.to_json(
            include_raw_snapshot=inc_snapshot, include_raw_records=inc_records
        )
        ctx = payload.get("context", {}) or {}
        agent_name = (
            ctx.get("agent")
            or getattr(self, "__class__", type("X", (object,), {})).__name__
            or "UnknownAgent"
        )
        thread_id = (
            ctx.get("thread_id") or getattr(self, "thread_id", None) or "—"
        )
        run_id = ctx.get("run_id", "—")
        started_at = ctx.get("started_at")
        ended_at = ctx.get("ended_at")
        start_dt = _parse_iso(started_at)
        end_dt = _parse_iso(ended_at)
        wall_secs = (
            (end_dt - start_dt).total_seconds()
            if (start_dt and end_dt)
            else None
        )

        # Optional human alias set post-construction: executor.name = "exec-A"
        human_alias = getattr(self, "name", None) or getattr(
            self, "alias", None
        )
        base_label = human_alias or agent_name

        # Lazily create a per-instance short id (stable for the object's lifetime)
        if not hasattr(self, "_short_id"):
            try:
                import uuid as _uuid

                self._short_id = _uuid.uuid4().hex[:6]
            except Exception:
                self._short_id = format(id(self) & 0xFFFFFF, "06x")
        agent_label = f"{base_label} [{self._short_id}]"

        # --- Totals (use wall clock for unattributed) ---
        def _total(rows):
            return sum((row[2] for row in rows), 0.0)

        llm_total = _total(l_rows)
        tool_total = _total(t_rows)
        unattributed = (
            max(0.0, wall_secs - (llm_total + tool_total))
            if wall_secs is not None
            else None
        )
        graph_bucket_sum = _total(r_rows)  # informative only (overlaps)

        # --- Pricing (optional) ---
        pricing_text_lines = []
        pricing_mod = _get_pricing_module()
        if pricing_mod and (payload.get("llm_events") or []):
            registry_path = os.environ.get("URSA_PRICING_JSON")
            registry = pricing_mod.load_registry(path=registry_path)
            payload = pricing_mod.price_payload(
                payload, registry=registry, overwrite=False
            )
            costs = payload.get("costs") or {}
            total_usd = costs.get("total_usd", 0.0)
            by_model = costs.get("by_model_usd", {})
            src_counts = costs.get("event_sources", {})
            pricing_text_lines.append("[bold]Cost Summary (USD)[/]")
            pricing_text_lines.append(
                f"  total: [bold green]${total_usd:,.6f}[/]"
            )
            for model, amt in (by_model or {}).items():
                pricing_text_lines.append(f"  {model}: ${amt:,.6f}")
            if src_counts:
                pricing_text_lines.append(
                    f"  (events: provider={src_counts.get('provider', 0)}, "
                    f"computed={src_counts.get('computed', 0)}, "
                    f"no_usage={src_counts.get('no_usage', 0)}, "
                    f"no_pricing={src_counts.get('no_pricing', 0)})"
                )

        # --- Save JSON (if requested) ---
        do_save = self.save_json_default if save_json is None else save_json
        saved_path = None
        if do_save:
            saved_path = self._save_json(payload, filepath=filepath)

        # --- Build Rich renderables ---
        # console = get_console()

        # --- Build header & attribution lines (markup-aware) ---
        header_lines = []
        header_lines.append(
            f"[bold magenta]{agent_label}[/] [dim]•[/] thread [bold]{thread_id}[/] [dim]•[/] run [bold]{run_id}[/]"
        )
        if start_dt and end_dt:
            header_lines.append(
                f"[dim]{started_at} → {ended_at}[/dim]   [bold]wall[/]: {wall_secs:,.2f}s"
            )

        attrib_lines = []
        attrib_lines.append("[bold]Attribution[/]")
        if wall_secs is not None:
            attrib_lines.append(
                f"  Total run (wall): [bold]{wall_secs:,.2f}s[/]"
            )
        attrib_lines.append(f"  LLM total:         {llm_total:,.2f}s")
        attrib_lines.append(f"  Tool total:        {tool_total:,.2f}s")
        if unattributed is not None:
            attrib_lines.append(f"  Unattributed:      {unattributed:,.2f}s")
        attrib_lines.append(
            f"[dim]  Sum of runnable buckets (non-additive): {graph_bucket_sum:,.2f}s[/]"
        )
        if saved_path:
            attrib_lines.append(f"[dim]Saved metrics JSON to:[/] {saved_path}")

        header_str = "\n".join(
            header_lines
        )  # these strings contain [bold], [dim], etc.
        attrib_str = "\n".join(attrib_lines)
        pricing_str = (
            "\n".join(pricing_text_lines) if pricing_text_lines else None
        )

        tbl_nodes = _mk_table("Per-Node / Runnable Timing", r_rows)
        tbl_tools = _mk_table("Per-Tool Timing", t_rows)
        tbl_llms = _mk_table("Per-LLM Timing", l_rows)

        renderables = [
            Text.from_markup(header_str),  # <- parse markup
            Rule(),
            tbl_nodes,
            tbl_tools,
            tbl_llms,
            Rule(),
            Text.from_markup(attrib_str),  # <- parse markup
        ]
        if pricing_str:
            renderables += [
                Rule(),
                Text.from_markup(pricing_str),
            ]  # <- parse markup

        # panel = Panel.fit(
        #     Group(*renderables),  # <- pass a single renderable
        #     title=f"[bold white]Metrics[/] • [cyan]{agent_label}[/]",
        #     border_style="bright_magenta",
        #     padding=(1, 2),
        #     box=HEAVY,  # <- beefy border with corners
        # )
        # console.print(panel)

        _session_ingest(payload)

        # parts = []
        # parts.append(f"{agent_label} • thread {thread_id} • run {run_id}")
        # if start_dt and end_dt:
        #     parts.append(f"{started_at} → {ended_at}   wall: {wall_secs:.2f}s")
        # parts.append("\nPer-Node / Runnable Timing\n" + _plain_table(r_rows))
        # parts.append("\nPer-Tool Timing\n" + _plain_table(t_rows))
        # parts.append("\nPer-LLM Timing\n" + _plain_table(l_rows))
        # parts.append(
        #     "\nAttribution\n"
        #     + (f"  Total run (wall): {wall_secs:.2f}s\n" if wall_secs is not None else "")
        #     + f"  LLM total: {llm_total:.2f}s\n"
        #     + f"  Tool total: {tool_total:.2f}s\n"
        #     + (f"  Unattributed: {unattributed:.2f}s\n" if unattributed is not None else "")
        #     + f"  Graph bucket sum (overlaps): {graph_bucket_sum:.2f}s"
        # )
        # if pricing_text_lines:
        #     sanitized = []
        #     for line in pricing_text_lines:
        #         sanitized.append(
        #             line.replace("[bold]", "")
        #                 .replace("[/bold]", "")
        #                 .replace("[dim]", "")
        #                 .replace("[/dim]", "")
        #                 .replace("[bold green]", "")
        #                 .replace("[/bold green]", "")
        #         )
        #     parts.append("\n" + "\n".join(sanitized))
        # if saved_path:
        #     parts.append(f"\nSaved metrics JSON to: {saved_path}")

        # include_raw = self.debug_raw if raw is None else raw
        # if include_raw:
        #     import pprint as _pp
        #     parts.append("\nRaw Debug Snapshot\n" + _pp.pformat(self._snapshot(), sort_dicts=True, width=120))

        # return ""
        # # return "\n".join(parts)
