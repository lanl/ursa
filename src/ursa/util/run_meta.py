from __future__ import annotations

import json
from pathlib import Path
from typing import Any

"""
run_meta.py

This module manages per-workspace "run metadata" stored in `run_meta.json`.

A workspace is a directory created for a single run (or a series of resumed runs)
of an agentic workflow. We store small bits of bookkeeping state that should
persist across restarts, e.g.:

- which planning mode the workspace is locked to ("single" vs "hierarchical")
- whether one-time setup steps were already performed (e.g., logo generation)
- plan signature/hash and plan step count from the last planning phase
- the thread_id and model name used for the run
"""


def _run_meta_file(workspace: str) -> Path:
    return Path(workspace) / "run_meta.json"


def load_run_meta(workspace: str) -> dict:
    p = _run_meta_file(workspace)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_run_meta(workspace: str, **fields) -> dict:
    """
    Update run_meta.json with provided key/value fields.
    Fields set to None are ignored (so callers can do save_run_meta(..., foo=maybe_none)).
    """
    p = _run_meta_file(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)

    meta = load_run_meta(workspace)
    meta.update({k: v for k, v in fields.items() if v is not None})
    p.write_text(json.dumps(meta, indent=2))
    return meta


def lock_or_warn_planning_mode(
    workspace: str,
    chosen_mode: str,
    console: Any | None = None,
) -> tuple[str, bool]:
    """
    Ensure a workspace has a fixed planning_mode.
    Returns (effective_mode, locked_already).

    If planning_mode is already stored in run_meta.json and differs from chosen_mode,
    we warn and keep the existing mode. If `console` (rich console) is provided, we
    print a nice Panel; otherwise we fall back to plain print().
    """
    meta = load_run_meta(workspace)
    existing = meta.get("planning_mode")

    if existing:
        if existing != chosen_mode:
            msg = (
                f"Workspace already locked to planning_mode={existing}\n"
                f"Ignoring requested mode '{chosen_mode}'. Use a new --workspace if you want a different mode."
            )
            if console is not None:
                try:
                    from rich.panel import Panel

                    console.print(Panel.fit(msg, border_style="yellow"))
                except Exception:
                    print(msg)
            else:
                print(msg)
        return existing, True

    # First run for this workspace: lock it
    save_run_meta(workspace, planning_mode=chosen_mode)
    return chosen_mode, False
