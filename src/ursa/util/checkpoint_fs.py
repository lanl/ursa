"""
checkpoint_fs.py

Helpers for filesystem-based checkpoint discovery, snapshotting, and restore.

Tiny utilities used by URSA runners to find and restore executor
checkpoints, snapshot SQLite DBs in a WAL-safe way, and manage
executor checkpoint naming conventions.

Public functions:
  - ckpt_dir(workspace) -> Path
  - snapshot_sqlite_db(src_path, dst_path) -> None
  - parse_snapshot_indices(Path) -> (int|None, int|None)
  - list_executor_checkpoints(workspace) -> list[Path]
  - choose_checkpoint(workspace, timeout=60, input_fn=None) -> Path|None
  - resolve_resume_checkpoint(workspace, resume_from, timeout=60, choose_fn=None, input_fn=None) -> Path|None
  - restore_executor_from_snapshot(workspace, snapshot) -> None
  - sync_progress_for_snapshot_single(workspace, snapshot, plan_sig, progress_file=None) -> None

Design notes:
  - `input_fn(prompt, timeout) -> str|None` is injected; pass your interactive
    timed_input_with_countdown from your runner for the same UX.
  - Defaults are non-interactive: choose_checkpoint returns the default live DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable, Optional


def ckpt_dir(workspace: str) -> Path:
    """Return and ensure the checkpoints directory for a workspace."""
    p = Path(workspace) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def snapshot_sqlite_db(src_path: Path, dst_path: Path) -> None:
    """
    Make a consistent copy of the SQLite database at src_path into dst_path,
    using the sqlite3 backup API. Safe with WAL; no need to copy -wal/-shm files.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_uri = f"file:{Path(src_path).resolve().as_posix()}?mode=ro"
    src = dst = None
    try:
        src = sqlite3.connect(src_uri, uri=True)
        dst = sqlite3.connect(str(dst_path))
        with dst:
            src.backup(dst)
    finally:
        try:
            if dst:
                dst.close()
        except Exception:
            pass
        try:
            if src:
                src.close()
        except Exception:
            pass


def parse_snapshot_indices(p: Path) -> tuple[Optional[int], Optional[int]]:
    """
    Parse snapshot filename to indices.

    Examples:
      executor_5.db / executor_checkpoint_5.db => (5, None)
      executor_3_2.db / executor_checkpoint_3_2.db => (3, 2)
    """
    import re

    m = re.match(
        r"(?:executor|executor_checkpoint)_(\d+)(?:_(\d+))?\.db$", p.name
    )
    if not m:
        return None, None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else None
    return a, b


def _ckpt_sort_key(p: Path):
    """Internal sort key: numbered snapshots first, then live default, then others."""
    import re

    name = p.name
    pat = r"executor_checkpoint_(\d+)(?:_(\d+))?\.db$"
    m = re.match(pat, name)
    if m:
        a = int(m.group(1))
        b = int(m.group(2) or 0)
        return (0, a, b, name)  # numbered snapshots first
    if name == "executor_checkpoint.db":
        return (1, float("inf"), float("inf"), name)  # live default next
    # anything else sinks to the bottom
    return (2, float("inf"), float("inf"), name)


def list_executor_checkpoints(workspace: str) -> list[Path]:
    """
    Find executor checkpoint DB files in checkpoint dir and workspace root.

    Returns a sorted list (newest/most-relevant first according to our sort key).
    """
    ws = Path(workspace)
    ckdir = ckpt_dir(workspace)
    seen = {}
    for base in (ckdir, ws):
        for pat in ("executor_checkpoint_*.db", "executor_checkpoint.db"):
            for p in base.glob(pat):
                seen[p.resolve()] = p
    return sorted(seen.values(), key=_ckpt_sort_key)


def choose_checkpoint(
    workspace: str,
    timeout: int = 60,
    input_fn: Optional[Callable[[str, int], Optional[str]]] = None,
) -> Optional[Path]:
    """
    Present available checkpoints and let user select one.

    - input_fn(prompt, timeout) should return the user's input string or None for timeout/no-input.
    - If input_fn is None, this will immediately return the default live checkpoint path.

    Returns a Path (possibly the default live DB), or None if nothing is available.
    """
    ckpts = list_executor_checkpoints(workspace)
    default = ckpt_dir(workspace) / "executor_checkpoint.db"

    # Non-interactive default behavior: return the default live DB
    if input_fn is None:
        return default

    print("\nAvailable executor checkpoints:")
    if ckpts:
        for i, p in enumerate(ckpts, 1):
            tag = " (default)" if p.resolve() == default.resolve() else ""
            print(f"  {i}. {p.name}{tag}")
        prompt = (
            f"Select checkpoint [1-{len(ckpts)} or filename] "
            f"(Enter for default: {default.name}; auto in {timeout}s) > "
        )
        sel = input_fn(prompt, timeout)
    else:
        print("  (none found)")
        prompt = (
            f"Press Enter to start fresh ({default.name}; auto in {timeout}s), "
            f"or type a checkpoint filename to restore > "
        )
        sel = input_fn(prompt, timeout)

    if not sel:
        return default

    sel = sel.strip()
    if sel.isdigit() and ckpts:
        idx = int(sel)
        if 1 <= idx <= len(ckpts):
            return ckpts[idx - 1]
        print(f"[warn] Invalid selection {sel}; using default.")
        return default

    cand = Path(sel)
    if not cand.is_absolute():
        cand = Path(workspace) / sel
    if cand.exists():
        # Optional: accept legacy names beginning with executor_
        return cand

    print(f"[warn] '{sel}' not found; using default.")
    return default


def resolve_resume_checkpoint(
    workspace: str,
    resume_from: Optional[str],
    timeout: int,
    choose_fn: Optional[Callable[..., Optional[Path]]] = None,
    input_fn: Optional[Callable[[str, int], Optional[str]]] = None,
) -> Optional[Path]:
    """
    Resolve a resume target from CLI value (resume_from) or via interactive chooser.

    - If resume_from is provided and exists (either absolute or relative to checkpoints/ or workspace/),
      it is returned.
    - Otherwise, choose_fn(workspace, timeout, input_fn) is called if provided; otherwise
      choose_checkpoint is used with the provided input_fn.
    """
    if resume_from:
        p = Path(resume_from)
        if not p.is_absolute():
            cand = ckpt_dir(workspace) / p
            if cand.exists():
                print(
                    f"[resume] Using checkpoint from CLI (checkpoints): {cand.name}"
                )
                return cand
            p = Path(workspace) / p
        if p.exists():
            print(f"[resume] Using checkpoint from CLI: {p.name}")
            return p
        print(
            f"[warn] --resume-from '{resume_from}' not found; falling back to interactive/default."
        )

    chooser = choose_fn or choose_checkpoint
    return chooser(workspace, timeout=timeout, input_fn=input_fn)


def restore_executor_from_snapshot(workspace: str, snapshot: Path) -> None:
    """
    Copy the selected snapshot database into the live executor DB (checkpoints/executor_checkpoint.db).
    """
    live = ckpt_dir(workspace) / "executor_checkpoint.db"
    if not snapshot.exists():
        print(
            f"[resume] No snapshot to restore (missing: {snapshot}); starting fresh."
        )
        return
    if snapshot.resolve() == live.resolve():
        print(f"[resume] Live DB already at desired checkpoint: {live.name}")
        return
    try:
        snapshot_sqlite_db(snapshot, live)
        for suffix in ("-wal", "-shm"):
            side = live.with_name(live.name + suffix)
            if side.exists():
                side.unlink()
        print(f"[resume] Restored: {snapshot.name} → {live.name}")
    except Exception as e:
        print(
            f"[warn] Failed to restore '{snapshot}': {e}. Continuing with current live DB."
        )


def sync_progress_for_snapshot_single(
    workspace: str,
    snapshot: Path,
    plan_sig: str,
    progress_file: Optional[Path] = None,
) -> None:
    """
    For SINGLE mode numbered snapshots, align executor_progress.json so the engine resumes at the right step.

    executor_<k>.db means 'k' steps completed ⇒ next_index = k (0-based start from k).
    """
    k, _ = parse_snapshot_indices(snapshot)
    if not k:
        # Not a numbered snapshot (e.g., executor_checkpoint.db) — leave JSON as-is
        print(
            "[resume] Using live/default checkpoint; not altering executor_progress.json."
        )
        return
    prog_path = progress_file or progress_file(workspace)
    payload = {
        "next_index": int(
            k
        ),  # start loop at idx=k (i.e., step k+1 in 1-based terms)
        "plan_hash": str(plan_sig),
        "last_summary": f"Resumed from snapshot {snapshot.name}",
    }
    prog_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[resume] Wrote {prog_path.name}: next_index={k}, plan_hash={plan_sig[:8]}. . ."
    )


def hash_plan(plan_steps) -> str:
    """
    Hash the structure so we can detect if the plan changed between runs.
    Deterministic by sorting keys and using default=str for objects.
    """
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(plan_steps, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# Progress JSON helpers (executor progress tracking)
def progress_file(workspace: str) -> Path:
    """Default executor progress JSON location."""
    return Path(workspace) / "executor_progress.json"


def load_exec_progress(workspace: str) -> dict:
    p = progress_file(workspace)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_exec_progress(
    workspace: str,
    next_index: int,
    plan_hash: str,
    last_summary: str | None = None,
) -> None:
    p = progress_file(workspace)
    payload = {"next_index": int(next_index), "plan_hash": plan_hash}
    if last_summary is not None:
        payload["last_summary"] = last_summary
    p.write_text(json.dumps(payload, indent=2))
