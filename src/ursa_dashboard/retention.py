from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .storage import read_json


@dataclass(frozen=True)
class RetentionPolicy:
    max_runs_per_agent: int = 100
    max_age_days: int = 7
    max_disk_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB


def _parse_ts(ts: str) -> datetime:
    # expecting RFC3339 with Z
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for fn in files:
            fp = Path(root) / fn
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def enforce_retention(
    *,
    workspace_root: Path,
    policy: RetentionPolicy,
) -> dict:
    """Delete old terminal runs to satisfy retention policy.

    Returns stats about deletions.
    """
    meta_dir = workspace_root / "_meta" / "runs"
    if not meta_dir.exists():
        return {"deleted": 0, "reason": "no_meta"}

    runs = []
    for p in meta_dir.glob("*.json"):
        try:
            rec = read_json(p)
        except Exception:
            continue
        status = rec.get("status")
        if status not in {"succeeded", "failed", "cancelled"}:
            continue
        agent_id = rec.get("agent_id")
        created_at = rec.get("created_at")
        if not (agent_id and created_at):
            continue
        runs.append((agent_id, _parse_ts(created_at), p, rec))

    deleted = 0

    # Age-based deletion
    cutoff = datetime.now(timezone.utc) - timedelta(days=policy.max_age_days)
    for agent_id, created, meta_path, rec in sorted(runs, key=lambda x: x[1]):
        if created >= cutoff:
            continue
        run_rel = rec.get("run_dir")
        if run_rel:
            run_dir = (workspace_root / run_rel).resolve()
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        deleted += 1

    # Count-based deletion per agent
    by_agent: dict[str, list[tuple[datetime, Path, dict]]] = {}
    for agent_id, created, meta_path, rec in runs:
        # might have been deleted above
        if not meta_path.exists():
            continue
        by_agent.setdefault(agent_id, []).append((created, meta_path, rec))

    for agent_id, lst in by_agent.items():
        lst.sort(key=lambda x: x[0], reverse=True)  # newest first
        for created, meta_path, rec in lst[policy.max_runs_per_agent :]:
            run_rel = rec.get("run_dir")
            if run_rel:
                run_dir = (workspace_root / run_rel).resolve()
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
            try:
                meta_path.unlink(missing_ok=True)
            except Exception:
                pass
            deleted += 1

    # Disk cap deletion (oldest-first across all agents)
    runs_left = []
    for agent_id, created, meta_path, rec in runs:
        if meta_path.exists():
            runs_left.append((created, meta_path, rec))
    runs_left.sort(key=lambda x: x[0])

    total_bytes = (
        _dir_size_bytes(workspace_root / "runs")
        if (workspace_root / "runs").exists()
        else 0
    )
    idx = 0
    while total_bytes > policy.max_disk_bytes and idx < len(runs_left):
        created, meta_path, rec = runs_left[idx]
        idx += 1
        run_rel = rec.get("run_dir")
        if run_rel:
            run_dir = (workspace_root / run_rel).resolve()
            if run_dir.exists():
                total_bytes -= _dir_size_bytes(run_dir)
                shutil.rmtree(run_dir, ignore_errors=True)
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        deleted += 1

    return {
        "deleted": deleted,
        "remaining_meta": len(list(meta_dir.glob("*.json"))),
    }
