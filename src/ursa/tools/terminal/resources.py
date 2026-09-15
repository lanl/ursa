"""Best-effort, read-only resource sampling for a local process tree."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import psutil

RESOURCE_SAMPLE_SECONDS = 0.2
_RESOURCE_ERRORS = (psutil.Error, NotImplementedError)


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """A PID tied to its creation time, captured before the child is reaped.

    If creation time is unavailable, keep the PID for display but never sample
    it: a later lookup could silently attach to an unrelated, reused PID.
    """

    pid: int
    create_time: float | None


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Capture identity without making failed inspection a launch failure."""
    try:
        created = psutil.Process(pid).create_time()
    except _RESOURCE_ERRORS:
        created = None
    return ProcessIdentity(pid, created)


@dataclass
class _Snapshot:
    sampled_at: float = 0.0
    processes: dict[ProcessIdentity, dict[str, int | float]] = field(
        default_factory=dict
    )
    issues: set[str] = field(default_factory=set)
    tree_complete: bool = True


def _require_live(process: psutil.Process) -> None:
    # psutil's is_running checks creation time as well as PID. Do not use only
    # create_time() on an existing Process: that method caches its first value.
    if not process.is_running():
        raise psutil.NoSuchProcess(process.pid)
    if process.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
        raise psutil.ZombieProcess(process.pid)


def _read_process(process: psutil.Process, snapshot: _Snapshot) -> None:
    _require_live(process)
    identity = ProcessIdentity(process.pid, process.create_time())
    metrics: dict[str, int | float] = {}
    # Each metric fails independently. In particular, lack of permission to
    # inspect memory must not suppress otherwise available CPU/thread data.
    try:
        cpu = process.cpu_times()
        # Exclude children_user/children_system: descendants are sampled once
        # each, and a parent's waited-for children are historical, not live.
        metrics["cpu_seconds"] = cpu.user + cpu.system
        metrics["cpu_sampled_at"] = time.monotonic()
    except _RESOURCE_ERRORS as error:
        snapshot.issues.add(f"cpu: {type(error).__name__}")
    try:
        memory = process.memory_info()
        metrics["rss_bytes"] = memory.rss
        metrics["vms_bytes"] = memory.vms
    except _RESOURCE_ERRORS as error:
        snapshot.issues.add(f"memory: {type(error).__name__}")
    try:
        metrics["thread_count"] = process.num_threads()
    except _RESOURCE_ERRORS as error:
        snapshot.issues.add(f"threads: {type(error).__name__}")
    # Discard counters if the process exited or its PID was reused mid-read.
    _require_live(process)
    snapshot.processes[identity] = metrics


def _take_snapshot(identity: ProcessIdentity) -> _Snapshot:
    snapshot = _Snapshot()
    try:
        root = psutil.Process(identity.pid)
        if root.create_time() != identity.create_time:
            snapshot.issues.add("root PID was reused")
            snapshot.tree_complete = False
            return snapshot
        _require_live(root)
        processes = [root, *root.children(recursive=True)]
        for process in processes:
            try:
                _read_process(process, snapshot)
            except _RESOURCE_ERRORS as error:
                snapshot.tree_complete = False
                snapshot.issues.add(f"process: {type(error).__name__}")
        _require_live(root)
    except _RESOURCE_ERRORS as error:
        snapshot.tree_complete = False
        snapshot.issues.add(f"process tree: {type(error).__name__}")
    finally:
        snapshot.sampled_at = time.monotonic()
    return snapshot


def _summarize(
    before: _Snapshot, after: _Snapshot
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "sample_seconds": after.sampled_at - before.sampled_at,
    }
    issues = before.issues | after.issues
    # Never present a known-incomplete subtotal as the process-tree total.
    if after.tree_complete and after.processes:
        result["process_count"] = len(after.processes)
        for metric in ("rss_bytes", "vms_bytes", "thread_count"):
            if all(metric in values for values in after.processes.values()):
                result[metric] = sum(
                    values[metric] for values in after.processes.values()
                )

        if before.processes.keys() != after.processes.keys():
            issues.add("process tree changed during sampling; CPU omitted")
        elif before.tree_complete:
            percentages = []
            for identity, current in after.processes.items():
                previous = before.processes[identity]
                if (
                    "cpu_seconds" not in current
                    or "cpu_seconds" not in previous
                ):
                    break
                elapsed = current["cpu_sampled_at"] - previous["cpu_sampled_at"]
                used = current["cpu_seconds"] - previous["cpu_seconds"]
                if elapsed <= 0 or used < 0:
                    issues.add("invalid CPU counter interval; CPU omitted")
                    break
                percentages.append(100.0 * used / elapsed)
            else:
                result["cpu_percent"] = sum(percentages)

    metrics_available = any(
        metric in result
        for metric in ("cpu_percent", "rss_bytes", "vms_bytes", "thread_count")
    )
    result["status"] = (
        ("partial" if issues else "running")
        if metrics_available
        else "unavailable"
    )
    if issues:
        result["reason"] = "; ".join(sorted(issues))
    return result


async def collect_resources(
    identity: ProcessIdentity,
) -> dict[str, int | float | str]:
    """Sample a root and recursive descendants without terminal input.

    All psutil inspection runs off the event loop. Calls have independent CPU
    baselines, so concurrent requests do not interfere. CPU uses user+system
    deltas for the same processes in both snapshots (100% is one logical CPU).
    Memory and thread counts describe the second snapshot. Missing metrics are
    omitted, not zero-filled; known-incomplete tree totals are also omitted.
    """
    if identity.create_time is None:
        return {
            "status": "unavailable",
            "reason": "process creation time was unavailable at launch",
        }
    before = await asyncio.to_thread(_take_snapshot, identity)
    await asyncio.sleep(RESOURCE_SAMPLE_SECONDS)
    after = await asyncio.to_thread(_take_snapshot, identity)
    return _summarize(before, after)
