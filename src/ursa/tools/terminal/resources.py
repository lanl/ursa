"""Best-effort, read-only resource sampling for a local process tree."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import psutil

RESOURCE_SAMPLE_SECONDS = 0.2
_ERRORS = (psutil.Error, NotImplementedError)
_TOTALS = ("rss_bytes", "vms_bytes", "thread_count")


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """A PID tied to creation time, preventing accidental PID-reuse sampling."""

    pid: int
    create_time: float | None


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Capture identity without making failed inspection a launch failure."""
    try:
        created = psutil.Process(pid).create_time()
    except _ERRORS:
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


def _live(process: psutil.Process) -> None:
    # is_running checks creation time too; create_time alone may be cached.
    if not process.is_running():
        raise psutil.NoSuchProcess(process.pid)
    if process.status() in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}:
        raise psutil.ZombieProcess(process.pid)


def _cpu_metrics(process: psutil.Process) -> dict[str, float]:
    cpu = process.cpu_times()
    return {
        "cpu_seconds": cpu.user + cpu.system,
        "cpu_sampled_at": time.monotonic(),
    }


def _read_process(process: psutil.Process, snapshot: _Snapshot) -> None:
    _live(process)
    values: dict[str, int | float] = {}
    readers = (
        (
            "cpu",
            lambda: _cpu_metrics(process),
        ),
        (
            "memory",
            lambda: {
                "rss_bytes": process.memory_info().rss,
                "vms_bytes": process.memory_info().vms,
            },
        ),
        ("threads", lambda: {"thread_count": process.num_threads()}),
    )
    for name, read in readers:
        try:
            values.update(read())
        except _ERRORS as error:
            snapshot.issues.add(f"{name}: {type(error).__name__}")
    _live(process)  # discard counters for an exited/reused PID.
    snapshot.processes[ProcessIdentity(process.pid, process.create_time())] = (
        values
    )


def _take_snapshot(identity: ProcessIdentity) -> _Snapshot:
    snapshot = _Snapshot()
    try:
        root = psutil.Process(identity.pid)
        if root.create_time() != identity.create_time:
            snapshot.issues.add("root PID was reused")
            snapshot.tree_complete = False
            return snapshot
        _live(root)
        for process in (root, *root.children(recursive=True)):
            try:
                _read_process(process, snapshot)
            except _ERRORS as error:
                snapshot.tree_complete = False
                snapshot.issues.add(f"process: {type(error).__name__}")
        _live(root)
    except _ERRORS as error:
        snapshot.tree_complete = False
        snapshot.issues.add(f"process tree: {type(error).__name__}")
    finally:
        snapshot.sampled_at = time.monotonic()
    return snapshot


def _summarize(
    before: _Snapshot, after: _Snapshot
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {
        "sample_seconds": after.sampled_at - before.sampled_at
    }
    issues = before.issues | after.issues
    if after.tree_complete and after.processes:
        result["process_count"] = len(after.processes)
        for name in _TOTALS:
            if all(name in values for values in after.processes.values()):
                result[name] = sum(
                    values[name] for values in after.processes.values()
                )
        if before.processes.keys() != after.processes.keys():
            issues.add("process tree changed during sampling; CPU omitted")
        elif before.tree_complete:
            rates: list[float] = []
            for identity, current in after.processes.items():
                previous = before.processes[identity]
                if (
                    not {"cpu_seconds", "cpu_sampled_at"}
                    <= current.keys() | previous.keys()
                ):
                    break
                elapsed = current["cpu_sampled_at"] - previous["cpu_sampled_at"]
                used = current["cpu_seconds"] - previous["cpu_seconds"]
                if elapsed <= 0 or used < 0:
                    issues.add("invalid CPU counter interval; CPU omitted")
                    break
                rates.append(100.0 * used / elapsed)
            else:
                result["cpu_percent"] = sum(rates)
    available = any(name in result for name in (*_TOTALS, "cpu_percent"))
    result["status"] = (
        ("partial" if issues else "running") if available else "unavailable"
    )
    if issues:
        result["reason"] = "; ".join(sorted(issues))
    return result


async def collect_resources(
    identity: ProcessIdentity,
) -> dict[str, int | float | str]:
    """Sample a root and descendants, keeping CPU baselines request-local."""
    if identity.create_time is None:
        return {
            "status": "unavailable",
            "reason": "process creation time was unavailable at launch",
        }
    before = await asyncio.to_thread(_take_snapshot, identity)
    await asyncio.sleep(RESOURCE_SAMPLE_SECONDS)
    return _summarize(before, await asyncio.to_thread(_take_snapshot, identity))
