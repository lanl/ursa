"""Contract tests for best-effort terminal resource sampling."""

from types import SimpleNamespace

import psutil
import pytest

from ursa.tools.terminal import resources
from ursa.tools.terminal.resources import ProcessIdentity, _Snapshot

ROOT = ProcessIdentity(123, 10.0)
CHILD = ProcessIdentity(456, 20.0)


def snapshot(*ids, at=1.0, cpu=2.0):
    return _Snapshot(
        sampled_at=at,
        processes={
            identity: {
                "cpu_seconds": cpu,
                "cpu_sampled_at": at,
                "rss_bytes": 100,
                "vms_bytes": 200,
                "thread_count": 3,
            }
            for identity in ids
        },
    )


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ((ROOT, CHILD), (ROOT, CHILD), "running"),
        ((ROOT,), (ROOT, CHILD), "partial"),
        ((ROOT, CHILD), (ROOT,), "partial"),
        ((ROOT, CHILD), (ROOT, ProcessIdentity(456, 30.0)), "partial"),
    ],
)
def test_summary_tracks_complete_trees_and_marks_tree_churn(
    before, after, expected
):
    result = resources._summarize(snapshot(*before), snapshot(*after, at=1.2))
    assert result["status"] == expected
    assert result["process_count"] == len(after)
    if expected == "running":
        assert result["cpu_percent"] == 0.0
        assert result["rss_bytes"] == 200
    else:
        assert "cpu_percent" not in result


@pytest.mark.parametrize("corruption", ["metric", "cpu_interval", "tree"])
def test_summary_keeps_only_reliable_metrics(corruption):
    before, after = snapshot(ROOT), snapshot(ROOT, at=1.2)
    if corruption == "metric":
        del after.processes[ROOT]["rss_bytes"]
        after.issues.add("permission denied")
    elif corruption == "cpu_interval":
        after.processes[ROOT]["cpu_sampled_at"] = 1.0
    else:
        after.tree_complete = False
        after.issues.add("process tree: AccessDenied")
    result = resources._summarize(before, after)
    assert result["status"] in {"partial", "unavailable"}
    if corruption != "metric":
        assert "cpu_percent" not in result


class FakeProcess:
    def __init__(self, pid=123, created=10.0):
        self.pid, self.created, self.descendants = pid, created, []
        self.state, self.alive, self.memory_error, self.tree_error = (
            psutil.STATUS_RUNNING,
            True,
            None,
            None,
        )

    def create_time(self):
        return self.created

    def is_running(self):
        return self.alive

    def status(self):
        return self.state

    def children(self, *, recursive):
        assert recursive
        if self.tree_error:
            raise self.tree_error
        return self.descendants

    def cpu_times(self):
        return SimpleNamespace(user=2.0, system=3.0)

    def memory_info(self):
        if self.memory_error:
            raise self.memory_error
        return SimpleNamespace(rss=100, vms=200)

    def num_threads(self):
        return 3


@pytest.fixture
def root(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(resources.psutil, "Process", lambda _pid: process)
    return process


@pytest.mark.parametrize(
    ("change", "complete", "reason"),
    [
        ("child", True, None),
        ("denied", False, "process tree: AccessDenied"),
        ("reused", False, "root PID was reused"),
        ("zombie", False, "process tree: ZombieProcess"),
    ],
)
def test_snapshot_handles_children_identity_and_process_errors(
    root, change, complete, reason
):
    if change == "child":
        root.descendants = [FakeProcess(CHILD.pid, CHILD.create_time)]
    elif change == "denied":
        root.tree_error = psutil.AccessDenied(root.pid)
    elif change == "reused":
        root.created += 1
    else:
        root.state = psutil.STATUS_ZOMBIE
    result = resources._take_snapshot(ROOT)
    assert result.tree_complete is complete
    if complete:
        assert set(result.processes) == {ROOT, CHILD}
    else:
        assert reason in result.issues


def test_snapshot_preserves_independent_metrics_when_one_probe_fails(root):
    root.memory_error = psutil.AccessDenied(root.pid)
    result = resources._take_snapshot(ROOT)
    assert "rss_bytes" not in result.processes[ROOT]
    assert result.processes[ROOT]["thread_count"] == 3
