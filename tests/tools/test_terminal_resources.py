"""Best-effort process-tree resource sampling and identity safety."""

import asyncio
import threading
from types import SimpleNamespace

import psutil
import pytest

from ursa.tools.terminal import resources
from ursa.tools.terminal.resources import ProcessIdentity, _Snapshot

ROOT = ProcessIdentity(123, 10.0)
CHILD = ProcessIdentity(456, 20.0)


def snapshot(*identities, at=1.0, cpu=2.0):
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
            for identity in identities
        },
    )


def test_tree_totals_and_cpu_above_one_core():
    before = snapshot(ROOT, CHILD, at=1.0)
    after = snapshot(ROOT, CHILD, at=1.2, cpu=2.15)
    result = resources._summarize(before, after)
    assert result["status"] == "running"
    assert result["sample_seconds"] == pytest.approx(0.2)
    assert result["cpu_percent"] == pytest.approx(150.0)
    assert result["rss_bytes"] == 200
    assert result["vms_bytes"] == 400
    assert result["process_count"] == 2
    assert result["thread_count"] == 6


def test_cpu_uses_each_process_reading_interval():
    before = snapshot(ROOT, CHILD, at=1.0)
    after = snapshot(ROOT, CHILD, at=1.2, cpu=2.1)
    after.processes[CHILD]["cpu_sampled_at"] = 1.4
    result = resources._summarize(before, after)
    assert result["cpu_percent"] == pytest.approx(75.0)


def test_idle_cpu_is_a_real_zero():
    result = resources._summarize(
        snapshot(ROOT, at=1.0), snapshot(ROOT, at=1.2)
    )
    assert result["cpu_percent"] == 0.0
    assert result["status"] == "running"


@pytest.mark.parametrize(
    ("before_ids", "after_ids"),
    [
        ((ROOT,), (ROOT, CHILD)),
        ((ROOT, CHILD), (ROOT,)),
        ((ROOT, CHILD), (ROOT, ProcessIdentity(CHILD.pid, 30.0))),
    ],
)
def test_tree_churn_including_child_pid_reuse_omits_cpu(before_ids, after_ids):
    result = resources._summarize(
        snapshot(*before_ids), snapshot(*after_ids, at=1.2)
    )
    assert result["status"] == "partial"
    assert "cpu_percent" not in result
    assert "tree changed" in result["reason"]
    assert result["process_count"] == len(after_ids)
    assert result["rss_bytes"] == 100 * len(after_ids)


@pytest.mark.parametrize("metric", ["rss_bytes", "vms_bytes", "thread_count"])
def test_incomplete_metric_is_not_reported_as_a_tree_total(metric):
    before = snapshot(ROOT, CHILD)
    after = snapshot(ROOT, CHILD, at=1.2)
    del after.processes[CHILD][metric]
    after.issues.add("permission denied")
    result = resources._summarize(before, after)
    assert result["status"] == "partial"
    assert metric not in result
    assert result["cpu_percent"] == 0.0


def test_missing_cpu_does_not_prevent_memory_totals():
    before = snapshot(ROOT)
    del before.processes[ROOT]["cpu_seconds"]
    before.issues.add("cpu: AccessDenied")
    result = resources._summarize(before, snapshot(ROOT, at=1.2))
    assert result["status"] == "partial"
    assert "cpu_percent" not in result
    assert result["rss_bytes"] == 100


@pytest.mark.parametrize(("at", "cpu"), [(1.0, 2.0), (1.2, 1.0)])
def test_invalid_cpu_interval_is_not_clamped_to_zero(at, cpu):
    result = resources._summarize(
        snapshot(ROOT), snapshot(ROOT, at=at, cpu=cpu)
    )
    assert result["status"] == "partial"
    assert "cpu_percent" not in result
    assert "invalid CPU" in result["reason"]


def test_incomplete_tree_does_not_report_subtotals():
    after = snapshot(ROOT, at=1.2)
    after.tree_complete = False
    after.issues.add("process tree: AccessDenied")
    result = resources._summarize(snapshot(ROOT), after)
    assert result == {
        "sample_seconds": pytest.approx(0.2),
        "status": "unavailable",
        "reason": "process tree: AccessDenied",
    }


def test_process_count_alone_is_not_available_usage():
    before = _Snapshot(sampled_at=1.0, processes={ROOT: {}})
    after = _Snapshot(sampled_at=1.2, processes={ROOT: {}})
    after.issues.add("metrics unsupported")
    result = resources._summarize(before, after)
    assert result["status"] == "unavailable"
    assert result["process_count"] == 1
    assert "rss_bytes" not in result
    assert "cpu_percent" not in result


class FakeProcess:
    def __init__(self, pid=ROOT.pid, created=ROOT.create_time):
        self.pid = pid
        self.created = created
        self.descendants = []
        self.alive = True
        self.state = psutil.STATUS_RUNNING
        self.memory_error = None
        self.tree_error = None

    def create_time(self):
        return self.created

    def is_running(self):
        return self.alive

    def status(self):
        return self.state

    def children(self, *, recursive):
        assert recursive is True
        if self.tree_error:
            raise self.tree_error
        return self.descendants

    def cpu_times(self):
        return SimpleNamespace(
            user=2.0, system=3.0, children_user=100.0, children_system=200.0
        )

    def memory_info(self):
        if self.memory_error:
            raise self.memory_error
        return SimpleNamespace(rss=100, vms=200)

    def num_threads(self):
        return 3


@pytest.fixture
def root(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(resources.psutil, "Process", lambda pid: process)
    return process


def test_snapshot_reads_recursive_children_without_waited_cpu(root):
    root.descendants = [FakeProcess(CHILD.pid, CHILD.create_time)]
    result = resources._take_snapshot(ROOT)
    assert result.tree_complete
    assert not result.issues
    assert set(result.processes) == {ROOT, CHILD}
    assert result.processes[ROOT]["cpu_seconds"] == 5.0
    assert result.processes[CHILD]["cpu_seconds"] == 5.0


@pytest.mark.parametrize(
    "error", [psutil.AccessDenied(123), NotImplementedError()]
)
def test_metric_inspection_errors_preserve_independent_metrics(root, error):
    root.memory_error = error
    result = resources._take_snapshot(ROOT)
    assert result.tree_complete
    assert f"memory: {type(error).__name__}" in result.issues
    assert "rss_bytes" not in result.processes[ROOT]
    assert result.processes[ROOT]["thread_count"] == 3
    assert result.processes[ROOT]["cpu_seconds"] == 5.0


def test_denied_tree_discovery_does_not_become_shell_only_totals(root):
    root.tree_error = psutil.AccessDenied(root.pid)
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert result.processes == {}
    assert "process tree: AccessDenied" in result.issues


def test_reused_root_is_never_sampled(root, monkeypatch):
    root.created += 1

    def unexpected(*args, **kwargs):
        pytest.fail("must not inspect the reused root")

    monkeypatch.setattr(root, "children", unexpected)
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert result.processes == {}
    assert result.issues == {"root PID was reused"}


@pytest.mark.parametrize("state", [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD])
def test_zombie_is_not_live_usage(root, state):
    root.state = state
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert result.processes == {}
    assert "process tree: ZombieProcess" in result.issues


def test_disappeared_root_is_unavailable(monkeypatch):
    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(resources.psutil, "Process", missing)
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert result.processes == {}
    assert "process tree: NoSuchProcess" in result.issues


def test_child_reused_or_exited_during_read_discards_counters(
    root, monkeypatch
):
    child = FakeProcess(CHILD.pid, CHILD.create_time)
    root.descendants = [child]

    def threads():
        # psutil.is_running() becomes false after an exit OR identity mismatch.
        child.alive = False
        return 3

    monkeypatch.setattr(child, "num_threads", threads)
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert set(result.processes) == {ROOT}
    assert "process: NoSuchProcess" in result.issues


def test_root_exit_during_tree_read_invalidates_totals(root, monkeypatch):
    child = FakeProcess(CHILD.pid, CHILD.create_time)
    root.descendants = [child]

    def threads():
        root.alive = False
        return 3

    monkeypatch.setattr(child, "num_threads", threads)
    result = resources._take_snapshot(ROOT)
    assert not result.tree_complete
    assert "process tree: NoSuchProcess" in result.issues


def test_capture_identity(root):
    assert resources.capture_process_identity(ROOT.pid) == ROOT


@pytest.mark.parametrize(
    "error", [psutil.AccessDenied(123), NotImplementedError()]
)
async def test_failed_launch_identity_is_not_recaptured(monkeypatch, error):
    calls = []

    def denied(pid):
        calls.append(pid)
        raise error

    monkeypatch.setattr(resources.psutil, "Process", denied)
    identity = resources.capture_process_identity(ROOT.pid)
    assert identity == ProcessIdentity(ROOT.pid, None)
    result = await resources.collect_resources(identity)
    assert calls == [ROOT.pid]
    assert result["status"] == "unavailable"
    assert "creation time" in result["reason"]
    assert "cpu_percent" not in result


async def test_collection_uses_two_worker_snapshots_and_async_delay(
    monkeypatch,
):
    loop_thread = threading.get_ident()
    threads = []
    delays = []
    samples = iter([snapshot(ROOT), snapshot(ROOT, at=1.2, cpu=2.1)])

    def take(identity):
        assert identity == ROOT
        threads.append(threading.get_ident())
        return next(samples)

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(resources, "_take_snapshot", take)
    monkeypatch.setattr(resources.asyncio, "sleep", sleep)
    result = await resources.collect_resources(ROOT)
    assert delays == [0.2]
    assert len(threads) == 2
    assert all(thread != loop_thread for thread in threads)
    assert result["cpu_percent"] == pytest.approx(50.0)


async def test_inspection_does_not_block_event_loop_and_can_be_cancelled(
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def take(identity):
        calls.append(identity)
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(timeout=5)
            return snapshot(ROOT)
        finally:
            finished.set()

    monkeypatch.setattr(resources, "_take_snapshot", take)
    task = asyncio.create_task(resources.collect_resources(ROOT))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
    assert calls == [ROOT]


async def test_concurrent_collection_has_independent_cpu_baselines(monkeypatch):
    lock = threading.Lock()
    tick = 0

    def take(identity):
        nonlocal tick
        with lock:
            tick += 1
            return snapshot(identity, at=float(tick), cpu=2.0 * tick)

    monkeypatch.setattr(resources, "_take_snapshot", take)
    monkeypatch.setattr(resources, "RESOURCE_SAMPLE_SECONDS", 0.01)
    first, second = await asyncio.gather(
        resources.collect_resources(ROOT), resources.collect_resources(ROOT)
    )
    assert tick == 4
    for result in (first, second):
        assert result["status"] == "running"
        assert result["cpu_percent"] == 200.0
