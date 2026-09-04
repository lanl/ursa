"""Regression tests for the tqdm/Textual ``bad value(s) in fds_to_keep`` bug.

See ``ursa.util.tqdm_lock`` for the full mechanism. In short: under Textual's
TUI, ``sys.stderr`` is a proxy whose ``fileno()`` is invalid (-1). The first
``tqdm`` progress bar builds a ``multiprocessing.RLock``, which starts the
resource tracker and passes ``sys.stderr.fileno()`` (== -1) to
``_posixsubprocess.fork_exec`` -> ``ValueError: bad value(s) in fds_to_keep``.
The failure also leaks tqdm's process-wide threading lock, deadlocking any
subsequent tqdm use on another thread.

``install_thread_only_tqdm_lock`` installs a pure-threading global lock so the
multiprocessing lock (and resource tracker) is never created.
"""

import subprocess
import sys
import textwrap

import pytest

from ursa.util.tqdm_lock import install_thread_only_tqdm_lock

# The scenario relies on POSIX ``fork_exec`` fd validation used by the
# multiprocessing resource tracker; it does not apply on Windows.
posix_only = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="resource_tracker fd validation is POSIX-specific",
)


def test_install_thread_only_tqdm_lock_sets_thread_lock():
    """After install, tqdm's global lock is a pure-threading lock.

    ``tqdm.std.TRLock`` is a factory that returns a ``_thread.RLock`` (a plain
    threading reentrant lock), never a ``multiprocessing`` lock. Installing it
    means tqdm never calls ``create_mp_lock`` / starts the resource tracker,
    which is what triggers the ``fds_to_keep`` failure under redirected stderr.
    """
    from tqdm import tqdm

    install_thread_only_tqdm_lock()
    lock = tqdm.get_lock()
    # A usable reentrant threading lock: acquires and releases without spawning.
    assert hasattr(lock, "acquire") and hasattr(lock, "release")
    assert lock.acquire(blocking=False) is True
    lock.release()
    # It is NOT a multiprocessing synchronization primitive.
    assert type(lock).__module__ not in {
        "multiprocessing.synchronize",
        "multiprocessing",
    }


def test_install_thread_only_tqdm_lock_is_idempotent_and_safe():
    """Calling it repeatedly is safe and keeps a thread-only lock."""
    from tqdm import tqdm

    install_thread_only_tqdm_lock()
    first = tqdm.get_lock()
    install_thread_only_tqdm_lock()
    second = tqdm.get_lock()
    for lock in (first, second):
        assert type(lock).__module__ not in {
            "multiprocessing.synchronize",
            "multiprocessing",
        }


# Shared preamble: install a Textual-like stderr proxy (fileno() == -1) and a
# fresh resource tracker so a multiprocessing lock would actually spawn.
_PREAMBLE = """
import sys, threading
from multiprocessing import resource_tracker

class TextualStderrProxy:
    def write(self, t): return len(t)
    def flush(self): pass
    def isatty(self): return True
    def fileno(self): return -1  # mirrors textual/app.py invalid fileno

resource_tracker._resource_tracker._fd = None
resource_tracker._resource_tracker._pid = None
sys.stderr = TextualStderrProxy()
"""

_SCENARIO = """
res = {}
def call(tag):
    try:
        from tqdm import tqdm
        for _ in tqdm([1, 2], desc=tag, file=sys.stderr):
            pass
        res[tag] = "OK"
    except Exception as e:
        res[tag] = f"{type(e).__name__}: {e}"

# First call in its own thread; second in a DIFFERENT thread (a leaked tqdm
# lock would deadlock this one).
a = threading.Thread(target=call, args=("first",)); a.start(); a.join(timeout=15)
b = threading.Thread(target=call, args=("second",), daemon=True)
b.start(); b.join(timeout=8)

print("first=" + str(res.get("first")))
print("second=" + str(res.get("second", "*** DEADLOCK ***")))
print("deadlocked=" + str(b.is_alive()))
sys.stdout.flush()
import os
os._exit(0)  # bypass interpreter-shutdown join, which is itself part of the bug
"""


def _run_scenario(fix: bool) -> dict[str, str]:
    fix_stmt = (
        "from ursa.util.tqdm_lock import install_thread_only_tqdm_lock\n"
        "install_thread_only_tqdm_lock()\n"
        if fix
        else ""
    )
    # The fix must be applied BEFORE stderr is replaced (like the real TUI).
    script = fix_stmt + _PREAMBLE + _SCENARIO
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            out[key] = val
    return out


@posix_only
def test_bug_reproduces_without_fix():
    """Guard is meaningful: without the fix, the bug actually occurs.

    This makes the fix test below a genuine regression guard rather than a
    tautology. If the environment ever stops reproducing the bug, this test
    fails loudly so the fix test can be re-evaluated.
    """
    result = _run_scenario(fix=False)
    assert "fds_to_keep" in result.get("first", ""), result


@posix_only
def test_fix_prevents_fds_error_and_deadlock():
    """With the fix, both tqdm calls succeed and no thread deadlocks."""
    result = _run_scenario(fix=True)
    assert result.get("first") == "OK", result
    assert result.get("second") == "OK", result
    assert result.get("deadlocked") == "False", result
