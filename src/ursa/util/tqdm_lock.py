"""Make ``tqdm`` safe to use under stdio redirection (e.g. Textual's TUI).

Background
----------
When URSA runs inside the Textual TUI, Textual redirects ``sys.stderr`` to a
proxy whose ``fileno()`` returns ``-1`` (an intentionally invalid descriptor).

``tqdm`` lazily builds a *global write lock* the first time a progress bar is
constructed. On POSIX that lock is a ``multiprocessing.RLock``, whose creation
starts the ``multiprocessing.resource_tracker``. The resource tracker spawns a
helper process and passes ``sys.stderr.fileno()`` to
``_posixsubprocess.fork_exec`` as a descriptor to keep open. With the Textual
proxy installed, that descriptor is ``-1``, so CPython raises::

    ValueError: bad value(s) in fds_to_keep

Worse, ``tqdm``'s ``TqdmDefaultWriteLock.__init__`` acquires a process-wide,
class-level threading lock *before* creating the multiprocessing lock and only
releases it afterwards. Because ``create_mp_lock`` catches only
``(ImportError, OSError)`` -- not ``ValueError`` -- the exception propagates and
the release never runs. The leaked lock then deadlocks any *other* thread that
subsequently uses ``tqdm`` (e.g. a fresh asyncio executor thread), which is why
a first RAG-as-tool call surfaces the error but a second call hangs the TUI.

Fix
---
Install a pure-threading global lock via ``tqdm.set_lock(TRLock())``. This
never creates a multiprocessing lock, so the resource tracker is never started
and no descriptor is ever passed to ``fork_exec``. ``tqdm`` remains thread-safe
(URSA only uses threads, never process pools, around progress bars). Applying
this lock also *recovers* an already-poisoned lock state.
"""

from __future__ import annotations


def install_thread_only_tqdm_lock() -> None:
    """Force ``tqdm`` to use a thread-only global lock.

    Safe to call multiple times and safe to call even if ``tqdm`` is not
    installed. Prevents the ``bad value(s) in fds_to_keep`` error (and the
    follow-on deadlock) that occurs when ``tqdm`` builds its default
    multiprocessing lock while ``sys.stderr`` has been redirected to a stream
    whose ``fileno()`` is invalid (as Textual's TUI does).
    """
    try:
        from tqdm import tqdm
        from tqdm.std import TRLock
    except Exception:  # noqa: BLE001 - tqdm optional / import issues are non-fatal
        return

    tqdm.set_lock(TRLock())
