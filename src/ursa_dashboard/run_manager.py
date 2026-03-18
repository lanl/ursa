from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .artifacts import scan_artifacts
from .events import make_event
from .retention import RetentionPolicy, enforce_retention
from .security import safe_join, workspace_root_from_env
from .storage import (
    RunPaths,
    append_jsonl,
    ensure_dirs,
    file_size,
    read_json,
    utc_now,
    write_json,
)
from .ulid import new_ulid

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


@dataclass
class RunConfig:
    concurrency: int = 2
    stdout_cap_bytes: int = 25 * 1024 * 1024
    stderr_cap_bytes: int = 25 * 1024 * 1024
    events_cap_bytes: int = 50 * 1024 * 1024
    poll_interval_s: float = 0.25


@dataclass
class InFlight:
    run_id: str
    agent_id: str
    proc: asyncio.subprocess.Process
    cancel_requested: bool = False


class RunManager:
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        config: RunConfig | None = None,
        retention: RetentionPolicy | None = None,
    ):
        self.workspace_root = workspace_root or workspace_root_from_env()
        self.config = config or RunConfig()
        self.retention = retention or RetentionPolicy()

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "_meta" / "runs").mkdir(
            parents=True, exist_ok=True
        )

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._inflight: dict[str, InFlight] = {}
        self._lock = asyncio.Lock()
        self._seq_lock = asyncio.Lock()
        self._seq: dict[str, int] = {}
        self._started = False
        self._workers: list[asyncio.Task] = []

        # Lightweight index for listing (append-only)
        self._index_path = self.workspace_root / "_meta" / "runs_index.jsonl"

    # ----------------------------
    # Paths and persistence
    # ----------------------------

    def _run_paths(self, *, agent_id: str, run_id: str) -> RunPaths:
        run_dir = self.workspace_root / "runs" / agent_id / run_id
        logs_dir = run_dir / "logs"
        artifacts_dir = run_dir / "artifacts"
        meta_path = self.workspace_root / "_meta" / "runs" / f"{run_id}.json"
        return RunPaths(
            run_dir=run_dir,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            meta_path=meta_path,
            stdout_path=logs_dir / "stdout.log",
            stderr_path=logs_dir / "stderr.log",
            events_path=logs_dir / "events.jsonl",
            artifacts_manifest_path=artifacts_dir / "artifacts.json",
        )

    def _read_run(self, run_id: str) -> dict[str, Any]:
        meta_path = self.workspace_root / "_meta" / "runs" / f"{run_id}.json"
        return read_json(meta_path)

    def _write_run(self, run_id: str, rec: dict[str, Any]) -> None:
        meta_path = self.workspace_root / "_meta" / "runs" / f"{run_id}.json"
        write_json(meta_path, rec)

    async def _emit(
        self,
        *,
        run_id: str,
        agent_id: str,
        type: str,
        payload: dict[str, Any],
        level: str = "info",
    ) -> None:
        """Append an event to events.jsonl with a monotonic per-run seq."""
        paths = self._run_paths(agent_id=agent_id, run_id=run_id)

        async with self._seq_lock:
            last = self._seq.get(run_id, 0)
            seq = last + 1
            self._seq[run_id] = seq

        ev = make_event(
            run_id=run_id,
            agent_id=agent_id,
            seq=seq,
            type=type,
            payload=payload,
            level=level,
        )

        # If events file is too large, suppress high-volume log events but keep critical events.
        if (
            type == "log"
            and file_size(paths.events_path) >= self.config.events_cap_bytes
        ):
            return
        append_jsonl(paths.events_path, ev)

    # ----------------------------
    # Public API
    # ----------------------------

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        # Enforce retention once at startup
        enforce_retention(
            workspace_root=self.workspace_root, policy=self.retention
        )

        # Recovery: if the dashboard restarts, mark in-progress runs as failed,
        # and requeue runs that were still queued.
        await self._recover_runs()

        for i in range(self.config.concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))

    async def shutdown(self) -> None:
        # Best-effort cancel workers
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _recover_runs(self) -> None:
        meta_dir = self.workspace_root / "_meta" / "runs"
        if not meta_dir.exists():
            return

        for meta_path in meta_dir.glob("*.json"):
            try:
                rec = read_json(meta_path)
            except Exception:
                continue

            run_id = rec.get("run_id")
            agent_id = rec.get("agent_id")
            if not run_id or not agent_id:
                continue

            paths = self._run_paths(agent_id=agent_id, run_id=run_id)
            last_seq = self._load_last_seq(paths.events_path)
            async with self._seq_lock:
                self._seq.setdefault(run_id, last_seq)

            status = rec.get("status")
            if status == "queued":
                # Requeue
                await self._queue.put(run_id)
            elif status in {"starting", "running", "cancelling"}:
                # Mark failed due to restart
                prev = status
                rec["status"] = "failed"
                rec["finished_at"] = utc_now()
                rec["error"] = {
                    "error_type": "DashboardRestart",
                    "message": "Dashboard restarted during run",
                }
                self._write_run(run_id, rec)
                await self._emit(
                    run_id=run_id,
                    agent_id=agent_id,
                    type="state_change",
                    payload={
                        "from": prev,
                        "to": "failed",
                        "reason": "dashboard_restart",
                    },
                    level="warn",
                )

    async def create_run(
        self,
        *,
        agent_id: str,
        params: dict[str, Any],
        agent_init: dict[str, Any],
        llm: dict[str, Any],
        runner: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = new_ulid()
        paths = self._run_paths(agent_id=agent_id, run_id=run_id)
        ensure_dirs(paths)

        rec: dict[str, Any] = {
            "run_id": run_id,
            "agent_id": agent_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "cancel_requested_at": None,
            "cancelled_at": None,
            "queue_reason": "user_request",
            "params": params,
            "agent_init": agent_init,
            "llm": llm,
            "runner": runner,
            "run_dir": str(
                paths.run_dir.relative_to(self.workspace_root)
            ).replace("\\", "/"),
            "logs": {
                "stdout": str(
                    paths.stdout_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "stderr": str(
                    paths.stderr_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "events": str(
                    paths.events_path.relative_to(self.workspace_root)
                ).replace("\\", "/"),
            },
            "artifacts": {
                "dir": str(
                    paths.artifacts_dir.relative_to(self.workspace_root)
                ).replace("\\", "/"),
                "manifest": str(
                    paths.artifacts_manifest_path.relative_to(
                        self.workspace_root
                    )
                ).replace("\\", "/"),
            },
            "result": None,
            "error": None,
            "runtime": {"pid": None},
        }
        if extra:
            rec.update(extra)

        self._write_run(run_id, rec)
        append_jsonl(
            self._index_path,
            {"ts": rec["created_at"], "run_id": run_id, "agent_id": agent_id},
        )

        # Initialize per-run event sequence.
        async with self._seq_lock:
            self._seq[run_id] = 0

        # Initial state event
        await self._emit(
            run_id=run_id,
            agent_id=agent_id,
            type="state_change",
            payload={"from": None, "to": "queued", "reason": "created"},
        )

        await self._queue.put(run_id)
        return rec

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return self._read_run(run_id)

    async def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        meta_dir = self.workspace_root / "_meta" / "runs"
        recs = []
        for p in meta_dir.glob("*.json"):
            try:
                recs.append(read_json(p))
            except Exception:
                continue
        recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return recs[:limit]

    async def cancel(
        self, run_id: str, *, reason: str = "user_request"
    ) -> dict[str, Any]:
        async with self._lock:
            rec = self._read_run(run_id)
            if rec["status"] in TERMINAL_STATUSES:
                return rec

            agent_id = rec["agent_id"]
            rec["cancel_requested_at"] = utc_now()
            prev = rec["status"]
            if prev == "queued":
                rec["status"] = "cancelled"
                rec["cancelled_at"] = utc_now()
                rec["finished_at"] = rec["cancelled_at"]
                self._write_run(run_id, rec)
                # state event
                await self._emit(
                    run_id=run_id,
                    agent_id=agent_id,
                    type="state_change",
                    payload={"from": prev, "to": "cancelled", "reason": reason},
                )
                return rec

            # running/starting
            rec["status"] = "cancelling"
            self._write_run(run_id, rec)
            await self._emit(
                run_id=run_id,
                agent_id=agent_id,
                type="state_change",
                payload={"from": prev, "to": "cancelling", "reason": reason},
            )

            inflight = self._inflight.get(run_id)
            if inflight:
                inflight.cancel_requested = True
                with contextlib.suppress(ProcessLookupError, OSError):
                    inflight.proc.send_signal(signal.SIGTERM)
                # Escalate to SIGKILL if it doesn't exit promptly.
                asyncio.create_task(self._kill_after(run_id, delay_s=5.0))
            return rec

    # ----------------------------
    # Event reading (for SSE)
    # ----------------------------

    def events_path(self, run_id: str) -> Path:
        rec = self._read_run(run_id)
        agent_id = rec["agent_id"]
        return self._run_paths(agent_id=agent_id, run_id=run_id).events_path

    def artifacts_manifest_path(self, run_id: str) -> Path:
        rec = self._read_run(run_id)
        agent_id = rec["agent_id"]
        return self._run_paths(
            agent_id=agent_id, run_id=run_id
        ).artifacts_manifest_path

    # ----------------------------
    # Internals
    # ----------------------------

    def _load_last_seq(self, events_path: Path) -> int:
        """Infer last seq from events.jsonl (best-effort)."""
        if not events_path.exists() or events_path.stat().st_size == 0:
            return 0
        try:
            with events_path.open("rb") as f:
                f.seek(-min(8192, events_path.stat().st_size), os.SEEK_END)
                tail = (
                    f.read()
                    .decode("utf-8", errors="ignore")
                    .strip()
                    .splitlines()
                )
            if not tail:
                return 0
            last = json.loads(tail[-1])
            return int(last.get("seq", 0))
        except Exception:
            return 0

    async def _kill_after(self, run_id: str, *, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        async with self._lock:
            inflight = self._inflight.get(run_id)
            if not inflight:
                return
            proc = inflight.proc
            if proc.returncode is not None:
                return
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()

    async def _worker_loop(self, worker_idx: int) -> None:
        while True:
            run_id = await self._queue.get()
            try:
                # In case the run was cancelled while queued.
                rec = self._read_run(run_id)
                if rec.get("status") == "cancelled":
                    continue
                await self._execute_run(run_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Best-effort: mark failed.
                try:
                    rec = self._read_run(run_id)
                    if rec.get("status") not in TERMINAL_STATUSES:
                        rec["status"] = "failed"
                        rec["finished_at"] = utc_now()
                        rec["error"] = {"message": "run manager failure"}
                        self._write_run(run_id, rec)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    async def _execute_run(self, run_id: str) -> None:
        async with self._lock:
            rec = self._read_run(run_id)
            agent_id = rec["agent_id"]
            paths = self._run_paths(agent_id=agent_id, run_id=run_id)

            prev = rec["status"]
            rec["status"] = "starting"
            rec["started_at"] = utc_now()
            self._write_run(run_id, rec)
            await self._emit(
                run_id=run_id,
                agent_id=agent_id,
                type="state_change",
                payload={"from": prev, "to": "starting", "reason": "dequeued"},
            )

        # Write config JSON blobs for the worker.
        params_json = paths.run_dir / "params.json"
        agent_init_json = paths.run_dir / "agent_init.json"
        llm_json = paths.run_dir / "llm.json"
        mcp_json = paths.run_dir / "mcp.json"
        output_json = paths.run_dir / "output.json"
        params_json.write_text(
            json.dumps(rec.get("params") or {}, indent=2), encoding="utf-8"
        )
        agent_init_json.write_text(
            json.dumps(rec.get("agent_init") or {}, indent=2), encoding="utf-8"
        )
        llm_json.write_text(
            json.dumps(rec.get("llm") or {}, indent=2), encoding="utf-8"
        )
        mcp_json.write_text(
            json.dumps(rec.get("mcp") or {}, indent=2), encoding="utf-8"
        )

        timeout = None
        runner_cfg = rec.get("runner") or {}
        if runner_cfg.get("timeout_seconds"):
            timeout = float(runner_cfg["timeout_seconds"])

        # Spawn worker subprocess
        workspace_dir_rel = rec.get("workspace_dir")
        agent_workspace_dir = paths.run_dir
        if workspace_dir_rel:
            try:
                agent_workspace_dir = safe_join(
                    self.workspace_root, str(workspace_dir_rel)
                )
            except Exception:
                # Fall back to per-run directory if misconfigured.
                agent_workspace_dir = paths.run_dir

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "ursa_dashboard.worker_main",
            "--agent-id",
            agent_id,
            "--run-id",
            run_id,
            "--workspace-dir",
            str(agent_workspace_dir),
            "--params-json",
            str(params_json),
            "--agent-init-json",
            str(agent_init_json),
            "--llm-json",
            str(llm_json),
            "--mcp-json",
            str(mcp_json),
            "--output-json",
            str(output_json),
        ]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure the worker can import `ursa_dashboard` even when the dashboard
        # is run from a source checkout (not installed as a package).
        project_root = str(Path(__file__).resolve().parent.parent)
        existing_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            project_root
            if not existing_pp
            else (project_root + os.pathsep + existing_pp)
        )

        # Strongly discourage interactive behavior (no stdin). For log readability,
        # prefer *plain* output when stdout/stderr are pipes (the default behavior of
        # rich/tqdm). Users can opt-in to forced ANSI output if they want.
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        force_no_ansi = str(
            os.environ.get("URSA_DASHBOARD_NO_ANSI", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not force_no_ansi:
            env.setdefault("COLORTERM", "truecolor")
            env.setdefault("FORCE_COLOR", "1")
            env.setdefault("CLICOLOR", "1")
            env.setdefault("RICH_FORCE_TERMINAL", "1")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(paths.run_dir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        async with self._lock:
            rec = self._read_run(run_id)
            prev = rec["status"]
            rec["status"] = "running"
            rec["runtime"]["pid"] = proc.pid
            self._write_run(run_id, rec)
            await self._emit(
                run_id=run_id,
                agent_id=agent_id,
                type="state_change",
                payload={
                    "from": prev,
                    "to": "running",
                    "reason": "process_spawned",
                },
            )
            self._inflight[run_id] = InFlight(
                run_id=run_id, agent_id=agent_id, proc=proc
            )

        # Stream stdout/stderr
        stdout_task = asyncio.create_task(
            self._drain_stream(
                run_id=run_id,
                agent_id=agent_id,
                stream_name="stdout",
                stream=proc.stdout,
                log_path=paths.stdout_path,
                cap_bytes=self.config.stdout_cap_bytes,
            )
        )
        stderr_task = asyncio.create_task(
            self._drain_stream(
                run_id=run_id,
                agent_id=agent_id,
                stream_name="stderr",
                stream=proc.stderr,
                log_path=paths.stderr_path,
                cap_bytes=self.config.stderr_cap_bytes,
            )
        )

        try:
            if timeout:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            else:
                await proc.wait()
        except asyncio.TimeoutError:
            # Timeout: terminate then kill.
            async with self._lock:
                rec = self._read_run(run_id)
                rec["error"] = {
                    "error_type": "Timeout",
                    "message": f"Timed out after {timeout}s",
                }
                self._write_run(run_id, rec)
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        finally:
            # Ensure streams drained
            await asyncio.gather(
                stdout_task, stderr_task, return_exceptions=True
            )

        rc = proc.returncode

        async with self._lock:
            self._inflight.pop(run_id, None)

        # Determine terminal status
        rec = self._read_run(run_id)
        if rec.get("status") == "cancelling":
            status = "cancelled"
        elif rc == 0:
            status = "succeeded"
        else:
            status = "failed"

        # Read output JSON from worker
        result_obj: dict[str, Any] | None = None
        if output_json.exists():
            try:
                result_obj = json.loads(output_json.read_text(encoding="utf-8"))
            except Exception:
                result_obj = None

        if result_obj and result_obj.get("text"):
            await self._emit(
                run_id=run_id,
                agent_id=agent_id,
                type="final_output",
                payload={
                    "content_type": result_obj.get(
                        "content_type", "text/markdown"
                    ),
                    "text": result_obj.get("text"),
                    "message_id": new_ulid(),
                },
                level="info" if status == "succeeded" else "error",
            )

        # If this run is part of a multi-turn session, append an assistant message
        # to the session transcript and update session pointers.
        session_id = rec.get("session_id")
        if session_id:
            try:
                from .sessions import append_message as _append_session_message
                from .sessions import update_session as _update_session

                if status == "cancelled":
                    assistant_text = "(cancelled)"
                elif result_obj and result_obj.get("text"):
                    assistant_text = str(result_obj.get("text"))
                elif status == "failed" and rec.get("error"):
                    assistant_text = (
                        f"(failed) {rec['error'].get('message') or ''}".strip()
                    )
                else:
                    assistant_text = f"({status})"

                _append_session_message(
                    self.workspace_root,
                    session_id=str(session_id),
                    role="assistant",
                    text=assistant_text,
                    run_id=run_id,
                )

                # Clear active run if this was it.
                sess_patch = {"last_run_id": run_id}
                try:
                    sess = read_json(
                        self.workspace_root
                        / "sessions"
                        / str(session_id)
                        / "session.json"
                    )
                    if sess.get("active_run_id") == run_id:
                        sess_patch["active_run_id"] = None
                except Exception:
                    sess_patch["active_run_id"] = None
                _update_session(
                    self.workspace_root, str(session_id), sess_patch
                )
            except Exception:
                pass

        # Build artifacts manifest
        manifest = scan_artifacts(
            paths.run_dir,
            exclude_dirs={"logs", "metrics", "agent_store", "__pycache__"},
        )
        paths.artifacts_manifest_path.write_text(
            json.dumps(
                {"run_id": run_id, "agent_id": agent_id, "artifacts": manifest},
                indent=2,
            ),
            encoding="utf-8",
        )

        rec["status"] = status
        rec["finished_at"] = utc_now()
        if status == "cancelled":
            rec["cancelled_at"] = rec["finished_at"]
        if result_obj and status != "succeeded":
            rec["error"] = {
                "error_type": result_obj.get("error_type"),
                "message": result_obj.get("message"),
            }
        rec["result"] = (result_obj or {}).get("text")
        self._write_run(run_id, rec)

        await self._emit(
            run_id=run_id,
            agent_id=agent_id,
            type="state_change",
            payload={
                "from": "running",
                "to": status,
                "reason": "process_exit",
                "returncode": rc,
            },
            level="info" if status == "succeeded" else "warn",
        )

        # Enforce retention after each run
        enforce_retention(
            workspace_root=self.workspace_root, policy=self.retention
        )

    async def _drain_stream(
        self,
        *,
        run_id: str,
        agent_id: str,
        stream_name: str,
        stream: Optional[asyncio.StreamReader],
        log_path: Path,
        cap_bytes: int,
    ) -> None:
        if stream is None:
            return

        log_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        truncated_notice_emitted = False

        with log_path.open("ab") as f:
            chunk_id = 0
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                chunk_id += 1

                # Always drain the pipe. Only persist/emit up to cap.
                if written < cap_bytes:
                    remain = cap_bytes - written
                    to_write = chunk[:remain]
                    if to_write:
                        f.write(to_write)
                        f.flush()
                        written += len(to_write)

                        # Emit log event (text), but cap per event to 64KiB
                        text = to_write.decode("utf-8", errors="replace")
                        # split if huge
                        for i in range(0, len(text), 65536):
                            part = text[i : i + 65536]
                            await self._emit(
                                run_id=run_id,
                                agent_id=agent_id,
                                type="log",
                                payload={
                                    "stream": stream_name,
                                    "text": part,
                                    "chunk_id": chunk_id,
                                    "truncated": False,
                                },
                                level="warn"
                                if stream_name == "stderr"
                                else "info",
                            )

                    if len(chunk) > remain and not truncated_notice_emitted:
                        truncated_notice_emitted = True
                        marker = f"\n[dashboard] {stream_name} truncated after {cap_bytes} bytes\n"
                        f.write(marker.encode("utf-8"))
                        f.flush()
                        await self._emit(
                            run_id=run_id,
                            agent_id=agent_id,
                            type="log",
                            payload={
                                "stream": stream_name,
                                "text": marker,
                                "chunk_id": chunk_id,
                                "truncated": True,
                            },
                            level="warn",
                        )
                else:
                    if not truncated_notice_emitted:
                        truncated_notice_emitted = True
                        marker = f"\n[dashboard] {stream_name} truncated after {cap_bytes} bytes\n"
                        try:
                            f.write(marker.encode("utf-8"))
                            f.flush()
                        except Exception:
                            pass
                        await self._emit(
                            run_id=run_id,
                            agent_id=agent_id,
                            type="log",
                            payload={
                                "stream": stream_name,
                                "text": marker,
                                "chunk_id": chunk_id,
                                "truncated": True,
                            },
                            level="warn",
                        )


# Missing import fix: contextlib is only used in cancel/timeout sections.
import contextlib  # noqa: E402
