from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import scan_artifacts
from .credentials import (
    CredentialConfigurationError,
    CredentialStore,
    KeyringCredentialStore,
    assert_no_raw_api_key,
    resolve_api_key,
)
from .events import make_event
from .retention import RetentionPolicy, enforce_retention
from .security import dashboard_root_from_env, safe_join
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
    concurrency: int = 5
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
        dashboard_root: Path | None = None,
        config: RunConfig | None = None,
        retention: RetentionPolicy | None = None,
        credential_store: CredentialStore | None = None,
        dashboard_group: str = "default",
    ):
        self.dashboard_root = dashboard_root or dashboard_root_from_env()
        self.config = config or RunConfig()
        self.retention = retention or RetentionPolicy()
        self.credential_store = credential_store or KeyringCredentialStore()
        self.dashboard_group = dashboard_group

        self.dashboard_root.mkdir(parents=True, exist_ok=True)
        (self.dashboard_root / "_meta" / "runs").mkdir(
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
        self._index_path = self.dashboard_root / "_meta" / "runs_index.jsonl"

    # ----------------------------
    # Paths and persistence
    # ----------------------------

    def _run_paths(self, *, agent_id: str, run_id: str) -> RunPaths:
        run_dir = self.dashboard_root / "runs" / agent_id / run_id
        logs_dir = run_dir / "logs"
        artifacts_dir = run_dir / "artifacts"
        meta_path = self.dashboard_root / "_meta" / "runs" / f"{run_id}.json"
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
        meta_path = self.dashboard_root / "_meta" / "runs" / f"{run_id}.json"
        return read_json(meta_path)

    def _write_run(self, run_id: str, rec: dict[str, Any]) -> None:
        meta_path = self.dashboard_root / "_meta" / "runs" / f"{run_id}.json"
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
            dashboard_root=self.dashboard_root, policy=self.retention
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
        meta_dir = self.dashboard_root / "_meta" / "runs"
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
        assert_no_raw_api_key(llm, context="run.llm")
        if extra and isinstance(extra.get("embedding"), dict):
            assert_no_raw_api_key(extra["embedding"], context="run.embedding")
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
                paths.run_dir.relative_to(self.dashboard_root)
            ).replace("\\", "/"),
            "logs": {
                "stdout": str(
                    paths.stdout_path.relative_to(self.dashboard_root)
                ).replace("\\", "/"),
                "stderr": str(
                    paths.stderr_path.relative_to(self.dashboard_root)
                ).replace("\\", "/"),
                "events": str(
                    paths.events_path.relative_to(self.dashboard_root)
                ).replace("\\", "/"),
            },
            "artifacts": {
                "dir": str(
                    paths.artifacts_dir.relative_to(self.dashboard_root)
                ).replace("\\", "/"),
                "manifest": str(
                    paths.artifacts_manifest_path.relative_to(
                        self.dashboard_root
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

    def validate_credentials(
        self, *, llm: dict[str, Any], embedding: dict[str, Any]
    ) -> None:
        """Fail before run creation when required credentials are unavailable."""
        self._resolve_credentials(llm=llm, embedding=embedding)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return self._read_run(run_id)

    async def list_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        meta_dir = self.dashboard_root / "_meta" / "runs"
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

    def _resolve_credentials(
        self, *, llm: dict[str, Any], embedding: dict[str, Any]
    ) -> dict[str, str | None]:
        llm_disabled = bool(llm.get("disabled")) or str(
            llm.get("model") or ""
        ).strip().lower() in {"none", "disabled"}
        llm_key = None
        if not llm_disabled:
            llm_key = resolve_api_key(
                llm,
                group=self.dashboard_group,
                kind="llm",
                store=self.credential_store,
            )

        embedding_key = None
        embedding_model = str(embedding.get("model") or "").strip()
        if embedding_model and embedding_model.lower() not in {
            "none",
            "disabled",
        }:
            embedding_key = resolve_api_key(
                embedding,
                group=self.dashboard_group,
                kind="embedding",
                store=self.credential_store,
            )
        return {
            "llm_api_key": llm_key,
            "embedding_api_key": embedding_key,
        }

    @staticmethod
    def _worker_environment(
        rec: dict[str, Any], *, project_root: str
    ) -> dict[str, str]:
        """Build the worker environment without model-provider credentials."""
        env = dict(os.environ)
        sensitive_names = {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        }
        for section in ("llm", "embedding"):
            config = rec.get(section) or {}
            env_name = str(config.get("api_key_env") or "").strip()
            if env_name:
                sensitive_names.add(env_name)
        secret_markers = (
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "secret",
            "password",
            "credential",
            "bearer",
        )
        for name in list(env):
            lowered = name.lower()
            if (
                name in sensitive_names
                or any(marker in lowered for marker in secret_markers)
                or lowered.endswith("_token")
            ):
                env.pop(name, None)

        env["PYTHONUNBUFFERED"] = "1"
        existing_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            project_root
            if not existing_pp
            else (project_root + os.pathsep + existing_pp)
        )
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    async def _fail_before_spawn(
        self, run_id: str, *, error: Exception
    ) -> None:
        rec = self._read_run(run_id)
        agent_id = rec["agent_id"]
        previous = rec.get("status") or "starting"
        rec["status"] = "failed"
        rec["finished_at"] = utc_now()
        rec["error"] = {
            "error_type": error.__class__.__name__,
            "message": str(error),
        }
        self._write_run(run_id, rec)
        await self._emit(
            run_id=run_id,
            agent_id=agent_id,
            type="state_change",
            payload={
                "from": previous,
                "to": "failed",
                "reason": "credential_preflight",
            },
            level="error",
        )

        session_id = rec.get("session_id")
        if not session_id:
            return
        try:
            from .sessions import append_message as _append_session_message
            from .sessions import update_session as _update_session

            _append_session_message(
                self.dashboard_root,
                session_id=str(session_id),
                role="assistant",
                text=f"(failed) {error}",
                run_id=run_id,
            )
            _update_session(
                self.dashboard_root,
                str(session_id),
                {"active_run_id": None, "last_run_id": run_id},
            )
        except Exception:
            pass

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

        try:
            resolved_secrets = await asyncio.to_thread(
                self._resolve_credentials,
                llm=rec.get("llm") or {},
                embedding=rec.get("embedding") or {},
            )
        except (CredentialConfigurationError, RuntimeError) as e:
            await self._fail_before_spawn(run_id, error=e)
            return

        # Write config JSON blobs for the worker.
        params_json = paths.run_dir / "params.json"
        agent_init_json = paths.run_dir / "agent_init.json"
        llm_json = paths.run_dir / "llm.json"
        embedding_json = paths.run_dir / "embedding.json"
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
        embedding_json.write_text(
            json.dumps(rec.get("embedding") or {}, indent=2), encoding="utf-8"
        )
        mcp_json.write_text(
            json.dumps(rec.get("mcp") or {}, indent=2), encoding="utf-8"
        )

        timeout = None
        runner_cfg = rec.get("runner") or {}
        if runner_cfg.get("timeout_seconds"):
            timeout = float(runner_cfg["timeout_seconds"])

        # Spawn worker subprocess
        workspace_dir_value = rec.get("workspace_dir")
        agent_workspace_dir = paths.run_dir
        if workspace_dir_value:
            try:
                raw_workspace_dir = Path(str(workspace_dir_value)).expanduser()
                if raw_workspace_dir.is_absolute():
                    agent_workspace_dir = raw_workspace_dir.resolve()
                else:
                    agent_workspace_dir = safe_join(
                        self.dashboard_root, str(workspace_dir_value)
                    )
                agent_workspace_dir.mkdir(parents=True, exist_ok=True)
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
            "--embedding-json",
            str(embedding_json),
            "--mcp-json",
            str(mcp_json),
            "--output-json",
            str(output_json),
            "--secrets-stdin",
        ]

        # Ensure the worker can import `ursa_dashboard` even when the dashboard
        # is run from a source checkout (not installed as a package).
        project_root = str(Path(__file__).resolve().parent.parent)
        env = self._worker_environment(rec, project_root=project_root)

        # The worker reads one credential message from stdin and then the pipe is
        # closed, preventing interactive behavior. Prefer plain output when logs
        # are pipes; users can opt in to forced ANSI output.
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
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        if proc.stdin is None:  # pragma: no cover - asyncio contract
            proc.kill()
            await proc.wait()
            await self._fail_before_spawn(
                run_id,
                error=RuntimeError("Worker secret channel is unavailable"),
            )
            return
        pipe_error: Exception | None = None
        try:
            secret_message = (
                json.dumps(resolved_secrets, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            proc.stdin.write(secret_message)
            await proc.stdin.drain()
        except Exception as e:
            pipe_error = e
        finally:
            proc.stdin.close()
            with contextlib.suppress(Exception):
                await proc.stdin.wait_closed()
            secret_message = b""
            resolved_secrets = {}
        if pipe_error is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            await self._fail_before_spawn(
                run_id,
                error=RuntimeError("Could not deliver credentials to worker"),
            )
            return

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
                    self.dashboard_root,
                    session_id=str(session_id),
                    role="assistant",
                    text=assistant_text,
                    run_id=run_id,
                )

                # Clear active run if this was it.
                sess_patch = {"last_run_id": run_id}
                try:
                    sess = read_json(
                        self.dashboard_root
                        / "sessions"
                        / str(session_id)
                        / "session.json"
                    )
                    if sess.get("active_run_id") == run_id:
                        sess_patch["active_run_id"] = None
                except Exception:
                    sess_patch["active_run_id"] = None
                _update_session(
                    self.dashboard_root, str(session_id), sess_patch
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
            dashboard_root=self.dashboard_root, policy=self.retention
        )

    async def _drain_stream(
        self,
        *,
        run_id: str,
        agent_id: str,
        stream_name: str,
        stream: asyncio.StreamReader | None,
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
