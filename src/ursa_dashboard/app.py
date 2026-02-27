from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import html
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .api_models import (
    ErrorResponse,
    FileMetaResponse,
    RunCancelRequest,
    RunCreateRequest,
    RunEventListResponse,
    RunListResponse,
    RunRecord,
    SessionCreateRequest,
    SessionDetail,
    SessionFileMetaResponse,
    SessionListResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionPatchRequest,
    SessionWorkspaceListResponse,
    SettingsPatchRequest,
    SettingsResponse,
    WorkspaceAllFilesResponse,
    WorkspaceListResponse,
)
from .artifacts import scan_artifacts
from .models import AgentListResponse
from .registry import REGISTRY
from .run_manager import RunManager
from .security import WorkspaceJailError, safe_join
from .sessions import (
    append_message as session_append_message,
)
from .sessions import (
    build_prompt_from_messages,
    session_paths,
)
from .sessions import (
    create_session as session_create_session,
)
from .sessions import (
    delete_session as session_delete_session,
)
from .sessions import (
    list_sessions as session_list_sessions,
)
from .sessions import (
    read_messages as session_read_messages,
)
from .sessions import (
    read_session as session_read_session,
)
from .sessions import (
    update_session as session_update_session,
)
from .settings import AuthConfig, SettingsStore


def create_app() -> FastAPI:
    auth = AuthConfig.from_env()

    security = HTTPBearer(auto_error=False)

    def require_auth(
        creds: HTTPAuthorizationCredentials | None = Depends(security),
    ) -> None:
        if auth.mode == "local":
            return
        # remote mode
        if not auth.token:
            raise HTTPException(
                status_code=500,
                detail="Remote mode enabled but URSA_DASHBOARD_TOKEN is not set",
            )
        if (
            creds is None
            or creds.scheme.lower() != "bearer"
            or creds.credentials != auth.token
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

    app = FastAPI(
        title="URSA Dashboard",
        description="Backend API for launching and monitoring URSA agents",
        version="0.1.0",
        responses={
            401: {"model": ErrorResponse},
            403: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
        },
    )

    if auth.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=auth.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    rm = RunManager()
    settings_store = SettingsStore(rm.workspace_root)

    @app.on_event("startup")
    async def _startup() -> None:
        # Validate auth config early
        if auth.mode == "remote" and not auth.token:
            raise RuntimeError(
                "URSA_DASHBOARD_MODE=remote requires URSA_DASHBOARD_TOKEN"
            )
        await rm.start()
        settings_store.load()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await rm.shutdown()

    # ----------------------------
    # Agents
    # ----------------------------

    def _include_demo_agents() -> bool:
        return str(
            os.environ.get("URSA_DASHBOARD_INCLUDE_DEMO_AGENTS", "")
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @app.get(
        "/agents",
        response_model=AgentListResponse,
        dependencies=[Depends(require_auth)],
    )
    def list_agents() -> AgentListResponse:
        include_demos = _include_demo_agents()
        agents = [
            e.spec
            for e in REGISTRY.values()
            if include_demos or not e.spec.agent_id.startswith("demo_")
        ]
        return AgentListResponse(agents=agents)

    # Back-compat alias
    @app.get(
        "/api/agents", response_model=AgentListResponse, include_in_schema=False
    )
    def list_agents_api() -> AgentListResponse:
        return list_agents()

    # ----------------------------
    # Settings (apply to new runs)
    # ----------------------------

    @app.get(
        "/settings",
        response_model=SettingsResponse,
        dependencies=[Depends(require_auth)],
    )
    def get_settings() -> SettingsResponse:
        s = settings_store.load()
        return SettingsResponse(settings=s.model_dump(mode="json"))

    @app.patch(
        "/settings",
        response_model=SettingsResponse,
        dependencies=[Depends(require_auth)],
    )
    def patch_settings(req: SettingsPatchRequest) -> SettingsResponse:
        s = settings_store.patch(req.patch)
        return SettingsResponse(settings=s.model_dump(mode="json"))

    # ----------------------------
    # Sessions (multi-turn chat)
    # ----------------------------

    @app.get(
        "/sessions",
        response_model=SessionListResponse,
        dependencies=[Depends(require_auth)],
    )
    def list_sessions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> SessionListResponse:
        recs = session_list_sessions(rm.workspace_root, limit=limit)
        return SessionListResponse(sessions=recs)

    @app.post(
        "/sessions",
        response_model=SessionDetail,
        dependencies=[Depends(require_auth)],
    )
    def create_session(req: SessionCreateRequest) -> SessionDetail:
        if req.agent_id not in REGISTRY or (
            req.agent_id.startswith("demo_") and not _include_demo_agents()
        ):
            raise HTTPException(status_code=404, detail="Unknown agent_id")
        sess = session_create_session(
            rm.workspace_root, agent_id=req.agent_id, title=req.title
        )
        return SessionDetail(session=sess, messages=[])

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionDetail,
        dependencies=[Depends(require_auth)],
    )
    def get_session(
        session_id: str, limit: int = Query(default=200, ge=1, le=2000)
    ) -> SessionDetail:
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        msgs = session_read_messages(rm.workspace_root, session_id, limit=limit)
        return SessionDetail(session=sess, messages=msgs)

    @app.patch(
        "/sessions/{session_id}",
        response_model=SessionDetail,
        dependencies=[Depends(require_auth)],
    )
    def patch_session(
        session_id: str, req: SessionPatchRequest
    ) -> SessionDetail:
        try:
            # Handle unknown session ID by trying to read and failing gracefully
            _ = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        # Only allow renaming for now.
        title = str(req.title).strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        sess2 = session_update_session(
            rm.workspace_root, session_id, {"title": title}
        )
        msgs = session_read_messages(rm.workspace_root, session_id, limit=200)
        return SessionDetail(session=sess2, messages=msgs)

    @app.delete("/sessions/{session_id}", dependencies=[Depends(require_auth)])
    def delete_session(session_id: str) -> Response:
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        if sess.get("active_run_id"):
            raise HTTPException(
                status_code=409,
                detail="Cannot delete a session with an active run",
            )

        try:
            session_delete_session(rm.workspace_root, session_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return Response(status_code=204)

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=list[dict[str, Any]],
        dependencies=[Depends(require_auth)],
    )
    def get_session_messages(
        session_id: str, limit: int = Query(default=200, ge=1, le=2000)
    ):
        # Lightweight endpoint for polling
        try:
            _ = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")
        return session_read_messages(rm.workspace_root, session_id, limit=limit)

    @app.post(
        "/sessions/{session_id}/message",
        response_model=SessionMessageResponse,
        dependencies=[Depends(require_auth)],
    )
    async def post_session_message(
        session_id: str, req: SessionMessageRequest
    ) -> SessionMessageResponse:
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        agent_id = str(sess.get("agent_id") or "")
        if agent_id not in REGISTRY:
            raise HTTPException(
                status_code=400,
                detail="Session agent_id is no longer available",
            )

        # Build conversational prompt from prior transcript.
        prior = session_read_messages(rm.workspace_root, session_id, limit=200)
        prompt = build_prompt_from_messages(prior, new_user_text=req.text)

        user_msg = session_append_message(
            rm.workspace_root,
            session_id=session_id,
            role="user",
            text=req.text,
        )

        # Merge global defaults (apply only if caller didn't provide)
        s = settings_store.load().model_dump(mode="json")
        llm = {**(s.get("llm") or {}), **(req.llm or {})}
        runner = {**(s.get("runner") or {}), **(req.runner or {})}
        mcp = s.get("mcp") or {}

        # Demo agents should work without external credentials.
        if agent_id.startswith("demo_") and "disabled" not in llm:
            llm["disabled"] = True

        params = dict(req.params or {})
        params.setdefault("prompt", prompt)

        agent_init = dict(req.agent_init or {})

        # Use a shared per-session workspace directory so artifacts persist across turns.
        sp = session_paths(rm.workspace_root, session_id)
        workspace_dir_rel = sp.workspace_dir.relative_to(
            rm.workspace_root
        ).as_posix()

        run = await rm.create_run(
            agent_id=agent_id,
            params=params,
            agent_init=agent_init,
            llm=llm,
            runner=runner,
            extra={
                "session_id": session_id,
                "session_user_message_id": user_msg.get("message_id"),
                "workspace_dir": workspace_dir_rel,
                "mcp": mcp,
            },
        )

        sess2 = session_update_session(
            rm.workspace_root,
            session_id,
            {"active_run_id": run["run_id"], "last_run_id": run["run_id"]},
        )
        return SessionMessageResponse(
            session=sess2, user_message=user_msg, run=run
        )

    # ----------------------------
    # Runs
    # ----------------------------

    @app.post(
        "/runs", response_model=RunRecord, dependencies=[Depends(require_auth)]
    )
    async def create_run(req: RunCreateRequest) -> RunRecord:
        if req.agent_id not in REGISTRY or (
            req.agent_id.startswith("demo_") and not _include_demo_agents()
        ):
            raise HTTPException(status_code=404, detail="Unknown agent_id")

        # Merge with global defaults (apply only if caller didn't provide)
        s = settings_store.load().model_dump(mode="json")
        llm = {**(s.get("llm") or {}), **(req.llm or {})}
        runner = {**(s.get("runner") or {}), **(req.runner or {})}
        mcp = s.get("mcp") or {}

        # Demo agents should work without external credentials.
        if req.agent_id.startswith("demo_") and "disabled" not in llm:
            llm["disabled"] = True

        rec = await rm.create_run(
            agent_id=req.agent_id,
            params=req.params,
            agent_init=req.agent_init,
            llm=llm,
            runner=runner,
            extra={"mcp": mcp},
        )
        return rec

    # Back-compat alias
    @app.post("/api/runs", include_in_schema=False)
    async def create_run_api(req: RunCreateRequest):
        return await create_run(req)

    @app.get(
        "/runs",
        response_model=RunListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def list_runs(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> RunListResponse:
        return RunListResponse(runs=await rm.list_runs(limit=limit))

    @app.get(
        "/runs/{run_id}",
        response_model=RunRecord,
        dependencies=[Depends(require_auth)],
    )
    async def get_run(run_id: str) -> RunRecord:
        try:
            return await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

    @app.post(
        "/runs/{run_id}/cancel",
        response_model=RunRecord,
        dependencies=[Depends(require_auth)],
    )
    async def cancel_run(run_id: str, req: RunCancelRequest) -> RunRecord:
        try:
            return await rm.cancel(run_id, reason=req.reason)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

    # ----------------------------
    # Events
    # ----------------------------

    @app.get(
        "/runs/{run_id}/events",
        response_model=RunEventListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_events(
        run_id: str,
        after_seq: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=5000),
    ) -> RunEventListResponse:
        try:
            await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        events_path = rm.events_path(run_id)
        out: list[dict[str, Any]] = []
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if int(ev.get("seq", 0)) <= after_seq:
                        continue
                    out.append(ev)
                    if len(out) >= limit:
                        break
        return RunEventListResponse(events=out)

    async def _sse_events(run_id: str, after_seq: int) -> AsyncIterator[bytes]:
        # Tail events.jsonl and stream as SSE.
        #
        # Notes:
        # - We include SSE `id:` as the event sequence number so the browser can
        #   automatically resume after reconnect via the Last-Event-ID header.
        try:
            await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        events_path = rm.events_path(run_id)
        last_keepalive = asyncio.get_event_loop().time()

        def format_sse(event: dict[str, Any]) -> bytes:
            etype = event.get("type", "message")
            seq = int(event.get("seq", 0))
            data = json.dumps(event, ensure_ascii=False)
            # Provide id for resume + retry hint.
            return (
                f"id: {seq}\nevent: {etype}\nretry: 1500\ndata: {data}\n\n"
            ).encode("utf-8")

        cursor_seq = after_seq
        offset = 0

        while True:
            if events_path.exists():
                with events_path.open("rb") as f:
                    if offset:
                        f.seek(offset)
                    while True:
                        line_b = f.readline()
                        if not line_b:
                            offset = f.tell()
                            break
                        try:
                            ev = json.loads(
                                line_b.decode("utf-8", errors="replace")
                            )
                        except Exception:
                            continue
                        seq = int(ev.get("seq", 0))
                        if seq <= cursor_seq:
                            continue
                        cursor_seq = seq
                        yield format_sse(ev)

            run = await rm.get_run(run_id)
            if run.get("status") in {"succeeded", "failed", "cancelled"}:
                # final pass
                if events_path.exists():
                    with events_path.open("rb") as f:
                        f.seek(offset)
                        for line_b in f:
                            try:
                                ev = json.loads(
                                    line_b.decode("utf-8", errors="replace")
                                )
                            except Exception:
                                continue
                            seq = int(ev.get("seq", 0))
                            if seq <= cursor_seq:
                                continue
                            cursor_seq = seq
                            yield format_sse(ev)
                break

            now = asyncio.get_event_loop().time()
            if now - last_keepalive > 15:
                yield b": keepalive\n\n"
                last_keepalive = now

            await asyncio.sleep(0.5)

    @app.get("/runs/{run_id}/stream", dependencies=[Depends(require_auth)])
    async def stream_events(
        run_id: str,
        after_seq: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        # Prefer Last-Event-ID (standard SSE resume mechanism) if present.
        try:
            if last_event_id is not None and str(last_event_id).strip() != "":
                after_seq = max(after_seq, int(str(last_event_id).strip()))
        except Exception:
            pass

        return StreamingResponse(
            _sse_events(run_id, after_seq),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # helps with reverse proxies
                "Connection": "keep-alive",
            },
        )

    # Back-compat alias
    @app.get("/api/runs/{run_id}/events", include_in_schema=False)
    async def stream_events_api(
        run_id: str, after_seq: int = Query(default=0, ge=0)
    ):
        return await stream_events(run_id, after_seq)

    # ----------------------------
    # Workspace / artifacts
    # ----------------------------

    def _run_dir_for(run: dict[str, Any]) -> Path:
        return (rm.workspace_root / run["run_dir"]).resolve()

    def _file_sha256(
        path: Path, *, max_bytes: int = 50 * 1024 * 1024
    ) -> str | None:
        try:
            if path.stat().st_size > max_bytes:
                return None
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @app.get(
        "/runs/{run_id}/workspace",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def list_workspace(
        run_id: str, refresh: bool = False
    ) -> WorkspaceListResponse:
        try:
            run = await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        run_dir = _run_dir_for(run)

        # Prefer persisted manifest unless refresh requested.
        manifest_path = rm.artifacts_manifest_path(run_id)
        if (not refresh) and manifest_path.exists():
            try:
                obj = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = obj.get("artifacts", [])
                return WorkspaceListResponse(
                    run_id=run_id, agent_id=run["agent_id"], files=files
                )
            except Exception:
                pass

        files = scan_artifacts(
            run_dir,
            exclude_dirs={"logs", "metrics", "agent_store", "__pycache__"},
        )
        return WorkspaceListResponse(
            run_id=run_id, agent_id=run["agent_id"], files=files
        )

    # Alias (older name)
    @app.get(
        "/runs/{run_id}/artifacts",
        response_model=WorkspaceListResponse,
        include_in_schema=False,
    )
    async def list_artifacts_alias(
        run_id: str, refresh: bool = False
    ) -> WorkspaceListResponse:
        return await list_workspace(run_id, refresh=refresh)

    def _parse_rfc3339(ts: str) -> datetime:
        # Accept "...Z" or explicit offsets.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)

    @app.get("/workspace/runs", dependencies=[Depends(require_auth)])
    async def list_workspace_by_run(
        agent_id: str | None = None,
        status: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        include_files: bool = True,
        refresh: bool = False,
    ):
        """Default workspace browser view: group files by run_id with filters."""
        runs = await rm.list_runs(limit=500)

        if agent_id:
            runs = [r for r in runs if r.get("agent_id") == agent_id]
        if status:
            runs = [r for r in runs if r.get("status") == status]
        if created_after:
            ca = _parse_rfc3339(created_after)
            runs = [
                r
                for r in runs
                if r.get("created_at") and _parse_rfc3339(r["created_at"]) >= ca
            ]
        if created_before:
            cb = _parse_rfc3339(created_before)
            runs = [
                r
                for r in runs
                if r.get("created_at") and _parse_rfc3339(r["created_at"]) <= cb
            ]

        runs = runs[:limit]

        items: list[dict[str, Any]] = []
        for r in runs:
            item: dict[str, Any] = {"run": r}
            if include_files:
                manifest_path = rm.artifacts_manifest_path(r["run_id"])
                files: list[dict[str, Any]] = []
                if (not refresh) and manifest_path.exists():
                    try:
                        obj = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        files = obj.get("artifacts", [])
                    except Exception:
                        files = []
                else:
                    run_dir = _run_dir_for(r)
                    files = scan_artifacts(
                        run_dir,
                        exclude_dirs={
                            "logs",
                            "metrics",
                            "agent_store",
                            "__pycache__",
                        },
                    )
                item["files"] = files
            items.append(item)

        return {"items": items}

    @app.get(
        "/workspace/files",
        response_model=WorkspaceAllFilesResponse,
        dependencies=[Depends(require_auth)],
    )
    async def list_all_workspace_files(
        limit: int = Query(default=5000, ge=1, le=20000),
    ) -> WorkspaceAllFilesResponse:
        # Optional global scan across all runs.
        base = rm.workspace_root / "runs"
        files: list[dict[str, Any]] = []
        if base.exists():
            for root, dirnames, filenames in os.walk(base):
                # Skip runtime dirs
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d
                    not in {"logs", "metrics", "agent_store", "__pycache__"}
                ]
                for fn in filenames:
                    p = Path(root) / fn
                    try:
                        rel = p.relative_to(rm.workspace_root).as_posix()
                        st = p.stat()
                    except Exception:
                        continue
                    files.append({
                        "rel_path": rel,
                        "size_bytes": st.st_size,
                        "mtime": st.st_mtime,
                    })
                    if len(files) >= limit:
                        return WorkspaceAllFilesResponse(
                            files=sorted(files, key=lambda x: x["rel_path"])
                        )
        return WorkspaceAllFilesResponse(
            files=sorted(files, key=lambda x: x["rel_path"])
        )

    @app.get(
        "/runs/{run_id}/workspace/file/meta",
        response_model=FileMetaResponse,
        dependencies=[Depends(require_auth)],
    )
    async def file_meta(
        run_id: str, path: str = Query(..., min_length=1)
    ) -> FileMetaResponse:
        try:
            run = await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        run_dir = _run_dir_for(run)
        try:
            fp = safe_join(run_dir, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        mime, _ = mimetypes.guess_type(fp.name)
        sha256 = _file_sha256(fp)

        return FileMetaResponse(
            run_id=run_id,
            path=path,
            size_bytes=st.st_size,
            mtime=st.st_mtime,
            mime=mime,
            sha256=sha256,
        )

    def _html_page(title: str, body: str) -> HTMLResponse:
        html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }}
    .muted {{ color: #555; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    details {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px 12px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f6f6; padding: 12px; border-radius: 8px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #eee; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
        return HTMLResponse(
            html_doc,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-src 'self'; object-src 'none'",
            },
        )

    def _looks_binary(sample: bytes) -> bool:
        return b"\x00" in sample

    @app.get(
        "/runs/{run_id}/workspace/file/preview",
        dependencies=[Depends(require_auth)],
    )
    async def preview_file(
        run_id: str, path: str = Query(..., min_length=1)
    ) -> HTMLResponse:
        """Safe previewer (HTML-escaped text, inline images, basic PDF embed)."""
        try:
            run = await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        run_dir = _run_dir_for(run)
        try:
            fp = safe_join(run_dir, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        size = st.st_size
        mime, _ = mimetypes.guess_type(fp.name)
        mime = mime or "application/octet-stream"

        # Safety caps
        TEXT_CAP = 1_000_000  # 1MB
        EMBED_CAP = 50 * 1024 * 1024  # 50MB for images/pdf embed

        file_url = f"/runs/{run_id}/workspace/file?path={quote(path)}&disposition=inline"
        download_url = f"/runs/{run_id}/workspace/file?path={quote(path)}&disposition=attachment"

        header = (
            f'<div><a href="/ui/workspace">Workspace</a> / '
            f'<span class="pill">{html.escape(run.get("agent_id", ""))}</span> '
            f'<span class="muted">run {html.escape(run_id)}</span></div>'
            f'<h2 style="margin-top:8px">{html.escape(path)}</h2>'
            f'<div class="muted">{html.escape(mime)} · {size} bytes · <a href="{download_url}">download</a></div>'
        )

        # Image preview
        if mime.startswith("image/"):
            if size > EMBED_CAP:
                return _html_page(
                    "Preview",
                    header
                    + f'<p>Image too large to preview (> {EMBED_CAP} bytes). <a href="{download_url}">Download</a></p>',
                )
            body = (
                header
                + f'<div style="margin-top:12px"><img src="{file_url}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px" /></div>'
            )
            return _html_page("Preview", body)

        # PDF preview
        if mime == "application/pdf":
            if size > EMBED_CAP:
                return _html_page(
                    "Preview",
                    header
                    + f'<p>PDF too large to embed (> {EMBED_CAP} bytes). <a href="{download_url}">Download</a></p>',
                )
            body = header + (
                f'<div style="margin-top:12px">'
                f'<iframe src="{file_url}" style="width: 100%; height: 80vh; border: 1px solid #ddd; border-radius: 8px"></iframe>'
                f"</div>"
            )
            return _html_page("Preview", body)

        # Text/code preview (escaped)
        # Treat many common code/data formats as text even if mimetypes is generic.
        text_exts = {
            ".py",
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".csv",
            ".tsv",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".sh",
            ".bat",
            ".ps1",
            ".log",
            ".tex",
            ".rst",
        }
        is_texty = (
            mime.startswith("text/")
            or fp.suffix.lower() in text_exts
            or mime
            in {
                "application/json",
                "application/xml",
                "application/x-yaml",
                "application/yaml",
            }
        )

        if is_texty:
            with fp.open("rb") as f:
                sample = f.read(8192)
                if _looks_binary(sample):
                    return _html_page(
                        "Preview",
                        header
                        + f'<p>Binary file (detected NUL bytes). <a href="{download_url}">Download</a></p>',
                    )
                # Read up to cap
                f.seek(0)
                data = f.read(TEXT_CAP + 1)

            truncated = len(data) > TEXT_CAP
            if truncated:
                data = data[:TEXT_CAP]
            text = data.decode("utf-8", errors="replace")
            escaped = html.escape(text)
            note = (
                ""
                if not truncated
                else f'<div class="muted">Preview truncated to {TEXT_CAP} bytes.</div>'
            )
            body = (
                header
                + f'<div style="margin-top:12px">{note}<pre><code>{escaped}</code></pre></div>'
            )
            return _html_page("Preview", body)

        # Fallback: no preview
        return _html_page(
            "Preview",
            header
            + f'<p>No safe preview available for this file type. <a href="{download_url}">Download</a></p>',
        )

    def _parse_range(h: str, size: int) -> tuple[int, int] | None:
        # Supports a single range: bytes=start-end | bytes=start- | bytes=-suffix
        if not h:
            return None
        if not h.startswith("bytes="):
            return None
        spec = h[len("bytes=") :].strip()
        if "," in spec:
            return None
        if "-" not in spec:
            return None
        a, b = spec.split("-", 1)
        a = a.strip()
        b = b.strip()
        if a == "":
            # suffix bytes
            try:
                suffix = int(b)
            except Exception:
                return None
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
            return (start, end)
        try:
            start = int(a)
        except Exception:
            return None
        if start < 0 or start >= size:
            return None
        if b == "":
            return (start, size - 1)
        try:
            end = int(b)
        except Exception:
            return None
        if end < start:
            return None
        end = min(end, size - 1)
        return (start, end)

    def _iter_file(
        fp: Path, start: int, end: int, chunk_size: int = 1024 * 1024
    ):
        with fp.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    @app.get(
        "/runs/{run_id}/workspace/file", dependencies=[Depends(require_auth)]
    )
    async def open_file(
        request: Request,
        run_id: str,
        path: str = Query(..., min_length=1),
        disposition: str = Query(
            default="inline", pattern="^(inline|attachment)$"
        ),
    ):
        try:
            run = await rm.get_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Unknown run_id")

        run_dir = _run_dir_for(run)
        try:
            fp = safe_join(run_dir, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        size = st.st_size
        mime, _ = mimetypes.guess_type(fp.name)

        base_headers = {
            "Content-Disposition": f'{disposition}; filename="{fp.name}"',
            "Accept-Ranges": "bytes",
        }

        rng = _parse_range(
            request.headers.get("range") or request.headers.get("Range") or "",
            size,
        )
        if rng is None:
            # Full file
            return FileResponse(fp, headers=base_headers, media_type=mime)

        start, end = rng
        content_length = end - start + 1
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(content_length),
        }
        return StreamingResponse(
            _iter_file(fp, start, end),
            status_code=206,
            headers=headers,
            media_type=mime or "application/octet-stream",
        )

    # Back-compat alias
    @app.get("/api/runs/{run_id}/file", include_in_schema=False)
    async def open_file_api(request: Request, run_id: str, path: str):
        return await open_file(request=request, run_id=run_id, path=path)

    # ----------------------------
    # Session workspace (shared across turns)
    # ----------------------------

    def _session_workspace_dir(session_id: str) -> Path:
        sp = session_paths(rm.workspace_root, session_id)
        return sp.workspace_dir.resolve()

    @app.get(
        "/sessions/{session_id}/workspace",
        response_model=SessionWorkspaceListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def list_session_workspace(
        session_id: str,
    ) -> SessionWorkspaceListResponse:
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        ws = _session_workspace_dir(session_id)
        files = scan_artifacts(ws, exclude_dirs={"__pycache__"})
        return SessionWorkspaceListResponse(
            session_id=session_id,
            agent_id=str(sess.get("agent_id") or ""),
            files=files,
        )

    async def _save_upload(
        upload: UploadFile, dest: Path, *, max_bytes: int
    ) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(
            dest.name + f".tmp_upload_{datetime.now(timezone.utc).timestamp()}"
        )

        total = 0
        try:
            with tmp.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload too large (>{max_bytes} bytes)",
                        )
                    out.write(chunk)

            # Atomic replace (overwrites if target exists)
            tmp.replace(dest)
        finally:
            try:
                await upload.close()
            except Exception:
                pass
            with contextlib.suppress(Exception):
                if tmp.exists():
                    tmp.unlink()
        return total

    @app.post(
        "/sessions/{session_id}/workspace/upload",
        dependencies=[Depends(require_auth)],
    )
    async def upload_session_files(
        session_id: str,
        files: list[UploadFile] = File(...),
        dir: str = Form(default=""),
        overwrite: bool = Form(default=False),
    ):
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        ws = _session_workspace_dir(session_id)

        MAX_FILE_BYTES = 100 * 1024 * 1024  # 100MB per file
        saved: list[dict[str, Any]] = []

        for up in files:
            name = Path(str(up.filename or "")).name
            if not name:
                raise HTTPException(status_code=400, detail="Missing filename")
            rel = (
                (Path(dir) / name).as_posix()
                if dir and str(dir).strip()
                else name
            )

            try:
                dest = safe_join(ws, rel)
            except WorkspaceJailError:
                raise HTTPException(status_code=403, detail="Forbidden")

            if dest.exists() and not overwrite:
                raise HTTPException(
                    status_code=409, detail=f"File already exists: {rel}"
                )

            size = await _save_upload(up, dest, max_bytes=MAX_FILE_BYTES)
            saved.append({"path": rel, "size_bytes": size})

        # Touch session updated_at.
        session_update_session(rm.workspace_root, session_id, {})

        # Return updated file listing for convenience.
        files_out = scan_artifacts(ws, exclude_dirs={"__pycache__"})
        return {
            "session_id": session_id,
            "saved": saved,
            "files": files_out,
            "active_run_id": sess.get("active_run_id"),
        }

    @app.get(
        "/sessions/{session_id}/workspace/file/meta",
        response_model=SessionFileMetaResponse,
        dependencies=[Depends(require_auth)],
    )
    async def session_file_meta(
        session_id: str, path: str = Query(..., min_length=1)
    ) -> SessionFileMetaResponse:
        try:
            _ = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        ws = _session_workspace_dir(session_id)
        try:
            fp = safe_join(ws, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        mime, _ = mimetypes.guess_type(fp.name)
        sha256 = _file_sha256(fp)
        return SessionFileMetaResponse(
            session_id=session_id,
            path=path,
            size_bytes=st.st_size,
            mtime=st.st_mtime,
            mime=mime,
            sha256=sha256,
        )

    @app.get(
        "/sessions/{session_id}/workspace/file/preview",
        dependencies=[Depends(require_auth)],
    )
    async def preview_session_file(
        session_id: str, path: str = Query(..., min_length=1)
    ) -> HTMLResponse:
        try:
            sess = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        ws = _session_workspace_dir(session_id)
        try:
            fp = safe_join(ws, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        size = st.st_size
        mime, _ = mimetypes.guess_type(fp.name)
        mime = mime or "application/octet-stream"

        TEXT_CAP = 1_000_000
        EMBED_CAP = 50 * 1024 * 1024

        file_url = f"/sessions/{session_id}/workspace/file?path={quote(path)}&disposition=inline"
        download_url = f"/sessions/{session_id}/workspace/file?path={quote(path)}&disposition=attachment"

        header = (
            f'<div><a href="/ui">Dashboard</a> / '
            f'<span class="pill">{html.escape(str(sess.get("agent_id", "")))}</span> '
            f'<span class="muted">session {html.escape(session_id)}</span></div>'
            f'<h2 style="margin-top:8px">{html.escape(path)}</h2>'
            f'<div class="muted">{html.escape(mime)} · {size} bytes · <a href="{download_url}">download</a></div>'
        )

        if mime.startswith("image/"):
            if size > EMBED_CAP:
                return _html_page(
                    "Preview",
                    header
                    + f'<p>Image too large to preview (&gt; {EMBED_CAP} bytes). <a href="{download_url}">Download</a></p>',
                )
            body = (
                header
                + f'<div style="margin-top:12px"><img src="{file_url}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px" /></div>'
            )
            return _html_page("Preview", body)

        if mime == "application/pdf":
            if size > EMBED_CAP:
                return _html_page(
                    "Preview",
                    header
                    + f'<p>PDF too large to embed (&gt; {EMBED_CAP} bytes). <a href="{download_url}">Download</a></p>',
                )
            body = header + (
                f'<div style="margin-top:12px">'
                f'<iframe src="{file_url}" style="width: 100%; height: 80vh; border: 1px solid #ddd; border-radius: 8px"></iframe>'
                f"</div>"
            )
            return _html_page("Preview", body)

        text_exts = {
            ".py",
            ".txt",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".csv",
            ".tsv",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".sh",
            ".bat",
            ".ps1",
            ".log",
            ".tex",
            ".rst",
        }
        is_texty = (
            mime.startswith("text/")
            or fp.suffix.lower() in text_exts
            or mime
            in {
                "application/json",
                "application/xml",
                "application/x-yaml",
                "application/yaml",
            }
        )
        if is_texty:
            with fp.open("rb") as f:
                sample = f.read(8192)
                if _looks_binary(sample):
                    return _html_page(
                        "Preview",
                        header
                        + f'<p>Binary file (detected NUL bytes). <a href="{download_url}">Download</a></p>',
                    )
                f.seek(0)
                data = f.read(TEXT_CAP + 1)

            truncated = len(data) > TEXT_CAP
            if truncated:
                data = data[:TEXT_CAP]
            text = data.decode("utf-8", errors="replace")
            escaped = html.escape(text)
            note = (
                ""
                if not truncated
                else f'<div class="muted">Preview truncated to {TEXT_CAP} bytes.</div>'
            )
            body = (
                header
                + f'<div style="margin-top:12px">{note}<pre><code>{escaped}</code></pre></div>'
            )
            return _html_page("Preview", body)

        return _html_page(
            "Preview",
            header
            + f'<p>No safe preview available for this file type. <a href="{download_url}">Download</a></p>',
        )

    @app.get(
        "/sessions/{session_id}/workspace/file",
        dependencies=[Depends(require_auth)],
    )
    async def open_session_file(
        request: Request,
        session_id: str,
        path: str = Query(..., min_length=1),
        disposition: str = Query(
            default="inline", pattern="^(inline|attachment)$"
        ),
    ):
        try:
            _ = session_read_session(rm.workspace_root, session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Unknown session_id")

        ws = _session_workspace_dir(session_id)
        try:
            fp = safe_join(ws, path)
        except WorkspaceJailError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if not fp.exists() or not fp.is_file():
            raise HTTPException(status_code=404, detail="Not found")

        st = fp.stat()
        size = st.st_size
        mime, _ = mimetypes.guess_type(fp.name)

        base_headers = {
            "Content-Disposition": f'{disposition}; filename="{fp.name}"',
            "Accept-Ranges": "bytes",
        }

        rng = _parse_range(
            request.headers.get("range") or request.headers.get("Range") or "",
            size,
        )
        if rng is None:
            return FileResponse(fp, headers=base_headers, media_type=mime)

        start, end = rng
        content_length = end - start + 1
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(content_length),
        }
        return StreamingResponse(
            _iter_file(fp, start, end),
            status_code=206,
            headers=headers,
            media_type=mime or "application/octet-stream",
        )

    # ----------------------------
    # Minimal UI: workspace browser + previewer (run-first)
    # ----------------------------

    @app.get(
        "/ui/workspace",
        response_class=HTMLResponse,
        include_in_schema=False,
        dependencies=[Depends(require_auth)],
    )
    async def ui_workspace(
        agent_id: str | None = None,
        status: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = Query(default=25, ge=1, le=200),
    ) -> HTMLResponse:
        runs = await rm.list_runs(limit=500)
        if agent_id:
            runs = [r for r in runs if r.get("agent_id") == agent_id]
        if status:
            runs = [r for r in runs if r.get("status") == status]
        if created_after:
            ca = _parse_rfc3339(created_after)
            runs = [
                r
                for r in runs
                if r.get("created_at") and _parse_rfc3339(r["created_at"]) >= ca
            ]
        if created_before:
            cb = _parse_rfc3339(created_before)
            runs = [
                r
                for r in runs
                if r.get("created_at") and _parse_rfc3339(r["created_at"]) <= cb
            ]
        runs = runs[:limit]

        # Build filter options from registry and known statuses.
        agent_opts = "".join(
            f'<option value="{html.escape(a)}" {"selected" if agent_id == a else ""}>{html.escape(a)}</option>'
            for a in sorted(REGISTRY.keys())
        )
        status_vals = [
            "queued",
            "starting",
            "running",
            "succeeded",
            "failed",
            "cancelling",
            "cancelled",
        ]
        status_opts = (
            '<option value=""'
            + (" selected" if not status else "")
            + ">all</option>"
            + "".join(
                f'<option value="{s}" {"selected" if status == s else ""}>{s}</option>'
                for s in status_vals
            )
        )

        form = f"""
<form method=\"get\" style=\"display:flex; gap:12px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 12px\">
  <div><div class=\"muted\">Agent</div><select name=\"agent_id\"><option value=\"\"{" selected" if not agent_id else ""}>all</option>{agent_opts}</select></div>
  <div><div class=\"muted\">Status</div><select name=\"status\">{status_opts}</select></div>
  <div><div class=\"muted\">Created after (RFC3339)</div><input name=\"created_after\" value=\"{html.escape(created_after or "")}\" size=\"26\" /></div>
  <div><div class=\"muted\">Created before (RFC3339)</div><input name=\"created_before\" value=\"{html.escape(created_before or "")}\" size=\"26\" /></div>
  <div><div class=\"muted\">Limit</div><input name=\"limit\" type=\"number\" min=\"1\" max=\"200\" value=\"{limit}\" style=\"width:90px\" /></div>
  <div><button type=\"submit\">Apply</button></div>
  <div class=\"muted\" style=\"margin-left:auto\">Tip: expand a run to see files. Click a file to preview.</div>
</form>
"""

        blocks: list[str] = []
        for r in runs:
            rid = r["run_id"]
            aid = r.get("agent_id", "")
            st = r.get("status", "")
            created = r.get("created_at", "")
            manifest_path = rm.artifacts_manifest_path(rid)
            files: list[dict[str, Any]] = []
            if manifest_path.exists():
                try:
                    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
                    files = obj.get("artifacts", [])
                except Exception:
                    files = []

            file_lines = []
            for fobj in files:
                rel = fobj.get("rel_path") or fobj.get("path") or ""
                kind = fobj.get("kind") or "file"
                size_b = fobj.get("size_bytes")
                size_txt = f" · {size_b} B" if isinstance(size_b, int) else ""
                href = f"/runs/{rid}/workspace/file/preview?path={quote(rel)}"
                file_lines.append(
                    f'<li><a href="{href}">{html.escape(rel)}</a> <span class="muted">({html.escape(kind)}{size_txt})</span></li>'
                )

            inner = (
                "<ul>" + "".join(file_lines) + "</ul>"
                if file_lines
                else '<div class="muted">No files recorded yet.</div>'
            )
            blocks.append(
                f"""
<details>
  <summary>{html.escape(rid)} · <span class=\"pill\">{html.escape(aid)}</span> <span class=\"pill\">{html.escape(st)}</span> <span class=\"muted\">{html.escape(created)}</span></summary>
  <div style=\"margin-top:8px\">{inner}</div>
</details>
"""
            )

        body = f'<h1>Workspace (grouped by run)</h1>{form}<div class="grid">{"".join(blocks)}</div>'
        return _html_page("Workspace", body)

    @app.get(
        "/ui/workspace/all",
        response_class=HTMLResponse,
        include_in_schema=False,
        dependencies=[Depends(require_auth)],
    )
    async def ui_workspace_all(
        limit: int = Query(default=2000, ge=1, le=20000),
    ) -> HTMLResponse:
        # Secondary view: global scan
        obj = await list_all_workspace_files(limit=limit)
        lines = [
            f'<li><code>{html.escape(f["rel_path"])}</code> <span class="muted">({f["size_bytes"]} B)</span></li>'
            for f in obj.files
        ]
        body = f'<h1>All workspace files</h1><div class="muted">Global scan across runs (limit {limit}).</div><ul>{"".join(lines)}</ul>'
        return _html_page("All workspace files", body)

    # ----------------------------
    # Main UI: dashboard
    # ----------------------------

    DASHBOARD_JS = r"""
(() => {
  const $ = (sel, root=document) => root.querySelector(sel);
  const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

  const state = {
    agents: [],
    agentsById: new Map(),
    sessions: [],
    activeSessionId: null,
    activeSession: null, // {session, messages}
    viewRunId: null,
    viewRunKind: 'none', // 'none' | 'static' | 'stream'
    es: null,
    logs: { stdout: '', stderr: '' },

    // Panel visibility (left sidebar is always shown)
    showChat: true,
    showRunLogs: true,
    showArtifacts: true,

    settings: null,
    _renderTimers: { stdout: null, stderr: null },
    _logToken: 0,
  };

  function escHtml(s) {
    return String(s).replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;','\'':'&#39;'}[c]));
  }

  async function api(method, path, body=null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const resp = await fetch(path, opts);
    if (!resp.ok) {
      let msg = resp.status + ' ' + resp.statusText;
      try { const j = await resp.json(); if (j && j.detail) msg = j.detail; } catch {}
      throw new Error(msg);
    }
    const ct = resp.headers.get('content-type') || '';
    if (ct.includes('application/json')) return await resp.json();
    return await resp.text();
  }

  function savePref(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} }
  function loadPref(k, def=null) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : def; } catch { return def; } }

  function applyPanelVisibility() {
    const app = $('.app');
    const left = $('#leftPanel');
    const main = $('#mainPanel');
    const right = $('#rightPanel');
    const leftSplit = $('#leftSplitter');
    const rightSplit = $('#rightSplitter');

    const conv = $('#conversationSection');
    const run = $('#runSection');
    const midSplit = $('#midSplitter');
    const mainBody = $('#mainBody');

    const showChat = !!state.showChat;
    const showRunLogs = !!state.showRunLogs;
    const showArtifacts = !!state.showArtifacts;
    const showMain = (showChat || showRunLogs);

    // Left sidebar is always visible.
    if (left) left.classList.toggle('hidden', false);

    if (main) main.classList.toggle('hidden', !showMain);
    if (right) right.classList.toggle('hidden', !showArtifacts);
    if (app) app.classList.toggle('mainHidden', !showMain);

    if (conv) conv.classList.toggle('hidden', !showChat);
    if (run) run.classList.toggle('hidden', !showRunLogs);

    // Only show the middle splitter when both chat and run logs are visible.
    if (midSplit) midSplit.classList.toggle('hidden', !(showChat && showRunLogs));

    // When chat is hidden, let run logs expand.
    if (mainBody) {
      mainBody.classList.toggle('noChat', !showChat);
      mainBody.classList.toggle('noLogs', !showRunLogs);
    }

    // If chat isn't visible, don't constrain log height with a stored splitter height.
    if (run && !(showChat && showRunLogs)) {
      run.style.height = '';
    } else if (run && (showChat && showRunLogs)) {
      const savedH = loadPref('ursa.ui.runSectionHeight', null);
      if (typeof savedH === 'number' && Number.isFinite(savedH) && savedH > 0) {
        run.style.height = `${Math.round(savedH)}px`;
      }
    }

    // Splitters only make sense when both adjacent panels are visible.
    if (leftSplit) leftSplit.classList.toggle('hidden', !showMain);
    if (rightSplit) rightSplit.classList.toggle('hidden', !showMain || !showArtifacts);

    // Update toggle button labels (all toggles live on the left panel).
    const tChat = $('#toggleChatBtn');
    if (tChat) tChat.textContent = showChat ? 'Hide chat' : 'Show chat';
    const tLogs = $('#toggleLogsBtn');
    if (tLogs) tLogs.textContent = showRunLogs ? 'Hide logs' : 'Show logs';
    const tArt = $('#toggleArtifactsBtn');
    if (tArt) tArt.textContent = showArtifacts ? 'Hide artifacts' : 'Show artifacts';

    savePref('ursa.ui.showChat', showChat);
    savePref('ursa.ui.showRunLogs', showRunLogs);
    savePref('ursa.ui.showArtifacts', showArtifacts);
  }

  function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }

  function setupSplitters() {
    const left = $('#leftPanel');
    const right = $('#rightPanel');

    const leftW = loadPref('ursa.ui.leftWidth', null);
    if (left && typeof leftW === 'number' && Number.isFinite(leftW)) left.style.width = `${Math.round(leftW)}px`;

    const rightW = loadPref('ursa.ui.rightWidth', null);
    if (right && typeof rightW === 'number' && Number.isFinite(rightW)) right.style.width = `${Math.round(rightW)}px`;

    initSplitter($('#leftSplitter'), left, { side: 'left', minPx: 220, maxPx: 700, prefKey: 'ursa.ui.leftWidth' });
    initSplitter($('#rightSplitter'), right, { side: 'right', minPx: 260, maxPx: 1200, prefKey: 'ursa.ui.rightWidth' });
    setupMiddleSplitter();
  }

  function initSplitter(splitterEl, panelEl, cfg) {
    if (!splitterEl || !panelEl) return;

    let dragging = false;
    let startX = 0;
    let startW = 0;

    const onMove = (e) => {
      if (!dragging) return;
      const x = e.clientX;
      let w = startW;
      if (cfg.side === 'left') w = startW + (x - startX);
      else w = startW - (x - startX);
      w = clamp(w, cfg.minPx, cfg.maxPx);
      panelEl.style.width = `${Math.round(w)}px`;
    };

    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      try {
        const w = panelEl.getBoundingClientRect().width;
        savePref(cfg.prefKey, Math.round(w));
      } catch {}
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    splitterEl.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      startX = e.clientX;
      startW = panelEl.getBoundingClientRect().width;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
      e.preventDefault();
    });

    splitterEl.addEventListener('dblclick', () => {
      panelEl.style.width = '';
      savePref(cfg.prefKey, null);
    });
  }

  function setupMiddleSplitter() {
    const split = $('#midSplitter');
    const mainBody = $('#mainBody');
    const run = $('#runSection');
    if (!split || !mainBody || !run) return;

    const savedH = loadPref('ursa.ui.runSectionHeight', null);
    if (state.showChat && state.showRunLogs && typeof savedH === 'number' && Number.isFinite(savedH) && savedH > 0) {
      run.style.height = `${Math.round(savedH)}px`;
    }

    let dragging = false;
    let startY = 0;
    let startH = 0;

    const onMove = (e) => {
      if (!dragging) return;
      const dy = e.clientY - startY;
      const bodyH = mainBody.getBoundingClientRect().height;

      const minRun = 160;
      const minConv = 220;
      const maxRun = Math.max(minRun, bodyH - minConv - 24);

      // Moving mouse up should increase log height.
      let h = startH - dy;
      h = clamp(h, minRun, maxRun);
      run.style.height = `${Math.round(h)}px`;
    };

    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      try {
        const h = run.getBoundingClientRect().height;
        savePref('ursa.ui.runSectionHeight', Math.round(h));
      } catch {}
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    split.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      startY = e.clientY;
      startH = run.getBoundingClientRect().height;
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
      e.preventDefault();
    });

    split.addEventListener('dblclick', () => {
      run.style.height = '';
      savePref('ursa.ui.runSectionHeight', null);
    });
  }

  function sanitizeUrl(u) {
    u = String(u || '').trim();
    if (u.startsWith('http://') || u.startsWith('https://') || u.startsWith('/')) return u;
    return '#';
  }

  function mdToHtml(md) {
    // Minimal, safe markdown renderer:
    // - escapes HTML
    // - supports fenced code blocks, inline code, basic emphasis, links
    md = String(md ?? '').replace(/\r\n/g, '\n');
    const src = escHtml(md);

    const out = [];
    const re = /```([a-zA-Z0-9_+-]+)?\n([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = re.exec(src)) !== null) {
      out.push({type:'text', text: src.slice(last, m.index)});
      out.push({type:'code', lang: (m[1] || '').trim(), code: m[2] || ''});
      last = re.lastIndex;
    }
    out.push({type:'text', text: src.slice(last)});

    function renderText(t) {
      // inline code
      t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
      // bold/italic (very small subset)
      t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      // links
      t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (all, label, url) => {
        const href = sanitizeUrl(url);
        return `<a href="${escHtml(href)}" target="_blank" rel="noreferrer">${label}</a>`;
      });

      // headings (line-based)
      t = t.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
      t = t.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
      t = t.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');

      // paragraphs: split on blank lines
      const blocks = t.split(/\n{2,}/).map(b => b.trim()).filter(Boolean);
      return blocks.map(b => {
        // preserve single newlines
        b = b.replace(/\n/g, '<br>');
        // if already a heading tag, don't wrap
        if (/^<h[1-3]>/.test(b)) return b;
        return `<p>${b}</p>`;
      }).join('\n');
    }

    const html = out.map(part => {
      if (part.type === 'code') {
        const lang = part.lang ? ` data-lang="${escHtml(part.lang)}"` : '';
        return `<pre class="codeblock"><code${lang}>${part.code}</code></pre>`;
      }
      return renderText(part.text);
    }).join('\n');

    return `<div class="md">${html || '<p class="muted">(empty)</p>'}</div>`;
  }

  function fmtTime(ts) {
    try {
      const d = new Date(ts);
      return d.toLocaleString();
    } catch { return ts || ''; }
  }

  function setStatus(text) {
    const el = $('#runStatus');
    if (el) el.textContent = text || '';
  }

  function setRunLogMeta(text) {
    const el = $('#runLogMeta');
    if (el) el.textContent = text || '';
  }

  function clearLogs() {
    state.logs.stdout = '';
    state.logs.stderr = '';
    const o = $('#stdoutLog');
    const e = $('#stderrLog');
    if (o) o.innerHTML = '';
    if (e) e.innerHTML = '';
  }

  function xterm256ToRgb(n) {
    n = Number(n);
    if (!Number.isFinite(n)) return [255, 255, 255];
    n = Math.max(0, Math.min(255, Math.floor(n)));

    // 0-15: system colors
    const sys = [
      [0,0,0],[205,49,49],[13,188,121],[229,229,16],[36,114,200],[188,63,188],[17,168,205],[229,229,229],
      [102,102,102],[241,76,76],[35,209,139],[245,245,67],[59,142,234],[214,112,214],[41,184,219],[255,255,255]
    ];
    if (n < 16) return sys[n];

    // 16-231: 6x6x6 cube
    if (n >= 16 && n <= 231) {
      const idx = n - 16;
      const r = Math.floor(idx / 36);
      const g = Math.floor((idx % 36) / 6);
      const b = idx % 6;
      const conv = (v) => (v === 0 ? 0 : 55 + v * 40);
      return [conv(r), conv(g), conv(b)];
    }

    // 232-255: grayscale
    const gray = 8 + (n - 232) * 10;
    return [gray, gray, gray];
  }

  function ansiToHtml(raw) {
    raw = String(raw ?? '');

    // Drop OSC sequences (e.g. hyperlinks) and other non-printing controls that
    // commonly appear in rich output.
    raw = raw.replace(/\x00/g, '');
    raw = raw.replace(/\x1b\][\s\S]*?(?:\x07|\x1b\\)/g, '');

    let out = '';
    let i = 0;
    let currentCss = '';
    const st = { fg: null, bg: null, bold: false, dim: false, italic: false, underline: false };

    function styleToCss() {
      const props = [];
      if (st.fg) props.push(`color:${st.fg}`);
      if (st.bg) props.push(`background-color:${st.bg}`);
      if (st.bold) props.push('font-weight:700');
      if (st.underline) props.push('text-decoration: underline');
      if (st.italic) props.push('font-style: italic');
      if (st.dim) props.push('opacity:0.85');
      return props.join(';');
    }

    function applyCss(nextCss) {
      if (nextCss === currentCss) return;
      if (currentCss) out += '</span>';
      currentCss = nextCss;
      if (currentCss) out += `<span style="${currentCss}">`;
    }

    function setFgXterm(n) {
      const [r,g,b] = xterm256ToRgb(n);
      st.fg = `rgb(${r},${g},${b})`;
    }
    function setBgXterm(n) {
      const [r,g,b] = xterm256ToRgb(n);
      st.bg = `rgb(${r},${g},${b})`;
    }

    function applySgr(params) {
      const parts = (params === '' ? [0] : params.split(';').map(x => parseInt(x, 10)).filter(Number.isFinite));
      for (let k = 0; k < parts.length; k++) {
        const c = parts[k];
        if (c === 0) { st.fg = null; st.bg = null; st.bold = false; st.dim = false; st.italic = false; st.underline = false; }
        else if (c === 1) st.bold = true;
        else if (c === 2) st.dim = true;
        else if (c === 3) st.italic = true;
        else if (c === 4) st.underline = true;
        else if (c === 22) { st.bold = false; st.dim = false; }
        else if (c === 23) st.italic = false;
        else if (c === 24) st.underline = false;
        else if (c === 39) st.fg = null;
        else if (c === 49) st.bg = null;
        else if (c >= 30 && c <= 37) setFgXterm(c - 30);
        else if (c >= 90 && c <= 97) setFgXterm(8 + (c - 90));
        else if (c >= 40 && c <= 47) setBgXterm(c - 40);
        else if (c >= 100 && c <= 107) setBgXterm(8 + (c - 100));
        else if (c === 38 || c === 48) {
          const isFg = (c === 38);
          const mode = parts[k + 1];
          if (mode === 2 && (k + 4) < parts.length) {
            const r = parts[k + 2], g = parts[k + 3], b = parts[k + 4];
            const rgb = `rgb(${r},${g},${b})`;
            if (isFg) st.fg = rgb; else st.bg = rgb;
            k += 4;
          } else if (mode === 5 && (k + 2) < parts.length) {
            const n = parts[k + 2];
            if (isFg) setFgXterm(n); else setBgXterm(n);
            k += 2;
          }
        }
      }
      applyCss(styleToCss());
    }

    while (i < raw.length) {
      const esc = raw.indexOf('\x1b[', i);
      if (esc === -1) {
        out += escHtml(raw.slice(i));
        break;
      }
      out += escHtml(raw.slice(i, esc));

      // Parse CSI sequence: ESC [ ... <cmd>
      let j = esc + 2;
      while (j < raw.length && !/[A-Za-z]/.test(raw[j])) j++;
      if (j >= raw.length) break;
      const cmd = raw[j];
      const params = raw.slice(esc + 2, j);

      if (cmd === 'm') {
        applySgr(params);
      }
      // Ignore other CSI commands (cursor movement, erase, etc.)
      i = j + 1;
    }

    if (currentCss) out += '</span>';
    return out;
  }

  function scheduleLogRender(which) {
    const key = (which === 'stderr') ? 'stderr' : 'stdout';
    if (state._renderTimers[key]) return;
    state._renderTimers[key] = setTimeout(() => {
      state._renderTimers[key] = null;
      const el = (key === 'stderr') ? $('#stderrLog') : $('#stdoutLog');
      if (!el) return;
      const raw = (key === 'stderr') ? state.logs.stderr : state.logs.stdout;
      el.innerHTML = ansiToHtml(raw);
      el.scrollTop = el.scrollHeight;
    }, 60);
  }

  function stripBackspaces(s) {
    s = String(s ?? '');
    if (!s.includes('\b')) return s;
    const out = [];
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (ch === '\b') {
        if (out.length) out.pop();
      } else {
        out.push(ch);
      }
    }
    return out.join('');
  }

  function appendLog(stream, text) {
    const cap = 250000;
    const key = (stream === 'stderr') ? 'stderr' : 'stdout';
    text = String(text ?? '');

    // Some console UIs emit backspaces for spinners.
    text = stripBackspaces(text);

    // Make progress-style output readable even without a real terminal.
    text = text.replace(/\r(?!\n)/g, '\n');

    if (key === 'stderr') {
      state.logs.stderr = (state.logs.stderr + text);
      if (state.logs.stderr.length > cap) state.logs.stderr = state.logs.stderr.slice(-cap);
    } else {
      state.logs.stdout = (state.logs.stdout + text);
      if (state.logs.stdout.length > cap) state.logs.stdout = state.logs.stdout.slice(-cap);
    }

    scheduleLogRender(key);
  }

  function closeRunStream() {
    if (state.es) {
      try { state.es.close(); } catch {}
      state.es = null;
    }
  }

  async function loadRunEvents(runId, token) {
    let after = 0;
    let pages = 0;
    const limit = 5000;

    while (pages < 50) {
      if (token !== state._logToken) return;
      const res = await api('GET', `/runs/${encodeURIComponent(runId)}/events?after_seq=${after}&limit=${limit}`);
      const events = res.events || [];
      if (!events.length) break;

      for (const e of events) {
        if (token !== state._logToken) return;
        const seq = Number(e.seq || 0);
        if (Number.isFinite(seq) && seq > after) after = seq;

        if (e.type === 'log') {
          const p = e.payload || {};
          appendLog(p.stream || 'stdout', p.text || '');
        } else if (e.type === 'state_change') {
          const to = (e.payload && e.payload.to) || '';
          if (to) setStatus(`run ${runId} \u00b7 ${to}`);
        }
      }

      pages += 1;
      if (events.length < limit) break;
    }
  }

  function clearRunView() {
    closeRunStream();
    state.viewRunId = null;
    state.viewRunKind = 'none';
    clearLogs();
    setStatus('');
    setRunLogMeta('No run selected.');

    const cancelBtn = $('#cancelRunBtn');
    if (cancelBtn) cancelBtn.style.display = 'none';
  }

  async function showRunStatic(runId) {
    if (!runId) { clearRunView(); return; }
    if (state.viewRunId === runId && state.viewRunKind === 'static') return;

    state._logToken += 1;
    const token = state._logToken;

    closeRunStream();
    clearLogs();
    state.viewRunId = runId;
    state.viewRunKind = 'static';

    const cancelBtn = $('#cancelRunBtn');
    if (cancelBtn) cancelBtn.style.display = 'none';

    setRunLogMeta(`run ${runId} \u00b7 last run`);

    try {
      await loadRunEvents(runId, token);
    } catch (e) {
      appendLog('stderr', `\n[dashboard] Failed to load run logs: ${e.message}\n`);
    }
  }

  function showRunStream(runId) {
    if (!runId) return;
    if (state.viewRunId === runId && state.viewRunKind === 'stream' && state.es) return;

    state._logToken += 1;
    const token = state._logToken;

    closeRunStream();
    clearLogs();
    state.viewRunId = runId;
    state.viewRunKind = 'stream';

    const cancelBtn = $('#cancelRunBtn');
    if (cancelBtn) cancelBtn.style.display = '';

    setRunLogMeta(`run ${runId} \u00b7 running`);

    const es = new EventSource(`/runs/${encodeURIComponent(runId)}/stream`);
    state.es = es;

    const onState = (ev) => {
      if (token !== state._logToken) return;
      try {
        const e = JSON.parse(ev.data);
        const to = (e.payload && e.payload.to) || '';
        setStatus(to ? `run ${runId} \u00b7 ${to}` : `run ${runId}`);
        if (to) setRunLogMeta(`run ${runId} \u00b7 ${to}`);

        if (['succeeded','failed','cancelled'].includes(to)) {
          // Refresh session + workspace after terminal.
          setTimeout(async () => {
            if (token !== state._logToken) return;
            closeRunStream();
            await refreshSessions();
            if (state.activeSessionId) await loadSession(state.activeSessionId);
          }, 250);
        }
      } catch {}
    };

    const onLog = (ev) => {
      if (token !== state._logToken) return;
      try {
        const e = JSON.parse(ev.data);
        const p = e.payload || {};
        appendLog(p.stream || 'stdout', p.text || '');
      } catch {}
    };

    const onFinal = async (_ev) => {
      if (token !== state._logToken) return;
      // Assistant message is appended server-side at run completion.
      await refreshSessions();
      if (state.activeSessionId) await loadSession(state.activeSessionId);
    };

    es.addEventListener('state_change', onState);
    es.addEventListener('log', onLog);
    es.addEventListener('final_output', onFinal);

    es.onerror = async () => {
      if (token !== state._logToken) return;
      // If connection drops, poll session/run state.
      try {
        await refreshSessions();
        if (state.activeSessionId) await loadSession(state.activeSessionId);
      } catch {}
    };
  }

  function renderAgents() {
    const list = $('#agentList');
    if (!list) return;
    list.innerHTML = '';

    const agents = state.agents.slice().sort((a,b) => (a.display_name||a.agent_id).localeCompare(b.display_name||b.agent_id));
    for (const a of agents) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'agentBtn';
      btn.onclick = () => startSession(a.agent_id);

      const top = document.createElement('div');
      top.className = 'row';

      const name = document.createElement('div');
      name.className = 'agentName';
      name.textContent = a.display_name || a.agent_id;
      top.appendChild(name);

      const start = document.createElement('span');
      start.className = 'pill action';
      start.textContent = 'New session';
      top.appendChild(start);

      const desc = document.createElement('div');
      desc.className = 'agentDesc';
      desc.textContent = a.description || '';

      btn.appendChild(top);
      btn.appendChild(desc);
      list.appendChild(btn);
    }

    if (!agents.length) {
      list.innerHTML = '<div class="muted">No agents found.</div>';
    }
  }

  function renderSessions() {
    const list = $('#sessionList');
    if (!list) return;
    list.innerHTML = '';

    for (const s of state.sessions) {
      const row = document.createElement('div');
      row.className = 'sessionRow';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'histItem' + (state.activeSessionId === s.session_id ? ' selected' : '');
      btn.onclick = () => loadSession(s.session_id);

      const title = document.createElement('div');
      title.textContent = s.title || s.session_id;

      const meta = document.createElement('div');
      meta.className = 'muted small';
      const agentName = state.agentsById.get(s.agent_id)?.display_name || s.agent_id;
      const active = s.active_run_id ? ' (running)' : '';
      meta.textContent = `${agentName} \u00b7 ${fmtTime(s.updated_at)}${active}`;

      btn.appendChild(title);
      btn.appendChild(meta);

      const renameBtn = document.createElement('button');
      renameBtn.type = 'button';
      renameBtn.className = 'sessActBtn';
      renameBtn.textContent = 'Rename';
      renameBtn.title = 'Rename session';
      renameBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const cur = s.title || '';
        const next = prompt('Rename session', cur);
        if (next === null) return;
        await renameSession(s.session_id, next);
      };

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'sessActBtn danger';
      delBtn.textContent = 'Delete';
      delBtn.title = s.active_run_id ? 'Cannot delete while a run is active' : 'Delete session';
      delBtn.disabled = !!s.active_run_id;
      delBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        await deleteSessionUI(s.session_id);
      };

      row.appendChild(btn);
      row.appendChild(renameBtn);
      row.appendChild(delBtn);
      list.appendChild(row);
    }

    if (!state.sessions.length) {
      list.innerHTML = '<div class="muted">No sessions yet. Start one from the Agents list.</div>';
    }
  }

  function renderActiveSession() {
    const title = $('#activeSessionTitle');
    const meta = $('#activeSessionMeta');
    const msgs = $('#sessionMessages');
    const wsTitle = $('#workspaceTitle');

    if (!state.activeSession) {
      if (title) title.textContent = 'No session selected';
      if (meta) meta.textContent = '';
      if (msgs) msgs.innerHTML = '<div class="muted">Pick an agent to start a new session, or select a session on the left.</div>';
      if (wsTitle) wsTitle.textContent = 'Session artifacts';
      setStatus('');
      clearRunView();
      return;
    }

    const s = state.activeSession.session;
    const agentName = state.agentsById.get(s.agent_id)?.display_name || s.agent_id;

    if (title) title.textContent = s.title || 'Session';
    if (meta) meta.textContent = `${agentName} \u00b7 created ${fmtTime(s.created_at)}`;

    if (wsTitle) wsTitle.textContent = `Artifacts \u00b7 ${s.session_id}`;

    const activeRunId = s.active_run_id || null;
    const lastRunId = s.last_run_id || null;
    const viewRunId = activeRunId || lastRunId || null;

    if (activeRunId) setStatus(`run ${activeRunId} \u00b7 running`);
    else if (lastRunId) setStatus(`run ${lastRunId} \u00b7 last run`);
    else setStatus('');

    if (!msgs) return;
    msgs.innerHTML = '';

    for (const m of (state.activeSession.messages || [])) {
      const row = document.createElement('div');
      row.className = 'msgRow ' + (m.role || '');

      const head = document.createElement('div');
      head.className = 'msgHead';

      const who = document.createElement('span');
      who.className = 'who';
      who.textContent = (m.role === 'assistant') ? 'Assistant' : (m.role === 'system' ? 'System' : 'You');
      head.appendChild(who);

      const t = document.createElement('span');
      t.className = 'muted small';
      t.textContent = ' \u00b7 ' + fmtTime(m.ts);
      head.appendChild(t);

      if (m.run_id) {
        const r = document.createElement('span');
        r.className = 'muted small mono';
        r.textContent = ' \u00b7 ' + m.run_id;
        head.appendChild(r);
      }

      const body = document.createElement('div');
      body.className = 'bubble ' + (m.role || '');
      if (m.role === 'assistant' || m.role === 'system') {
        body.innerHTML = mdToHtml(m.text || '');
      } else {
        body.textContent = m.text || '';
      }

      row.appendChild(head);
      row.appendChild(body);
      msgs.appendChild(row);
    }

    // Scroll to bottom
    msgs.scrollTop = msgs.scrollHeight;

    // Show run logs affiliated with this session:
    // - If a run is currently active, stream it.
    // - Otherwise, show the last completed run logs.
    if (activeRunId) {
      showRunStream(activeRunId);
    } else if (lastRunId) {
      showRunStatic(lastRunId).catch(err => console.error(err));
    } else {
      clearRunView();
    }
  }

  async function refreshAgents() {
    const res = await api('GET', '/agents');
    state.agents = res.agents || [];
    state.agentsById = new Map(state.agents.map(a => [a.agent_id, a]));
    renderAgents();
    renderSessions();
  }

  async function refreshSessions() {
    const res = await api('GET', '/sessions?limit=200');
    state.sessions = res.sessions || [];
    renderSessions();
  }

  async function startSession(agentId) {
    const res = await api('POST', '/sessions', { agent_id: agentId });
    await refreshSessions();
    await loadSession(res.session.session_id);
  }

  async function renameSession(sessionId, newTitle) {
    const title = String(newTitle || '').trim();
    if (!title) {
      alert('Title cannot be empty');
      return;
    }
    await api('PATCH', `/sessions/${encodeURIComponent(sessionId)}`, { title });
    await refreshSessions();
    if (state.activeSessionId === sessionId) {
      state.activeSession = await api('GET', `/sessions/${encodeURIComponent(sessionId)}`);
      renderActiveSession();
    }
  }

  async function deleteSessionUI(sessionId) {
    if (state.activeSessionId === sessionId && state.activeSession?.session?.active_run_id) {
      alert('Cannot delete a session with an active run');
      return;
    }
    if (!confirm('Delete this session? This will remove its messages and workspace files.')) return;

    await api('DELETE', `/sessions/${encodeURIComponent(sessionId)}`);

    if (state.activeSessionId === sessionId) {
      state.activeSessionId = null;
      state.activeSession = null;
      savePref('ursa.ui.activeSessionId', '');
      renderActiveSession();
      renderWorkspace([]);
      clearLogs();
    }

    await refreshSessions();
  }

  async function loadSession(sessionId) {
    state.activeSessionId = sessionId;
    savePref('ursa.ui.activeSessionId', sessionId);
    state.activeSession = await api('GET', `/sessions/${encodeURIComponent(sessionId)}`);
    renderSessions();
    renderActiveSession();
    await refreshWorkspace();
  }

  function isMdFile(path) {
    return String(path || '').toLowerCase().endsWith('.md');
  }

  async function openArtifact(file) {
    const iframe = $('#previewFrame');
    const txt = $('#artifactTextPreview');
    if (!state.activeSessionId) return;

    $$('#workspaceFiles .fileItem').forEach(x => x.classList.remove('selected'));
    const btn = $(`#workspaceFiles .fileItem[data-path="${CSS.escape(file.rel_path)}"]`);
    if (btn) btn.classList.add('selected');

    // Prefer client-side rendering for small text files
    const isText = file.kind === 'text';
    const smallEnough = (file.size_bytes || 0) <= 1_000_000;
    if (isText && smallEnough) {
      const url = `/sessions/${encodeURIComponent(state.activeSessionId)}/workspace/file?path=${encodeURIComponent(file.rel_path)}&disposition=inline`;
      const resp = await fetch(url);
      const raw = await resp.text();

      if (iframe) { iframe.style.display = 'none'; iframe.src = 'about:blank'; }
      if (txt) {
        txt.style.display = '';
        if (isMdFile(file.rel_path)) {
          txt.innerHTML = mdToHtml(raw);
        } else {
          txt.innerHTML = `<pre class="plain"><code>${escHtml(raw)}</code></pre>`;
        }
      }
      return;
    }

    // Fallback to safe server preview for images, PDFs, large files.
    const prev = `/sessions/${encodeURIComponent(state.activeSessionId)}/workspace/file/preview?path=${encodeURIComponent(file.rel_path)}`;
    if (txt) { txt.style.display = 'none'; txt.innerHTML = ''; }
    if (iframe) { iframe.style.display = ''; iframe.src = prev; }
  }

  function renderWorkspace(files) {
    const list = $('#workspaceFiles');
    const hint = $('#workspaceHint');
    if (!list) return;
    list.innerHTML = '';

    if (!state.activeSessionId) {
      if (hint) hint.textContent = 'Select a session to view its workspace.';
      return;
    }
    if (hint) hint.textContent = 'Click a file to preview, or download it.';

    for (const f of (files || [])) {
      const row = document.createElement('div');
      row.className = 'fileRow';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fileItem';
      btn.dataset.path = f.rel_path;
      btn.textContent = f.rel_path;
      btn.onclick = () => openArtifact(f);

      const dl = document.createElement('a');
      dl.className = 'fileDl';
      dl.textContent = 'Download';
      dl.href = `/sessions/${encodeURIComponent(state.activeSessionId)}/workspace/file?path=${encodeURIComponent(f.rel_path)}&disposition=attachment`;
      dl.setAttribute('download', '');
      dl.onclick = (e) => { e.stopPropagation(); };

      row.appendChild(btn);
      row.appendChild(dl);
      list.appendChild(row);
    }

    if (!files || !files.length) {
      list.innerHTML = '<div class="muted">No files in this session workspace yet.</div>';
    }
  }

  async function refreshWorkspace() {
    if (!state.activeSessionId) {
      renderWorkspace([]);
      return;
    }
    try {
      const res = await api('GET', `/sessions/${encodeURIComponent(state.activeSessionId)}/workspace`);
      renderWorkspace(res.files || []);
    } catch (e) {
      const list = $('#workspaceFiles');
      if (list) list.innerHTML = `<div class="muted">Failed to load workspace: ${escHtml(e.message)}</div>`;
    }
  }

  async function sendMessage() {
    if (!state.activeSessionId) return;
    const ta = $('#messageInput');
    const text = (ta && ta.value || '').trim();
    if (!text) return;

    const sendBtn = $('#sendMsgBtn');
    if (sendBtn) sendBtn.disabled = true;

    try {
      await api('POST', `/sessions/${encodeURIComponent(state.activeSessionId)}/message`, { text });
      if (ta) ta.value = '';
      await loadSession(state.activeSessionId);
      await refreshSessions();
    } catch (e) {
      alert('Failed to send: ' + e.message);
    } finally {
      if (sendBtn) sendBtn.disabled = false;
    }
  }

  async function cancelActiveRun() {
    const runId = state.activeSession?.session?.active_run_id;
    if (!runId) return;
    try {
      await api('POST', `/runs/${encodeURIComponent(runId)}/cancel`, { reason: 'user_request' });
    } catch (e) {
      alert('Cancel failed: ' + e.message);
    }
  }

  function _cloneJson(obj) {
    return JSON.parse(JSON.stringify(obj || {}));
  }

  function setMcpStatus(msg) {
    const el = $('#mcpStatus');
    if (el) el.textContent = msg || '';
  }

  function renderMcpServers() {
    const list = $('#mcpServersList');
    if (!list) return;
    const servers = state._mcpServers || {};
    list.innerHTML = '';
    const names = Object.keys(servers).sort();
    if (!names.length) {
      const span = document.createElement('span');
      span.className = 'muted small';
      span.textContent = '(none)';
      list.appendChild(span);
      return;
    }
    for (const name of names) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn';
      btn.style.padding = '6px 8px';
      btn.style.borderRadius = '999px';
      btn.textContent = name;
      if (state._mcpEditKey === name) {
        btn.style.borderColor = '#0b57d0';
        btn.style.boxShadow = '0 0 0 2px rgba(11,87,208,0.10)';
      }
      btn.onclick = () => selectMcpServer(name);
      list.appendChild(btn);
    }
  }

  function clearMcpEditor() {
    state._mcpEditKey = null;
    const n = $('#mcp_name');
    const j = $('#mcp_json');
    if (n) n.value = '';
    if (j) j.value = '';
    setMcpStatus('');
    renderMcpServers();
  }

  function selectMcpServer(name) {
    const servers = state._mcpServers || {};
    if (!(name in servers)) return;
    state._mcpEditKey = name;
    $('#mcp_name').value = name;
    $('#mcp_json').value = JSON.stringify(servers[name], null, 2);
    setMcpStatus('Editing ' + name);
    renderMcpServers();
  }

  function upsertMcpServerFromEditor() {
    const name = ($('#mcp_name').value || '').trim();
    const txt = ($('#mcp_json').value || '').trim();
    if (!name) {
      setMcpStatus('Server name is required.');
      return;
    }
    if (!txt) {
      setMcpStatus('Server JSON is required.');
      return;
    }
    let cfg;
    try {
      cfg = JSON.parse(txt);
    } catch (e) {
      setMcpStatus('Invalid JSON: ' + (e && e.message ? e.message : String(e)));
      return;
    }
    if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) {
      setMcpStatus('Server config must be a JSON object.');
      return;
    }

    if (!state._mcpServers) state._mcpServers = {};
    const prevKey = state._mcpEditKey;
    if (prevKey && prevKey !== name) delete state._mcpServers[prevKey];
    state._mcpServers[name] = cfg;
    state._mcpEditKey = name;
    setMcpStatus('Staged: ' + name);
    renderMcpServers();
  }

  function removeMcpServerFromEditor() {
    const name = state._mcpEditKey || ($('#mcp_name').value || '').trim();
    if (!name) {
      setMcpStatus('Select a server to remove.');
      return;
    }
    const servers = state._mcpServers || {};
    if (!(name in servers)) {
      setMcpStatus('Not found: ' + name);
      return;
    }
    if (!confirm('Remove MCP server ' + name + '?')) return;
    delete servers[name];
    clearMcpEditor();
    setMcpStatus('Removed (staged). Click Save to persist.');
    renderMcpServers();
  }

  async function loadSettings() {
    const res = await api('GET', '/settings');
    state.settings = res.settings || {};
    const llm = state.settings.llm || {};
    const runner = state.settings.runner || {};
    const mcp = state.settings.mcp || {};

    $('#set_base_url').value = llm.base_url || '';
    $('#set_model').value = llm.model || '';
    $('#set_api_key_env_var').value = llm.api_key_env_var || '';
    $('#set_max_tokens').value = llm.max_tokens ?? '';
    $('#set_temperature').value = llm.temperature ?? '';
    $('#set_timeout').value = runner.timeout_seconds ?? '';

    // MCP
    state._mcpServers = _cloneJson(mcp.servers || {});
    state._mcpEditKey = null;
    clearMcpEditor();
    renderMcpServers();

    const upd = $('#settingsUpdated');
    if (upd) upd.textContent = 'Loaded.';
  }

  async function saveSettings() {
    const patch = {
      llm: {
        base_url: ($('#set_base_url').value || '').trim() || null,
        model: ($('#set_model').value || '').trim() || null,
        api_key_env_var: ($('#set_api_key_env_var').value || '').trim() || null,
        max_tokens: ($('#set_max_tokens').value === '' ? null : Number($('#set_max_tokens').value)),
        temperature: ($('#set_temperature').value === '' ? null : Number($('#set_temperature').value)),
      },
      runner: {
        timeout_seconds: ($('#set_timeout').value === '' ? null : Number($('#set_timeout').value)),
      },
      mcp: {
        servers: (state._mcpServers || {}),
      }
    };

    // remove nulls to avoid overwriting with null unless explicitly intended
    // (but preserve empty objects where we need to allow clearing settings, e.g. mcp.servers).
    function compact(o, path='') {
      if (!o || typeof o !== 'object') return o;
      const out = Array.isArray(o) ? [] : {};
      for (const [k,v] of Object.entries(o)) {
        const p = path ? (path + '.' + k) : k;
        if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) continue;
        if (typeof v === 'object' && !Array.isArray(v)) {
          const c = compact(v, p);
          const empty = c && typeof c === 'object' && !Array.isArray(c) && Object.keys(c).length === 0;
          if (!empty || p === 'mcp.servers') out[k] = c;
        } else out[k] = v;
      }
      return out;
    }

    const payload = { patch: compact(patch) };
    const res = await api('PATCH', '/settings', payload);
    state.settings = res.settings || {};

    const saved = $('#settingsSaved');
    if (saved) {
      saved.textContent = 'Saved.';
      setTimeout(() => { saved.textContent = ''; }, 1500);
    }
  }

  function setupUi() {
    // New panel toggles (sidebar is always visible)
    // Migrate old prefs if present.
    const oldHideMain = loadPref('ursa.ui.hideMain', null);
    const oldHideRight = loadPref('ursa.ui.hideRight', null);

    state.showChat = !!loadPref('ursa.ui.showChat', oldHideMain === null ? true : !oldHideMain);
    state.showRunLogs = !!loadPref('ursa.ui.showRunLogs', oldHideMain === null ? true : !oldHideMain);
    state.showArtifacts = !!loadPref('ursa.ui.showArtifacts', oldHideRight === null ? true : !oldHideRight);

    applyPanelVisibility();

    const tChat = $('#toggleChatBtn');
    if (tChat) tChat.onclick = () => { state.showChat = !state.showChat; applyPanelVisibility(); };

    const tLogs = $('#toggleLogsBtn');
    if (tLogs) tLogs.onclick = () => { state.showRunLogs = !state.showRunLogs; applyPanelVisibility(); };

    const tArt = $('#toggleArtifactsBtn');
    if (tArt) tArt.onclick = () => { state.showArtifacts = !state.showArtifacts; applyPanelVisibility(); };

    $('#refreshSessionsBtn').onclick = async () => { await refreshSessions(); };
    $('#refreshFilesBtn').onclick = async () => { await refreshWorkspace(); };

    // Upload files into the active session workspace
    const uploadBtn = $('#uploadFilesBtn');
    const uploadInput = $('#uploadInput');
    if (uploadBtn && uploadInput) {
      uploadBtn.onclick = () => {
        if (!state.activeSessionId) {
          alert('Select a session first');
          return;
        }
        uploadInput.value = '';
        uploadInput.click();
      };

      uploadInput.onchange = async () => {
        if (!state.activeSessionId) return;
        const picked = Array.from(uploadInput.files || []);
        if (!picked.length) return;

        const dir = prompt('Upload into subfolder (optional)', '')
        if (dir === null) return;

        async function doUpload(overwrite) {
          const fd = new FormData();
          for (const f of picked) fd.append('files', f, f.name);
          fd.append('dir', dir || '');
          fd.append('overwrite', overwrite ? 'true' : 'false');
          const resp = await fetch(`/sessions/${encodeURIComponent(state.activeSessionId)}/workspace/upload`, { method: 'POST', body: fd });
          if (!resp.ok) {
            let msg = resp.status + ' ' + resp.statusText;
            try { const j = await resp.json(); if (j && j.detail) msg = j.detail; } catch {}
            throw new Error(msg);
          }
          return await resp.json();
        }

        try {
          await doUpload(false);
        } catch (e) {
          const msg = String(e && e.message ? e.message : e);
          if (msg.includes('already exists') || msg.includes('409')) {
            if (confirm(msg + '\n\nOverwrite existing files?')) {
              await doUpload(true);
            } else {
              return;
            }
          } else {
            alert(msg);
            return;
          }
        }

        await refreshWorkspace();
      };
    }

    $('#clearLogsBtn').onclick = () => { clearLogs(); };

    $('#sendMsgBtn').onclick = sendMessage;
    $('#cancelRunBtn').onclick = cancelActiveRun;

    $('#messageInput').addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
      }
    });

    setupSplitters();

    // Settings modal
    const modal = $('#settingsModal');
    $('#openSettingsBtn').onclick = async () => {
      modal.classList.add('open');
      await loadSettings();
    };
    $('#closeSettingsBtn').onclick = () => modal.classList.remove('open');
    $('#settingsBackdrop').onclick = () => modal.classList.remove('open');

    const mAdd = $('#mcpAddUpdateBtn');
    if (mAdd) mAdd.onclick = upsertMcpServerFromEditor;
    const mRem = $('#mcpRemoveBtn');
    if (mRem) mRem.onclick = removeMcpServerFromEditor;
    const mClr = $('#mcpClearBtn');
    if (mClr) mClr.onclick = clearMcpEditor;

    $('#saveSettingsBtn').onclick = saveSettings;
  }

  async function init() {
    setupUi();
    await refreshAgents();
    await refreshSessions();

    // Restore last selected session if possible; otherwise select most recent.
    const remembered = loadPref('ursa.ui.activeSessionId', null);
    if (!state.activeSessionId && state.sessions.length) {
      const exists = remembered && state.sessions.some(s => s.session_id === remembered);
      await loadSession(exists ? remembered : state.sessions[0].session_id);
    }

    // periodic refresh
    setInterval(() => { refreshSessions().catch(() => {}); }, 5000);
  }

  init().catch(err => {
    console.error(err);
    const main = $('#sessionMessages');
    if (main) main.innerHTML = `<div class="muted">Failed to initialize: ${escHtml(err.message)}</div>`;
  });
})();
"""

    DASHBOARD_CSS = r"""
:root {
  --bg: #ffffff;
  --panel: rgba(250, 250, 250, 0.92);
  --panelSolid: #fafafa;
  --border: #e2e2e2;
  --text: #111;
  --muted: #666;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --ursa-logo-url: url("/ui/ursa_logo.png");
}
body { margin: 0; color: var(--text); background: var(--bg); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; position: relative; }

/* Subtle branding background */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image: var(--ursa-logo-url);
  background-repeat: no-repeat;
  background-position: center;
  background-size: min(70vmin, 820px) auto;
  opacity: 0.05;
  pointer-events: none;
  z-index: 0;
  filter: grayscale(100%);
}

.app { display:flex; height: 100vh; width: 100vw; position: relative; z-index: 1; }
.sidebar, .workspace { overflow:auto; }
.main { overflow:hidden; }

.splitter {
  flex: 0 0 8px;
  cursor: col-resize;
  background: linear-gradient(to right, transparent, rgba(0,0,0,0.08), transparent);
}
.splitter.hidden { display:none; }

/* Generic utility used throughout JS */
.hidden { display:none !important; }

.hsplitter {
  flex: 0 0 8px;
  cursor: row-resize;
  background: linear-gradient(to bottom, transparent, rgba(0,0,0,0.08), transparent);
  border-radius: 6px;
}
.hsplitter.hidden { display:none; }

.sidebar {
  flex: 0 0 auto;
  width: 320px;
  min-width: 220px;
  max-width: 700px;
  border-right: 1px solid var(--border);
  padding: 12px;
  background: var(--panel);
}
.sidebar.hidden { display:none; }

.workspace {
  flex: 0 0 auto;
  width: 520px;
  min-width: 260px;
  max-width: 1200px;
  border-left: 1px solid var(--border);
  padding: 12px;
  background: var(--panel);
  display:flex;
  flex-direction:column;
  gap: 10px;
}
.workspace.hidden { display:none; }

.main {
  flex: 1 1 auto;
  min-width: 320px;
  padding: 12px;
  display:flex;
  flex-direction:column;
  min-height: 0;
}
.main.hidden { display:none; }

.mainBody {
  flex: 1 1 auto;
  display:flex;
  flex-direction:column;
  gap: 10px;
  min-height: 0;
}

.conversation { flex: 1 1 auto; display:flex; flex-direction:column; overflow:hidden; min-height: 220px; }
.runSection { flex: 0 0 auto; overflow:auto; min-height: 160px; }

/* If chat is hidden, let the run logs fill the center column. */
.mainBody.noChat .runSection { flex: 1 1 auto; min-height: 0; }

/* When the main chat is hidden, let the workspace expand to fill the window. */
.app.mainHidden .workspace {
  flex: 1 1 auto;
  width: auto;
  max-width: none;
}

.topbar { display:flex; align-items:center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.topbarCol { flex-direction: column; align-items: stretch; justify-content: flex-start; }
.topbarCol > .row { width: 100%; }
.brandRow { display:flex; align-items:flex-start; gap: 12px; }
.brandLogo { width: 75px; height: 75px; object-fit: contain; border-radius: 10px; background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.06); }

/* Brand sizing (left sidebar only) */
#leftPanel .brand .title { font-size: 16pt; line-height: 1.15; }
#leftPanel .brand .muted.small { font-size: 14pt; line-height: 1.2; }

.title { font-weight: 700; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
.row { display:flex; align-items:center; justify-content: space-between; gap: 8px; }
.mono { font-family: var(--mono); font-size: 12px; }

.section { border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: rgba(255,255,255,0.90); margin-bottom: 12px; backdrop-filter: blur(1px); }
.sectionHead { font-weight: 650; margin-bottom: 8px; }
.mainBody .section { margin-bottom: 0; }

.btn { border: 1px solid var(--border); background: #fff; padding: 8px 10px; border-radius: 10px; cursor: pointer; }
.btn:hover { border-color: #bbb; }
.btn.primary { background: #0b57d0; border-color: #0b57d0; color: #fff; }
.btn.danger { border-color: #cc3a3a; color: #cc3a3a; }
.btn.danger:hover { background: rgba(204,58,58,0.06); }

.agentBtn { width: 100%; text-align: left; border: 1px solid var(--border); background: #fff; padding: 10px; border-radius: 10px; margin-bottom: 10px; cursor: pointer; }
.agentName { font-weight: 650; }
.agentDesc { margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.3; }

.pill { font-size: 11px; padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); background: #fff; color: var(--muted); }
.pill.action { color: #0b57d0; border-color: rgba(11,87,208,0.35); }

.sessionRow { display:flex; gap: 8px; align-items: stretch; margin-bottom: 10px; }
.sessionRow .histItem { margin-bottom: 0; flex: 1 1 auto; }
.sessionRow .sessActBtn { flex: 0 0 auto; padding: 8px 10px; border-radius: 10px; border: 1px solid var(--border); background: #fff; cursor: pointer; font-size: 12px; }
.sessionRow .sessActBtn:hover { border-color: #bbb; }
.sessionRow .sessActBtn.danger { border-color: #cc3a3a; color: #cc3a3a; }
.sessionRow .sessActBtn.danger:hover { background: rgba(204,58,58,0.06); }

.histItem { width: 100%; text-align:left; border: 1px solid var(--border); background: #fff; padding: 10px; border-radius: 10px; margin-bottom: 10px; cursor: pointer; }
.histItem.selected { border-color: #0b57d0; box-shadow: 0 0 0 2px rgba(11,87,208,0.10); }

.messages {
  flex: 1 1 auto;
  min-height: 0;
  overflow:auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  background: #fff;
}

.msgRow { margin-bottom: 12px; }
.msgHead { font-weight: 650; margin-bottom: 6px; display:flex; gap: 8px; align-items: baseline; }
.msgHead .who { font-weight: 700; }

.bubble {
  border-radius: 12px;
  padding: 10px;
  background: #f6f6f6;
  white-space: pre-wrap;
  overflow-wrap:anywhere;
}
.bubble.user { background: #eef5ff; }

.composer { margin-top: 10px; }
textarea, input, select { font: inherit; }
textarea { width: 100%; min-height: 90px; resize: vertical; padding: 10px; border-radius: 10px; border: 1px solid var(--border); }

.logDetails { border: 1px solid var(--border); border-radius: 12px; padding: 10px; background: #fff; margin-bottom: 10px; }
.logDetails:last-child { margin-bottom: 0; }
.logDetails > summary { cursor: pointer; user-select: none; list-style: none; }
.logDetails > summary::-webkit-details-marker { display:none; }
.logDetails[open] > summary { margin-bottom: 8px; }

.logHead { font-weight: 650; font-family: var(--mono); font-size: 12px; color: var(--muted); }
.log { margin: 0; white-space: pre-wrap; overflow-wrap:anywhere; background: #0b0f14; color: #eaeef3; border-radius: 10px; padding: 10px; min-height: 140px; max-height: 52vh; overflow:auto; font-family: var(--mono); font-size: 12px; resize: vertical; box-sizing: border-box; width: 100%; }

.fileRow { display:flex; gap: 8px; align-items: stretch; margin-bottom: 8px; }

.fileItem { flex: 1 1 auto; text-align:left; border: 1px solid var(--border); background: #fff; padding: 8px 10px; border-radius: 10px; cursor:pointer; font-family: var(--mono); font-size: 12px; }
.fileItem.selected { border-color: #0b57d0; }

.fileDl { flex: 0 0 auto; display:inline-flex; align-items:center; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: #fff; color: #0b57d0; text-decoration: none; font-family: var(--mono); font-size: 12px; }
.fileDl:hover { border-color: #bbb; }

#filesDetails { flex: 0 0 auto; }
#workspaceFiles { max-height: 32vh; overflow:auto; padding-right: 4px; }

iframe { width: 100%; flex: 1 1 auto; min-height: 320px; border: 1px solid var(--border); border-radius: 12px; background:#fff; }

.artifactText { border: 1px solid var(--border); border-radius: 12px; padding: 10px; background: #fff; flex: 1 1 auto; min-height: 320px; overflow:auto; }
pre.plain { margin:0; white-space: pre; overflow:auto; font-family: var(--mono); font-size: 12px; }

/* Markdown styling */
.md h1, .md h2, .md h3 { margin: 10px 0 8px; }
.md p { margin: 0 0 10px; }
.md code { font-family: var(--mono); background: rgba(0,0,0,0.06); padding: 1px 4px; border-radius: 6px; }
.md pre.codeblock { margin: 10px 0; padding: 10px; border-radius: 10px; background: #0b0f14; color: #eaeef3; overflow:auto; }
.md pre.codeblock code { background: none; padding: 0; }
.md a { color: #0b57d0; text-decoration: none; }
.md a:hover { text-decoration: underline; }

/* Settings modal */
.modal { position: fixed; inset: 0; display:none; z-index: 30; }
.modal.open { display:block; }
.backdrop { position:absolute; inset:0; background: rgba(0,0,0,0.25); }
.modalCard { position:absolute; top: 8vh; left: 50%; transform: translateX(-50%); width: min(720px, 94vw); background:#fff; border-radius: 14px; border: 1px solid var(--border); padding: 14px; }
.fieldRow { display:grid; grid-template-columns: 170px 1fr; gap: 8px; align-items: center; margin-bottom: 8px; }
.label { color: var(--muted); font-size: 12px; }
.input { padding: 8px 10px; border-radius: 10px; border: 1px solid var(--border); }
textarea.input { width: 100%; box-sizing: border-box; resize: vertical; }

@media (max-width: 1100px) {
  .workspace { display:none; }
}
@media (max-width: 820px) {
  .sidebar { display:none; }
}
"""

    def _find_logo_path() -> Path | None:
        repo_root = Path(__file__).resolve().parent
        candidates = [
            repo_root / "logo.png",
        ]
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    return p
            except Exception:
                continue
        return None

    @app.get(
        "/ui/ursa_logo.png",
        include_in_schema=False,
        dependencies=[Depends(require_auth)],
    )
    def ui_logo() -> FileResponse:
        p = _find_logo_path()
        if not p:
            raise HTTPException(status_code=404, detail="URSA logo not found")
        return FileResponse(
            str(p),
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/", include_in_schema=False, dependencies=[Depends(require_auth)])
    async def ui_root() -> HTMLResponse:
        return await ui_dashboard()

    @app.get(
        "/ui", include_in_schema=False, dependencies=[Depends(require_auth)]
    )
    async def ui_dashboard() -> HTMLResponse:
        # If ursa_logo.png is available, inline it as a data: URL so it works
        # even when the dashboard is in authenticated "remote" mode (where
        # <img src="/ui/ursa_logo.png"> won't include Authorization headers).
        logo_inline = ""
        logo_img_src = ""
        try:
            lp = _find_logo_path()
            if lp:
                b = lp.read_bytes()
                # Keep the HTML payload reasonable.
                if len(b) <= 2_000_000:
                    data = base64.b64encode(b).decode("ascii")
                    logo_img_src = f"data:image/png;base64,{data}"
                    logo_inline = (
                        "<style>:root{--ursa-logo-url:url('data:image/png;base64,"
                        + data
                        + "');}</style>"
                    )
                else:
                    logo_img_src = "/ui/ursa_logo.png"
        except Exception:
            logo_inline = ""
            logo_img_src = ""

        logo_img_style = "" if str(logo_img_src).strip() else "display:none"

        body = f"""
<div class="app">
  <div class="sidebar" id="leftPanel">
    <div class="topbar topbarCol">
      <div class="brand">
        <div class="brandRow">
          <img class="brandLogo" id="brandLogo" src="{logo_img_src}" style="{logo_img_style}" alt="URSA" onerror="this.style.display='none'" />
          <div>
            <div class="title">URSA Dashboard</div>
            <div class="muted small">Start a new session or continue an existing one.</div>
          </div>
        </div>
      </div>

      <div class="row" style="margin-top:10px; justify-content:flex-start; flex-wrap:wrap">
        <button class="btn" id="toggleChatBtn" type="button">Hide chat</button>
        <button class="btn" id="toggleLogsBtn" type="button">Hide logs</button>
        <button class="btn" id="toggleArtifactsBtn" type="button">Hide artifacts</button>
        <button class="btn" id="openSettingsBtn" type="button">Settings</button>
      </div>
    </div>

    <div class="section">
      <div class="sectionHead">Agents</div>
      <div id="agentList"></div>
    </div>

    <div class="section">
      <div class="row" style="margin-bottom: 8px">
        <div class="sectionHead" style="margin:0">Sessions</div>
        <button class="btn" id="refreshSessionsBtn" type="button">Refresh</button>
      </div>
      <div id="sessionList"></div>
    </div>
  </div>

  <div class="splitter" id="leftSplitter" title="Drag to resize"></div>

  <div class="main" id="mainPanel">
    <div class="topbar">
      <div>
        <div class="title" id="activeSessionTitle">No session selected</div>
        <div class="muted small" id="activeSessionMeta"></div>
      </div>
      <div class="row" style="justify-content:flex-end">
        <button class="btn danger" id="cancelRunBtn" type="button" style="display:none">Cancel run</button>
      </div>
    </div>

    <div class="mainBody" id="mainBody">
      <div class="section conversation" id="conversationSection">
        <div class="messages" id="sessionMessages"></div>
        <div class="composer">
          <div class="muted small" style="margin: 8px 0">Ctrl/⌘ + Enter to send</div>
          <textarea id="messageInput" placeholder="Message the agent..."></textarea>
          <div class="row" style="margin-top: 8px">
            <button class="btn primary" id="sendMsgBtn" type="button">Send</button>
            <div class="muted small" id="runStatus"></div>
            <a class="muted small" href="/ui/workspace" style="margin-left:auto">Run workspace browser</a>
          </div>
        </div>
      </div>

      <div class="hsplitter" id="midSplitter" title="Drag to resize"></div>

      <div class="section runSection" id="runSection">
        <div class="row" style="margin-bottom: 8px; align-items:flex-start">
          <div>
            <div class="sectionHead" style="margin-bottom:2px">Run logs</div>
            <div class="muted small mono" id="runLogMeta">No run selected.</div>
          </div>
          <div class="row" style="justify-content:flex-end">
            <button class="btn" id="clearLogsBtn" type="button">Clear</button>
          </div>
        </div>

        <details class="logDetails" id="stdoutDetails" open>
          <summary class="logHead">stdout</summary>
          <pre class="log" id="stdoutLog"></pre>
        </details>

        <details class="logDetails" id="stderrDetails">
          <summary class="logHead">stderr</summary>
          <pre class="log" id="stderrLog"></pre>
        </details>
      </div>
    </div>
  </div>

  <div class="splitter" id="rightSplitter" title="Drag to resize"></div>

  <div class="workspace" id="rightPanel">
    <div class="topbar">
      <div>
        <div class="title" id="workspaceTitle">Session artifacts</div>
        <div class="muted small" id="workspaceHint">Select a session to view its workspace.</div>
      </div>
      <div class="row" style="justify-content:flex-end">
        <button class="btn" id="refreshFilesBtn" type="button">Refresh</button>
        <button class="btn" id="uploadFilesBtn" type="button">Upload</button>
      </div>
    </div>

    <input id="uploadInput" type="file" multiple style="display:none" />

    <details class="section" id="filesDetails" open>
      <summary class="muted">Files</summary>
      <div id="workspaceFiles" style="margin-top:10px"></div>
    </details>
    <div class="artifactText" id="artifactTextPreview" style="display:none"></div>
    <iframe id="previewFrame" src="about:blank" referrerpolicy="no-referrer"></iframe>
  </div>
</div>

<div class="modal" id="settingsModal" aria-hidden="true">
  <div class="backdrop" id="settingsBackdrop"></div>
  <div class="modalCard">
    <div class="topbar">
      <div>
        <div class="title">Settings</div>
        <div class="muted small">Applies to new runs only.</div>
      </div>
      <button class="btn" id="closeSettingsBtn" type="button">Close</button>
    </div>

    <div class="section">
      <div class="sectionHead">LLM</div>
      <div class="fieldRow"><div class="label">Base URL</div><input class="input" id="set_base_url" placeholder="http://127.0.0.1:8000/v1" /></div>
      <div class="fieldRow"><div class="label">Model</div><input class="input" id="set_model" placeholder="openai:gpt-5-mini" /></div>
      <div class="fieldRow"><div class="label">API key env var</div><input class="input" id="set_api_key_env_var" placeholder="OPENAI_API_KEY" /></div>
      <div class="muted small" style="margin: 2px 0 10px">The dashboard does not store API keys. Set the key in the dashboard server environment and reference its variable name here.</div>
      <div class="fieldRow"><div class="label">Max tokens</div><input class="input" id="set_max_tokens" type="number" min="1" /></div>
      <div class="fieldRow"><div class="label">Temperature</div><input class="input" id="set_temperature" type="number" step="0.1" min="0" max="2" /></div>
    </div>

    <div class="section">
      <div class="sectionHead">Runner</div>
      <div class="fieldRow"><div class="label">Timeout (seconds)</div><input class="input" id="set_timeout" type="number" min="1" placeholder="(none)" /></div>
    </div>

    <div class="section">
      <div class="sectionHead">MCP tools</div>
      <div class="muted small" style="margin: 2px 0 10px">
        MCP servers configured here will be started in the worker subprocess and their tools will be attached to the ExecutionAgent (and the executor inside the Planning + Execution Workflow) for new runs.
      </div>

      <div class="fieldRow">
        <div class="label">Configured servers</div>
        <div>
          <div class="row" id="mcpServersList" style="justify-content:flex-start; gap:6px; flex-wrap:wrap"></div>
          <div class="muted small" style="margin-top:6px">Click a server name to edit it.</div>
        </div>
      </div>

      <div class="fieldRow"><div class="label">Server name</div><input class="input" id="mcp_name" placeholder="my_server" /></div>
      <div class="fieldRow">
        <div class="label">Server config (JSON)</div>
        <textarea class="input" id="mcp_json" rows="6" placeholder='{{"transport":"sse","url":"http://127.0.0.1:3000/sse"}}' style="font-family: var(--mono);"></textarea>
      </div>

      <div class="row" style="justify-content:flex-start; gap: 10px">
        <button class="btn" id="mcpAddUpdateBtn" type="button">Add/Update</button>
        <button class="btn danger" id="mcpRemoveBtn" type="button">Remove</button>
        <button class="btn" id="mcpClearBtn" type="button">Clear</button>
        <div class="muted small" id="mcpStatus"></div>
      </div>
    </div>

    <div class="row" style="gap: 10px">
      <div class="muted small" id="settingsUpdated"></div>
      <div class="muted small" id="settingsSaved"></div>
      <div style="margin-left:auto">
        <button class="btn primary" id="saveSettingsBtn" type="button">Save</button>
      </div>
    </div>
  </div>
</div>

        """

        html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>URSA Dashboard</title>
  <style>{DASHBOARD_CSS}</style>
  {logo_inline}
</head>
<body>
{body}
<script>{DASHBOARD_JS}</script>
</body>
</html>"""
        return HTMLResponse(
            html_doc,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-src 'self'; object-src 'none'",
            },
        )

    return app
