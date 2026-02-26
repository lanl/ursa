from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    agent_id: str
    # Optional: if omitted, values are taken from global settings.
    params: dict[str, Any] = Field(default_factory=dict)
    agent_init: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    runner: dict[str, Any] = Field(default_factory=dict)


class RunCancelRequest(BaseModel):
    reason: str = "user_request"


class RunRecord(BaseModel):
    run_id: str
    agent_id: str
    status: str
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    agent_init: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    runner: dict[str, Any] = Field(default_factory=dict)
    logs: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: Any | None = None
    runtime: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class RunListResponse(BaseModel):
    runs: list[RunRecord]


class RunEvent(BaseModel):
    id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    ts: str | None = None
    seq: int | None = None
    type: str
    level: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class RunEventListResponse(BaseModel):
    events: list[RunEvent]


class WorkspaceListResponse(BaseModel):
    run_id: str
    agent_id: str
    files: list[dict[str, Any]]


class WorkspaceAllFilesResponse(BaseModel):
    files: list[dict[str, Any]]


class FileMetaResponse(BaseModel):
    run_id: str
    path: str
    size_bytes: int
    mtime: float
    mime: str | None = None
    sha256: str | None = None


class SettingsResponse(BaseModel):
    settings: dict[str, Any]


class SettingsPatchRequest(BaseModel):
    patch: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


# -----------------------------
# Sessions (multi-turn chat)
# -----------------------------


class SessionCreateRequest(BaseModel):
    agent_id: str
    title: str | None = None


class SessionPatchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SessionMessageRequest(BaseModel):
    text: str
    # Optional per-message overrides (advanced)
    params: dict[str, Any] = Field(default_factory=dict)
    agent_init: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    runner: dict[str, Any] = Field(default_factory=dict)


class SessionMessage(BaseModel):
    message_id: str
    ts: str
    role: Literal["user", "assistant", "system"]
    text: str
    run_id: str | None = None


class SessionRecord(BaseModel):
    session_id: str
    agent_id: str
    title: str
    created_at: str
    updated_at: str
    active_run_id: str | None = None
    last_run_id: str | None = None

    model_config = {"extra": "allow"}


class SessionDetail(BaseModel):
    session: SessionRecord
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: list[SessionRecord] = Field(default_factory=list)


class SessionMessageResponse(BaseModel):
    session: SessionRecord
    user_message: SessionMessage
    run: RunRecord


class SessionWorkspaceListResponse(BaseModel):
    session_id: str
    agent_id: str
    files: list[dict[str, Any]]


class SessionFileMetaResponse(BaseModel):
    session_id: str
    path: str
    size_bytes: int
    mtime: float
    mime: str | None = None
    sha256: str | None = None
