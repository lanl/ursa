from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.sessions import create_session, read_session, session_paths


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa-cache")
    app = create_app(credential_store=MemoryCredentialStore())
    with TestClient(app) as test_client:
        yield test_client, tmp_path


def test_session_creation_requires_explicit_workspace(client) -> None:
    test_client, _ = client

    response = test_client.post("/sessions", json={"agent_id": "chat_agent"})

    assert response.status_code == 400
    assert "workspace is required" in response.json()["detail"].lower()
    assert test_client.get("/sessions").json()["sessions"] == []


def test_folder_workspace_is_used_without_cached_default(client) -> None:
    test_client, tmp_path = client
    workspace = tmp_path / "visible-workspace"

    response = test_client.post(
        "/sessions",
        json={
            "agent_id": "chat_agent",
            "workspace_mode": "folder",
            "workspace_path": str(workspace),
        },
    )

    assert response.status_code == 200
    session = response.json()["session"]
    info = test_client.get(
        f"/sessions/{session['session_id']}/workspace"
    ).json()
    assert info["configured"] is True
    assert info["workspace_mode"] == "folder"
    assert info["workspace_path"] == str(workspace.resolve())
    assert info["default_workspace_path"] is None
    cached_paths = session_paths(
        tmp_path / "ursa-cache" / "default" / "dashboard",
        session["session_id"],
    )
    assert not cached_paths.workspace_dir.exists()


def test_temporary_workspace_is_outside_cache_and_deleted_with_session(
    client,
) -> None:
    test_client, tmp_path = client

    response = test_client.post(
        "/sessions",
        json={"agent_id": "chat_agent", "workspace_mode": "temporary"},
    )

    assert response.status_code == 200
    session = response.json()["session"]
    workspace = Path(session["workspace_path"])
    assert workspace.is_dir()
    assert not workspace.is_relative_to(tmp_path / "ursa-cache")

    deleted = test_client.delete(f"/sessions/{session['session_id']}")
    assert deleted.status_code == 204
    assert not workspace.exists()


def test_unconfigured_legacy_session_is_guarded_until_workspace_is_set(
    client,
) -> None:
    test_client, tmp_path = client
    dashboard_root = tmp_path / "ursa-cache" / "default" / "dashboard"
    session = create_session(dashboard_root, agent_id="chat_agent")
    session_id = session["session_id"]

    info = test_client.get(f"/sessions/{session_id}/workspace")
    assert info.status_code == 200
    assert info.json()["configured"] is False
    assert info.json()["workspace_path"] is None

    blocked = test_client.post(
        f"/sessions/{session_id}/message", json={"text": "hello"}
    )
    assert blocked.status_code == 409
    assert "does not have a workspace" in blocked.json()["detail"]
    assert test_client.get(f"/sessions/{session_id}/messages").json() == []

    configured = test_client.patch(
        f"/sessions/{session_id}/workspace", json={"mode": "temporary"}
    )
    assert configured.status_code == 200
    assert configured.json()["workspace_mode"] == "temporary"
    assert configured.json()["configured"] is True


def test_replacing_temporary_workspace_removes_it(client) -> None:
    test_client, tmp_path = client
    created = test_client.post(
        "/sessions",
        json={"agent_id": "chat_agent", "workspace_mode": "temporary"},
    ).json()["session"]
    temporary_workspace = Path(created["workspace_path"])
    visible_workspace = tmp_path / "follow-on-work"

    changed = test_client.patch(
        f"/sessions/{created['session_id']}/workspace",
        json={"mode": "folder", "path": str(visible_workspace)},
    )

    assert changed.status_code == 200
    assert changed.json()["workspace_path"] == str(visible_workspace.resolve())
    assert not temporary_workspace.exists()


def test_dashboard_shutdown_removes_temporary_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa-cache")
    app = create_app(credential_store=MemoryCredentialStore())

    with TestClient(app) as test_client:
        session = test_client.post(
            "/sessions",
            json={"agent_id": "chat_agent", "workspace_mode": "temporary"},
        ).json()["session"]
        session_id = session["session_id"]
        workspace = Path(session["workspace_path"])
        assert workspace.is_dir()

    assert not workspace.exists()
    saved = read_session(
        tmp_path / "ursa-cache" / "default" / "dashboard", session_id
    )
    assert saved["workspace_mode"] is None
    assert saved["workspace_path"] is None


def test_dashboard_contains_workspace_choice_dialog(client) -> None:
    test_client, _ = client

    html = test_client.get("/ui").text

    assert 'id="workspaceChoiceModal"' in html
    assert "Use temporary workspace" in html
    assert "chooseWorkspaceSelection" in html
    assert "dashboard-managed default workspace" not in html
