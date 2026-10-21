import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.run_manager import RunManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("URSA_DASHBOARD_TOKEN", raising=False)
    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as test_client:
        yield test_client


def create_session(client, workspace):
    response = client.post(
        "/sessions",
        json={
            "agent_id": "chat_agent",
            "workspace_mode": "folder",
            "workspace_path": str(workspace),
        },
    )
    assert response.status_code == 200
    return response.json()["session"]["session_id"]


def test_dashboard_catalog_uses_session_workspace_and_does_not_start_run(
    client, tmp_path, make_skill, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_skill(first, "first-only")
    make_skill(second, "second-only")
    session = create_session(client, first)

    async def unexpected_run(*args, **kwargs):
        pytest.fail("Listing skills must not create a run")

    monkeypatch.setattr(RunManager, "create_run", unexpected_run)
    result = client.get(f"/sessions/{session}/skills")
    assert result.status_code == 200
    assert [skill["name"] for skill in result.json()["skills"]] == [
        "first-only"
    ]
    assert result.json()["skills"][0]["source"] == "workspace"
    assert client.get(f"/sessions/{session}/messages").json() == []
    assert (
        client.patch(
            f"/sessions/{session}/workspace",
            json={"mode": "folder", "path": str(second)},
        ).status_code
        == 200
    )
    assert [
        s["name"]
        for s in client.get(f"/sessions/{session}/skills").json()["skills"]
    ] == ["second-only"]


def test_dashboard_catalog_unknown_session_and_empty_workspace(
    client, tmp_path
):
    assert client.get("/sessions/not-a-session/skills").status_code == 404
    session = create_session(client, tmp_path / "empty")
    result = client.get(f"/sessions/{session}/skills").json()
    assert result["skills"] == []
    assert "No skills found" in result["text"]


def test_message_keeps_raw_request_separate_from_history(
    client, tmp_path, monkeypatch
):
    session = create_session(client, tmp_path / "workspace")
    calls = []

    async def capture_run(self, **kwargs):
        calls.append(kwargs)
        return {
            "run_id": "test-run",
            "agent_id": "chat_agent",
            "status": "completed",
        }

    monkeypatch.setattr(RunManager, "create_run", capture_run)
    monkeypatch.setattr(
        RunManager, "validate_credentials", lambda *args, **kwargs: None
    )
    assert (
        client.post(
            f"/sessions/{session}/message",
            json={"text": "$old-skill previous task"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/sessions/{session}/message", json={"text": "current task"}
        ).status_code
        == 200
    )
    assert calls[-1]["params"]["_ursa_skill_request"] == "current task"
    assert "$old-skill" in calls[-1]["params"]["prompt"]
    assert calls[-1]["extra"]["workspace_dir"] == str(tmp_path / "workspace")
