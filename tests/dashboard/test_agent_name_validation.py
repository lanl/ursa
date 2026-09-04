"""Regression coverage for issue #319: invalid agent names must surface as
clear 400s, not silent 500s, at every dashboard endpoint that names an
agent, and the served UI must pre-check names with the same rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa.cli.agent_management import _AGENT_NAME_RE
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa-cache")
    app = create_app(credential_store=MemoryCredentialStore())
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, tmp_path


def _create_payload(tmp_path: Path, name: str) -> dict:
    return {
        "agent_id": "chat_agent",
        "agent_name": name,
        "workspace_mode": "folder",
        "workspace_path": str(tmp_path / "ws"),
    }


def test_create_session_rejects_spaced_name_with_400(client) -> None:
    test_client, tmp_path = client

    response = test_client.post(
        "/sessions", json=_create_payload(tmp_path, "Swiss Roll")
    )

    assert response.status_code == 400
    assert "letters, numbers" in response.json()["detail"]


def test_agent_save_rejects_spaced_name_with_400(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/agent-management/save", json={"agent_name": "Swiss Roll"}
    )

    assert response.status_code == 400
    assert "letters, numbers" in response.json()["detail"]


def test_agent_delete_rejects_spaced_name_with_400(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/agent-management/delete", json={"agent_name": "Swiss Roll"}
    )

    assert response.status_code == 400
    assert "letters, numbers" in response.json()["detail"]


def test_agent_copy_rejects_spaced_new_name_with_400(client) -> None:
    test_client, _ = client

    response = test_client.post(
        "/agent-management/copy",
        json={"source_agent_name": "SwissRoll", "new_agent_name": "My Copy"},
    )

    assert response.status_code == 400
    assert "letters, numbers" in response.json()["detail"]


def test_create_session_rejects_leading_dot_with_clear_message(client) -> None:
    # The rule requires an alphanumeric first character; the message must
    # say so rather than implying dot is always allowed.
    test_client, tmp_path = client

    response = test_client.post(
        "/sessions", json=_create_payload(tmp_path, ".hidden")
    )

    assert response.status_code == 400
    assert "must start with a letter or number" in response.json()["detail"]


def test_create_session_accepts_valid_name(client) -> None:
    test_client, tmp_path = client

    response = test_client.post(
        "/sessions", json=_create_payload(tmp_path, "SwissRoll")
    )

    assert response.status_code == 200
    assert response.json()["session"]["agent_name"] == "SwissRoll"


def test_ui_embeds_the_backend_name_rule(client) -> None:
    # Drift guard: the client-side pre-check must mirror the backend rule;
    # if _AGENT_NAME_RE changes, this containment check fails until the
    # served JS follows.
    test_client, _ = client

    html = test_client.get("/ui").text

    assert _AGENT_NAME_RE.pattern in html
