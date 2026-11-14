"""Acceptance tests for the #301 send-feedback repairs.

Pre-registered before the fix: a fresh install's stored-credential error
must name the real problem, a send rejected before a run starts must not
orphan the user message in the transcript, and the favicon must resolve.
"""

from __future__ import annotations

import pytest

from tests.dashboard import test_session_workspaces
from ursa_dashboard.credentials import (
    CredentialConfigurationError,
    MemoryCredentialStore,
    resolve_api_key,
)

client = test_session_workspaces.client


def test_fresh_install_stored_source_names_the_real_problem():
    # A fresh install defaults to credential_source="stored" with no
    # credential_id; the error must say no key has been saved yet, not
    # that a stored reference is invalid.
    with pytest.raises(CredentialConfigurationError, match="saved yet"):
        resolve_api_key(
            {"credential_source": "stored", "credential_id": None},
            group="default",
            kind="llm",
            store=MemoryCredentialStore(),
        )


def test_mismatched_stored_reference_stays_invalid():
    # Characterization: a non-empty wrong reference keeps the
    # invalid-reference wording.
    with pytest.raises(CredentialConfigurationError, match="invalid"):
        resolve_api_key(
            {
                "credential_source": "stored",
                "credential_id": "llm:not-this-group",
            },
            group="default",
            kind="llm",
            store=MemoryCredentialStore(),
        )


def test_failed_send_does_not_orphan_the_user_message(client):
    test_client, _ = client
    response = test_client.post(
        "/sessions",
        json={"agent_id": "chat_agent", "workspace_mode": "temporary"},
    )
    assert response.status_code == 200
    session_id = response.json()["session"]["session_id"]

    response = test_client.post(
        f"/sessions/{session_id}/message", json={"text": "hello"}
    )

    assert response.status_code == 400
    transcript = test_client.get(f"/sessions/{session_id}/messages").json()
    assert transcript == [], (
        "a send rejected before a run starts must not persist the user "
        "message as an unanswered transcript entry"
    )


def test_favicon_is_served(client):
    test_client, _ = client

    response = test_client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
