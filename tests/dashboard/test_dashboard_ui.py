"""Focused regression coverage for the dashboard's first-run UI."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore


@pytest.fixture
def dashboard_html(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa-cache")
    app = create_app(credential_store=MemoryCredentialStore())
    with TestClient(app) as client:
        response = client.get("/ui")
    assert response.status_code == 200
    return response.text


def test_dashboard_always_uses_welcome_screen_and_closed_panels(
    dashboard_html: str,
) -> None:
    assert '<main class="welcomePanel" id="welcomePanel">' in dashboard_html
    assert "Welcome to URSA" in dashboard_html
    assert "WELCOME_TASK_LIBRARY" in dashboard_html
    assert "planning_executor_workflow" in dashboard_html
    assert "showChat: false" in dashboard_html
    assert "showRunLogs: false" in dashboard_html
    assert "showArtifacts: false" in dashboard_html
    assert "state.showChat = false;" in dashboard_html
    assert "Restore last selected session" not in dashboard_html
    assert 'id="startBlankChatBtn"' in dashboard_html
    assert 'id="agentList"' not in dashboard_html
    assert "Start a session</strong> in the sidebar" not in dashboard_html
    assert "background-size: cover;" in dashboard_html


def test_composer_uses_registry_driven_agent_buttons(
    dashboard_html: str,
) -> None:
    assert (
        'class="composerAgentButtons" id="composerAgentType"' in dashboard_html
    )
    assert '<select id="composerAgentType"' not in dashboard_html
    assert "showComposerAgentTooltip(btn, agent)" in dashboard_html
    assert "className = 'composerAgentTooltip'" in dashboard_html
    assert "composerAgentTooltipCopy" in dashboard_html
    assert "btn.title = agent.description" not in dashboard_html
    assert "role', 'radio'" in dashboard_html
    assert "Choose behavior" in dashboard_html
    assert "composerBehaviorLabel(agent)" in dashboard_html


def test_session_creation_moves_to_composer_menu(dashboard_html: str) -> None:
    assert 'id="sessionCreateMenuBtn"' in dashboard_html
    assert 'id="sessionCreateMenu"' in dashboard_html
    assert "Create persistent agent" in dashboard_html
    assert "Use an existing persistent agent" in dashboard_html

    create_menu = dashboard_html.split("function renderSessionCreateMenu()", 1)[
        1
    ].split("function renderSessions()", 1)[0]
    assert "Non-persistent session" not in create_menu

    load_session = dashboard_html.split(
        "async function loadSession(sessionId)", 1
    )[1].split("function clearArtifactPreview", 1)[0]
    assert "state.showChat = true;" in load_session
    assert "applyPanelVisibility();" in load_session

    send_message = dashboard_html.split("async function sendMessage()", 1)[
        1
    ].split("async function cancelActiveRun", 1)[0]
    assert "if (!state.activeSessionId)" in send_message
    assert "startSession(agentId, '', { draftPrompt: text })" in send_message


def test_settings_default_to_llm_without_runner_or_ui_sections(
    dashboard_html: str,
) -> None:
    llm_nav = (
        '<button class="settingsNavBtn active" data-settings-section="llm"'
    )
    assert llm_nav in dashboard_html
    assert 'data-settings-section="runner"' not in dashboard_html
    assert 'data-settings-section="ui"' not in dashboard_html
    assert "STDOUT buffer lines" not in dashboard_html
    assert 'id="cycleThemeBtn"' in dashboard_html
