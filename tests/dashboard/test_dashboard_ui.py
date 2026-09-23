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


def test_settings_include_inference_provider_model_pickers(
    dashboard_html: str,
) -> None:
    assert 'id="set_llm_inference_provider"' in dashboard_html
    assert 'id="set_embedding_inference_provider"' in dashboard_html
    assert 'list="set_llm_model_options"' in dashboard_html
    assert 'list="set_embedding_model_options"' in dashboard_html
    assert 'id="refresh_llm_models"' in dashboard_html
    assert "refreshProviderModels(kind)" in dashboard_html
    assert "stageInferenceProvider('llm'" in dashboard_html


def test_session_settings_match_global_model_and_tool_options(
    dashboard_html: str,
) -> None:
    assert (
        'data-settings-section="embedding" data-settings-scope="global"'
        not in dashboard_html
    )
    assert (
        'data-settings-section="tools" data-settings-scope="global"'
        not in dashboard_html
    )
    assert (
        'data-settings-section="mcp" data-settings-scope="global"'
        not in dashboard_html
    )
    assert "credentialApiPath(kind=null)" in dashboard_html
    assert "/sessions/${encodeURIComponent(state._settingsSessionId)}" in (
        dashboard_html
    )
    assert "? scopedSettings" in dashboard_html
    assert "globalCredentialOnly').forEach" not in dashboard_html


def test_settings_use_cancel_and_update_actions(
    dashboard_html: str,
) -> None:
    assert (
        'id="closeSettingsBtn" type="button">Cancel</button>' in dashboard_html
    )
    assert (
        'id="saveSettingsBtn" type="button">Update</button>' in dashboard_html
    )
    assert (
        "if (await saveSettings()) modal.classList.remove('open');"
        in dashboard_html
    )
    assert "Theme preview. Click Update to keep it." in dashboard_html
    assert (
        "applyTheme(state.settings?.ui?.theme || 'system');" in dashboard_html
    )


def test_walkthrough_has_friendly_three_step_connection_setup(
    dashboard_html: str,
) -> None:
    assert 'id="guidedConfig"' in dashboard_html
    for step in range(3):
        assert f'data-guided-step="{step}"' in dashboard_html
    assert "Where is your model hosted?" in dashboard_html
    assert "It is not your account password." in dashboard_html
    assert "Enter a key · store securely" in dashboard_html
    assert "Enter the variable’s name, not the key itself." in dashboard_html
    assert "function validateGuidedConfigStep()" in dashboard_html
    assert "if (guidedConfigStep < 2)" in dashboard_html
    assert "const directConnection = index < 0;" in dashboard_html
    assert "Loaded your direct endpoint." in dashboard_html
    # The normal editor remains available outside the tour.
    assert 'id="configProviders"' in dashboard_html
    assert 'id="configEmbeddingFields"' in dashboard_html
    assert ".tourConfigActive .defaultConfigEditor" in dashboard_html


def test_config_tests_show_local_explicit_results(dashboard_html: str) -> None:
    for result in (
        "configLlmTestResult",
        "configEmbTestResult",
        "guidedTestResult",
    ):
        assert f'id="{result}" role="status" aria-live="polite"' in (
            dashboard_html
        )
    assert "Passed — ${result.model} responded successfully." in dashboard_html
    assert "configTestFeedback(kind, 'failed'" in dashboard_html
    assert "Settings changed during the test — test again" in dashboard_html
    assert "function invalidateConfigTests()" in dashboard_html


def test_walkthrough_example_uses_chat_and_explains_workspaces(
    dashboard_html: str,
) -> None:
    assert "openComposerDraft('chat_agent', FIRST_TASK)" in dashboard_html
    assert (
        "openComposerDraft('execution_agent', FIRST_TASK)" not in dashboard_html
    )
    assert "workspace picker opens" in dashboard_html
    assert "Artifacts → Set workspace" in dashboard_html
    assert "Temporary files are removed" in dashboard_html


def test_walkthrough_reserves_space_for_dialogs_and_compact_artifacts(
    dashboard_html: str,
) -> None:
    assert "bottom:20px; left:16px; width:304px" in dashboard_html
    assert ".tourActive .modal .modalCard { left:344px" in dashboard_html
    assert "--tour-dock-height" in dashboard_html
    assert "new ResizeObserver" in dashboard_html
    assert (
        "document.body.classList.remove('tourActive', 'tourConfigActive')"
        in (dashboard_html)
    )
