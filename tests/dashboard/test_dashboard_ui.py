"""Focused regression coverage for the dashboard's first-run UI."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
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
    assert "WELCOME_MODES" in dashboard_html
    assert "hypothesis_symposium_agent" in dashboard_html
    assert "showChat: false" in dashboard_html
    assert "showRunLogs: false" in dashboard_html
    assert "showArtifacts: false" in dashboard_html
    assert "state.showChat = false;" in dashboard_html
    assert "Restore last selected session" not in dashboard_html
    assert 'data-start-mode="simple"' in dashboard_html
    assert 'id="agentList"' not in dashboard_html
    assert "Start a session</strong> in the sidebar" not in dashboard_html
    assert "background-size: cover;" in dashboard_html


def test_welcome_has_three_start_modes_instead_of_examples(
    dashboard_html: str,
) -> None:
    soup = BeautifulSoup(dashboard_html, "html.parser")
    modes = soup.select("#welcomeModeList button")
    assert [button["data-start-mode"] for button in modes] == [
        "simple",
        "complex",
        "custom",
    ]
    assert [
        button.select_one(".welcomeModeTitle").text for button in modes
    ] == ["Simple", "Complex", "Custom"]
    assert soup.select_one("#startWalkthroughBtn") is not None
    assert soup.select_one("#welcomeTaskList") is None
    assert "WELCOME_TASK_LIBRARY" not in dashboard_html
    assert "New examples" not in dashboard_html


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
    assert 'id="toggleBehaviorsBtn"' in dashboard_html
    assert 'aria-controls="composerBehaviorOptions"' in dashboard_html
    assert 'class="hidden" id="composerBehaviorOptions"' in dashboard_html
    assert "Choose a behavior" in dashboard_html
    assert "composerBehaviorLabel(agent)" in dashboard_html


def test_expanded_behavior_collapse_control_shares_the_choices_row(
    dashboard_html: str,
) -> None:
    soup = BeautifulSoup(dashboard_html, "html.parser")
    options = soup.select_one("#composerBehaviorOptions")
    row = options.select_one(".composerBehaviorRow")
    assert row.find_previous_sibling(class_="composerAgentLabel") is not None
    collapse = row.select_one("#collapseBehaviorsBtn")
    assert collapse["aria-label"] == "Collapse behaviors"
    assert collapse["aria-expanded"] == "true"
    assert collapse.get_text(strip=True) == ""
    assert collapse.select_one("svg path")["d"] == "m9 6 6 6-6 6"
    assert collapse.find_next_sibling(id="composerAgentType") is not None
    assert soup.select_one("#toggleBehaviorsBtn").get_text(strip=True) == (
        "Behaviors"
    )


def test_sidebar_collapses_to_an_accessible_logo_control(dashboard_html: str):
    soup = BeautifulSoup(dashboard_html, "html.parser")
    sidebar = soup.select_one("#leftPanel")
    assert sidebar.select_one("#sidebarLogoBtn #brandLogo") is not None
    assert sidebar.select_one("#sidebarLogoBtn")["aria-expanded"] == "true"
    for selector in ("#sessionList", "#openSettingsBtn", "#toggleChatBtn"):
        assert sidebar.select_one(f"#sidebarContent {selector}") is not None
    assert sidebar.find_next_sibling(id="leftSplitter") is not None
    assert "ursa.ui.sidebarCollapsed" in dashboard_html
    assert ".sidebar.collapsed { width:72px !important" in dashboard_html
    assert ".sidebar { display:none; }" not in dashboard_html


def _run_dashboard_js(dashboard_html: str, checks: str) -> None:
    """Run the actual UI state functions with a tiny DOM, without a browser.

    No JavaScript packages are required. Python-only environments still run the
    HTML/registry checks above; CI and local installs with Node run these too.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for dashboard JavaScript behavior checks")
    modes = re.search(
        r"  const WELCOME_MODES = \{.*?\n  \};", dashboard_html, re.S
    )
    functions = []
    for name in (
        "startWelcomeMode",
        "setComposerBehaviorsExpanded",
        "setSidebarCollapsed",
        "welcomeIsVisible",
        "applyPanelVisibility",
        "composerBehaviorLabel",
        "renderWelcomeModes",
    ):
        match = re.search(
            rf"  function {name}\(.*?\n  \}}", dashboard_html, re.S
        )
        assert match is not None, name
        functions.append(match[0])
    harness = r"""
const assert = require('node:assert/strict');
const nodes = new Map();
function $(selector) {
  if (!nodes.has(selector)) {
    const classes = new Set();
    nodes.set(selector, {
      attributes: {}, style: {}, dataset: {}, textContent: '',
      classList: {toggle(name, on) { on ? classes.add(name) : classes.delete(name); }, contains(name) {return classes.has(name);}},
      setAttribute(name, value) {this.attributes[name] = value;},
    });
  }
  return nodes.get(selector);
}
const modeButtons = ['simple','complex','custom'].map(mode => { const button = $(mode); button.dataset.startMode = mode; return button; });
const $$ = () => modeButtons;
const calls = [];
const prefs = new Map();
const savePref = (key, value) => prefs.set(key, value);
const loadPref = (key, fallback) => prefs.has(key) ? prefs.get(key) : fallback;
const hideComposerAgentTooltip = () => {};
const state = {
  agentsById: new Map([
    ['chat_agent', {display_name:'Base'}],
    ['hypothesis_symposium_agent', {display_name:'Hypothesize - Symposium'}],
  ]),
  selectedComposerAgentId:'old-agent', activeSessionId:'old-session',
  behaviorsExpanded:false, sidebarCollapsed:false,
  showChat:false, showRunLogs:true, showArtifacts:true,
};
function clearActiveSessionForDraft(id) {
  calls.push(['clear', id]);
  state.activeSessionId = null;
  state.selectedComposerAgentId = id;
}
function openComposerDraft(id, prompt) {
  calls.push(['draft', id, prompt]);
  state.showChat = true;
  setComposerBehaviorsExpanded(state.behaviorsExpanded);
}
"""
    result = subprocess.run(
        [node, "-"],
        input=harness + modes[0] + "\n" + "\n".join(functions) + checks,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("mode", "agent_id", "expanded"),
    [
        ("simple", "chat_agent", False),
        ("complex", "hypothesis_symposium_agent", False),
        ("custom", "chat_agent", True),
    ],
)
def test_start_modes_select_behavior_without_starting_a_run(
    dashboard_html,
    mode,
    agent_id,
    expanded,
):
    _run_dashboard_js(
        dashboard_html,
        f"""
state.behaviorsExpanded = {str(not expanded).lower()};
startWelcomeMode('{mode}');
assert.deepEqual(calls, [['clear','{agent_id}'], ['draft','{agent_id}','']]);
assert.equal(state.selectedComposerAgentId, '{agent_id}');
assert.equal(state.activeSessionId, null);
assert.equal(state.behaviorsExpanded, {str(expanded).lower()});
assert.equal(state.showChat, true);
assert.equal(state.showRunLogs, false);
assert.equal(state.showArtifacts, false);
assert.equal($('#composerBehaviorOptions').classList.contains('hidden'), {str(not expanded).lower()});
""",
    )


def test_missing_complex_behavior_never_silently_falls_back(dashboard_html):
    _run_dashboard_js(
        dashboard_html,
        """
state.agentsById.delete('hypothesis_symposium_agent');
renderWelcomeModes();
assert.equal(modeButtons[1].disabled, true);
assert.equal(modeButtons[0].disabled, false);
startWelcomeMode('complex');
startWelcomeMode('unknown');
assert.deepEqual(calls, []);
assert.equal(state.activeSessionId, 'old-session');
""",
    )


def test_behavior_disclosure_does_not_change_selection(dashboard_html):
    _run_dashboard_js(
        dashboard_html,
        """
state.selectedComposerAgentId = 'hypothesis_symposium_agent';
setComposerBehaviorsExpanded(true);
assert.equal($('#toggleBehaviorsBtn').attributes['aria-expanded'], 'true');
assert.equal($('#toggleBehaviorsBtn').classList.contains('hidden'), true);
assert.equal($('#composerBehaviorOptions').classList.contains('hidden'), false);
assert.match($('#collapseBehaviorsBtn').title, /Hypothesize - Symposium.*collapse/);
setComposerBehaviorsExpanded(false);
assert.equal($('#toggleBehaviorsBtn').attributes['aria-expanded'], 'false');
assert.equal($('#toggleBehaviorsBtn').classList.contains('hidden'), false);
assert.equal($('#composerBehaviorOptions').classList.contains('hidden'), true);
assert.equal(state.selectedComposerAgentId, 'hypothesis_symposium_agent');
assert.match($('#toggleBehaviorsBtn').title, /Hypothesize - Symposium/);
""",
    )


def test_sidebar_toggle_preserves_width_session_and_panel_state(dashboard_html):
    _run_dashboard_js(
        dashboard_html,
        """
$('#leftPanel').style.width = '470px';
setSidebarCollapsed(true);
applyPanelVisibility();
assert.equal($('#sidebarContent').classList.contains('hidden'), true);
assert.equal($('#sidebarBrandCopy').classList.contains('hidden'), true);
assert.equal($('#collapseSidebarBtn').classList.contains('hidden'), true);
assert.equal($('#leftSplitter').classList.contains('hidden'), true);
assert.equal($('#sidebarLogoBtn').attributes['aria-expanded'], 'false');
assert.equal(prefs.get('ursa.ui.sidebarCollapsed'), true);
setSidebarCollapsed(false, {persist:false});
assert.equal(prefs.get('ursa.ui.sidebarCollapsed'), true);
assert.equal($('#sidebarLogoBtn').attributes['aria-expanded'], 'true');
assert.equal($('#leftPanel').style.width, '470px');
assert.equal(state.activeSessionId, 'old-session');
assert.equal(state.showArtifacts, true);
assert.equal($('#leftSplitter').classList.contains('hidden'), false);
""",
    )


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
