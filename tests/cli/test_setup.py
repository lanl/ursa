import asyncio
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import keyring
import pytest
import yaml
from textual.widgets import (
    Checkbox,
    Collapsible,
    Input,
    LoadingIndicator,
    Select,
    Static,
    Tab,
    TabbedContent,
    TabPane,
    TextArea,
)

from ursa.cli.config import ChatModelConfig, EmbModelConfig
from ursa.cli.setup import (
    CUSTOM_MODEL,
    CUSTOM_PROVIDER_URL,
    MCP_TOKEN_NEW,
    MCP_TOKEN_NONE,
    ApiKeyEntry,
    MCPDraft,
    ProviderDraft,
    ProviderUrlSelect,
    SetupApp,
    SetupState,
    _friendly_validation_error,
    probe_mcp_server,
    run_setup,
    save_setup_state,
)
from ursa.cli.tui.widgets import FuzzySelect
from ursa.util.inference_providers import ProviderModel


async def _wait_for(predicate, attempts=40):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition did not become true")


def configured_state(path: Path) -> SetupState:
    state = SetupState(
        path=path,
        base_config={"use_web": True},
    )
    state.providers["company"] = ProviderDraft(
        name="company",
        kind="litellm",
        base_url="https://models.example/v1",
        pending_secret="provider-secret",
    )
    state.llm_model = ChatModelConfig(
        model="openai/gpt-test",
        model_provider="openai",
        inference_provider="company",
    )
    state.embedding_model = EmbModelConfig(
        model="text-embedding-test",
        model_provider="openai",
        inference_provider="company",
    )
    state.mcp_servers["tools"] = MCPDraft(
        name="tools",
        transport="streamable_http",
        url="https://tools.example/mcp",
        auth=True,
        pending_secret="mcp-secret",
    )
    return state


def test_run_setup_injects_truststore(monkeypatch, tmp_path):
    inject = MagicMock()
    app = MagicMock()
    app.run.return_value = True
    create_app = MagicMock(return_value=app)
    monkeypatch.setattr("ursa.cli.setup.inject_truststore_into_ssl", inject)
    monkeypatch.setattr("ursa.cli.setup.SetupApp", create_app)
    path = tmp_path / "config.yaml"

    assert run_setup(path) is True

    inject.assert_called_once_with()
    create_app.assert_called_once_with(path)


def test_setup_state_builds_keyring_only_configuration(tmp_path):
    config = configured_state(tmp_path / "config.yaml").configuration()

    assert config["use_web"] is True
    assert config["inference_providers"]["company"] == {
        "model_provider": "litellm",
        "base_url": "https://models.example/v1",
        "api_key": {"keyring": True},
    }
    assert config["llm_model"] == {
        "model": "openai/gpt-test",
        "inference_provider": "company",
        "model_provider": "openai",
    }
    assert config["mcp_servers"]["tools"]["headers"]["Authorization"] == {
        "keyring": True,
        "template": "Bearer %s",
    }
    assert "provider-secret" not in yaml.safe_dump(config)
    assert "mcp-secret" not in yaml.safe_dump(config)


def test_existing_keyring_reference_resolves_to_provider_name():
    provider = ProviderDraft(
        name="company",
        kind="openai",
        base_url="https://models.example/v1",
        api_key_reference={"keyring": True},
    )

    config = provider.validation_config().resolve(provider.name)

    assert config.api_key is not None
    assert config.api_key.keyring == "company"


@pytest.mark.asyncio
async def test_mcp_probe_uses_unsaved_auth_key(monkeypatch):
    captured = {}

    class Client:
        async def get_tools(self, *, server_name):
            captured["server_name"] = server_name
            return [object()]

    def start(connections):
        captured["connections"] = connections
        return Client()

    monkeypatch.setattr(
        "ursa.cli.setup.start_mcp_client",
        start,
    )
    server = MCPDraft(
        name="tools",
        transport="streamable_http",
        url="https://tools.example/mcp",
        auth=True,
        pending_secret="unsaved-secret",
    )

    assert await probe_mcp_server(server) == 1
    assert captured["connections"]["tools"]["headers"] == {
        "Authorization": "Bearer unsaved-secret"
    }
    assert captured["server_name"] == "tools"


def test_setup_preserves_unmanaged_provider_model_and_mcp_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    model_provider: openai
    base_url: https://models.example/v1
    api_key: {keyring: true}
    organization: research
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
  max_completion_tokens: 512
mcp_servers:
  tools:
    transport: streamable_http
    url: https://tools.example/mcp
    timeout: 30
"""
    )

    config = SetupState.load(path).configuration()

    assert config["inference_providers"]["company"]["organization"] == (
        "research"
    )
    assert config["llm_model"]["max_completion_tokens"] == 512
    assert config["mcp_servers"]["tools"]["timeout"] == 30


def test_setup_always_enables_provider_tls_verification(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    model_provider: openai
    base_url: https://models.example/v1
    api_key: {keyring: true}
    ssl_verify: false
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
"""
    )

    state = SetupState.load(path)

    assert state.providers["company"].ssl_verify is True
    assert (
        "ssl_verify"
        not in state.configuration()["inference_providers"]["company"]
    )


def test_save_setup_state_writes_keyring_and_atomic_config(
    monkeypatch, tmp_path
):
    path = tmp_path / "nested" / "config.yaml"
    state = configured_state(path)
    stored = []
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.set_password",
        lambda service, username, secret: stored.append((
            service,
            username,
            secret,
        )),
    )

    assert save_setup_state(state) == path

    assert stored == [
        ("ursa", "company", "provider-secret"),
        ("ursa", "tools", "mcp-secret"),
    ]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = yaml.safe_load(path.read_text())
    assert loaded["inference_providers"]["company"]["api_key"] == {
        "keyring": True
    }


def test_save_does_not_write_config_when_keyring_fails(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    state = configured_state(path)
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.set_password",
        lambda *_args: (_ for _ in ()).throw(keyring.errors.NoKeyringError()),
    )

    with pytest.raises(keyring.errors.NoKeyringError):
        save_setup_state(state)

    assert not path.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("401 Unauthorized"), "API key was rejected"),
        (RuntimeError("404 Not Found"), "endpoint was not found"),
        (RuntimeError("Connection refused"), "could not connect"),
    ],
)
def test_provider_errors_are_actionable(error, message):
    assert message in _friendly_validation_error(error)


@pytest.mark.asyncio
async def test_setup_app_railroads_first_run(monkeypatch, tmp_path):
    models = [
        ProviderModel("gpt-test", "openai"),
        ProviderModel("text-embedding-test", "openai", type="embedding"),
    ]
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models", lambda _config: models
    )
    chat_probes = []
    embedding_probes = []
    monkeypatch.setattr(
        "ursa.cli.setup.probe_chat_model",
        lambda config, _providers: chat_probes.append(config.model),
    )
    monkeypatch.setattr(
        "ursa.cli.setup.probe_embedding_model",
        lambda config, _providers: embedding_probes.append(config.model),
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        app.query_one("#provider-name", Input).value = "company"
        app.query_one("#provider-url", FuzzySelect).value = CUSTOM_PROVIDER_URL
        app.query_one(
            "#provider-url-custom", Input
        ).value = "https://models.example/v1"
        app.query_one("#provider-key", Input).value = "secret"
        await pilot.pause()
        await pilot.click("#provider-save")
        await _wait_for(lambda: app.current_page == "providers")

        assert app.query_one("#providers-tabs", TabbedContent)
        assert app.state.provider_verified["company"] is True
        await pilot.click("#providers-next")
        await _wait_for(lambda: app.current_page == "models")

        assert app.query_one("#llm-choice", Select).value == "gpt-test"
        app.query_one("#embedding-choice", Select).value = "text-embedding-test"
        await pilot.pause()
        assert "Emb: text-embeddi..." in str(
            app.query_one("#setup-progress", Static).content
        )
        await pilot.click("#models-save")
        await _wait_for(lambda: app.current_page == "mcps")
        assert chat_probes == ["gpt-test"]
        assert embedding_probes == ["text-embedding-test"]
        assert "✓" in str(
            app.query_one("#llm-validation-status", Static).content
        )
        progress = str(app.query_one("#setup-progress", Static).content)
        assert "✓ LLM: gpt-test" in progress
        assert "✓ Emb: text-embeddi..." in progress
        await pilot.click("#mcps-next")
        assert app.current_page == "review"
        assert "company" in str(
            app.query_one("#review-summary", Static).content
        )


@pytest.mark.asyncio
async def test_setup_app_fits_standard_terminal_width(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(80, 24)):
        button = app.query_one("#welcome-next")

        assert button.region.right <= 80


@pytest.mark.asyncio
async def test_provider_base_url_fuzzy_select_allows_custom_urls(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        assert isinstance(app.query_one("#provider-name"), Input)
        urls = app.query_one("#provider-url", ProviderUrlSelect)
        assert [value for _label, value in urls._options] == [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "http://localhost:4000",
            "http://localhost:11434",
            CUSTOM_PROVIDER_URL,
        ]
        assert [str(label) for label, _value in urls._options] == [
            "openai (https://api.openai.com/v1)",
            "anthropic (https://api.anthropic.com)",
            "litellm (http://localhost:4000)",
            "ollama (http://localhost:11434)",
            "Other",
        ]

        name = app.query_one("#provider-name", Input)
        name.value = "company"
        urls.value = "http://localhost:11434"
        await pilot.pause()
        assert name.value == "company"

        urls.value = CUSTOM_PROVIDER_URL
        await pilot.pause()
        assert name.value == "company"
        custom_url = app.query_one("#provider-url-custom", Input)
        assert [str(label) for label, _value in urls._options] == [
            "openai (https://api.openai.com/v1)",
            "anthropic (https://api.anthropic.com)",
            "litellm (http://localhost:4000)",
            "ollama (http://localhost:11434)",
            "Other",
        ]
        assert not custom_url.has_class("hidden")
        assert custom_url.parent is urls.parent
        assert urls.parent.has_class("custom-url")
        assert urls.region.width == 22
        assert custom_url.region.width > urls.region.width

        urls.expanded = True
        await pilot.pause()
        assert urls.parent.has_class("selecting")
        assert urls.region.width > custom_url.region.width

        urls.expanded = False
        await pilot.pause()
        assert not urls.parent.has_class("selecting")
        assert custom_url.region.width > urls.region.width
        custom_url.value = "https://models.example/v1"
        assert app._provider_url() == "https://models.example/v1"

        urls.value = "https://api.openai.com/v1"
        await pilot.pause()
        assert str(urls._options[0][0]) == (
            "openai (https://api.openai.com/v1)"
        )
        assert custom_url.has_class("hidden")


@pytest.mark.asyncio
async def test_provider_validation_uses_ascii_spinner(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)):
        app._set_provider_checking(True)
        app._provider_spinner_timer.pause()
        message = app.query_one("#provider-error", Static)
        assert str(message.content) == (
            "| Checking the API key and loading models"
        )

        app._advance_provider_spinner()
        assert str(message.content) == (
            "/ Checking the API key and loading models"
        )
        app._set_provider_checking(False)


@pytest.mark.asyncio
async def test_model_tree_truncates_names_and_animates_ascii_spinner(tmp_path):
    llm_name = "language-model-name-that-is-too-long"
    embedding_name = "embedding-model-name-that-is-too-long"
    app = SetupApp(tmp_path / "config.yaml")
    app.state.llm_model = ChatModelConfig(
        model=llm_name,
        model_provider="openai",
        inference_provider="company",
    )
    app.state.embedding_model = EmbModelConfig(
        model=embedding_name,
        model_provider="openai",
        inference_provider="company",
    )

    async with app.run_test(size=(120, 42)):
        assert app.MODEL_SPINNER_FRAMES == ("|", "/", "-", "\\")
        assert not app.query(LoadingIndicator)
        app._update_progress()
        progress = str(app.query_one("#setup-progress", Static).content)
        assert f"LLM: {llm_name[:12]}..." in progress
        assert f"Emb: {embedding_name[:12]}..." in progress

        app._set_model_validation_status(
            "llm", None, "Testing...", loading=True
        )
        app._set_model_validation_status(
            "embedding", None, "Testing...", loading=True
        )
        app._model_spinner_timer.pause()
        progress = str(app.query_one("#setup-progress", Static).content)
        assert f"| LLM: {llm_name[:12]}..." in progress
        assert f"| Emb: {embedding_name[:12]}..." in progress
        assert (
            str(app.query_one("#llm-validation-status", Static).content)
            == "| Testing..."
        )

        app._advance_model_spinner()
        progress = str(app.query_one("#setup-progress", Static).content)
        assert f"/ LLM: {llm_name[:12]}..." in progress
        assert f"/ Emb: {embedding_name[:12]}..." in progress
        assert (
            str(app.query_one("#llm-validation-status", Static).content)
            == "/ Testing..."
        )

        app._set_mcp_validation_status(None, "Connecting...", loading=True)
        app._mcp_spinner_timer.pause()
        mcp_status = app.query_one("#mcp-validation-status", Static)
        assert str(mcp_status.content) == "| Connecting..."
        app._advance_mcp_spinner()
        assert str(mcp_status.content) == "/ Connecting..."


@pytest.mark.asyncio
async def test_setup_app_allows_custom_model_and_back(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda _config: [ProviderModel("listed", "openai")],
    )
    monkeypatch.setattr(
        "ursa.cli.setup.probe_chat_model", lambda _config, _providers: None
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        app.query_one("#provider-key", Input).value = "secret"
        await pilot.click("#provider-save")
        await _wait_for(lambda: app.current_page == "providers")
        await pilot.click("#providers-next")
        await _wait_for(lambda: app.current_page == "models")
        app.query_one("#llm-choice", Select).value = CUSTOM_MODEL
        app.query_one("#llm-custom", Input).value = "private-model"
        await pilot.pause()
        await pilot.click("#models-save")
        await _wait_for(lambda: app.current_page == "mcps")
        assert app.state.llm_model is not None
        assert app.state.llm_model.model == "private-model"
        app.action_back()
        assert app.current_page == "models"


@pytest.mark.asyncio
async def test_model_advanced_editors_validate_and_preserve_yaml(
    monkeypatch, tmp_path
):
    models = [
        ProviderModel("gpt-test", "openai"),
        ProviderModel("text-embedding-test", "openai", type="embedding"),
    ]
    monkeypatch.setattr(
        "ursa.cli.setup.probe_chat_model", lambda _config, _providers: None
    )
    monkeypatch.setattr(
        "ursa.cli.setup.probe_embedding_model",
        lambda _config, _providers: None,
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        kind="openai",
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_models["company"] = models
    app.state.provider_verified["company"] = True

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await _wait_for(lambda: bool(app._provider_tab_names))
        await pilot.click("#providers-next")
        await _wait_for(lambda: app.current_page == "models")

        llm_advanced = app.query_one("#llm-advanced", Collapsible)
        embedding_advanced = app.query_one("#embedding-advanced", Collapsible)
        assert llm_advanced.collapsed is True
        assert embedding_advanced.collapsed is True
        assert app.query_one("#llm-choice", Select).region.bottom <= (
            llm_advanced.region.y
        )
        assert (
            llm_advanced.region.bottom
            <= app.query_one("#llm-validation-status", Static).region.y
        )

        llm_editor = app.query_one("#llm-config-yaml", TextArea)
        llm_values = yaml.safe_load(llm_editor.text)
        llm_values["temperature"] = 0.2
        llm_editor.text = yaml.safe_dump(llm_values, sort_keys=False)

        embedding_editor = app.query_one("#embedding-config-yaml", TextArea)
        embedding_values = yaml.safe_load(embedding_editor.text)
        embedding_values["model"] = "text-embedding-test"
        embedding_values["dimensions"] = 512
        embedding_editor.text = yaml.safe_dump(
            embedding_values, sort_keys=False
        )

        await _wait_for(lambda: llm_editor.has_class("yaml-valid"))
        await _wait_for(lambda: embedding_editor.has_class("yaml-valid"))
        await _wait_for(
            lambda: app.query_one("#embedding-choice", Select).value
            == "text-embedding-test"
        )
        await pilot.click("#models-save")
        await _wait_for(lambda: app.current_page == "mcps")

        assert app.state.llm_model is not None
        assert app.state.llm_model.model_extra == {"temperature": 0.2}
        assert app.state.embedding_model is not None
        assert app.state.embedding_model.model_extra == {"dimensions": 512}


@pytest.mark.asyncio
async def test_setup_page_leaves_gutter_before_scrollbar(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 30)) as pilot:
        app._show("models")
        await pilot.pause()

        page = app.query_one("#models-page")
        scrollbar = page.vertical_scrollbar
        assert page.show_vertical_scrollbar is True
        for selector in ("#llm-choice", "#llm-advanced", "#models-error"):
            content = app.query_one(selector)
            assert scrollbar.region.x - content.region.right == 1


@pytest.mark.asyncio
async def test_existing_provider_key_is_masked_and_revealable(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    model_provider: openai
    base_url: https://models.example/v1
    api_key: {keyring: true}
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
"""
    )

    def get_password(service, username):
        print(
            f"Error: ursa-keyring/{service}/{username} is not in the password "
            "store",
            file=sys.stderr,
        )
        return (
            "stored-secret"
            if (service, username) == ("ursa", "company")
            else None
        )

    monkeypatch.setattr("ursa.cli.setup.keyring.get_password", get_password)
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda _config: [ProviderModel("gpt-test", "openai")],
    )
    app = SetupApp(path)

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await _wait_for(lambda: bool(app._provider_tab_names))
        await pilot.click("#provider-edit")

        key = app.query_one("#provider-key", Input)
        await _wait_for(lambda: key.value == "stored-secret")
        assert key.value == "stored-secret"
        assert key.password is True
        assert key.placeholder == "Enter the API Key for the provider"

        app.query_one("#provider-show-key", Checkbox).value = True
        await pilot.pause()
        assert key.password is False

    assert "is not in the password store" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_api_key_entries_share_layout_and_load_by_name(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.get_password",
        lambda service, username: (
            f"{username}-secret" if service == "ursa" else None
        ),
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        assert len(list(app.query(ApiKeyEntry))) == 2
        assert not app.query("#provider-key-source")
        assert not app.query("#provider-keyring-name")
        assert not app.query("#provider-load-keyring")
        assert not app.query("#mcp-keyring-name")
        assert not app.query("#mcp-load-keyring")
        assert app.query_one("#mcp-token-source", FuzzySelect)

        provider_key = app.query_one("#provider-key", Input)
        provider_show = app.query_one("#provider-show-key", Checkbox)
        assert provider_key.parent is provider_show.parent
        assert provider_key.parent.has_class("api-key-row")

        await pilot.click("#welcome-next")
        await _wait_for(lambda: provider_key.value == "openai-secret")
        app.query_one("#provider-name", Input).value = "company"
        await _wait_for(lambda: provider_key.value == "company-secret")

        app._begin_mcp()
        mcp_key = app.query_one("#mcp-token", Input)
        mcp_show = app.query_one("#mcp-show-key", Checkbox)
        assert mcp_key.parent is mcp_show.parent
        assert mcp_key.parent.has_class("api-key-row")
        app.query_one("#mcp-token-source", FuzzySelect).value = MCP_TOKEN_NEW
        app.query_one("#mcp-name", Input).value = "tools"
        await _wait_for(lambda: mcp_key.value == "tools-secret")


@pytest.mark.asyncio
async def test_missing_named_api_key_uses_entry_placeholder(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.get_password", lambda _service, _name: None
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await pilot.pause()

        key = app.query_one("#provider-key", Input)
        assert key.value == ""
        assert key.placeholder == "Enter the API Key for the provider"


@pytest.mark.asyncio
async def test_existing_providers_are_checked_on_page_entry(
    monkeypatch, tmp_path
):
    checked = []
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda config: checked.append(config.base_url)
        or [ProviderModel("gpt-test", "openai")],
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        kind="openai",
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_verified["company"] = None

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await _wait_for(
            lambda: app.state.provider_verified.get("company") is True
        )

        assert checked == ["https://models.example/v1"]
        assert "✓ All providers verified" in str(
            app.query_one("#providers-validation-status", Static).content
        )
        assert app.query_one("#providers-next").disabled is False


@pytest.mark.asyncio
async def test_provider_tabs_remain_interactive_during_validation(
    monkeypatch, tmp_path
):
    validation_started = threading.Event()
    release_validation = threading.Event()

    def validate(_config):
        validation_started.set()
        release_validation.wait(timeout=5)
        return [ProviderModel("gpt-test", "openai")]

    monkeypatch.setattr("ursa.cli.setup.list_provider_models", validate)
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        kind="openai",
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.providers["backup"] = ProviderDraft(
        name="backup",
        kind="openai",
        base_url="https://backup.example/v1",
        pending_secret="secret",
    )
    app.state.provider_verified["company"] = None
    app.state.provider_verified["backup"] = None

    try:
        async with app.run_test(size=(120, 42)) as pilot:
            try:
                await pilot.click("#welcome-next")
                await _wait_for(validation_started.is_set)

                assert app._provider_tab_names == {
                    "provider-pane-0": "company",
                    "provider-pane-1": "backup",
                }
                assert len(list(app.query(TabPane))) == 2
                assert app.state.provider_verified["company"] is None
                provider_tabs = list(
                    app.query_one("#providers-tabs", TabbedContent).query(Tab)
                )
                assert (
                    str(provider_tabs[0].label)[0] in app.MODEL_SPINNER_FRAMES
                )
                assert "company" in str(provider_tabs[0].label)
                assert any(
                    f"├─ {frame} company"
                    in str(app.query_one("#setup-progress", Static).content)
                    for frame in app.MODEL_SPINNER_FRAMES
                )
                connection_status = str(
                    app.query_one(
                        "#providers-validation-status", Static
                    ).content
                )
                assert connection_status[0] in app.MODEL_SPINNER_FRAMES
                assert connection_status[2:] == (
                    "Checking provider connection..."
                )
                initial_frame = connection_status[0]
                await _wait_for(
                    lambda: str(
                        app.query_one(
                            "#providers-validation-status", Static
                        ).content
                    )[0]
                    != initial_frame
                )

                await pilot.click(provider_tabs[1])
                await _wait_for(
                    lambda: app.query_one(
                        "#providers-tabs", TabbedContent
                    ).active
                    == "provider-pane-1"
                )
                assert app.state.provider_verified["company"] is None
            finally:
                release_validation.set()
                await _wait_for(lambda: not app._providers_validating)
    finally:
        release_validation.set()


@pytest.mark.asyncio
async def test_all_providers_must_verify_before_models(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    app = SetupApp(path)
    app.state.providers["broken"] = ProviderDraft(
        name="broken",
        kind="openai",
        base_url="https://models.example/v1",
        pending_secret="bad-secret",
    )
    app.state.provider_verified["broken"] = None
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda _config: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")),
    )

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await _wait_for(lambda: bool(app._provider_tab_names))
        await _wait_for(
            lambda: app.state.provider_verified.get("broken") is False
        )

        assert app.current_page == "providers"
        assert app.state.provider_verified["broken"] is False
        assert app.query_one("#providers-next").disabled is True
        await pilot.click("#providers-next")
        assert app.current_page == "providers"
        assert "✗ broken" in str(
            app.query_one("#setup-progress", Static).content
        )
        assert str(app.query_one("#providers-error", Static).content) == (
            "Fix or remove the providers that failed verification:\n"
            "- broken: The API key was rejected. Check that it was copied "
            "completely and has permission to list models."
        )

        monkeypatch.setattr(
            "ursa.cli.setup.list_provider_models",
            lambda _config: [ProviderModel("gpt-test", "openai")],
        )
        await pilot.click("#providers-retry")
        await _wait_for(
            lambda: app.state.provider_verified.get("broken") is True
        )
        assert app.query_one("#providers-next").disabled is False


@pytest.mark.asyncio
async def test_provider_failure_summary_shows_only_first_three(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    for index in range(5):
        name = f"broken-{index}"
        app.state.providers[name] = ProviderDraft(
            name=name,
            kind="openai",
            base_url="https://models.example/v1",
            pending_secret="bad-secret",
        )
        app.state.provider_verified[name] = False
        app.state.provider_errors[name] = f"error {index}"

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await _wait_for(
            lambda: bool(str(app.query_one("#providers-error", Static).content))
        )

        assert str(app.query_one("#providers-error", Static).content) == (
            "3+ providers failed verification, the first three are:\n"
            "- broken-0: error 0\n"
            "- broken-1: error 1\n"
            "- broken-2: error 2"
        )


@pytest.mark.asyncio
async def test_mcp_token_selector_reuses_keyring_or_creates_new_entry(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.get_password",
        lambda service, username: (
            "shared-secret"
            if (service, username) == ("ursa", "shared")
            else None
        ),
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["shared"] = ProviderDraft(
        name="shared",
        kind="openai",
        base_url="https://models.example/v1",
        api_key_reference={"keyring": True},
        keyring_username="shared",
    )

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_mcp()
        await pilot.pause()
        advanced = app.query_one("#mcp-advanced", Collapsible)
        assert advanced.collapsed is True
        assert (
            advanced.region.y == app.query_one("#mcp-token-row").region.bottom
        )
        assert advanced not in app.query_one("#mcp-token", Input).ancestors
        assert advanced in app.query_one("#mcp-config-yaml", TextArea).ancestors
        assert not app.query("#mcp-auth")
        assert not app.query("#mcp-header")
        assert not app.query("#mcp-template")
        assert (
            app.query_one("#mcp-name", Input).region.y
            < app.query_one("#mcp-transport", Select).region.y
        )

        token_source = app.query_one("#mcp-token-source", FuzzySelect)
        assert token_source.value == MCP_TOKEN_NONE
        assert app.query_one("#mcp-auth-fields").has_class("hidden")
        app.query_one("#mcp-name", Input).value = "tools"
        app.query_one("#mcp-url", Input).value = "https://tools.example/mcp"
        token_source.value = "shared"
        await _wait_for(
            lambda: app.query_one("#mcp-token", Input).value == "shared-secret"
        )
        assert not app.query_one("#mcp-auth-fields").has_class("hidden")
        await pilot.pause()
        auth_fields = app.query_one("#mcp-auth-fields")
        assert token_source.region.width == 22
        assert auth_fields.region.x == token_source.region.right + 1

        draft = app._mcp_values()
        assert draft.pending_secret is None
        assert draft.keyring_username == "shared"
        assert draft.as_config()["headers"]["Authorization"] == {
            "keyring": "shared",
            "template": "Bearer %s",
        }

        token_source.value = MCP_TOKEN_NEW
        await pilot.pause()
        assert not app.query_one("#mcp-auth-fields").has_class("hidden")
        app.query_one("#mcp-token", Input).value = "new-secret"
        draft = app._mcp_values()
        assert draft.pending_secret == "new-secret"
        assert draft.keyring_username == "tools"
        assert draft.as_config()["headers"]["Authorization"] == {
            "keyring": True,
            "template": "Bearer %s",
        }


@pytest.mark.asyncio
async def test_mcp_advanced_yaml_updates_config_and_token_source(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.cli.setup.keyring.get_password",
        lambda service, username: (
            "shared-secret"
            if (service, username) == ("ursa", "shared-token")
            else None
        ),
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_mcp()
        app.query_one("#mcp-name", Input).value = "tools"
        await pilot.pause()
        editor = app.query_one("#mcp-config-yaml", TextArea)
        editor.text = yaml.safe_dump(
            {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
                "headers": {
                    "X-API-Key": {
                        "keyring": "shared-token",
                        "template": "Token %s",
                    },
                    "X-Client": "ursa",
                },
                "timeout": 12,
            },
            sort_keys=False,
        )

        await _wait_for(
            lambda: app.query_one("#mcp-token-source", FuzzySelect).value
            == "shared-token"
        )
        await _wait_for(lambda: editor.has_class("yaml-valid"))
        await _wait_for(
            lambda: app.query_one("#mcp-token", Input).value == "shared-secret"
        )
        draft = app._mcp_values()
        config = draft.as_config()

        assert config["timeout"] == 12
        assert config["headers"] == {
            "X-Client": "ursa",
            "X-API-Key": {
                "keyring": "shared-token",
                "template": "Token %s",
            },
        }


@pytest.mark.asyncio
async def test_mcp_server_is_validated_before_it_is_saved(
    monkeypatch, tmp_path
):
    checked = []

    async def validate(server):
        checked.append(server.name)
        return 2

    monkeypatch.setattr("ursa.cli.setup.probe_mcp_server", validate)
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)):
        app._begin_mcp()
        app.query_one("#mcp-name", Input).value = "tools"
        app.query_one("#mcp-url", Input).value = "https://tools.example/mcp"
        await app._save_mcp()
        await _wait_for(lambda: app.current_page == "mcps")

        assert checked == ["tools"]
        assert app.state.mcp_verified["tools"] is True
        assert app.state.mcp_servers["tools"].url == (
            "https://tools.example/mcp"
        )
        assert "✓ tools" in str(
            app.query_one("#setup-progress", Static).content
        )


@pytest.mark.asyncio
async def test_mcp_validation_failure_stays_on_form(monkeypatch, tmp_path):
    notifications = []

    async def fail(_server):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("ursa.cli.setup.probe_mcp_server", fail)
    monkeypatch.setattr(
        SetupApp,
        "notify",
        lambda _self, message, **kwargs: notifications.append((
            message,
            kwargs,
        )),
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)):
        app._begin_mcp()
        app.query_one("#mcp-name", Input).value = "tools"
        app.query_one("#mcp-url", Input).value = "https://tools.example/mcp"
        await app._save_mcp()

        assert app.current_page == "mcp"
        assert "tools" not in app.state.mcp_servers
        assert "✗" in str(
            app.query_one("#mcp-validation-status", Static).content
        )
        assert "rejected the auth key" in notifications[-1][0]


@pytest.mark.asyncio
async def test_failed_mcp_server_disables_review_until_retry_succeeds(
    monkeypatch, tmp_path
):
    async def fail(_server):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("ursa.cli.setup.probe_mcp_server", fail)
    app = SetupApp(tmp_path / "config.yaml")
    app.state.mcp_servers["tools"] = MCPDraft(
        name="tools",
        transport="streamable_http",
        url="https://tools.example/mcp",
    )
    app.state.mcp_verified["tools"] = None

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("mcps")
        await _wait_for(lambda: app.state.mcp_verified["tools"] is False)

        assert app.query_one("#mcps-next").disabled is True
        await pilot.click("#mcps-next")
        assert app.current_page == "mcps"

        async def succeed(_server):
            return 2

        monkeypatch.setattr("ursa.cli.setup.probe_mcp_server", succeed)
        await pilot.click("#mcps-retry")
        await _wait_for(lambda: app.state.mcp_verified["tools"] is True)
        assert app.query_one("#mcps-next").disabled is False


@pytest.mark.asyncio
async def test_review_save_is_disabled_until_all_sections_are_valid(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    app.state = configured_state(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("review")
        await pilot.pause()
        assert app.query_one("#review-save").disabled is True

        app.state.provider_verified["company"] = True
        app.model_validation = {"llm": True, "embedding": True}
        app.state.mcp_verified["tools"] = True
        app._refresh_review()

        assert app.query_one("#review-save").disabled is False


@pytest.mark.asyncio
async def test_model_probe_failure_stays_on_models_and_notifies(
    monkeypatch, tmp_path
):
    notifications = []
    monkeypatch.setattr(
        SetupApp,
        "notify",
        lambda _self, message, **kwargs: notifications.append((
            message,
            kwargs,
        )),
    )
    monkeypatch.setattr(
        "ursa.cli.setup.probe_chat_model",
        lambda _config, _providers: (_ for _ in ()).throw(
            RuntimeError("401 Unauthorized")
        ),
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        kind="openai",
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_models["company"] = [ProviderModel("gpt-test", "openai")]
    app.state.provider_verified["company"] = True

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await pilot.pause()
        await pilot.click("#models-save")

        assert app.current_page == "models"
        assert app.model_validation["llm"] is False
        assert "✗" in str(
            app.query_one("#llm-validation-status", Static).content
        )
        assert notifications
        assert notifications[-1][1]["severity"] == "error"
        assert "API key was rejected" in notifications[-1][0]
