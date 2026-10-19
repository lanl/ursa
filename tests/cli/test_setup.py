import asyncio
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import keyring
import pytest
import yaml
from textual.widgets import (
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from ursa.cli.config import ChatModelConfig, EmbModelConfig
from ursa.cli.setup import (
    API_KEY_ENV,
    CUSTOM_MODEL,
    CUSTOM_PROVIDER_URL,
    MCP_TOKEN_NONE,
    PROVIDER_DEFAULT_URL,
    PROVIDER_DEFAULT_URLS,
    ApiKeyEntry,
    ConfiguredTabs,
    ExpandingFuzzySelect,
    MCPDraft,
    ProviderDraft,
    SetupAction,
    SetupApp,
    SetupError,
    SetupState,
    _friendly_validation_error,
    load_setup_defaults,
    probe_mcp_server,
    run_setup,
    save_setup_state,
)
from ursa.cli.tui.widgets import AdvancedYamlEditor, FuzzySelect, HelpLabel
from ursa.util.inference_providers import ProviderModel


def _tab_names(app: SetupApp, tab_id: str) -> dict[str, str]:
    return app.query_one(f"#{tab_id}", ConfiguredTabs).names


def configured_state(path: Path) -> SetupState:
    state = SetupState(
        path=path,
        base_config={"use_web": True},
    )
    state.providers["company"] = ProviderDraft(
        name="company",
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
    app.run.return_value = SetupAction.TERMINAL
    create_app = MagicMock(return_value=app)
    monkeypatch.setattr("ursa.cli.setup.inject_truststore_into_ssl", inject)
    monkeypatch.setattr("ursa.cli.setup.SetupApp", create_app)
    path = tmp_path / "config.yaml"

    assert run_setup(path) is SetupAction.TERMINAL

    inject.assert_called_once_with()
    create_app.assert_called_once_with(path, None)


def test_run_setup_treats_window_close_as_exit(monkeypatch, tmp_path):
    app = MagicMock()
    app.run.return_value = None
    monkeypatch.setattr(
        "ursa.cli.setup.inject_truststore_into_ssl", MagicMock()
    )
    monkeypatch.setattr("ursa.cli.setup.SetupApp", MagicMock(return_value=app))

    assert run_setup(tmp_path / "config.yaml") is SetupAction.EXIT


@pytest.mark.asyncio
async def test_done_actions_return_selected_launch_action(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    exit_app = MagicMock()
    monkeypatch.setattr(app, "exit", exit_app)

    for button_id, action in (
        ("setup-launch-terminal", SetupAction.TERMINAL),
        ("setup-launch-dashboard", SetupAction.DASHBOARD),
        ("setup-exit", SetupAction.EXIT),
    ):
        event = MagicMock()
        event.button.id = button_id
        await app.button_pressed(event)
        exit_app.assert_called_once_with(action)
        exit_app.reset_mock()


def test_done_page_offers_terminal_dashboard_and_exit():
    buttons = SetupApp._done_buttons()

    assert [(button.id, str(button.label)) for button in buttons] == [
        ("setup-launch-terminal", "Launch URSA - Terminal"),
        ("setup-launch-dashboard", "Launch URSA - Dashboard"),
        ("setup-exit", "Exit"),
    ]


@pytest.mark.asyncio
async def test_setup_input_labels_expose_contextual_help(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    expected = {
        "provider-name-label": SetupApp.FIELD_HELP["provider-name"],
        "provider-url-label": SetupApp.FIELD_HELP["provider-url"],
        "models-provider-label": SetupApp.FIELD_HELP["model-provider"],
        "embedding-provider-label": SetupApp.FIELD_HELP["model-provider"],
        "llm-label": SetupApp.FIELD_HELP["llm"],
        "embedding-label": SetupApp.FIELD_HELP["embedding"],
        "mcp-name-label": SetupApp.FIELD_HELP["mcp-name"],
        "mcp-transport-label": SetupApp.FIELD_HELP["mcp-transport"],
        "mcp-url-label": SetupApp.FIELD_HELP["mcp-url"],
        "mcp-command-label": SetupApp.FIELD_HELP["mcp-command"],
        "mcp-args-label": SetupApp.FIELD_HELP["mcp-args"],
        "provider-key-source-label": (
            "Choose a keyring entry, ENV for an environment variable, or None "
            "when this connection does not need authentication."
        ),
        "provider-key-value-label": (
            "Enter the credential to save in the operating-system keyring. "
            "Existing values load automatically and remain hidden unless Show "
            "is selected."
        ),
        "mcp-token-source-label": (
            "Choose a keyring entry, ENV for an environment variable, or None "
            "when this connection does not need authentication."
        ),
        "mcp-token-value-label": (
            "Enter the credential to save in the operating-system keyring. "
            "Existing values load automatically and remain hidden unless Show "
            "is selected."
        ),
    }

    async with app.run_test(size=(120, 42)):
        labels = {label.id: label for label in app.query(HelpLabel)}

        assert set(labels) == set(expected)
        for label_id, tooltip in expected.items():
            label = labels[label_id]
            assert label.tooltip == tooltip
            assert (
                str(label.query_one(".help-label-mark", Static).content) == "?"
            )


@pytest.mark.asyncio
async def test_deferred_page_preparation_stops_after_navigation(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    app.current_page = "models"
    app._refresh_provider_tabs = AsyncMock()
    app._start_provider_validation = MagicMock()
    app._refresh_mcp_tabs = AsyncMock()
    app._start_mcp_validation = MagicMock()

    await app._prepare_providers_page()
    await app._prepare_mcps_page()

    app._refresh_provider_tabs.assert_not_awaited()
    app._start_provider_validation.assert_not_called()
    app._refresh_mcp_tabs.assert_not_awaited()
    app._start_mcp_validation.assert_not_called()


def test_setup_state_applies_non_secret_installer_defaults(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(
        PROVIDER_DEFAULT_URLS,
        "anthropic",
        PROVIDER_DEFAULT_URLS["anthropic"],
    )
    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text(
        """
{
  "python_version": "3.13",
  "inference_providers": [
    {"name": "anthropic", "base_url": "https://api.anthropic.test"}
  ],
  "mcp_servers": {
    "tools": {
      "transport": "streamable_http",
      "url": "https://tools.example/mcp",
      "headers": {
        "Authorization": {"template": "Bearer %s"}
      },
      "timeout": 30
    }
  }
}
"""
    )
    state = SetupState.load(tmp_path / "config.yaml")

    state.apply_defaults(load_setup_defaults(defaults_path))

    provider = state.providers["anthropic"]
    assert provider.base_url == "https://api.anthropic.test"
    assert PROVIDER_DEFAULT_URLS["anthropic"] == ("https://api.anthropic.test")
    assert provider.api_key_reference == {"keyring": True}
    server = state.mcp_servers["tools"]
    assert server.auth is True
    assert server.secret_reference == {
        "keyring": True,
        "template": "Bearer %s",
    }
    assert server.template == "Bearer %s"
    assert server.extra == {"timeout": 30}


def test_setup_defaults_do_not_replace_existing_entries(monkeypatch, tmp_path):
    monkeypatch.setitem(
        PROVIDER_DEFAULT_URLS, "company", "https://original.test/v1"
    )
    state = configured_state(tmp_path / "config.yaml")

    state.apply_defaults({
        "inference_providers": [
            {"name": "company", "base_url": "https://replacement.test/v1"}
        ],
        "mcp_servers": {
            "tools": {
                "transport": "streamable_http",
                "url": "https://replacement.test/mcp",
            }
        },
    })

    assert state.providers["company"].base_url == "https://models.example/v1"
    assert state.mcp_servers["tools"].url == "https://tools.example/mcp"
    assert PROVIDER_DEFAULT_URLS["company"] == ("https://replacement.test/v1")


def test_setup_defaults_reject_duplicate_provider_names(monkeypatch, tmp_path):
    monkeypatch.setitem(
        PROVIDER_DEFAULT_URLS, "duplicate", "https://original.test/v1"
    )
    state = SetupState.load(tmp_path / "config.yaml")

    with pytest.raises(SetupError, match="default 'duplicate' is duplicated"):
        state.apply_defaults({
            "inference_providers": [
                {
                    "name": "duplicate",
                    "base_url": "https://one.example/v1",
                },
                {
                    "name": "duplicate",
                    "base_url": "https://two.example/v1",
                },
            ]
        })
    assert PROVIDER_DEFAULT_URLS["duplicate"] == "https://original.test/v1"
    assert "duplicate" not in state.providers


def test_setup_rejects_provider_and_mcp_name_collisions(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  shared:
    base_url: https://models.example/v1
mcp_servers:
  shared:
    transport: streamable_http
    url: https://tools.example/mcp
"""
    )

    with pytest.raises(SetupError, match="used by both: shared"):
        SetupState.load(path)


def test_setup_defaults_reject_mcp_secret_references(tmp_path):
    state = SetupState.load(tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="must not include API keys"):
        state.apply_defaults({
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                    "headers": {
                        "Authorization": {
                            "keyring": True,
                            "template": "Bearer %s",
                        }
                    },
                }
            }
        })


@pytest.mark.parametrize(
    "header",
    [
        "Authorization",
        "X-API-Key",
        "X-Auth",
        "X-Token",
        "X-Secret",
        "X-Password",
        "X-Credential",
    ],
)
def test_setup_defaults_reject_literal_mcp_credentials(tmp_path, header):
    state = SetupState.load(tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="literal credential"):
        state.apply_defaults({
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                    "headers": {header: "literal-secret"},
                }
            }
        })


def test_setup_rejects_invalid_existing_model_configuration(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm_model:\n  model: [not, a, string]\n")

    with pytest.raises(SetupError, match="Existing llm_model.*model"):
        SetupState.load(path)


@pytest.mark.parametrize(
    ("section", "value"),
    [("llm_model", "[]"), ("emb_model", "invalid")],
)
def test_setup_rejects_non_mapping_model_configuration(
    tmp_path, section, value
):
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}: {value}\n")

    with pytest.raises(SetupError, match=f"Existing {section}.*mapping"):
        SetupState.load(path)


@pytest.mark.parametrize(
    ("section", "value"),
    [("inference_providers", "[]"), ("mcp_servers", "invalid")],
)
def test_setup_rejects_non_mapping_configuration_sections(
    tmp_path, section, value
):
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}: {value}\n")

    with pytest.raises(SetupError, match=f"Existing {section}.*mapping"):
        SetupState.load(path)


def test_setup_rejects_non_mapping_provider_entry(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inference_providers:\n  company: invalid\n")

    with pytest.raises(SetupError, match="provider 'company'.*mapping"):
        SetupState.load(path)


def test_setup_wraps_invalid_configuration_root(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("[]\n")

    with pytest.raises(SetupError, match="must contain a mapping at its root"):
        SetupState.load(path)


def test_setup_state_builds_keyring_only_configuration(tmp_path):
    state = configured_state(tmp_path / "config.yaml")
    config = state.configuration()

    assert config["use_web"] is True
    assert config["inference_providers"]["company"] == {
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
        base_url="https://models.example/v1",
        api_key_reference={"keyring": True},
    )

    config = provider.validation_config().resolve(provider.name)

    assert config.api_key is not None
    assert config.api_key.keyring == "company"


def test_environment_credentials_round_trip_for_providers_and_mcp():
    provider = ProviderDraft.from_config(
        "company",
        {
            "base_url": "https://models.example/v1",
            "api_key": {"env": "COMPANY_API_KEY"},
        },
    )
    server = MCPDraft.from_config(
        "tools",
        {
            "transport": "streamable_http",
            "url": "https://tools.example/mcp",
            "headers": {
                "Authorization": {
                    "env": "TOOLS_TOKEN",
                    "template": "Bearer %s",
                }
            },
        },
    )

    assert provider.as_config()["api_key"] == {"env": "COMPANY_API_KEY"}
    assert server.as_config()["headers"]["Authorization"] == {
        "env": "TOOLS_TOKEN",
        "template": "Bearer %s",
    }


def test_mcp_draft_rejects_unknown_configuration_fields():
    with pytest.raises(SetupError, match="unknown configuration field: typo"):
        MCPDraft.from_config(
            "tools",
            {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
                "typo": True,
            },
        )


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
    assert config["inference_providers"]["company"]["model_provider"] == (
        "openai"
    )
    assert config["llm_model"]["max_completion_tokens"] == 512
    assert config["mcp_servers"]["tools"]["timeout"] == 30


def test_existing_provider_without_url_uses_model_protocol_for_validation(
    tmp_path,
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    api_key: {env: COMPANY_API_KEY}
llm_model:
  model: claude-test
  model_provider: anthropic
  inference_provider: company
"""
    )

    state = SetupState.load(path)
    provider = state.providers["company"]
    validation = state.provider_validation_config("company")

    assert provider.base_url is None
    assert "base_url" not in provider.as_config()
    assert (
        "base_url"
        not in (state.configuration()["inference_providers"]["company"])
    )
    assert validation.base_url is None
    assert validation.model_provider == "anthropic"


@pytest.mark.asyncio
async def test_existing_provider_without_url_round_trips_through_edit(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda config: ([ProviderModel("claude-test", config.model_provider)]),
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        base_url=None,
        api_key_reference={"env": "COMPANY_API_KEY"},
    )
    app.state.llm_model = ChatModelConfig(
        model="claude-test",
        model_provider="anthropic",
        inference_provider="company",
    )
    app.state.provider_verified["company"] = True

    async with app.run_test(size=(120, 42)) as pilot:
        app.current_page = "providers"
        app.query_one("#setup-pages").current = "providers-page"
        await app._refresh_provider_tabs()
        await pilot.click("#provider-edit")

        url = app.query_one("#provider-url", ExpandingFuzzySelect)
        assert url.value == PROVIDER_DEFAULT_URL
        assert app.query_one("#provider-url-custom", Input).has_class("hidden")

        await app._validate_provider()
        assert app.current_page == "providers"
        assert app.state.providers["company"].base_url is None
        assert "base_url" not in app.state.providers["company"].as_config()


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

    assert state.providers["company"].validation_config().ssl_verify is True
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
    assert path.with_suffix(".yaml.lock").is_file()
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
        await app._validate_provider()

        assert app.query_one("#providers-tabs", TabbedContent)
        assert app.state.provider_verified["company"] is True
        await app._run_provider_validation(force=False, continue_to_models=True)
        assert app.current_page == "models"
        await pilot.pause()

        assert app.query_one("#llm-choice", Select).value == "gpt-test"
        app.query_one("#embedding-choice", Select).value = "text-embedding-test"
        await pilot.pause()
        app._update_progress()
        assert "Emb: text-embeddi..." in str(
            app.query_one("#setup-progress", Static).content
        )
        await app._save_models()
        assert app.current_page == "mcps"
        assert app.query_one("#mcps-tabs", TabbedContent)
        assert chat_probes == ["gpt-test"]
        assert embedding_probes == ["text-embedding-test"]
        assert "✓" in str(
            app.query_one("#llm-validation-status", Static).content
        )
        progress = str(app.query_one("#setup-progress", Static).content)
        assert "✓ LLM: gpt-test" in progress
        assert "✓ Emb: text-embeddi..." in progress
        await app._run_mcp_validation(force=False, continue_to_review=True)
        assert app.current_page == "review"
        assert "company" in str(
            app.query_one("#review-summary", Static).content
        )


@pytest.mark.asyncio
async def test_models_can_use_different_inference_providers(
    monkeypatch, tmp_path
):
    chat_probes = []
    embedding_probes = []
    monkeypatch.setattr(
        "ursa.cli.setup.probe_chat_model",
        lambda config, _providers: chat_probes.append((
            config.model,
            config.inference_provider,
        )),
    )
    monkeypatch.setattr(
        "ursa.cli.setup.probe_embedding_model",
        lambda config, _providers: embedding_probes.append((
            config.model,
            config.inference_provider,
        )),
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["chat"] = ProviderDraft(
        name="chat", base_url="https://chat.example/v1"
    )
    app.state.providers["vectors"] = ProviderDraft(
        name="vectors", base_url="https://vectors.example/v1"
    )
    app.state.provider_models["chat"] = [ProviderModel("chat-model", "openai")]
    app.state.provider_models["vectors"] = [
        ProviderModel("embedding-model", "openai", type="embedding")
    ]

    async with app.run_test(size=(120, 42)):
        app._show("models")
        embedding_provider = app.query_one("#embedding-provider", Select)
        with app.prevent(Select.Changed):
            embedding_provider.value = "vectors"
        app._populate_model_choices("embedding", "vectors")
        embedding_choice = app.query_one("#embedding-choice", Select)
        with app.prevent(Select.Changed):
            embedding_choice.value = "embedding-model"
        app._sync_model_yaml_from_controls("embedding")

        await app._save_models()

        assert app.state.llm_model is not None
        assert app.state.llm_model.inference_provider == "chat"
        assert app.state.embedding_model is not None
        assert app.state.embedding_model.inference_provider == "vectors"
        assert chat_probes == [("chat-model", "chat")]
        assert embedding_probes == [("embedding-model", "vectors")]


@pytest.mark.asyncio
async def test_model_discovery_ignores_result_after_leaving_models(
    monkeypatch, tmp_path
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_to_thread(_function, _config):
        started.set()
        await release.wait()
        return [ProviderModel("stale-model", "openai")]

    monkeypatch.setattr(
        "ursa.cli.setup.asyncio.to_thread", controlled_to_thread
    )
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company", base_url="https://models.example/v1"
    )

    async with app.run_test(size=(120, 42)):
        app.current_page = "models"
        app.query_one("#setup-pages").current = "models-page"
        provider = app.query_one("#model-provider", Select)
        with app.prevent(Select.Changed):
            provider.set_options([("company", "company")])
            provider.value = "company"
        generation = app._model_load_generation["llm"]
        task = asyncio.create_task(
            app._load_selected_provider_models("llm", "company", generation)
        )
        await started.wait()
        try:
            app._show("welcome")
        finally:
            release.set()
        await task

        assert "company" not in app.state.provider_models


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
        urls = app.query_one("#provider-url", ExpandingFuzzySelect)
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

        credential_source = app.query_one("#provider-key-source", FuzzySelect)
        custom_url.value = "http://localhost:1234/v1"
        await pilot.pause()
        assert credential_source.value == MCP_TOKEN_NONE
        custom_url.value = "https://models.example/v1"
        await pilot.pause()
        assert credential_source.value == "company"

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
async def test_model_tree_truncates_names(tmp_path):
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
        app._update_progress()
        progress = str(app.query_one("#setup-progress", Static).content)
        assert f"LLM: {llm_name[:12]}..." in progress
        assert f"Emb: {embedding_name[:12]}..." in progress


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
        await app._validate_provider()
        assert app.current_page == "providers"
        app._show("models")
        await pilot.pause()
        app.query_one("#llm-choice", Select).value = CUSTOM_MODEL
        app.query_one("#llm-custom", Input).value = "private-model"
        await pilot.pause()
        await app._save_models()
        assert app.current_page == "mcps"
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
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_models["company"] = models
    app.state.provider_verified["company"] = True

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await pilot.pause()

        llm_advanced = app.query_one("#llm-advanced", Collapsible)
        embedding_advanced = app.query_one("#embedding-advanced", Collapsible)
        assert llm_advanced.collapsed is True
        assert embedding_advanced.collapsed is True
        llm_editor = app.query_one("#llm-config-yaml", TextArea)
        embedding_editor = app.query_one("#embedding-config-yaml", TextArea)
        assert llm_editor.parent.display is False
        assert embedding_editor.parent.display is False
        llm_title = llm_advanced.query_one("CollapsibleTitle")
        embedding_title = embedding_advanced.query_one("CollapsibleTitle")
        assert str(llm_title.label) == "Advanced"
        assert str(embedding_title.label) == "Advanced"
        assert llm_title.region.height == 1
        assert embedding_title.region.height == 1
        assert app.query_one("#llm-choice", Select).region.bottom <= (
            llm_advanced.region.y
        )
        assert (
            llm_advanced.region.bottom
            <= app.query_one("#llm-validation-status", Static).region.y
        )

        embedding_advanced.collapsed = False
        llm_advanced.collapsed = False
        await pilot.pause()
        assert llm_advanced.collapsed is False
        assert embedding_advanced.collapsed is False
        assert llm_editor.parent.display is True
        assert embedding_editor.parent.display is True
        llm_values = yaml.safe_load(llm_editor.text)
        llm_values["temperature"] = 0.2
        llm_editor.text = yaml.safe_dump(llm_values, sort_keys=False)

        embedding_values = yaml.safe_load(embedding_editor.text)
        embedding_values["model"] = "text-embedding-test"
        embedding_values["dimensions"] = 512
        embedding_editor.text = yaml.safe_dump(
            embedding_values, sort_keys=False
        )

        for prefix in ("llm", "embedding"):
            app.query_one(
                f"#{prefix}-advanced", AdvancedYamlEditor
            ).cancel_validation()
            assert app._validate_model_yaml(prefix, update_controls=True)
        assert llm_editor.has_class("yaml-valid")
        assert embedding_editor.has_class("yaml-valid")
        assert (
            app.query_one("#embedding-choice", Select).value
            == "text-embedding-test"
        )
        await app._save_models()
        assert app.current_page == "mcps"

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
        app.current_page = "providers"
        app.query_one("#setup-pages").current = "providers-page"
        await app._refresh_provider_tabs()
        await pilot.click("#provider-edit")

        key = app.query_one("#provider-key", Input)
        await app.workers.wait_for_complete()
        await pilot.pause()
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
        assert app.query_one("#provider-key-source", FuzzySelect)
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
        await app.workers.wait_for_complete()
        await pilot.pause()
        provider_entry = app.query_one("#provider-api-key", ApiKeyEntry)
        provider_source = app.query_one("#provider-key-source")
        provider_source_label = provider_entry.query_one(
            ".credential-source-label", HelpLabel
        )
        provider_value_label = provider_entry.query_one(
            ".credential-value-label", HelpLabel
        )
        assert provider_entry.region.width > 0
        assert provider_source.region.y == provider_key.region.y
        assert provider_source_label.region.y == provider_value_label.region.y
        assert provider_source_label.region.bottom == provider_source.region.y
        assert provider_key.parent.region.right <= provider_entry.region.right
        assert provider_show.region.right <= provider_entry.region.right
        app.query_one("#provider-name", Input).value = "company"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        app._begin_mcp()
        mcp_key = app.query_one("#mcp-token", Input)
        mcp_show = app.query_one("#mcp-show-key", Checkbox)
        assert mcp_key.parent is mcp_show.parent
        assert mcp_key.parent.has_class("api-key-row")
        assert app.query_one("#mcp-token-source").region.y == mcp_key.region.y
        app.query_one("#mcp-name", Input).value = "tools"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one("#mcp-token-source", FuzzySelect).value = "tools"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        provider_source = app.query_one("#provider-key-source", FuzzySelect)
        provider_source.value = API_KEY_ENV
        await pilot.pause()
        provider_label = app.query_one(
            "#provider-api-key .credential-value-label", HelpLabel
        )
        assert provider_label.label == ("Environmental Variable with API Key")
        assert provider_label.region.bottom == provider_key.region.y
        assert provider_show.has_class("hidden")
        assert provider_key.password is False
        provider_key.value = "COMPANY_API_KEY"
        assert provider_entry.values() == (
            {"env": "COMPANY_API_KEY"},
            None,
            None,
        )

        mcp_source = app.query_one("#mcp-token-source", FuzzySelect)
        mcp_source.value = API_KEY_ENV
        await pilot.pause()
        mcp_key.value = "TOOLS_TOKEN"
        assert app.query_one("#mcp-api-key", ApiKeyEntry).values() == (
            {"env": "TOOLS_TOKEN"},
            None,
            None,
        )


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
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_verified["company"] = None

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()

        assert checked == ["https://models.example/v1"]
        assert "✓ All providers verified" in str(
            app.query_one("#providers-validation-status", Static).content
        )
        assert app.query_one("#providers-next").disabled is False


@pytest.mark.asyncio
async def test_provider_tabs_remain_interactive_during_validation(
    tmp_path,
):
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = ProviderDraft(
        name="company",
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.providers["backup"] = ProviderDraft(
        name="backup",
        base_url="https://backup.example/v1",
        pending_secret="secret",
    )
    app.state.provider_verified["company"] = None
    app.state.provider_verified["backup"] = None

    async with app.run_test(size=(120, 42)) as pilot:
        app.current_page = "providers"
        app.query_one("#setup-pages").current = "providers-page"
        app._providers_validating.add("company")
        await app._refresh_provider_tabs()
        await pilot.pause()

        assert _tab_names(app, "providers-tabs") == {
            "provider-pane-0": "company",
            "provider-pane-1": "backup",
        }
        provider_tabs_widget = app.query_one("#providers-tabs", ConfiguredTabs)
        assert len(list(provider_tabs_widget.query(TabPane))) == 2
        assert app.state.provider_verified["company"] is None

        tabs = app.query_one("#providers-tabs", TabbedContent)
        tabs.active = "provider-pane-1"
        assert tabs.active == "provider-pane-1"


@pytest.mark.asyncio
async def test_all_providers_must_verify_before_models(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    app = SetupApp(path)
    app.state.providers["broken"] = ProviderDraft(
        name="broken",
        base_url="https://models.example/v1",
        pending_secret="bad-secret",
    )
    app.state.provider_verified["broken"] = None
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda _config: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")),
    )

    async with app.run_test(size=(120, 42)):
        app.current_page = "providers"
        app.query_one("#setup-pages").current = "providers-page"
        await app._refresh_provider_tabs()
        await app._run_provider_validation(
            force=False, continue_to_models=False
        )

        assert app.current_page == "providers"
        assert app.state.provider_verified["broken"] is False
        assert app.query_one("#providers-next").disabled is True
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
        await app._run_provider_validation(force=True, continue_to_models=False)
        assert app.query_one("#providers-next").disabled is False


@pytest.mark.asyncio
async def test_provider_failure_summary_shows_only_first_three(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    for index in range(5):
        name = f"broken-{index}"
        app.state.providers[name] = ProviderDraft(
            name=name,
            base_url="https://models.example/v1",
            pending_secret="bad-secret",
        )
        app.state.provider_verified[name] = False
        app.state.provider_errors[name] = f"error {index}"

    async with app.run_test(size=(120, 42)):
        app.current_page = "providers"
        app.query_one("#setup-pages").current = "providers-page"
        await app._refresh_provider_tabs()
        await app._run_provider_validation(
            force=False, continue_to_models=False
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
        await pilot.pause()
        token_source.value = "shared"
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.query_one("#mcp-auth-fields").has_class("hidden")
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

        token_source.value = "tools"
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

        await pilot.pause()
        app.query_one("#mcp-advanced", AdvancedYamlEditor).cancel_validation()
        assert app._validate_mcp_yaml(update_controls=True)
        await app.workers.wait_for_complete()
        assert (
            app.query_one("#mcp-token-source", FuzzySelect).value
            == "shared-token"
        )
        assert editor.has_class("yaml-valid")
        assert app.query_one("#mcp-token", Input).value == "shared-secret"
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
        name = app.query_one("#mcp-name", Input)
        with name.prevent(Input.Changed):
            name.value = "tools"
        url = app.query_one("#mcp-url", Input)
        with url.prevent(Input.Changed):
            url.value = "https://tools.example/mcp"
        await app._save_mcp()
        assert app.current_page == "mcps"
        await app._refresh_mcp_tabs()

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
        name = app.query_one("#mcp-name", Input)
        with name.prevent(Input.Changed):
            name.value = "tools"
        url = app.query_one("#mcp-url", Input)
        with url.prevent(Input.Changed):
            url.value = "https://tools.example/mcp"
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
        app.current_page = "mcps"
        app.query_one("#setup-pages").current = "mcps-page"
        await app._refresh_mcp_tabs()
        await app._run_mcp_validation(force=False, continue_to_review=False)

        tabs = app.query_one("#mcps-tabs", TabbedContent)
        assert _tab_names(app, "mcps-tabs") == {"mcp-pane-0": "tools"}
        assert "✗ tools" in str(tabs.get_tab("mcp-pane-0").label)
        assert app.query_one("#mcps-next").disabled is True
        await pilot.click("#mcps-next")
        assert app.current_page == "mcps"

        async def succeed(_server):
            return 2

        monkeypatch.setattr("ursa.cli.setup.probe_mcp_server", succeed)
        await app._run_mcp_validation(force=True, continue_to_review=False)
        assert app.query_one("#mcps-next").disabled is False


@pytest.mark.asyncio
async def test_mcp_tabs_select_the_server_to_edit(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    for name in ("search", "storage"):
        app.state.mcp_servers[name] = MCPDraft(
            name=name,
            transport="streamable_http",
            url=f"https://{name}.example/mcp",
        )
        app.state.mcp_verified[name] = True

    async with app.run_test(size=(120, 42)):
        app.current_page = "mcps"
        app.query_one("#setup-pages").current = "mcps-page"
        await app._refresh_mcp_tabs()

        tabs = app.query_one("#mcps-tabs", TabbedContent)
        assert _tab_names(app, "mcps-tabs") == {
            "mcp-pane-0": "search",
            "mcp-pane-1": "storage",
        }
        tabs.active = "mcp-pane-1"
        assert tabs.active == "mcp-pane-1"
        selected = app._active_mcp_name()
        assert selected == "storage"
        app._begin_mcp(selected)

        assert app.editing_mcp == "storage"
        assert app.query_one("#mcp-name", Input).value == "storage"


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
        base_url="https://models.example/v1",
        pending_secret="secret",
    )
    app.state.provider_models["company"] = [ProviderModel("gpt-test", "openai")]
    app.state.provider_verified["company"] = True

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await pilot.pause()
        app.query_one("#models-save").press()
        await pilot.pause()

        assert app.current_page == "models"
        assert app.model_validation["llm"] is False
        assert "✗" in str(
            app.query_one("#llm-validation-status", Static).content
        )
        assert notifications
        assert notifications[-1][1]["severity"] == "error"
        assert "API key was rejected" in notifications[-1][0]
