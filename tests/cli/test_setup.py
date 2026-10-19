import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from keyring.backends.fail import Keyring as FailingSystemKeyring
from keyring.errors import KeyringError
from textual.widgets import (
    Checkbox,
    Input,
    Select,
    Static,
    TabbedContent,
    TextArea,
)

from ursa.cli.config import ChatModelConfig, EmbModelConfig
from ursa.cli.setup import (
    API_KEY_ENV,
    MCP_TOKEN_NONE,
    PROVIDER_DEFAULT_URLS,
    ApiKeyEntry,
    ConfiguredTabs,
    CredentialValue,
    SetupAction,
    SetupApp,
    SetupError,
    _friendly,
    _mcp,
    _save,
    _State,
    load_setup_defaults,
    run_setup,
)
from ursa.cli.tui.widgets import (
    AdvancedYamlEditor,
    HelpLabel,
    ModelEditor,
    ModelSelection,
    ModelTabs,
)
from ursa.util.inference_providers import ProviderModel
from ursa.util.secrets import (
    FallbackKeyring,
    FileKeyring,
    SecretReference,
    SecretTemplate,
)


@pytest.fixture(autouse=True)
def setup_keyring(monkeypatch):
    store = MagicMock()
    store.get_password.return_value = None
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)
    return store


def configured_state(path: Path) -> _State:
    state = _State(path, {"use_web": True})
    state.providers["company"] = {
        "base_url": "https://models.example/v1",
        "api_key": {"keyring": True},
    }
    state.models = {
        "llm": ChatModelConfig(
            model="gpt-test",
            model_provider="openai",
            inference_provider="company",
        ),
        "embedding": EmbModelConfig(
            model="embedding-test",
            model_provider="openai",
            inference_provider="company",
        ),
    }
    state.mcps["tools"] = {
        "transport": "streamable_http",
        "url": "https://tools.example/mcp",
        "headers": {
            "Authorization": {"keyring": True, "template": "Bearer %s"}
        },
    }
    state.pending = {"company": "provider-secret", "tools": "mcp-secret"}
    state.verified = {"provider": {"company": True}, "mcp": {"tools": True}}
    return state


def test_run_setup_injects_truststore(monkeypatch, tmp_path):
    inject = MagicMock()
    app = MagicMock()
    app.run.return_value = SetupAction.TERMINAL
    monkeypatch.setattr("ursa.cli.setup.inject_truststore_into_ssl", inject)
    monkeypatch.setattr("ursa.cli.setup.SetupApp", MagicMock(return_value=app))

    assert run_setup(tmp_path / "config.yaml") is SetupAction.TERMINAL
    inject.assert_called_once_with()


def test_run_setup_treats_window_close_as_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ursa.cli.setup.inject_truststore_into_ssl", lambda: None
    )
    app = MagicMock()
    app.run.return_value = None
    monkeypatch.setattr("ursa.cli.setup.SetupApp", MagicMock(return_value=app))

    assert run_setup(tmp_path / "config.yaml") is SetupAction.EXIT


def test_load_setup_defaults_wraps_installer_errors(tmp_path):
    path = tmp_path / "defaults.json"
    path.write_text('{"unknown": true}')

    with pytest.raises(SetupError, match="unreleased URSA version"):
        load_setup_defaults(path)


def test_state_loads_existing_config_and_always_enables_tls(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company:
    base_url: https://models.example/v1
    api_key: {env: COMPANY_KEY}
    ssl_verify: false
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
mcp_servers:
  tools:
    transport: streamable_http
    url: https://tools.example/mcp
"""
    )

    state = _State.load(path)

    assert state.providers["company"] == {
        "base_url": "https://models.example/v1",
        "api_key": SecretReference(env="COMPANY_KEY"),
    }
    assert state.models["llm"].model == "gpt-test"
    assert state.mcps["tools"]["transport"] == "streamable_http"


@pytest.mark.parametrize("section", ["inference_providers", "mcp_servers"])
def test_state_rejects_non_mapping_sections(tmp_path, section):
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}: []\n")

    with pytest.raises(SetupError, match="must be a mapping"):
        _State.load(path)


def test_state_rejects_provider_mcp_name_collisions(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  shared: {base_url: https://models.example/v1}
mcp_servers:
  shared: {transport: streamable_http, url: https://tools.example/mcp}
"""
    )

    with pytest.raises(SetupError, match="both provider and MCP"):
        _State.load(path)


def test_installer_defaults_add_providers_and_mcp_without_replacing(tmp_path):
    state = configured_state(tmp_path / "config.yaml")
    state.defaults({
        "inference_providers": [
            {"name": "company", "base_url": "https://replacement.test/v1"},
            {"name": "ollama", "base_url": "http://localhost:11434"},
            {
                "name": "hosted",
                "base_url": "https://models.example/v1",
                "reasoning_effort": "high",
                "api_key": {"keyring": "shared"},
            },
        ],
        "mcp_servers": {
            "search": {
                "transport": "streamable_http",
                "url": "https://search.example/mcp",
                "headers": {
                    "Authorization": {
                        "keyring": "shared",
                        "template": "Bearer %s",
                    }
                },
            }
        },
    })

    assert state.providers["company"]["base_url"] == "https://models.example/v1"
    assert state.providers["ollama"] == {"base_url": "http://localhost:11434"}
    assert state.providers["hosted"] == {
        "base_url": "https://models.example/v1",
        "reasoning_effort": "high",
        "api_key": SecretReference(keyring="shared"),
    }
    assert PROVIDER_DEFAULT_URLS["company"] == "https://replacement.test/v1"
    assert state.mcps["search"]["headers"]["Authorization"] == (
        SecretTemplate(keyring="shared", template="Bearer %s")
    )


def test_installer_defaults_reject_duplicate_names_and_secrets(tmp_path):
    state = _State.load(tmp_path / "config.yaml")
    with pytest.raises(SetupError, match="duplicated"):
        state.defaults({
            "inference_providers": [
                {"name": "same", "base_url": "https://one.test/v1"},
                {"name": "same", "base_url": "https://two.test/v1"},
            ]
        })
    with pytest.raises(SetupError, match="contain a secret"):
        state.defaults({
            "inference_providers": [
                {
                    "name": "unsafe",
                    "base_url": "https://models.example/v1",
                    "api_key": "secret",
                }
            ]
        })
    with pytest.raises(SetupError, match="contain a secret"):
        state.defaults({
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                    "headers": {"Authorization": "secret"},
                }
            }
        })


def test_mcp_rejects_unknown_configuration():
    with pytest.raises(SetupError, match="Unknown MCP configuration"):
        _mcp(
            "tools",
            {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
                "typo": True,
            },
        )


def test_output_preserves_unmanaged_config_and_secret_references(tmp_path):
    state = configured_state(tmp_path / "config.yaml")

    config = state.output()

    assert config["use_web"] is True
    assert config["inference_providers"]["company"]["api_key"] == {
        "keyring": True
    }
    assert config["mcp_servers"]["tools"]["headers"]["Authorization"] == {
        "keyring": True,
        "template": "Bearer %s",
    }
    assert "provider-secret" not in yaml.safe_dump(config)


def test_save_writes_file_fallback_then_atomic_private_config(
    monkeypatch, tmp_path
):
    state = configured_state(tmp_path / "nested/config.yaml")
    fallback = FileKeyring(tmp_path / "keyring.json")
    store = FallbackKeyring(
        FailingSystemKeyring(),
        fallback,
    )
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)

    assert _save(state) == state.path

    assert fallback.get_password("ursa", "company") == "provider-secret"
    assert fallback.get_password("ursa", "tools") == "mcp-secret"
    assert stat.S_IMODE(state.path.stat().st_mode) == 0o600
    assert state.path.with_suffix(".yaml.lock").exists()


def test_save_does_not_write_config_when_keyring_fails(setup_keyring, tmp_path):
    state = configured_state(tmp_path / "config.yaml")
    setup_keyring.set_password.side_effect = KeyringError("write failed")

    with pytest.raises(KeyringError, match="write failed"):
        _save(state)

    assert not state.path.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("401 Unauthorized"), "API key was rejected"),
        (RuntimeError("404 Not Found"), "endpoint was not found"),
        (RuntimeError("Connection refused"), "could not connect"),
    ],
)
def test_provider_errors_are_actionable(error, message):
    assert message in _friendly(error)


@pytest.mark.asyncio
async def test_labels_have_help_and_ready_page_has_launch_actions(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)):
        labels = list(app.query(HelpLabel))
        buttons = {
            button.id: str(button.label)
            for button in app.query("#done-page Button")
        }

        assert labels
        assert all(label.tooltip for label in labels)
        assert buttons == {
            "setup-launch-terminal": "Launch URSA - Terminal",
            "setup-launch-dashboard": "Launch URSA - Dashboard",
            "setup-exit": "Exit",
        }


@pytest.mark.asyncio
async def test_welcome_starts_the_railroaded_flow_with_progress_visible(
    tmp_path,
):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        assert app.page == "welcome"
        assert app.query_one("#progress-panel").display
        assert ">  Welcome" in str(
            app.query_one("#setup-progress", Static).content
        )

        await pilot.click("#welcome-next")
        assert app.page == "provider"
        assert app.query_one("#progress-panel").display


@pytest.mark.asyncio
async def test_provider_form_uses_shared_masked_key_widget(
    setup_keyring, tmp_path
):
    setup_keyring.get_password.side_effect = (
        lambda service, name: "stored-secret" if name == "openai" else None
    )
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()
        entry = app.query_one("#provider-key-api-key", ApiKeyEntry)
        key = app.query_one("#provider-key-key", Input)
        show = app.query_one("#provider-key-show", Checkbox)

        assert key.value == "stored-secret"
        assert key.password is True
        show.value = True
        await pilot.pause()
        assert key.password is False
        source = entry.query_one("#provider-key-source", Select)
        assert source.value == "openai"

        source.value = API_KEY_ENV
        await pilot.pause()
        assert key.value == ""
        assert key.placeholder == "Environment variable name"


@pytest.mark.asyncio
async def test_shared_key_widget_supports_environment_and_none(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()
        entry = app.query_one("#provider-key-api-key", ApiKeyEntry)
        source = entry.query_one("#provider-key-source", Select)
        key = entry.query_one("#provider-key-key", Input)

        source.value = API_KEY_ENV
        key.value = "COMPANY_API_KEY"
        assert entry.values() == CredentialValue(
            SecretReference(env="COMPANY_API_KEY")
        )
        source.value = MCP_TOKEN_NONE
        assert entry.values() == CredentialValue()


@pytest.mark.asyncio
async def test_empty_provider_name_stays_on_form_with_validation_error(
    tmp_path,
):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        name = app.query_one("#provider-name", Input)
        source = app.query_one("#provider-key-source", Select)

        name.value = ""
        await pilot.pause()

        assert source.value == "openai"
        await pilot.click("#provider-save")
        assert app.page == "provider"
        assert "must start with a letter" in str(
            app.query_one("#provider-error", Static).content
        )


@pytest.mark.asyncio
async def test_provider_is_verified_before_appearing_in_tabs(
    monkeypatch, tmp_path
):
    models = [ProviderModel("gpt-test", "openai")]
    monkeypatch.setattr("ursa.cli.setup.list_provider_models", lambda _: models)
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()
        app.query_one("#provider-key-key", Input).value = "secret"
        await pilot.click("#provider-save")
        await app.workers.wait_for_complete()

        tabs = app.query_one("#providers-tabs", ConfiguredTabs)
        assert app.page == "providers"
        assert app.state.verified["provider"] == {"openai": True}
        assert tabs.names == {"provider-pane-0": "openai"}
        assert app.query_one("#providers-next").disabled is False


@pytest.mark.asyncio
async def test_existing_provider_tabs_render_before_background_validation(
    monkeypatch, tmp_path
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company: {base_url: https://models.example/v1}
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
"""
    )
    checked = []
    monkeypatch.setattr(
        "ursa.cli.setup.list_provider_models",
        lambda _: checked.append(True) or [ProviderModel("gpt-test", "openai")],
    )
    app = SetupApp(path)

    async with app.run_test(size=(120, 42)):
        app._show("providers")
        await app.workers.wait_for_complete()

        assert app.query_one("#providers-tabs", ConfiguredTabs).names == {
            "provider-pane-0": "company"
        }
        assert checked == [True]


@pytest.mark.asyncio
async def test_provider_failures_block_progress_and_show_first_three(
    monkeypatch, tmp_path
):
    def fail(_):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("ursa.cli.setup.list_provider_models", fail)
    app = SetupApp(tmp_path / "config.yaml")
    for index in range(5):
        name = f"broken-{index}"
        app.state.providers[name] = {"base_url": "https://models.example/v1"}
        app.state.verified["provider"][name] = False
        app.state.errors["provider"][name] = f"error {index}"

    async with app.run_test(size=(120, 42)):
        app._show("providers")
        await app.workers.wait_for_complete()

        message = str(app.query_one("#providers-error", Static).content)
        assert app.query_one("#providers-next").disabled is True
        assert message.startswith(
            "3+ providers failed verification, the first three are:"
        )
        assert "broken-2" in message
        assert "broken-3" not in message


@pytest.mark.asyncio
async def test_models_use_existing_editor_and_validate_both_providers(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers.update({
        "chat": {"base_url": "https://chat.example/v1"},
        "vectors": {"base_url": "https://vectors.example/v1"},
    })
    app.state.verified["provider"] = {"chat": True, "vectors": True}
    checked = []
    monkeypatch.setattr(
        "ursa.cli.setup._probe_model",
        lambda model, _: checked.append((
            model.model,
            model.inference_provider,
        )),
    )

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await app.workers.wait_for_complete()
        editor = app.query_one(ModelEditor)
        assert app.query_one("#progress-panel").display
        tabs = editor.query_one(ModelTabs)
        selection = ModelSelection(
            ChatModelConfig(
                model="chat-model",
                model_provider="openai",
                inference_provider="chat",
            ),
            EmbModelConfig(
                model="embedding-model",
                model_provider="openai",
                inference_provider="vectors",
            ),
        )
        editor._submit(selection)
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert checked == [
            ("chat-model", "chat"),
            ("embedding-model", "vectors"),
        ]
        assert app.model_ok == {"llm": True, "embedding": True}
        assert str(tabs.get_tab("chat-model-tab").label) == (
            "✓ Chat: chat-model"
        )
        assert str(tabs.get_tab("embedding-model-tab").label) == (
            "✓ Embedding: embedding-model"
        )
        assert app.page == "mcps"


@pytest.mark.asyncio
async def test_model_failure_stays_on_models_and_raises_error_toast(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    app.state.providers["company"] = {"base_url": "https://models.example/v1"}
    app.state.verified["provider"]["company"] = True
    app.state.models["llm"] = ChatModelConfig(
        model="gpt-test",
        model_provider="openai",
        inference_provider="company",
    )
    app.state.models["embedding"] = None
    notices = []
    monkeypatch.setattr(
        "ursa.cli.setup._probe_model",
        MagicMock(side_effect=RuntimeError("401 Unauthorized")),
    )
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )

    async with app.run_test(size=(120, 42)):
        app.page = "models"
        app.query_one("#setup-pages").current = "models-page"
        await app._validate_models()

        assert app.page == "models"
        assert app.model_ok["llm"] is False
        assert notices[-1][1]["severity"] == "error"


@pytest.mark.asyncio
async def test_mcp_form_uses_tabs_advanced_yaml_and_shared_token(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock())

    async with app.run_test(size=(120, 42)):
        app._begin_mcp()
        app.query_one("#mcp-name", Input).value = "tools"
        app.query_one("#mcp-url", Input).value = "https://tools.example/mcp"
        source = app.query_one("#mcp-token-source", Select)
        source.set_options([("None", MCP_TOKEN_NONE), ("tools", "tools")])
        source.value = "tools"
        app.query_one("#mcp-token-key", Input).value = "secret"
        editor = app.query_one("#mcp-config-yaml", TextArea)
        editor.text = yaml.safe_dump({
            "transport": "streamable_http",
            "url": "https://tools.example/mcp",
            "headers": {"X-API-Key": {"keyring": True, "template": "Token %s"}},
            "timeout": 12,
        })
        await app._save_mcp()
        await app.workers.wait_for_complete()

        assert app.query_one("#mcp-advanced", AdvancedYamlEditor).collapsed
        assert app.state.mcps["tools"]["timeout"] == 12
        assert app.state.pending["tools"] == "secret"
        assert isinstance(app.query_one("#mcps-tabs"), TabbedContent)
        assert app.state.verified["mcp"]["tools"] is True


@pytest.mark.asyncio
async def test_mcp_failure_stays_on_form(monkeypatch, tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    monkeypatch.setattr(
        app,
        "_probe_mcp",
        AsyncMock(side_effect=RuntimeError("401 Unauthorized")),
    )

    async with app.run_test(size=(120, 42)):
        app._begin_mcp()
        app.query_one("#mcp-name", Input).value = "tools"
        app.query_one("#mcp-url", Input).value = "https://tools.example/mcp"
        await app._save_mcp()

        assert app.page == "mcp"
        assert "tools" not in app.state.mcps
        assert "✗" in str(
            app.query_one("#mcp-validation-status", Static).content
        )


@pytest.mark.asyncio
async def test_tree_truncates_model_names_and_marks_results(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    app.state.models["llm"] = ChatModelConfig(
        model="a-model-name-that-is-long",
        model_provider="openai",
    )
    app.model_ok["llm"] = True

    async with app.run_test(size=(120, 42)):
        app._progress()
        progress = str(app.query_one("#setup-progress", Static).content)

        assert "✓ LLM: a-model-name-that..." in progress
        assert "Emb: None" in progress


@pytest.mark.asyncio
async def test_done_buttons_return_requested_action(monkeypatch, tmp_path):
    app = SetupApp(tmp_path / "config.yaml")
    exit_app = MagicMock()
    monkeypatch.setattr(app, "exit", exit_app)

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("done")
        await pilot.pause()
        for button, action in (
            ("#setup-launch-terminal", SetupAction.TERMINAL),
            ("#setup-launch-dashboard", SetupAction.DASHBOARD),
            ("#setup-exit", SetupAction.EXIT),
        ):
            await pilot.click(button)
            exit_app.assert_called_once_with(action)
            exit_app.reset_mock()
