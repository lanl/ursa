import stat
from unittest.mock import AsyncMock, MagicMock

import pytest
from keyring.backends.fail import Keyring as FailingSystemKeyring
from keyring.errors import KeyringError
from textual.widgets import Button, Input, Select, Static

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
)
from ursa.cli.setup import (
    API_KEY_ENV,
    MCP_TOKEN_NONE,
    ApiKeyEntry,
    CredentialValue,
    SetupApp,
    SetupError,
    _load_config,
    _save,
)
from ursa.cli.tui.widgets import ModelEditor, ModelSelection
from ursa.util.inference_providers import ProviderModel
from ursa.util.secrets import (
    FallbackKeyring,
    FileKeyring,
    SecretReference,
)


@pytest.fixture(autouse=True)
def setup_keyring(monkeypatch):
    store = MagicMock()
    store.get_password.return_value = None
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)
    return store


def configured_config() -> UrsaConfig:
    return UrsaConfig.model_validate({
        "use_web": True,
        "inference_providers": {
            "company": {
                "base_url": "https://models.example/v1",
                "api_key": {"keyring": True},
            }
        },
        "llm_model": {
            "model": "gpt-test",
            "model_provider": "openai",
            "inference_provider": "company",
        },
        "emb_model": {
            "model": "embedding-test",
            "model_provider": "openai",
            "inference_provider": "company",
        },
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
        },
    })


def test_run_setup_uses_highest_precedence_user_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    app = SetupApp()

    assert app.path == tmp_path / "ursa/config.yaml"


def test_load_uses_canonical_config_and_always_enables_tls(tmp_path):
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

    config, llm_configured = _load_config(path)

    assert config.inference_providers["company"] == InferenceProviderConfig(
        base_url="https://models.example/v1",
        api_key=SecretReference(env="COMPANY_KEY"),
    )
    assert config.llm_model.model == "gpt-test"
    assert config.mcp_servers["tools"].url == "https://tools.example/mcp"
    assert llm_configured


@pytest.mark.parametrize("section", ["inference_providers", "mcp_servers"])
def test_load_rejects_non_mapping_sections(tmp_path, section):
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}: []\n")

    with pytest.raises(SetupError, match="mapping|dictionary"):
        _load_config(path)


def test_load_rejects_provider_mcp_name_collisions(tmp_path):
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
        _load_config(path)


def test_save_writes_file_fallback_then_atomic_private_config(
    monkeypatch, tmp_path
):
    path = tmp_path / "nested/config.yaml"
    config = configured_config()
    pending = {"company": "provider-secret", "tools": "mcp-secret"}
    fallback = FileKeyring(tmp_path / "keyring.json")
    store = FallbackKeyring(
        FailingSystemKeyring(),
        fallback,
    )
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)

    assert _save(path, config, pending) == path

    assert fallback.get_password("ursa", "company") == "provider-secret"
    assert fallback.get_password("ursa", "tools") == "mcp-secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.with_suffix(".yaml.lock").exists()


def test_save_does_not_write_config_when_keyring_fails(setup_keyring, tmp_path):
    path = tmp_path / "config.yaml"
    setup_keyring.set_password.side_effect = KeyringError("write failed")

    with pytest.raises(KeyringError, match="write failed"):
        _save(path, configured_config(), {"company": "secret"})

    assert not path.exists()


def test_save_restores_keyring_when_config_write_fails(monkeypatch, tmp_path):
    store = FileKeyring(tmp_path / "keyring.json")
    store.set_password("ursa", "company", "old-secret")
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)
    monkeypatch.setattr(
        "ursa.cli.setup.save_config_file",
        MagicMock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _save(
            tmp_path / "config.yaml",
            configured_config(),
            {"company": "new-secret", "tools": "mcp-secret"},
        )

    assert store.get_password("ursa", "company") == "old-secret"
    assert store.get_password("ursa", "tools") is None


def test_pending_shared_secret_is_kept_until_last_reference_is_removed(
    tmp_path,
):
    app = SetupApp(tmp_path / "config.yaml")
    shared = InferenceProviderConfig(api_key=SecretReference(keyring="shared"))
    app.config.inference_providers = {
        "first": shared,
        "second": shared.model_copy(),
    }
    app.pending["shared"] = "secret"

    app.config.inference_providers.pop("first")
    app._discard_unused_pending({"shared"})
    assert app.pending == {"shared": "secret"}

    app.config.inference_providers.clear()
    app._discard_unused_pending({"shared"})
    assert app.pending == {}


@pytest.mark.asyncio
async def test_write_page_saves_and_shows_ready(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    config_yaml = "use_web: true\n"

    def save(*_args):
        path.write_text(config_yaml)
        return path

    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    monkeypatch.setattr("ursa.cli.setup._save", save)
    monkeypatch.setattr("ursa.cli.setup.asyncio.to_thread", inline)
    app = SetupApp(path)

    async with app.run_test(size=(120, 42)):
        app._show("review")
        await app._save_config()
        assert app.page == "done"
        assert str(app.query_one("#done-copy", Static).content) == (
            f"Saved to {path}"
        )


@pytest.mark.asyncio
async def test_shared_key_widget_supports_environment_and_none(tmp_path):
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()
        assert app.query_one("#provider-next", Button).disabled
        entry = app.query_one("#provider-key-api-key", ApiKeyEntry)
        source = entry.query_one("#provider-key-source", Select)
        key = entry.query_one("#provider-key-key", Input)

        entry.loaded = key.value = "secret"
        entry.rename("renamed")
        assert entry.values().pending == "secret"
        source.value = API_KEY_ENV
        key.value = "COMPANY_API_KEY"
        assert entry.values() == CredentialValue(
            SecretReference(env="COMPANY_API_KEY")
        )
        source.value = MCP_TOKEN_NONE
        assert entry.values() == CredentialValue()


@pytest.mark.asyncio
async def test_provider_is_checked_before_becoming_configured(
    monkeypatch, tmp_path
):
    models = [ProviderModel("gpt-test", "openai")]

    def list_models(_):
        return models

    monkeypatch.setattr("ursa.cli.setup.list_provider_models", list_models)
    app = SetupApp(tmp_path / "config.yaml")

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await app.workers.wait_for_complete()

        async def run_inline(callback, *args):
            if callback is list_models:
                assert (
                    str(app.query_one("#provider-error", Static).content)
                    == "? Validating openai"
                )
            return callback(*args)

        monkeypatch.setattr("ursa.cli.setup.asyncio.to_thread", run_inline)
        app.query_one("#provider-key-key", Input).value = "secret"
        await pilot.click("#provider-save")
        await app.workers.wait_for_complete()

        assert app.page == "provider"
        assert app.query_one("#provider-configured", Select).value == "openai"
        assert not app.query_one("#provider-next", Button).disabled


@pytest.mark.asyncio
async def test_models_use_existing_editor_and_validate_both_providers(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    app.config.inference_providers.update({
        "chat": InferenceProviderConfig(base_url="https://chat.example/v1"),
        "vectors": InferenceProviderConfig(
            base_url="https://vectors.example/v1"
        ),
    })
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
        assert app.page == "mcp"


@pytest.mark.asyncio
async def test_model_failure_stays_on_models_and_raises_error_toast(
    monkeypatch, tmp_path
):
    app = SetupApp(tmp_path / "config.yaml")
    app.config.inference_providers["company"] = InferenceProviderConfig(
        base_url="https://models.example/v1"
    )
    app.config.llm_model = ChatModelConfig(
        model="gpt-test",
        model_provider="openai",
        inference_provider="company",
    )
    app.config.emb_model = None
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
async def test_mcp_form_uses_configured_selector_and_shared_token(
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
        await app._save_mcp()
        await app.workers.wait_for_complete()

        assert app.pending["tools"] == "secret"
        assert app.query_one("#mcp-configured", Select).value == "tools"


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
        assert "tools" not in app.config.mcp_servers
        assert "✗" in str(
            app.query_one("#mcp-validation-status", Static).content
        )
