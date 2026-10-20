import os
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest
from keyring.backends.fail import Keyring as FailingSystemKeyring
from keyring.errors import KeyringError
from textual.widgets import Button, Input, Select, Static, TextArea

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
    apply_setup_defaults,
    load_setup_defaults,
)
from ursa.cli.tui.widgets import ModelEditor, ModelSelection
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
    store.health = MagicMock(return_value=())
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)
    monkeypatch.setattr("ursa.cli.setup.check_config_health", store.health)
    return store


@pytest.fixture
def app(tmp_path):
    return SetupApp(tmp_path / "config.yaml")


@pytest.fixture
def saved_keyring(monkeypatch, tmp_path):
    store = FileKeyring(tmp_path / "keyring.json")
    store.set_password("ursa", "company", "old-secret")
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)
    monkeypatch.setattr("ursa.util.secrets.get_keyring", lambda: store)
    return store


def begin_mcp(app, name: str, **fields: str) -> None:
    app._begin_mcp()
    for field, value in {"name": name, **fields}.items():
        app.query_one(f"#mcp-{field}").value = value


def model_config(kind, name: str, provider: str):
    return kind(
        model=name, model_provider="openai", inference_provider=provider
    )


def configured_config() -> UrsaConfig:
    return UrsaConfig.model_validate({
        "inference_providers": {"company": {"api_key": {"keyring": True}}},
        "llm_model": model_config(ChatModelConfig, "gpt-test", "company"),
        "mcp_servers": {
            "tools": {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
                "headers": {"Authorization": {"keyring": True}},
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


def test_setup_applies_non_secret_installer_defaults(tmp_path):
    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text(
        """
{
  "inference_providers": [
    {
      "name": "anthropic",
      "base_url": "https://api.anthropic.test",
      "reasoning_effort": "high",
      "api_key": {"keyring": "shared"}
    }
  ],
  "llm_model": {
    "model": "claude-test",
    "model_provider": "anthropic",
    "inference_provider": "anthropic"
  },
  "emb_model": {
    "model": "embedding-test",
    "model_provider": "openai",
    "inference_provider": "anthropic"
  },
  "mcp_servers": {
    "tools": {
      "transport": "streamable_http",
      "url": "https://tools.example/mcp",
      "headers": {"Authorization": {"template": "Bearer %s"}},
      "timeout": 30
    }
  }
}
"""
    )

    app = SetupApp(tmp_path / "config.yaml", defaults_path)

    assert app.config.inference_providers["anthropic"] == (
        InferenceProviderConfig(
            base_url="https://api.anthropic.test",
            api_key=SecretReference(keyring="shared"),
            reasoning_effort="high",
        )
    )
    assert app.config.llm_model.model == "claude-test"
    assert app.config.emb_model.model == "embedding-test"
    assert app.llm_configured
    server = app.config.mcp_servers["tools"]
    assert server.url == "https://tools.example/mcp"
    assert server.headers == {
        "Authorization": SecretTemplate(keyring=True, template="Bearer %s")
    }
    assert server.timeout.total_seconds() == 30


def test_setup_defaults_do_not_replace_existing_entries(tmp_path):
    config = configured_config()

    apply_setup_defaults(
        config,
        {
            "inference_providers": [
                {
                    "name": "company",
                    "base_url": "https://replacement.test/v1",
                }
            ],
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://replacement.test/mcp",
                }
            },
        },
    )

    assert config.inference_providers["company"].base_url is None
    assert config.mcp_servers["tools"].url == "https://tools.example/mcp"


@pytest.mark.parametrize(
    "defaults",
    [
        {
            "inference_providers": [
                {
                    "name": "company",
                    "base_url": "https://models.example/v1",
                    "access_token": "secret",
                }
            ]
        },
        {
            "llm_model": {
                "model": "test",
                "inference_provider": "company",
                "password": "secret",
            }
        },
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "tool",
                    "env": {"API_KEY": "secret"},
                }
            }
        },
    ],
)
def test_setup_defaults_reject_literal_secrets(defaults):
    config = UrsaConfig.model_validate({"inference_providers": {}})

    with pytest.raises(SetupError, match="must not contain API keys"):
        apply_setup_defaults(config, defaults)


def test_setup_defaults_allow_environment_placeholders():
    config = UrsaConfig.model_validate({"inference_providers": {}})

    apply_setup_defaults(
        config,
        {
            "mcp_servers": {
                "tools": {
                    "transport": "stdio",
                    "command": "tool",
                    "env": {"API_KEY": "${TOOLS_API_KEY}"},
                }
            }
        },
    )

    assert config.mcp_servers["tools"].env == {"API_KEY": "${TOOLS_API_KEY}"}


@pytest.mark.parametrize("contents", ["[]", "not json"])
def test_setup_defaults_require_a_json_object(tmp_path, contents):
    path = tmp_path / "defaults.json"
    path.write_text(contents)

    with pytest.raises(SetupError, match="JSON object|not valid JSON"):
        load_setup_defaults(path)


def test_load_preserves_mcp_environment_placeholders(
    monkeypatch, setup_keyring, tmp_path
):
    monkeypatch.setenv("MCP_SECRET", "resolved-secret")
    monkeypatch.setenv("MODEL_URL", "https://models.example/v1")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
inference_providers:
  company: {base_url: "${MODEL_URL}"}
llm_model:
  model: gpt-test
  model_provider: openai
  inference_provider: company
mcp_servers:
  tools:
    transport: stdio
    command: tool
    env: {API_KEY: "${MCP_SECRET}"}
"""
    )

    config, _ = _load_config(path)

    assert config.mcp_servers["tools"].env == {"API_KEY": "${MCP_SECRET}"}
    _save(path, config, {})

    runtime = setup_keyring.health.call_args.args[0]
    assert set(runtime.inference_providers) == {"company"}
    assert (
        runtime.inference_providers["company"].base_url
        == "https://models.example/v1"
    )
    assert runtime.mcp_servers["tools"].env == {"API_KEY": "resolved-secret"}
    saved = path.read_text()
    assert "${MODEL_URL}" in saved
    assert "${MCP_SECRET}" in saved


def test_save_writes_file_fallback_then_atomic_private_config(
    monkeypatch, tmp_path
):
    path = tmp_path / "nested/config.yaml"
    config = configured_config()
    pending = {"company": "provider-secret", "tools": "mcp-secret"}
    fallback = FileKeyring(tmp_path / "keyring.json")
    store = FallbackKeyring(FailingSystemKeyring(), fallback)
    monkeypatch.setattr("ursa.cli.setup.get_keyring", lambda: store)

    assert _save(path, config, pending) == path

    assert fallback.get_password("ursa", "company") == "provider-secret"
    assert fallback.get_password("ursa", "tools") == "mcp-secret"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.with_suffix(".yaml.lock").exists()


def test_save_does_not_write_config_when_keyring_fails(setup_keyring, tmp_path):
    path = tmp_path / "config.yaml"
    setup_keyring.set_password.side_effect = KeyringError("write failed")

    with pytest.raises(KeyringError, match="write failed"):
        _save(path, configured_config(), {"company": "secret"})

    assert not path.exists()


@pytest.mark.parametrize("failure", ["config-write", "health-check"])
def test_save_restores_staged_keys(
    monkeypatch, saved_keyring, tmp_path, failure
):
    path = tmp_path / "config.yaml"
    if failure == "config-write":
        error, message = OSError, "disk full"
        monkeypatch.setattr(
            "ursa.cli.setup.save_config_file",
            MagicMock(side_effect=OSError(message)),
        )
    else:
        error = SetupError
        message = "Configuration checks failed: mcp_servers.tools"
        path.write_text("existing: config\n")

        def fail(candidate):
            resolved = candidate.resolve()
            api_key = resolved.inference_providers["company"].api_key
            assert api_key.get_secret_value() == "new-secret"
            return (("mcp_servers", "tools"),)

        monkeypatch.setattr("ursa.cli.setup.check_config_health", fail)

    with pytest.raises(error, match=message) as raised:
        _save(
            path,
            configured_config(),
            {"company": "new-secret", "tools": "mcp-secret"},
        )

    if failure == "health-check":
        assert str(raised.value) == message
        assert path.read_text() == "existing: config\n"
    assert saved_keyring.get_password("ursa", "company") == "old-secret"
    assert saved_keyring.get_password("ursa", "tools") is None


def test_pending_shared_secret_is_kept_until_last_reference_is_removed(app):
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


async def test_write_page_saves_and_shows_ready(monkeypatch, app):
    path = app.path

    async def inline(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    monkeypatch.setattr("ursa.cli.setup._save", lambda *_args: path)
    monkeypatch.setattr("ursa.cli.setup.asyncio.to_thread", inline)

    async with app.run_test(size=(120, 42)):
        app._show("review")
        await app._save_config()
        assert app.page == "done"
        assert (
            str(app.query_one("#done-copy", Static).content)
            == f"Saved to {path}"
        )


async def test_provider_is_checked_before_becoming_configured(monkeypatch, app):
    list_models = MagicMock(return_value=[ProviderModel("gpt-test", "openai")])

    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models", list_models
    )

    def run_probe(probe, *args, **_kwargs):
        return probe(*args)

    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", run_probe)

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
        entry.configure("openai", None, [])

        async def run_inline(callback, *args, **kwargs):
            if callback is run_probe:
                assert (
                    str(app.query_one("#provider-error", Static).content)
                    == "? Validating openai"
                )
            return callback(*args, **kwargs)

        monkeypatch.setattr("ursa.cli.setup.asyncio.to_thread", run_inline)
        app.query_one("#provider-key-key", Input).value = "secret"
        await pilot.click("#provider-save")
        await app.workers.wait_for_complete()

        assert app.query_one("#provider-configured", Select).value == "openai"
        assert not app.query_one("#provider-next", Button).disabled


async def test_models_use_existing_editor_and_validate_both_providers(
    monkeypatch, app
):
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
            model_config(ChatModelConfig, "chat-model", "chat"),
            model_config(EmbModelConfig, "embedding-model", "vectors"),
        )
        editor._submit(selection)
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert checked == [
            ("chat-model", "chat"),
            ("embedding-model", "vectors"),
        ]
        assert app.page == "mcp"


async def test_model_failure_stays_on_models_and_raises_error_toast(
    monkeypatch, app
):
    app.config = configured_config()
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


async def test_mcp_advanced_and_failure_paths(monkeypatch, app):
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock())

    async with app.run_test(size=(120, 42)):
        begin_mcp(
            app,
            "local_tools",
            transport="stdio",
            command="uv",
            args="run tool.py",
        )
        app.query_one("#mcp-config-yaml", TextArea).text = """\
env:
  TOOL_MODE: guided
cwd: /srv/tools
"""

        await app._save_mcp()

        saved = app.config.mcp_servers["local_tools"]
        assert saved.command == "uv"
        assert saved.args == ["run", "tool.py"]
        assert saved.env == {"TOOL_MODE": "guided"}
        assert str(saved.cwd) == "/srv/tools"

        begin_mcp(app, "remote_tools", url="https://tools.example/mcp")
        app.query_one("#mcp-config-yaml", TextArea).text = """\
headers:
  X-Tenant: guided
timeout: 12
sse_read_timeout: 45
terminate_on_close: false
"""
        token = app.query_one("#mcp-token-source", Select)
        token.set_options([
            ("None", MCP_TOKEN_NONE),
            ("remote_tools", "remote_tools"),
        ])
        token.value = "remote_tools"
        app.query_one("#mcp-token-key", Input).value = "secret"

        await app._save_mcp()

        saved = app.config.mcp_servers["remote_tools"]
        assert saved.headers["X-Tenant"] == "guided"
        auth = saved.headers["Authorization"]
        assert isinstance(auth, SecretTemplate)
        assert auth.keyring_username("remote_tools") == "remote_tools"
        assert auth.template == "Bearer %s"
        assert saved.timeout.total_seconds() == 12
        assert saved.sse_read_timeout.total_seconds() == 45
        assert saved.terminate_on_close is False
        assert app.pending == {"remote_tools": "secret"}
        assert app.query_one("#mcp-configured", Select).value == "remote_tools"

        probe = AsyncMock()
        monkeypatch.setattr(app, "_probe_mcp", probe)
        begin_mcp(app, "broken")
        editor = app.query_one("#mcp-config-yaml", TextArea)
        editor.text = "url: https://wrong.example/mcp\n"
        await app._save_mcp()
        assert editor.has_class("yaml-invalid")
        assert "broken" not in app.config.mcp_servers
        probe.assert_not_awaited()

        monkeypatch.setattr(
            app,
            "_probe_mcp",
            AsyncMock(side_effect=RuntimeError("401 Unauthorized")),
        )
        begin_mcp(app, "tools", url="https://tools.example/mcp")
        await app._save_mcp()
        assert app.page == "mcp"
        assert "tools" not in app.config.mcp_servers
        assert "✗" in str(
            app.query_one("#mcp-validation-status", Static).content
        )
