import asyncio
import os
import stat
from unittest.mock import MagicMock

import pytest
import yaml
from keyring.backends.fail import Keyring as FailingSystemKeyring
from keyring.errors import KeyringError
from pydantic import SecretStr
from textual.widgets import Button, Input, Static, TextArea

from tests.tui.utils import eventually
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
)
from ursa.cli.setup import (
    SetupApp,
    SetupError,
    _load_config,
    _save,
    apply_setup_defaults,
    load_setup_defaults,
)
from ursa.cli.setup_models import CUSTOM, SetupModelEditor, SetupModelSelection
from ursa.cli.tui.widgets import FuzzySelect
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
    monkeypatch.setattr("ursa.cli.setup_config.get_keyring", lambda: store)
    monkeypatch.setattr("ursa.cli.setup_entries.get_keyring", lambda: store)
    return store


@pytest.fixture
def app(tmp_path):
    return SetupApp(tmp_path / "config.yaml")


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


async def show_model_editor(app) -> SetupModelEditor:
    app._show("models")
    await app.workers.wait_for_complete()
    return app.query_one(SetupModelEditor)


async def set_custom_chat(editor, pilot, model):
    editor.query_one("#setup-chat-model-select", FuzzySelect).value = CUSTOM
    await pilot.pause()
    editor.query_one("#setup-chat-model", Input).value = model
    await pilot.pause()


async def click_and_wait(pilot, selector, condition):
    pilot.app.query_one(selector, Button).press()
    await eventually(pilot, condition, description=f"{selector} action")


async def click_page(pilot, selector, page):
    await click_and_wait(pilot, selector, lambda: pilot.app.page == page)


async def add_entry(app, pilot, kind, total):
    tabs = getattr(app, f"{kind}_tabs")
    await click_and_wait(
        pilot,
        f"#{kind}-add",
        lambda: len(tabs.editors()) == total
        and tabs.active_editor() is tabs.editors()[-1],
    )
    editor = tabs.active_editor()
    assert editor is not None and editor.raw_name() == ""
    assert yaml.safe_load(editor.query_one(TextArea).text) == {}
    return editor


def entry_message(editor):
    return str(editor.query_one(".draft-message", Static).content).lower()


def test_run_setup_uses_highest_precedence_user_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    app = SetupApp()

    assert app.path == tmp_path / "ursa/config.yaml"


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


@pytest.mark.parametrize(
    "defaults",
    [
        {
            "inference_providers": [
                {"name": "company", "access_token": "secret"}
            ]
        },
        {"llm_model": {"model": "test", "password": "secret"}},
        {"mcp_servers": {"tools": {"env": {"API_KEY": "secret"}}}},
    ],
)
def test_setup_defaults_reject_literal_secrets(defaults):
    config = UrsaConfig.model_validate({"inference_providers": {}})

    with pytest.raises(SetupError, match="must not contain API keys"):
        apply_setup_defaults(config, defaults)


@pytest.mark.parametrize("contents", ["[]", "not json"])
def test_setup_defaults_require_a_json_object(tmp_path, contents):
    path = tmp_path / "defaults.json"
    path.write_text(contents)

    with pytest.raises(SetupError, match="JSON object|not valid JSON"):
        load_setup_defaults(path)


def test_save_writes_file_fallback_then_atomic_private_config(
    monkeypatch, tmp_path
):
    path = tmp_path / "nested/config.yaml"
    config = configured_config()
    pending = {"company": "provider-secret", "tools": "mcp-secret"}
    fallback = FileKeyring(tmp_path / "keyring.json")
    store = FallbackKeyring(FailingSystemKeyring(), fallback)
    monkeypatch.setattr("ursa.cli.setup_config.get_keyring", lambda: store)

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


async def test_provider_tabs_add_empty_persistent_draft(app):
    app.config.inference_providers.clear()
    app.config.inference_providers.update({
        "public": InferenceProviderConfig(),
        "company": InferenceProviderConfig(
            base_url="https://models.example/v1",
            api_key=SecretReference(env="COMPANY_API_KEY"),
        ),
    })

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        public, company = app.provider_tabs.editors()
        assert public.raw_name() == "public"
        assert company.raw_name() == "company"
        assert str(app.query_one("#provider-add", Button).label) == "+"
        assert str(app.query_one("#provider-remove", Button).label) == "Remove"
        assert str(app.query_one("#provider-save", Button).label) == (
            "Validate and continue"
        )

        added = await add_entry(app, pilot, "provider", 3)
        assert added.query_one(".provider-url-custom", Input).value == ""
        added.query_one(".entry-name", Input).value = "draft"
        app.provider_tabs.select(public.draft_id)
        app.provider_tabs.select(added.draft_id)
        assert added.raw_name() == "draft"
        assert company.query_one(".provider-url-custom", Input).value == (
            "https://models.example/v1"
        )


async def test_provider_yaml_validation_debounces_and_stops_on_remove(
    monkeypatch, app
):
    class ControlledTimer:
        def __init__(self, callback):
            self.callback = callback
            self.stopped = False

        def stop(self):
            self.stopped = True

        def fire(self):
            assert not self.stopped
            self.callback()

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        editor = await add_entry(app, pilot, "provider", 2)
        validate = MagicMock()
        monkeypatch.setattr(editor, "_validate_yaml", validate)
        scheduled = []

        def set_timer(_delay, callback, *args, **kwargs):
            timer = ControlledTimer(callback)
            scheduled.append(timer)
            return timer

        monkeypatch.setattr(editor, "set_timer", set_timer)
        yaml_editor = editor.query_one(TextArea)
        yaml_editor.text = "base_url: https://first.example/v1\n"
        await pilot.pause()
        first = scheduled[-1]
        yaml_editor.text = "base_url: https://second.example/v1\n"
        await pilot.pause()

        assert first.stopped
        validate.assert_not_called()
        scheduled[-1].fire()
        validate.assert_called_once_with()

        yaml_editor.text = "base_url: https://third.example/v1\n"
        await pilot.pause()
        pending = scheduled[-1]
        await click_and_wait(
            pilot,
            "#provider-remove",
            lambda: pending.stopped
            and editor not in app.provider_tabs.editors(),
        )
        assert pending.stopped


async def test_provider_validation_failure_does_not_continue(monkeypatch, app):
    monkeypatch.setattr(
        "ursa.cli.setup.run_probe_with_deadline",
        MagicMock(side_effect=RuntimeError("connection refused")),
    )

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        editor = app.provider_tabs.active_editor()
        assert editor is not None
        editor.query_one(".entry-name", Input).value = "company"
        editor.query_one(
            ".provider-url-custom", Input
        ).value = "https://models.example/v1"

        await click_and_wait(
            pilot,
            "#provider-save",
            lambda: "could not connect" in entry_message(editor),
        )

        assert app.page == "provider"
        assert not app.config.inference_providers
        assert "could not connect" in entry_message(editor)
        assert app.provider_tabs.active_editor() is editor


@pytest.mark.parametrize("kind", ["provider", "mcp"])
@pytest.mark.parametrize("failing", [False, True])
async def test_entry_continue_checks_all_and_is_transactional(
    monkeypatch, app, kind, failing
):
    names = ("first", "broken", "last") if failing else ("first", "second")
    checked = []
    section = f"{kind}_servers" if kind == "mcp" else "inference_providers"
    if kind == "provider":
        for name in names:
            monkeypatch.setenv(f"{name.upper()}_KEY", "test-key")
        values = {
            name: InferenceProviderConfig(
                base_url=f"https://{name}.example/v1",
                api_key=SecretReference(env=f"{name.upper()}_KEY"),
            )
            for name in names
        }
        monkeypatch.setattr(
            "ursa.cli.setup_models.list_provider_models", lambda _config: []
        )
    else:
        values = {
            name: {
                "transport": "streamable_http",
                "url": f"https://{name}.example/mcp",
            }
            for name in names
        }
    if kind == "provider":
        app.config.inference_providers.update(values)
    else:
        app.config = UrsaConfig.model_validate({section: values})

    def probe(_probe, entry, *_args, **_kwargs):
        name = (
            entry
            if kind == "mcp"
            else str(entry.base_url).split("//", 1)[1].split(".", 1)[0]
        )
        checked.append(name)
        if failing and name == {"provider": "broken", "mcp": "renamed"}[kind]:
            raise RuntimeError("connection refused")

    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", probe)

    async with app.run_test(size=(120, 42)) as pilot:
        if kind == "provider":
            await pilot.click("#welcome-next")
        else:
            app._begin_entry(kind, names[0])
            await pilot.pause()
        tabs = getattr(app, f"{kind}_tabs")
        assert str(app.query_one(f"#{kind}-save", Button).label) == (
            "Validate and continue"
        )
        if failing:
            broken = tabs.editors()[1]
            broken.query_one(".entry-name", Input).value = "renamed"

        expected_page = {"provider": "models", "mcp": "review"}[kind]
        if failing:
            await click_and_wait(
                pilot,
                f"#{kind}-save",
                lambda: tabs.active_editor() is broken,
            )
        else:
            await click_page(pilot, f"#{kind}-save", expected_page)

        expected = (
            {"first", "renamed", "last"}
            if kind == "mcp" and failing
            else set(names)
        )
        assert set(checked) == expected
        assert app.page == (kind if failing else expected_page)
        if failing:
            entries = getattr(app.config, section)
            assert set(entries) == set(names)
            assert tabs.active_editor() is broken
            assert "could not connect" in entry_message(broken)


async def test_provider_remove_retargets_models_only_after_commit(
    monkeypatch, app
):
    app.config.inference_providers.clear()
    app.config.inference_providers.update({
        "old": InferenceProviderConfig(),
        "remaining": InferenceProviderConfig(),
    })
    app.config.llm_model = model_config(ChatModelConfig, "chat", "old")
    app.config.emb_model = model_config(EmbModelConfig, "embedding", "old")
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", MagicMock())

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        await click_and_wait(
            pilot,
            "#provider-remove",
            lambda: len(app.provider_tabs.editors()) == 1
            and app.provider_tabs.active_editor()
            is app.provider_tabs.editors()[0]
            and app.query_one("#provider-remove", Button).disabled,
        )

        assert app.config.llm_model.inference_provider == "old"
        assert app.config.emb_model.inference_provider == "old"
        assert await app._validate_entry_page("provider")
        assert app.config.llm_model.inference_provider == "remaining"
        assert app.config.emb_model.inference_provider == "remaining"
        assert app.query_one("#provider-remove", Button).disabled
        assert not app.query_one("#provider-add", Button).disabled


async def test_setup_model_editor_validates_both_models(monkeypatch, app):
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
        editor = await show_model_editor(app)
        assert editor.parent.id == "setup-model-host"
        selection = SetupModelSelection(
            model_config(ChatModelConfig, "chat-model", "chat"),
            model_config(EmbModelConfig, "embedding-model", "vectors"),
        )
        editor.post_message(editor.Submitted(selection))
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert checked == [
            ("chat-model", "chat"),
            ("embedding-model", "vectors"),
        ]
        assert app.page == "mcp"


async def test_setup_model_editor_shows_full_config_and_null_embedding(app):
    async with app.run_test(size=(120, 42)):
        editor = await show_model_editor(app)
        chat = yaml.safe_load(
            editor.query_one("#setup-chat-config-yaml", TextArea).text
        )
        embedding = yaml.safe_load(
            editor.query_one("#setup-embedding-config-yaml", TextArea).text
        )

        assert set(chat) == {
            "model",
            "model_provider",
            "base_url",
            "api_key",
            "inference_provider",
            "ssl_verify",
            "max_completion_tokens",
        }
        assert embedding is None


async def test_setup_model_editor_submits_absent_embedding_as_none(
    monkeypatch, app
):
    checked = []
    monkeypatch.setattr(
        "ursa.cli.setup._probe_model",
        lambda model, _: checked.append(model.model),
    )

    async with app.run_test(size=(120, 42)) as pilot:
        editor = await show_model_editor(app)
        await set_custom_chat(editor, pilot, "chat-model")
        await click_page(pilot, "#models-save", "mcp")

        assert checked == ["chat-model"]
        assert app.config.emb_model is None


async def test_setup_model_named_provider_inherits_credentials(
    monkeypatch, app
):
    app.config.inference_providers.clear()
    app.config.inference_providers["company"] = InferenceProviderConfig(
        base_url="https://models.example/v1",
        api_key=SecretStr("provider-secret"),
        model_provider="anthropic",
    )
    app.llm_configured = False
    probe = MagicMock()
    monkeypatch.setattr(
        "ursa.cli.setup.run_probe_with_deadline",
        lambda callback, *args, **_kwargs: callback(*args),
    )
    monkeypatch.setattr("ursa.cli.setup.probe_chat_config", probe)

    async with app.run_test(size=(120, 42)) as pilot:
        editor = await show_model_editor(app)
        await set_custom_chat(editor, pilot, "claude-test")
        await click_page(pilot, "#models-save", "mcp")

        probe.assert_called_once()
        resolved, kind = probe.call_args.args
        assert kind == "chat"
        assert resolved.base_url == "https://models.example/v1"
        assert resolved.model_provider == "anthropic"
        assert isinstance(resolved.api_key, SecretStr)
        assert resolved.api_key.get_secret_value() == "provider-secret"
        assert {
            "api_key",
            "base_url",
            "model_provider",
        }.isdisjoint(app.config.llm_model.model_fields_set)
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


async def test_mcp_tabs_add_empty_draft_and_advanced_is_complete(app):
    app.config = configured_config()

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("mcp", "tools")
        await pilot.pause()
        existing = app.mcp_tabs.active_editor()
        assert existing is not None
        values = yaml.safe_load(existing.query_one(TextArea).text)
        assert values["transport"] == "streamable_http"
        assert values["url"] == "https://tools.example/mcp"
        assert values["headers"]["Authorization"]["keyring"] is True
        assert "timeout" in values

        await add_entry(app, pilot, "mcp", 2)
        await click_and_wait(
            pilot,
            "#mcp-remove",
            lambda: app.mcp_tabs.active_editor() is existing,
        )
        assert app.mcp_tabs.active_editor() is existing


async def test_mcp_local_schema_validation_precedes_every_probe(
    monkeypatch, app
):
    probe = MagicMock()
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", probe)

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("mcp")
        await pilot.pause()
        editor = await add_entry(app, pilot, "mcp", 1)
        editor.query_one(".entry-name", Input).value = "broken"

        assert not await app._validate_entry_page("mcp")
        probe.assert_not_called()
        assert not app.config.mcp_servers
        assert app.mcp_tabs.active_editor() is editor


async def test_mcp_validation_uses_canonical_full_server_config(
    monkeypatch, app
):
    monkeypatch.setenv("MCP_TOKEN", "runtime-secret")
    app.config = configured_config()
    app.config.mcp_servers = {
        "tools": {
            "transport": "streamable_http",
            "url": "https://tools.example/mcp",
            "headers": {
                "Authorization": {
                    "env": "MCP_TOKEN",
                    "template": "Token <%s>",
                },
                "X-Mode": "advanced",
            },
            "timeout": 17,
            "sse_read_timeout": 91,
            "terminate_on_close": False,
        }
    }
    captured = {}

    class DummyClient:
        def __init__(self, connections):
            captured.update(connections)

        async def get_tools(self, *, server_name):
            assert server_name == "tools"
            return [object()]

    def run_inline(callback, *args, **_kwargs):
        return asyncio.run(callback(*args))

    monkeypatch.setattr("ursa.util.mcp.UrsaMCPClient", DummyClient)
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", run_inline)

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("mcp", "tools")
        await pilot.pause()

        assert await app._validate_entry_page("mcp")

    connection = captured["tools"]
    assert connection["headers"] == {
        "Authorization": "Token <runtime-secret>",
        "X-Mode": "advanced",
    }
    assert connection["timeout"].total_seconds() == 17
    assert connection["sse_read_timeout"].total_seconds() == 91
    assert connection["terminate_on_close"] is False
    assert app.config.mcp_servers["tools"].headers["Authorization"] == (
        SecretTemplate(env="MCP_TOKEN", template="Token <%s>")
    )


async def test_mcp_continue_allows_no_servers(monkeypatch, app):
    probe = MagicMock()
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", probe)

    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("mcp")
        await pilot.pause()
        assert not app.mcp_tabs.editors()
        await click_page(pilot, "#mcp-save", "review")

        probe.assert_not_called()
        assert app.page == "review"
