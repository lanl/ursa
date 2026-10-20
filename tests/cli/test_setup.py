import asyncio
import os
import stat
from unittest.mock import MagicMock

import pytest
import yaml
from keyring.backends.fail import Keyring as FailingSystemKeyring
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
    _save,
    apply_setup_defaults,
)
from ursa.cli.setup_models import SetupModelEditor, SetupModelSelection
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
    return UrsaConfig.model_validate(
        yaml.safe_load("""
inference_providers: {company: {api_key: {keyring: true}}}
llm_model: {model: gpt-test, model_provider: openai, inference_provider: company}
mcp_servers:
  tools: {transport: streamable_http, url: https://tools.example/mcp,
          headers: {Authorization: {keyring: true}}}
""")
    )


async def show_model_editor(app) -> SetupModelEditor:
    app._show("models")
    await app.workers.wait_for_complete()
    return app.query_one(SetupModelEditor)


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
    assert editor_yaml(editor) == {}
    return editor


async def begin_entry(app, pilot, kind, name=None):
    app._begin_entry(kind, name)
    await pilot.pause()


def query(widget, selector, expect_type):
    return widget.query_one(selector, expect_type)


def editor_yaml(editor, selector=None):
    return yaml.safe_load(editor.query_one(selector or TextArea).text)


def patch_setup(monkeypatch, name, value):
    monkeypatch.setattr(f"ursa.cli.setup.{name}", value)


def entry_message(editor):
    return str(query(editor, ".draft-message", Static).content).lower()


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
    monkeypatch.setattr(
        "ursa.cli.setup_config.save_config_file",
        MagicMock(side_effect=OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        _save(path, config, {"company": "replacement"})
    assert fallback.get_password("ursa", "company") == "provider-secret"


async def test_provider_tabs_add_empty_persistent_draft(app):
    app.config.inference_providers.clear()
    app.config.inference_providers["public"] = InferenceProviderConfig()

    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.click("#welcome-next")
        (public,) = app.provider_tabs.editors()
        added = await add_entry(app, pilot, "provider", 2)
        assert query(added, ".provider-url-custom", Input).value == ""
        query(added, ".entry-name", Input).value = "draft"
        app.provider_tabs.select(public.draft_id)
        app.provider_tabs.select(added.draft_id)
        assert added.raw_name() == "draft"


@pytest.mark.parametrize("kind", ["provider", "mcp"])
async def test_entry_continue_checks_all_and_is_transactional(
    monkeypatch, app, kind
):
    names = ("first", "broken", "last")
    checked = []
    provider = kind == "provider"
    section = f"{kind}_servers" if kind == "mcp" else "inference_providers"
    if provider:
        for name in names:
            monkeypatch.setenv(f"{name.upper()}_KEY", "test-key")
        monkeypatch.setattr(
            "ursa.cli.setup_models.list_provider_models", lambda _config: []
        )
    values = {
        name: (
            InferenceProviderConfig(
                base_url=f"https://{name}.example/v1",
                api_key=SecretReference(env=f"{name.upper()}_KEY"),
            )
            if provider
            else {
                "transport": "streamable_http",
                "url": f"https://{name}.example/mcp",
            }
        )
        for name in names
    }
    if provider:
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
        if name == {"provider": "broken", "mcp": "renamed"}[kind]:
            raise RuntimeError("connection refused")

    patch_setup(monkeypatch, "run_probe_with_deadline", probe)
    async with app.run_test(size=(120, 42)) as pilot:
        if kind == "provider":
            await pilot.click("#welcome-next")
        else:
            await begin_entry(app, pilot, kind, names[0])
        tabs = getattr(app, f"{kind}_tabs")
        assert str(query(app, f"#{kind}-save", Button).label) == (
            "Validate and continue"
        )
        broken = tabs.editors()[1]
        query(broken, ".entry-name", Input).value = "renamed"
        await click_and_wait(
            pilot,
            f"#{kind}-save",
            lambda: tabs.active_editor() is broken,
        )
        expected = {"first", "renamed", "last"} if kind == "mcp" else set(names)
        assert set(checked) == expected
        assert app.page == kind
        assert set(getattr(app.config, section)) == set(names)
        assert tabs.active_editor() is broken
        assert "could not connect" in entry_message(broken)


async def test_setup_model_editor_validates_both_models(monkeypatch, app):
    app.config.inference_providers.update({
        "chat": InferenceProviderConfig(
            base_url="https://chat.example/v1",
            api_key=SecretReference(keyring=True),
        ),
        "vectors": InferenceProviderConfig(
            base_url="https://vectors.example/v1"
        ),
    })
    app.pending["chat"] = "pending-secret"
    checked = []

    def probe(model, providers):
        checked.append((
            model.model,
            model.inference_provider,
            providers["chat"].api_key.get_secret_value(),
        ))

    patch_setup(monkeypatch, "_probe_model", probe)
    async with app.run_test(size=(120, 42)) as pilot:
        editor = await show_model_editor(app)
        selection = SetupModelSelection(
            model_config(ChatModelConfig, "chat-model", "chat"),
            model_config(EmbModelConfig, "embedding-model", "vectors"),
        )
        editor.post_message(editor.Submitted(selection))
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert checked == [
            ("chat-model", "chat", "pending-secret"),
            ("embedding-model", "vectors", "pending-secret"),
        ]
        assert app.page == "mcp"


async def test_mcp_validation_uses_canonical_full_server_config(
    monkeypatch, app
):
    monkeypatch.setenv("MCP_TOKEN", "runtime-secret")
    app.config = configured_config()
    app.config.mcp_servers = {
        "tools": yaml.safe_load("""
transport: streamable_http
url: https://tools.example/mcp
headers: {Authorization: {env: MCP_TOKEN, template: 'Token <%s>'}, X-Mode: advanced}
timeout: 17
sse_read_timeout: 91
terminate_on_close: false
""")
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
    patch_setup(monkeypatch, "run_probe_with_deadline", run_inline)
    async with app.run_test(size=(120, 42)) as pilot:
        await begin_entry(app, pilot, "mcp", "tools")
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
