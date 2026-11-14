from threading import Event
from unittest.mock import MagicMock

import pytest
import yaml
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Select,
    Static,
    TextArea,
)

from tests.tui.utils import await_dom, await_event, await_workers, eventually
from ursa.cli.config import InferenceProviderConfig, UrsaConfig
from ursa.cli.doctor import CheckResult, CheckStatus
from ursa.cli.setup import config as setup_config
from ursa.cli.setup.config import apply_setup_defaults
from ursa.cli.setup.entries import API_KEY_ENV, MCP_TOKEN_NONE
from ursa.cli.setup.models import SetupModelEditor, SetupModelSelection
from ursa.cli.setup.setup import SetupApp
from ursa.cli.tui.widgets import FuzzySelect, HelpLabel
from ursa.util.inference_providers import ProviderModel
from ursa.util.secrets import SecretReference


@pytest.fixture(autouse=True)
def setup_keyring(monkeypatch):
    store = MagicMock()
    store.get_password.return_value = None
    for module in ("config", "entries"):
        monkeypatch.setattr(
            f"ursa.cli.setup.{module}.get_keyring", lambda: store
        )
    return store


@pytest.fixture
def app(tmp_path):
    return SetupApp(tmp_path / "config.yaml")


def configured_config():
    return UrsaConfig.model_validate(
        yaml.safe_load("""
inference_providers: {company: {api_key: {keyring: true}}}
llm_model: {model: gpt-test, model_provider: openai, inference_provider: company}
mcp_servers:
  tools: {transport: streamable_http, url: https://tools.example/mcp,
          headers: {Authorization: {keyring: true}}}
""")
    )


def query(widget, selector, kind):
    return widget.query_one(selector, kind)


def test_setup_defaults_preserve_model_extras():
    config = UrsaConfig.model_validate({"inference_providers": {}})
    assert apply_setup_defaults(
        config,
        yaml.safe_load("""
inference_providers: [{name: company, base_url: https://models.example/v1}]
llm_model: {model: chat, inference_provider: company, template: hello}
"""),
    )
    values = config.llm_model.editable_values()
    assert values["template"] == "hello" and "keyring" not in values


def test_config_boundary(monkeypatch, tmp_path, setup_keyring):
    path = tmp_path / "config.yaml"
    path.write_text(
        'inference_providers: {company: {base_url: "${MODEL_URL}", api_key: {keyring: true}}}\n'
        "llm_model: {model: chat, inference_provider: company}\n"
        'mcp_servers: {tools: {transport: stdio, command: tool, env: {KEY: "${MCP_SECRET}"}}}\n'
    )
    config, _ = setup_config.load_setup_config(path)
    with pytest.raises(
        setup_config.SetupError, match="must not contain API keys"
    ):
        apply_setup_defaults(
            config, {"mcp_servers": {"bad": {"env": {"API_KEY": "secret"}}}}
        )
    setup_config.save_setup_config(path, config, {})
    assert all(
        value in path.read_text() for value in ("${MODEL_URL}", "${MCP_SECRET}")
    )
    setup_keyring.get_password.side_effect = ["old", None]
    monkeypatch.setattr(
        setup_config,
        "save_config_file",
        MagicMock(side_effect=OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        setup_config.save_setup_config(
            path, config, {"company": "new", "tools": "new"}
        )
    assert setup_keyring.set_password.call_args.args == (
        "ursa",
        "company",
        "old",
    )
    setup_keyring.delete_password.assert_called_once_with("ursa", "tools")


@pytest.mark.parametrize("base_url", [None, "http://localhost:4000"])
@pytest.mark.parametrize(
    ("reference", "source"),
    [
        (None, "company"),
        (SecretReference(keyring=True), "company"),
        (SecretReference(keyring="shared"), "shared"),
        (SecretReference(env="SETUP_TEST_KEY"), API_KEY_ENV),
    ],
)
async def test_provider_credential_selection(
    app, monkeypatch, setup_keyring, base_url, reference, source
):
    monkeypatch.setenv("SETUP_TEST_KEY", "env-secret")
    setup_keyring.get_password.side_effect = lambda _, name: f"{name}-secret"
    app.config.inference_providers = {
        "company": InferenceProviderConfig(base_url=base_url, api_key=reference)
    }
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("provider", "company")
        await await_workers(app, pilot)
        editor = app.provider_tabs.active_editor()
        credential = editor.credential
        select = credential.query_one(Select)
        key = credential.query_one(f"#{credential.prefix}-key", Input)
        assert select.value == source
        assert key.value == (
            "SETUP_TEST_KEY" if source == API_KEY_ENV else f"{source}-secret"
        )
        expected = reference or SecretReference(keyring=True)
        assert credential.values().reference == expected
        assert credential.values().pending is None
        assert yaml.safe_load(editor.query_one(TextArea).text)["api_key"] == (
            expected.model_dump(exclude_none=True)
        )

        editor.query_one(".entry-name", Input).value = "renamed"
        await eventually(
            pilot,
            lambda: credential.entry_name == "renamed",
            description="the credential to follow the renamed provider",
        )
        await await_workers(app, pilot)
        expected_source = "renamed" if source == "company" else source
        assert select.value == expected_source
        assert key.value == (
            "SETUP_TEST_KEY"
            if source == API_KEY_ENV
            else f"{expected_source}-secret"
        )
        assert credential.values().reference == expected
        assert credential.values().pending is None


async def test_new_provider_selects_matching_keyring(app, setup_keyring):
    setup_keyring.get_password.return_value = "stored-secret"
    app.config.inference_providers.clear()
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("provider")
        editor = app.provider_tabs.active_editor()
        credential = editor.credential
        select = credential.query_one(Select)
        name = editor.query_one(".entry-name", Input)
        name.value = "company"
        await eventually(
            pilot,
            lambda: select.value == "company",
            description="the matching keyring entry to be selected",
        )
        await await_workers(app, pilot)
        assert credential.query_one(Input).value == "stored-secret"
        assert credential.values().reference == SecretReference(keyring=True)

        select.value = MCP_TOKEN_NONE
        await eventually(
            pilot,
            lambda: not credential.query_one(".credential-value").display,
            description="the credential fields to hide for None",
        )
        name.value = "no-auth"
        await eventually(
            pilot,
            lambda: credential.entry_name == "no-auth",
            description="the provider rename to finish with None selected",
        )
        assert select.value == MCP_TOKEN_NONE
        assert credential.values().reference is None


@pytest.mark.parametrize(
    ("value", "display"),
    [
        (None, "ENV Variable is UNSET"),
        ("", "ENV Variable is EMPTY"),
        (" [red]test-secret[/red] ", " [red]test-secret[/red] "),
    ],
)
async def test_env_credential_preview(
    app, monkeypatch, setup_keyring, value, display
):
    env_name = "SETUP_TEST_KEY"
    if value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, value)
    monkeypatch.setenv("SETUP_OTHER_KEY", "other-secret")
    app.config.inference_providers = {
        "company": InferenceProviderConfig(
            api_key=SecretReference(env=env_name)
        )
    }
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("provider", "company")
        await await_workers(app, pilot)
        editor = app.provider_tabs.active_editor()
        credential = editor.credential
        name = credential.query_one(f"#{credential.prefix}-key", Input)
        contents = credential.query_one(".credential-env-contents", Input)
        show = credential.query_one(Checkbox)
        assert contents.display
        assert contents.disabled
        assert contents.value == display
        assert contents.password is bool(value)
        assert contents.has_class("unset") is (value is None)
        assert not name.password
        assert show.display
        assert show.disabled is (not value)
        help_label = credential.query_one(".credential-env-label", HelpLabel)
        assert "Read-only" in str(help_label.tooltip)
        assert "cannot be edited" in str(help_label.tooltip)
        assert credential.values().reference == SecretReference(env=env_name)
        assert credential.values().pending is None

        if value:
            assert value not in app.export_screenshot()
            await pilot.click(show)
            await eventually(
                pilot,
                lambda: not contents.password,
                description="Show to reveal the ENV contents",
            )
            assert contents.value == value
            assert not name.password
            await pilot.click(contents)
            await pilot.press("x", "backspace")
            assert contents.value == value
            await pilot.click(show)
            await eventually(
                pilot,
                lambda: contents.password,
                description="the ENV contents to be masked again",
            )

        name.value = "SETUP_OTHER_KEY"
        await eventually(
            pilot,
            lambda: contents.value == "other-secret",
            description="the ENV preview to follow the variable name",
        )
        assert contents.password
        assert not contents.has_class("unset")
        assert not show.disabled
        assert credential.values().reference == SecretReference(
            env="SETUP_OTHER_KEY"
        )
        await eventually(
            pilot,
            lambda: yaml.safe_load(editor.query_one(TextArea).text)["api_key"]
            == {"env": "SETUP_OTHER_KEY"},
            description="the YAML to reference the new environment variable",
        )
        raw = editor.query_one(TextArea).text
        assert "other-secret" not in raw
        setup_keyring.set_password.assert_not_called()

        source = credential.query_one(Select)
        source.value = "company"
        await eventually(
            pilot,
            lambda: not contents.display,
            description="the ENV preview to hide for a keyring entry",
        )
        await await_workers(app, pilot)
        assert contents.value == ""
        assert not help_label.display
        assert name.password
        assert not name.disabled
        assert show.display
        source.value = MCP_TOKEN_NONE
        await eventually(
            pilot,
            lambda: not credential.query_one(".credential-value").display,
            description="the credential fields to hide for None",
        )


@pytest.mark.parametrize("kind", ["provider", "mcp"])
async def test_page_validation_checks_all_transactionally(
    monkeypatch, app, kind
):
    names = ("first", "broken")
    if kind == "provider":
        for name in names:
            monkeypatch.setenv(f"{name.upper()}_KEY", "test-key")
        providers = {
            name: InferenceProviderConfig(
                base_url=f"https://{name}.example/v1",
                api_key=SecretReference(env=f"{name.upper()}_KEY"),
            )
            for name in names
        }
        app.config.inference_providers.clear()
        app.config.inference_providers.update(providers)
        monkeypatch.setattr(
            "ursa.cli.setup.models.list_provider_models", lambda _: []
        )
    else:
        monkeypatch.setenv("MCP_TOKEN", "runtime-secret")
        servers = {
            name: {
                "transport": "streamable_http",
                "url": f"https://{name}.example/mcp",
            }
            for name in names
        }
        servers["first"].update({
            "headers": {
                "Authorization": {
                    "env": "MCP_TOKEN",
                    "template": "Token <%s>",
                }
            },
            "timeout": 17,
        })
        app.config = UrsaConfig.model_validate({
            "mcp_servers": servers,
        })
    checked = []

    def probe(name, entry, **kwargs):
        checked.append(name)
        if kind == "mcp" and name == "first":
            runtime = entry
            assert (
                runtime.headers["Authorization"].get_secret_value("first")
                == "Token <runtime-secret>"
            )
            assert runtime.timeout.total_seconds() == 17
            assert kwargs["timeout"] == 15
        if kind == "provider":
            assert kwargs["require_models"] is True
        if name == "broken":
            return CheckResult(name, CheckStatus.ERROR, "connection refused")
        return CheckResult(name, CheckStatus.OK, "Connected")

    monkeypatch.setattr(
        f"ursa.cli.setup.setup.check_{'inference_provider' if kind == 'provider' else 'mcp_server'}",
        probe,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry(kind, names[0])
        await await_workers(app, pilot)
        if kind == "provider":
            app.query_one("#provider-add", Button).press()
            await eventually(
                pilot,
                lambda: (
                    app.provider_tabs.active_editor() is not None
                    and app.provider_tabs.active_editor().raw_name() == ""
                ),
                description="the new provider editor to become active",
            )
            app.query_one("#provider-remove", Button).press()
            await await_dom(
                pilot,
                app.provider_tabs,
                ".entry-draft-editor",
                count=len(names),
            )
        broken = getattr(app, f"{kind}_tabs").editors()[1]
        assert not await app._validate_entry_page(kind)
        assert checked == list(names)
        assert set(
            getattr(
                app.config,
                "inference_providers" if kind == "provider" else "mcp_servers",
            )
        ) == set(names)
        assert (
            "could not connect"
            in str(query(broken, ".draft-message", Static).content).lower()
        )


async def test_non_mcp_endpoint_is_a_validation_error(monkeypatch, app):
    app.config = UrsaConfig.model_validate({
        "mcp_servers": {
            "not-mcp": {
                "transport": "streamable_http",
                "url": "https://google.com",
            }
        }
    })
    message = (
        "The endpoint does not accept MCP requests. Check that the URL "
        "points to an MCP server."
    )
    monkeypatch.setattr(
        "ursa.cli.setup.setup.check_mcp_server",
        lambda name, _server, **_kwargs: CheckResult(
            name, CheckStatus.ERROR, message
        ),
    )
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry("mcp", "not-mcp")
        await await_workers(app, pilot)
        assert not await app._validate_entry_page("mcp")
        editor = app.mcp_tabs.editors()[0]
        assert message in str(query(editor, ".draft-message", Static).content)
        assert app.page == "mcp"


async def test_model_page_validates_both_models(monkeypatch, app):
    app.config.inference_providers.update({
        "chat": InferenceProviderConfig(api_key=SecretReference(keyring=True)),
        "vectors": InferenceProviderConfig(),
    })
    app.pending["chat"] = "pending-secret"
    checked = []
    monkeypatch.setattr(
        "ursa.cli.setup.models.list_provider_models",
        lambda _: [ProviderModel("catalog-model", "openai")],
    )
    monkeypatch.setattr(
        "ursa.cli.setup.setup._probe_model",
        lambda model, providers: checked.append((
            model.model,
            model.inference_provider,
            providers["chat"].api_key.get_secret_value(),
        )),
    )
    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await await_dom(pilot, app, "SetupModelEditor")
        await await_workers(app, pilot)
        editor = app.query_one(SetupModelEditor)
        models = query(editor, "#setup-chat-model-select", FuzzySelect)
        await eventually(
            pilot,
            lambda: "catalog-model" in dict(models._options).values(),
            description="the provider model catalog to load",
        )
        assert "openai" in {
            value
            for _, value in query(
                editor, "#setup-chat-model-provider", Select
            )._options
        }
        query(editor, "#setup-chat-config-yaml", TextArea).text = (
            "model: chat-model\nmodel_provider: openai\n"
            "inference_provider: chat\n"
        )
        query(editor, "#setup-embedding-config-yaml", TextArea).text = (
            "model: embedding-model\nmodel_provider: openai\n"
            "inference_provider: vectors\n"
        )
        editor.submit()
        await eventually(
            pilot,
            lambda: app.page == "mcp",
            description="model validation to advance to the MCP page",
        )
    assert checked == [
        ("chat-model", "chat", "pending-secret"),
        ("embedding-model", "vectors", "pending-secret"),
    ]
    assert app.page == "mcp"


async def test_back_cancels_model_validation(monkeypatch, app):
    app.config = configured_config()
    app.models_valid = True
    started, release = Event(), Event()

    def probe(*_):
        started.set()
        release.wait(2)

    monkeypatch.setattr("ursa.cli.setup.setup._probe_model", probe)
    async with app.run_test(size=(120, 42)) as pilot:
        app.page = "models"
        app._models_selected(SetupModelSelection(app.config.llm_model, None))
        worker = next(w for w in app.workers if w.group == "model-check")
        await await_event(
            pilot, started, description="model validation to start"
        )
        await app.action_back()
        release.set()
        await eventually(
            pilot,
            lambda: worker not in app.workers,
            description="the model validation worker to be cancelled",
        )
    assert app.page == "provider" and not app.models_valid


async def test_doctor_recheck_continue_and_stale_state(monkeypatch, app):
    def report(status):
        return CheckResult.group("URSA Doctor", [CheckResult("", status)])

    reports = iter(
        report(s) for s in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.OK)
    )
    monkeypatch.setattr("ursa.cli.setup.setup._save", lambda *_: app.path)
    monkeypatch.setattr("ursa.cli.setup.setup._doctor", lambda _: next(reports))
    async with app.run_test(size=(120, 42)) as pilot:
        app._show("review")
        await app._save_config()
        assert app.page == "done"
        await app.action_back()
        monkeypatch.setattr(app, "run_worker", lambda work, **_: work.close())
        app._models_selected(SetupModelSelection(app.config.llm_model, None))
        app._show("review")
        title = query(app, "#review-validation", Collapsible)
        assert title.title == "URSA Doctor — not run"

        await app._save_config()
        assert app.page == "review"
        assert [
            str(query(app, f"#review-{name}", Button).label)
            for name in ("continue", "recheck")
        ] == ["Continue", "Recheck"]
        app.query_one("#review-recheck", Button).press()
        await eventually(
            pilot,
            lambda: title.title == "URSA Doctor — 0 warning(s), 0 error(s)",
            description="the doctor recheck result to appear",
        )
        app.query_one("#review-continue", Button).press()
        await eventually(
            pilot,
            lambda: app.page == "done",
            description="Continue to advance to the done page",
        )
