from threading import Event
from unittest.mock import MagicMock

import pytest
import yaml
from textual.widgets import Button, Collapsible, Select, Static, TextArea

from tests.tui.utils import eventually
from ursa.cli import setup_config
from ursa.cli.config import InferenceProviderConfig, UrsaConfig
from ursa.cli.doctor import CheckResult, CheckStatus
from ursa.cli.setup import SetupApp, apply_setup_defaults
from ursa.cli.setup_models import SetupModelEditor, SetupModelSelection
from ursa.cli.tui.widgets import FuzzySelect
from ursa.util.inference_providers import ProviderModel
from ursa.util.secrets import SecretReference


@pytest.fixture(autouse=True)
def setup_keyring(monkeypatch):
    store = MagicMock()
    store.get_password.return_value = None
    for module in ("setup_config", "setup_entries"):
        monkeypatch.setattr(f"ursa.cli.{module}.get_keyring", lambda: store)
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


# fmt: off
def test_config_boundary(monkeypatch, tmp_path, setup_keyring):
    path = tmp_path / "config.yaml"
    path.write_text(
        'inference_providers: {company: {base_url: "${MODEL_URL}", api_key: {keyring: true}}}\n'
        'llm_model: {model: chat, inference_provider: company}\n'
        'mcp_servers: {tools: {transport: stdio, command: tool, env: {KEY: "${MCP_SECRET}"}}}\n')
    config, _ = setup_config.load_setup_config(path)
    with pytest.raises(setup_config.SetupError, match="must not contain API keys"):
        apply_setup_defaults(config, {"mcp_servers": {"bad": {"env": {"API_KEY": "secret"}}}})
    setup_config.save_setup_config(path, config, {})
    assert all(value in path.read_text() for value in ("${MODEL_URL}", "${MCP_SECRET}"))
    setup_keyring.get_password.side_effect = ["old", None]
    monkeypatch.setattr(setup_config, "save_config_file", MagicMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        setup_config.save_setup_config(path, config, {"company": "new", "tools": "new"})
    assert setup_keyring.set_password.call_args.args == ("ursa", "company", "old")
    setup_keyring.delete_password.assert_called_once_with("ursa", "tools")
# fmt: on


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
            "ursa.cli.setup_models.list_provider_models", lambda _: []
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
        f"ursa.cli.setup.check_{'inference_provider' if kind == 'provider' else 'mcp_server'}",
        probe,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        app._begin_entry(kind, names[0])
        await app.workers.wait_for_complete()
        if kind == "provider":
            app.query_one("#provider-add", Button).press()
            await eventually(
                pilot,
                lambda: app.provider_tabs.active_editor() is not None
                and app.provider_tabs.active_editor().raw_name() == "",
            )
            app.query_one("#provider-remove", Button).press()
            await eventually(
                pilot, lambda: len(app.provider_tabs.editors()) == len(names)
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
        "ursa.cli.setup.check_mcp_server",
        lambda name, _server, **_kwargs: CheckResult(
            name, CheckStatus.ERROR, message
        ),
    )
    async with app.run_test(size=(120, 42)):
        app._begin_entry("mcp", "not-mcp")
        await app.workers.wait_for_complete()
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
        "ursa.cli.setup_models.list_provider_models",
        lambda _: [ProviderModel("catalog-model", "openai")],
    )
    monkeypatch.setattr(
        "ursa.cli.setup._probe_model",
        lambda model, providers: checked.append((
            model.model,
            model.inference_provider,
            providers["chat"].api_key.get_secret_value(),
        )),
    )
    async with app.run_test(size=(120, 42)) as pilot:
        app._show("models")
        await app.workers.wait_for_complete()
        editor = app.query_one(SetupModelEditor)
        models = query(editor, "#setup-chat-model-select", FuzzySelect)
        await eventually(
            pilot, lambda: "catalog-model" in dict(models._options).values()
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
        await eventually(pilot, lambda: app.page == "mcp")
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

    monkeypatch.setattr("ursa.cli.setup._probe_model", probe)
    async with app.run_test(size=(120, 42)) as pilot:
        app.page = "models"
        app._models_selected(SetupModelSelection(app.config.llm_model, None))
        worker = next(w for w in app.workers if w.group == "model-check")
        await eventually(pilot, started.is_set)
        await app.action_back()
        release.set()
        await eventually(pilot, lambda: worker not in app.workers)
    assert app.page == "provider" and not app.models_valid


async def test_doctor_recheck_continue_and_stale_state(monkeypatch, app):
    def report(status):
        return CheckResult.group("URSA Doctor", [CheckResult("", status)])

    reports = iter(
        report(s) for s in (CheckStatus.OK, CheckStatus.WARN, CheckStatus.OK)
    )
    monkeypatch.setattr("ursa.cli.setup._save", lambda *_: app.path)
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _: next(reports))
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
        )
        app.query_one("#review-continue", Button).press()
        await eventually(pilot, lambda: app.page == "done")
