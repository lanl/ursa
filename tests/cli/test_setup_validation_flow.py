from unittest.mock import MagicMock, call

import pytest
from textual.widgets import Button, Collapsible, Static

from tests.cli.test_setup import (
    click_and_wait,
    click_page,
    configured_config,
    model_config,
    query,
)
from ursa.cli.config import EmbModelConfig
from ursa.cli.doctor import CheckResult, CheckStatus
from ursa.cli.setup import SetupApp
from ursa.cli.setup_models import SetupModelSelection


@pytest.fixture
def app(tmp_path):
    return SetupApp(tmp_path / "config.yaml")


def _report(status):
    return CheckResult.group("URSA Doctor", [CheckResult("", status)])


@pytest.mark.parametrize(
    ("kind", "selector", "label", "name"),
    [
        ("provider", "#provider-error", "providers", "company"),
        ("mcp", "#mcp-validation-status", "MCP servers", "tools"),
    ],
)
async def test_entry_validation_only_shows_page_level_progress(
    monkeypatch, app, kind, selector, label, name
):
    app.config = configured_config()
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", MagicMock())

    async with app.run_test(size=(120, 42)):
        app._begin_entry(kind, name)
        editor = getattr(app, f"{kind}_tabs").active_editor()
        assert editor is not None
        page_status = MagicMock(wraps=app._status)
        editor_status = MagicMock(wraps=editor.set_status)
        monkeypatch.setattr(app, "_status", page_status)
        monkeypatch.setattr(editor, "set_status", editor_status)

        assert await app._validate_entry_page(kind)
        page_status.assert_has_calls([
            call(selector, f"Validating {label}..."),
            call(selector, f"✓ All {label} connected"),
        ])
        editor_status.assert_called_with("✓ Connected", False)
        assert not any(
            "Validating" in item.args[0]
            for item in editor_status.call_args_list
        )


async def test_model_validation_shows_progress_for_each_probe(monkeypatch, app):
    app.config = configured_config()
    app.config.llm_model.model = "chat"
    app.config.emb_model = model_config(EmbModelConfig, "embedding", "company")
    status = MagicMock(wraps=app._status)
    monkeypatch.setattr("ursa.cli.setup._probe_model", MagicMock())
    monkeypatch.setattr(app, "_status", status)

    async with app.run_test(size=(120, 42)):
        app.page = "models"
        app.query_one("#setup-pages").current = "models-page"
        await app._validate_models()

    messages = [item.args[1] for item in status.call_args_list]
    assert all(
        any(f"Validating {label}..." in message for message in messages)
        for label in ("language model", "embedding model")
    )


async def test_doctor_success_warning_recheck_and_continue(monkeypatch, app):
    reports = iter(
        map(_report, (CheckStatus.OK, CheckStatus.WARN, CheckStatus.OK))
    )
    monkeypatch.setattr("ursa.cli.setup._save", lambda *_args: app.path)
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _path: next(reports))
    success_title = "URSA Doctor — 0 warning(s), 0 error(s)"

    async with app.run_test(size=(120, 42)) as pilot:
        app._show("review")
        await app._save_config()
        assert app.page == "done"
        assert str(query(app, "#done-copy", Static).content) == (
            f"Saved to {app.path}\nValidation: 0 warning(s), 0 error(s)"
        )
        await app.action_back()
        monkeypatch.setattr(app, "run_worker", lambda work, **_: work.close())
        app._models_selected(SetupModelSelection(app.config.llm_model, None))
        assert not app.doctor_summary
        app._show("review")
        assert query(app, "#review-validation", Collapsible).title == (
            "URSA Doctor — not run"
        )

        await app._save_config()
        assert app.page == "review"
        buttons = [
            query(app, f"#review-{name}", Button)
            for name in ("continue", "recheck")
        ]
        assert [str(button.label) for button in buttons] == [
            "Continue",
            "Recheck",
        ]
        assert not any(button.has_class("hidden") for button in buttons)

        await click_and_wait(
            pilot,
            "#review-recheck",
            lambda: query(app, "#review-validation", Collapsible).title
            == success_title,
        )
        assert app.page == "review"

        await click_page(pilot, "#review-continue", "done")
