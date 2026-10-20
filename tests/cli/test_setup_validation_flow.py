import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Collapsible, Static

from tests.cli.test_setup import model_config
from tests.tui.utils import eventually
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
)
from ursa.cli.doctor import CheckResult, CheckStatus
from ursa.cli.setup import SetupApp


@pytest.fixture
def app(tmp_path):
    return SetupApp(tmp_path / "config.yaml")


def _report(status):
    return CheckResult.group(
        "URSA Doctor",
        [CheckResult("Configuration", status, "issue" if status else "")],
    )


@pytest.mark.parametrize(
    ("kind", "selector", "progress", "complete"),
    [
        (
            "provider",
            "#provider-error",
            "Validating providers...",
            "✓ All providers connected",
        ),
        (
            "mcp",
            "#mcp-validation-status",
            "Validating MCP servers...",
            "✓ All MCP servers connected",
        ),
    ],
)
async def test_entry_validation_only_shows_page_level_progress(
    monkeypatch, app, kind, selector, progress, complete
):
    if kind == "provider":
        app.config.inference_providers = {
            "company": InferenceProviderConfig(
                base_url="https://models.example/v1"
            )
        }
        name = "company"
    else:
        app.config = UrsaConfig.model_validate({
            "mcp_servers": {
                "tools": {
                    "transport": "streamable_http",
                    "url": "https://tools.example/mcp",
                }
            }
        })
        name = "tools"
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", MagicMock())

    async with app.run_test(size=(120, 42)):
        app._begin_entry(kind, name)
        editor = getattr(app, f"{kind}_tabs").active_editor()
        assert editor is not None
        page_updates = []
        editor_updates = []
        original_page_status = app._status
        original_editor_status = editor.set_status

        def page_status(target, message, failed=False):
            if target == selector:
                page_updates.append(message)
            original_page_status(target, message, failed)

        def editor_status(message, failed=False):
            editor_updates.append(message)
            original_editor_status(message, failed)

        monkeypatch.setattr(app, "_status", page_status)
        monkeypatch.setattr(editor, "set_status", editor_status)

        assert await app._validate_entry_page(kind)
        assert page_updates == [progress, complete]
        assert editor_updates[-1] == "✓ Connected"
        assert not any("Validating" in message for message in editor_updates)


async def test_model_validation_shows_progress_for_each_probe(monkeypatch, app):
    app.config.inference_providers = {
        "company": InferenceProviderConfig(base_url="https://models.example/v1")
    }
    app.config.llm_model = model_config(ChatModelConfig, "chat", "company")
    app.config.emb_model = model_config(EmbModelConfig, "embedding", "company")
    loop = asyncio.get_running_loop()
    started = {
        "chat": asyncio.Event(),
        "embedding": asyncio.Event(),
    }
    release = threading.Event()

    def probe(model, _providers):
        loop.call_soon_threadsafe(started[model.model].set)
        assert release.wait(timeout=5)

    monkeypatch.setattr("ursa.cli.setup._probe_model", probe)

    async with app.run_test(size=(120, 42)):
        app.page = "models"
        app.query_one("#setup-pages").current = "models-page"
        validation = asyncio.create_task(app._validate_models())
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())),
            timeout=2,
        )

        try:
            status = str(
                app.query_one("#model-validation-status", Static).content
            )
            assert "Validating language model..." in status
            assert "Validating embedding model..." in status
        finally:
            release.set()
            await validation


@pytest.mark.parametrize(
    "status",
    [CheckStatus.OK, CheckStatus.WARN, CheckStatus.ERROR],
)
async def test_doctor_result_exposes_review_actions(monkeypatch, app, status):
    monkeypatch.setattr("ursa.cli.setup._save", lambda *_args: app.path)
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _path: _report(status))

    async with app.run_test(size=(120, 42)):
        app._show("review")
        await app._save_config()

        if status is CheckStatus.OK:
            assert app.page == "done"
            assert app.saved_path == app.path
            assert str(app.query_one("#done-copy", Static).content) == (
                f"Saved to {app.path}\nValidation: 0 warning(s), 0 error(s)"
            )
            await app.action_back()
        assert app.page == "review"
        for id, label in {
            "review-continue": "Continue",
            "review-recheck": "Recheck",
        }.items():
            button = app.query_one(f"#{id}", Button)
            assert str(button.label) == label
            assert not button.has_class("hidden")


async def test_recheck_stays_on_review_then_continue_finishes(monkeypatch, app):
    reports = iter(map(_report, (CheckStatus.WARN, CheckStatus.OK)))
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _path: next(reports))

    async with app.run_test(size=(120, 42)) as pilot:
        app.saved_path = app.path
        app._show("review")
        await app._run_doctor()

        app.query_one("#review-recheck", Button).press()
        await eventually(
            pilot,
            lambda: app.query_one("#review-validation", Collapsible).title
            == "URSA Doctor — 0 warning(s), 0 error(s)",
            description="doctor recheck",
        )
        assert app.page == "review"
        assert app.query_one("#review-validation", Collapsible).title == (
            "URSA Doctor — 0 warning(s), 0 error(s)"
        )

        app.query_one("#review-continue", Button).press()
        await eventually(
            pilot,
            lambda: app.page == "done",
            description="review continuation",
        )
        assert app.page == "done"
