import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Static

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


def _model(kind, name: str, provider: str):
    return kind(
        model=name,
        model_provider="openai",
        inference_provider=provider,
    )


async def test_provider_validation_only_shows_page_level_progress(
    monkeypatch, app
):
    app.config.inference_providers = {
        "company": InferenceProviderConfig(base_url="https://models.example/v1")
    }
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", MagicMock())

    async with app.run_test(size=(120, 42)):
        app._begin_provider("company")
        editor = app.provider_tabs.active_editor()
        assert editor is not None
        page_updates = []
        editor_updates = []
        original_page_status = app._status
        original_editor_status = editor.set_status

        def page_status(selector, message, failed=False):
            if selector == "#provider-error":
                page_updates.append(message)
            original_page_status(selector, message, failed)

        def editor_status(message, failed=False):
            editor_updates.append(message)
            original_editor_status(message, failed)

        monkeypatch.setattr(app, "_status", page_status)
        monkeypatch.setattr(editor, "set_status", editor_status)

        assert await app._validate_provider_page()

        assert page_updates == [
            "Validating providers...",
            "✓ All providers connected",
        ]
        assert editor_updates[-1] == "✓ Connected"
        assert not any("Validating" in message for message in editor_updates)


async def test_mcp_validation_only_shows_page_level_progress(monkeypatch, app):
    app.config = UrsaConfig.model_validate({
        "mcp_servers": {
            "tools": {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
            }
        }
    })
    monkeypatch.setattr("ursa.cli.setup.run_probe_with_deadline", MagicMock())

    async with app.run_test(size=(120, 42)):
        app._begin_mcp("tools")
        editor = app.mcp_tabs.active_editor()
        assert editor is not None
        page_updates = []
        editor_updates = []
        original_page_status = app._status
        original_editor_status = editor.set_status

        def page_status(selector, message, failed=False):
            if selector == "#mcp-validation-status":
                page_updates.append(message)
            original_page_status(selector, message, failed)

        def editor_status(message, failed=False):
            editor_updates.append(message)
            original_editor_status(message, failed)

        monkeypatch.setattr(app, "_status", page_status)
        monkeypatch.setattr(editor, "set_status", editor_status)

        assert await app._validate_mcp_page()

        assert page_updates == [
            "Validating MCP servers...",
            "✓ All MCP servers connected",
        ]
        assert editor_updates[-1] == "✓ Connected"
        assert not any("Validating" in message for message in editor_updates)


async def test_model_validation_shows_progress_for_each_probe(monkeypatch, app):
    app.config.inference_providers = {
        "company": InferenceProviderConfig(base_url="https://models.example/v1")
    }
    app.config.llm_model = _model(ChatModelConfig, "chat", "company")
    app.config.emb_model = _model(EmbModelConfig, "embedding", "company")
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


async def test_clean_doctor_report_advances_to_done(monkeypatch, app):
    report = CheckResult.group(
        "URSA Doctor",
        [CheckResult("Configuration", CheckStatus.OK)],
    )
    monkeypatch.setattr("ursa.cli.setup._save", lambda *_args: app.path)
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _path: report)

    async with app.run_test(size=(120, 42)):
        app._show("review")

        await app._save_config()

        assert app.page == "done"
        assert str(app.query_one("#done-copy", Static).content) == (
            f"Saved to {app.path}\nValidation: 0 warning(s), 0 error(s)"
        )


@pytest.mark.parametrize(
    ("status", "finish_label", "retry_hidden"),
    [
        (CheckStatus.WARN, "Finish", True),
        (CheckStatus.ERROR, "Finish anyway", False),
    ],
)
async def test_doctor_issues_remain_on_review(
    monkeypatch, app, status, finish_label, retry_hidden
):
    report = CheckResult.group(
        "URSA Doctor",
        [CheckResult("Configuration", status, "issue")],
    )
    monkeypatch.setattr("ursa.cli.setup._doctor", lambda _path: report)

    async with app.run_test(size=(120, 42)):
        app.saved_path = app.path
        app._show("review")

        await app._run_doctor()

        assert app.page == "review"
        assert (
            str(app.query_one("#review-finish", Button).label) == finish_label
        )
        assert (
            app.query_one("#review-retry", Button).has_class("hidden")
            is retry_hidden
        )
