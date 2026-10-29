"""Effective thresholds in /status follow the model without cluttering the footer."""

from types import SimpleNamespace

import httpx
import pytest
import yaml
from langchain_ollama import ChatOllama
from textual.widgets import Static

from tests.cli._app_fakes import FakeHITL
from tests.tui.utils import await_workers
from ursa.cli.config import ChatModelConfig
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import ModelSelection
from ursa.util.model_context import summarization_threshold


@pytest.mark.parametrize(
    "metadata,window,threshold",
    [
        ({"profile": {"max_input_tokens": 128000}}, "128,000", "89,600"),
        (
            {"num_ctx": 8192, "profile": {"max_input_tokens": 131072}},
            "8,192",
            "5,735",
        ),
        ({}, None, "50,000"),
        ({"profile": {"max_input_tokens": 0}}, None, "50,000"),
    ],
)
async def test_context_display(tmp_path, metadata, window, threshold):
    hitl = FakeHITL(tmp_path)
    hitl.model = SimpleNamespace(model_name="test-model", **metadata)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        status = app.query_one("#status", Static)
        content = status.content
        assert "context" not in content.plain
        assert "default trigger" not in content.plain
        assert "0 tokens" in content.plain
        assert not content.spans
        details = app._status_markdown()
        if window is None:
            assert "| Context window | `Unknown` |" in details
            assert "Default summarization threshold" in details
            assert (
                f"{threshold} tokens (fallback; agents may override)" in details
            )
        else:
            assert f"| Context window | `{window} tokens` |" in details
            assert f"{threshold} tokens (70%)" in details
        config_yaml = app._resolved_config_yaml()
        assert yaml.safe_load(config_yaml)["tokens_before_summarize"] == int(
            threshold.replace(",", "")
        )
        assert "Configured fallback (global): 50000 tokens" in config_yaml
        assert hitl.config.tokens_before_summarize == 50000


async def test_model_switch_refreshes_effective_threshold(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    hitl.model = SimpleNamespace(profile={"max_input_tokens": 128000})
    reconfigure = hitl.reconfigure_models

    async def reconfigure_models(chat, embedding):
        await reconfigure(chat, embedding)
        if chat.model == "custom-known":
            hitl.model.profile = {"max_input_tokens": 32768}

    monkeypatch.setattr(hitl, "reconfigure_models", reconfigure_models)
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        for model, expected in [
            ("custom-unknown", 50000),
            ("custom-known", 22938),
        ]:
            app._select_model(
                ModelSelection(
                    chat=ChatModelConfig(
                        model=model, inference_provider="openai"
                    ),
                    embedding=None,
                )
            )
            await await_workers(app, pilot)
            content = app.query_one("#status", Static).content
            assert model in content.plain
            assert "context" not in content.plain
            details = app._status_markdown()
            assert f"{expected:,}" in details
            assert (
                yaml.safe_load(app._resolved_config_yaml())[
                    "tokens_before_summarize"
                ]
                == expected
            )


async def test_ollama_response_refreshes_shared_context_and_display(tmp_path):
    def handler(request):
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [{"name": "gemma4:latest", "context_length": 8192}]
            },
        )

    hitl = FakeHITL(tmp_path)
    hitl.model = ChatOllama(
        model="gemma4",
        client_kwargs={"transport": httpx.MockTransport(handler)},
    )
    agent_model = hitl.model.bind(extra_headers={})
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("hello", tmp_path)
        await app.query_one("#conversation").mount(turn)
        assert (
            yaml.safe_load(app._resolved_config_yaml())[
                "tokens_before_summarize"
            ]
            == 50000
        )
        await TextualEventHandler(app, turn).on_llm_end(
            SimpleNamespace(llm_output={})
        )
        await pilot.pause()
        assert (
            yaml.safe_load(app._resolved_config_yaml())[
                "tokens_before_summarize"
            ]
            == 5735
        )
        assert "Ollama /api/ps" in app._status_markdown()
        assert summarization_threshold(agent_model) == 5735


@pytest.mark.parametrize(
    "window,expected", [(None, "12,000"), (10000, "7,000")]
)
async def test_configured_fallback_in_status(tmp_path, window, expected):
    hitl = FakeHITL(tmp_path)
    hitl.config.tokens_before_summarize = 12000
    hitl.model = SimpleNamespace(profile={"max_input_tokens": window})
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)):
        details = app._status_markdown()
        assert f"{expected} tokens" in details
        assert "50,000" not in details
        assert yaml.safe_load(app._resolved_config_yaml())[
            "tokens_before_summarize"
        ] == int(expected.replace(",", ""))
        assert hitl.config.tokens_before_summarize == 12000


@pytest.mark.parametrize("window,expected", [(None, 12000), (128000, 89600)])
def test_status_config_reports_effective_agent_overrides(
    tmp_path, window, expected
):
    hitl = FakeHITL(tmp_path)
    hitl.config.agent_config = {"chat": {"tokens_before_summarize": 12000}}
    hitl.model = SimpleNamespace(profile={"max_input_tokens": window})
    config_yaml = UrsaTextualApp(hitl)._resolved_config_yaml()
    assert (
        yaml.safe_load(config_yaml)["agent_config"]["chat"][
            "tokens_before_summarize"
        ]
        == expected
    )
    assert "Configured fallback (chat): 12000 tokens" in config_yaml
    assert hitl.config.agent_config["chat"]["tokens_before_summarize"] == 12000
