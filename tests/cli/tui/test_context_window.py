"""Effective thresholds in /status follow the model without cluttering the footer."""

from types import SimpleNamespace

import httpx
import pytest
import yaml
from langchain_ollama import ChatOllama
from textual.widgets import Static

from tests.cli._app_fakes import FakeHITL
from tests.tui.utils import await_workers
from ursa.cli.config import ChatModelConfig, UrsaConfig
from ursa.cli.runtime import HITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import InformationScreen, ModelSelection
from ursa.util.model_context import summarization_threshold


@pytest.mark.parametrize(
    "metadata,window,threshold",
    [
        ({"profile": {"max_input_tokens": 128000}}, "128,000", "89,600"),
        (
            {"num_ctx": 8192, "profile": {"max_input_tokens": 131072}},
            "8,192",
            "5,734",
        ),
        ({}, None, "12,000"),
        ({"profile": {"max_input_tokens": 0}}, None, "12,000"),
        (None, None, "12,000"),
    ],
)
async def test_context_display(
    tmp_path, monkeypatch, metadata, window, threshold
):
    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Status must not validate or initialize a model")

    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", unexpected_initialization
    )
    monkeypatch.setattr(
        ChatModelConfig, "init_chat_model", unexpected_initialization
    )
    hitl = HITL(
        UrsaConfig(
            workspace=tmp_path,
            tokens_before_summarize=12000,
            agent_config={"chat": {"tokens_before_summarize": 1000}},
        )
    )
    model = SimpleNamespace(**metadata) if metadata is not None else None
    if model is not None:
        hitl.model = model
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        status = app.query_one("#status", Static)
        content = status.content
        assert "context" not in content.plain
        assert "default trigger" not in content.plain
        assert "0 tokens" in content.plain
        assert not content.spans
        await app._show_command("status")
        assert isinstance(app.screen, InformationScreen)
        assert hitl.initialized_model is model
        details = app.screen.content
        if window is None:
            assert "| Context window | `Unknown` |" in details
            assert "Default summarization threshold" in details
            assert (
                f"{threshold} tokens (fallback; agents may override)" in details
            )
        else:
            assert f"| Context window | `{window} tokens` |" in details
            assert f"{threshold} tokens (70%)" in details
        config_yaml = app.screen.config_yaml
        config = yaml.safe_load(config_yaml)
        expected = int(threshold.replace(",", ""))
        assert config["tokens_before_summarize"] == expected
        assert config["agent_config"]["chat"]["tokens_before_summarize"] == (
            expected if window else 1000
        )
        assert "Configured fallback (global): 12000 tokens" in config_yaml
        assert "Configured fallback (chat): 1000 tokens" in config_yaml
        assert hitl.config.tokens_before_summarize == 12000
        assert (
            hitl.config.agent_config["chat"]["tokens_before_summarize"] == 1000
        )


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
            == 5734
        )
        assert "Ollama /api/ps" in app._status_markdown()
        assert summarization_threshold(agent_model) == 5734
