from __future__ import annotations

import asyncio

import yaml
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Select, TextArea

from ursa.cli.config import ChatModelConfig, InferenceProviderConfig
from ursa.cli.setup_models import CUSTOM, SetupModelEditor, SetupModelSelection
from ursa.cli.tui.widgets import FuzzySelect
from ursa.util.inference_providers import ProviderModel


class SetupModelHarness(App[None]):
    def __init__(self, providers, chat=None, embedding=None) -> None:
        super().__init__()
        self.editor = SetupModelEditor(
            providers,
            chat
            or ChatModelConfig(
                model="old-model", inference_provider=next(iter(providers))
            ),
            embedding,
        )
        self.submissions: list[SetupModelSelection] = []

    def compose(self) -> ComposeResult:
        yield self.editor

    @on(SetupModelEditor.Submitted)
    def submitted(self, event: SetupModelEditor.Submitted) -> None:
        event.stop()
        self.submissions.append(event.selection)


async def test_setup_models_use_fuzzy_catalog_for_selected_provider(
    monkeypatch,
):
    providers = {
        "first": InferenceProviderConfig(base_url="https://first.example/v1"),
        "second": InferenceProviderConfig(base_url="https://second.example/v1"),
    }
    discovered = []

    def models(config):
        discovered.append(config)
        name = (
            "first-model" if "first" in str(config.base_url) else "second-model"
        )
        return [ProviderModel(name, "openai")]

    monkeypatch.setattr("ursa.cli.setup_models.list_provider_models", models)
    app = SetupModelHarness(providers)

    async with app.run_test(size=(100, 36)) as pilot:
        await app.workers.wait_for_complete()
        model_select = app.editor.query_one(
            "#setup-chat-model-select", FuzzySelect
        )
        assert "first-model" in {value for _, value in model_select._options}
        assert providers["first"] in discovered

        app.editor.query_one(
            "#setup-chat-inference-provider", Select
        ).value = "second"
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert "second-model" in {value for _, value in model_select._options}
        assert app.editor.model_catalogs["chat"] == {
            "second-model": ProviderModel("second-model", "openai")
        }
        assert providers["second"] in discovered
        values = yaml.safe_load(
            app.editor.query_one("#setup-chat-config-yaml", TextArea).text
        )
        assert values["inference_provider"] == "second"


async def test_setup_model_provider_is_canonical_dropdown(monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.setup_models.supported_model_providers",
        lambda model_type="chat": (
            ("embed_one", "embed_two")
            if model_type == "embedding"
            else ("chat_one", "chat_two")
        ),
    )
    monkeypatch.setattr(
        "ursa.cli.setup_models.list_provider_models", lambda _config: []
    )
    app = SetupModelHarness({"provider": InferenceProviderConfig()})

    async with app.run_test(size=(100, 36)):
        await app.workers.wait_for_complete()
        chat = app.editor.query_one("#setup-chat-model-provider", Select)
        embedding = app.editor.query_one(
            "#setup-embedding-model-provider", Select
        )

        assert [value for _, value in chat._options] == [
            "__ursa_setup_none__",
            "chat_one",
            "chat_two",
            "openai",
        ]
        assert [value for _, value in embedding._options] == [
            "__ursa_setup_none__",
            "embed_one",
            "embed_two",
        ]


async def test_setup_model_other_choice_preserves_custom_model(monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.setup_models.list_provider_models",
        lambda _config: [ProviderModel("catalog-model", "openai")],
    )
    app = SetupModelHarness({"provider": InferenceProviderConfig()})

    async with app.run_test(size=(100, 36)) as pilot:
        await app.workers.wait_for_complete()
        select = app.editor.query_one("#setup-chat-model-select", FuzzySelect)
        custom = app.editor.query_one("#setup-chat-model", Input)

        assert CUSTOM in {value for _, value in select._options}
        select.value = CUSTOM
        await pilot.pause()
        assert not custom.has_class("hidden")

        custom.value = "private-model"
        await pilot.pause()
        values = yaml.safe_load(
            app.editor.query_one("#setup-chat-config-yaml", TextArea).text
        )
        assert values["model"] == "private-model"

        app.editor.submit()
        await pilot.pause()
        assert app.submissions[0].chat.model == "private-model"
        assert app.submissions[0].embedding is None


async def test_model_discovery_ignores_editor_teardown(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_listing(*_args):
        entered.set()
        await release.wait()
        return []

    monkeypatch.setattr(
        SetupModelEditor, "_request_model_load", lambda *_args: None
    )
    monkeypatch.setattr(
        "ursa.cli.setup_models.asyncio.to_thread", delayed_listing
    )
    provider = InferenceProviderConfig()
    app = SetupModelHarness({"provider": provider})

    async with app.run_test(size=(100, 36)):
        discovery = asyncio.create_task(
            app.editor._load_models("chat", provider)
        )
        await entered.wait()
        await app.editor.query_one("FuzzySelectOverlay").remove()
        release.set()
        await discovery


async def test_setup_advanced_yaml_deletions_replace_old_model_fields(
    monkeypatch,
):
    monkeypatch.setattr(
        "ursa.cli.setup_models.list_provider_models", lambda _config: []
    )
    chat = ChatModelConfig(
        model="old-model",
        model_provider="anthropic",
        inference_provider="provider",
        max_completion_tokens=4096,
        old_extra="remove-me",
    )
    app = SetupModelHarness({"provider": InferenceProviderConfig()}, chat=chat)

    async with app.run_test(size=(100, 36)) as pilot:
        await app.workers.wait_for_complete()
        app.editor.query_one(
            "#setup-chat-config-yaml", TextArea
        ).text = "model: replacement\ninference_provider: provider\n"
        app.editor.submit()
        await pilot.pause()

        parsed = app.submissions[0].chat
        assert parsed.model == "replacement"
        assert parsed.max_completion_tokens is None
        assert parsed.model_extra == {}
        assert parsed.model_fields_set == {"model", "inference_provider"}
