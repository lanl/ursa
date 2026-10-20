"""Setup-only model selection, independent of the runtime ``/models`` UI."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import yaml
from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Select, TabbedContent, TabPane, TextArea

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    ModelConfig,
    dict_diff,
)
from ursa.cli.setup_config import SetupError, parse_setup_model_yaml
from ursa.cli.tui.widgets import (
    MODEL_CUSTOM,
    MODEL_NONE,
    AdvancedYamlEditor,
    FuzzySelect,
    HelpLabel,
    ModelScreen,
    ModelSelection,
    discover_model_catalog,
    model_choice_options,
    model_choice_value,
    model_kind,
    model_provider_options,
    model_values_from_controls,
    update_model_choices,
)
from ursa.util.inference_providers import (
    ProviderConfig,
    ProviderModel,
    list_provider_models,
    supported_model_providers,
)

NONE = MODEL_NONE
CUSTOM = MODEL_CUSTOM
SetupModelSelection = ModelSelection


class SetupModelEditor(Vertical):
    class Submitted(Message):
        def __init__(self, selection: SetupModelSelection) -> None:
            super().__init__()
            self.selection = selection

    def __init__(
        self,
        providers: Mapping[str, InferenceProviderConfig],
        chat: ChatModelConfig,
        embedding: EmbModelConfig | None,
    ) -> None:
        super().__init__(id="setup-model-editor")
        self.providers = dict(providers)
        self.model_catalogs: dict[str, dict[str, ProviderModel]] = {}
        embedding_base = embedding or self._new_embedding_config(chat)
        self._bases = {"chat": chat, "embedding": embedding_base}
        self._initial = {
            prefix: self._complete_values(base)
            for prefix, base in self._bases.items()
        }
        self._values: dict[str, dict[str, Any] | None] = dict(self._initial)
        if embedding is None:
            self._values["embedding"] = None

    @staticmethod
    def _complete_values(config: ModelConfig) -> dict[str, Any]:
        configured = type(config).model_validate(config.editable_values())
        return configured.model_dump(mode="json")

    @classmethod
    def _new_embedding_config(cls, chat: ChatModelConfig) -> EmbModelConfig:
        if chat.inference_provider is not None:
            return EmbModelConfig(
                model="", inference_provider=chat.inference_provider
            )
        chat_values = cls._complete_values(chat)
        return EmbModelConfig(
            model="",
            model_provider=chat.model_provider,
            base_url=chat_values["base_url"],
            api_key=deepcopy(chat.api_key),
            ssl_verify=chat.ssl_verify,
        )

    @staticmethod
    def _yaml(values: dict[str, Any] | None) -> str:
        if values is None:
            return "null\n"
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _help(label: str, field: str, prefix: str) -> HelpLabel:
        return HelpLabel(
            label,
            ModelScreen.FIELD_HELP[field],
            id=f"setup-{prefix}-{field}-label",
        )

    def _fields(self, prefix: str) -> ComposeResult:
        current = self._values[prefix] or self._initial[prefix]
        model = str(current.get("model") or "")
        yield self._help("Model", "model", prefix)
        yield FuzzySelect(
            model_choice_options((model,) if model else ()),
            value=model or NONE,
            allow_blank=False,
            id=f"setup-{prefix}-model-select",
        )
        yield Input(
            placeholder="Required"
            if prefix == "chat"
            else "Optional — leave blank",
            id=f"setup-{prefix}-model",
            classes="hidden",
        )
        yield self._help("Model provider", "model-provider", prefix)
        provider_options, selected_model_provider = model_provider_options(
            prefix,
            current.get("model_provider"),
            providers=supported_model_providers(model_kind(prefix)),
        )
        yield Select(
            provider_options,
            value=selected_model_provider,
            allow_blank=False,
            id=f"setup-{prefix}-model-provider",
        )
        yield self._help("Inference provider", "inference-provider", prefix)
        yield Select(
            [
                ("None (direct model config)", NONE),
                *zip(self.providers, self.providers),
            ],
            value=current.get("inference_provider") or NONE,
            allow_blank=False,
            id=f"setup-{prefix}-inference-provider",
        )
        yield AdvancedYamlEditor(
            f"setup-{prefix}",
            self._yaml(self._values[prefix]),
        )

    def compose(self) -> ComposeResult:
        with TabbedContent(id="setup-model-tabs"):
            for prefix, title in (
                ("chat", "Language model"),
                ("embedding", "Embedding model"),
            ):
                with TabPane(title, id=f"setup-{prefix}-tab"):
                    yield from self._fields(prefix)

    def on_mount(self) -> None:
        self.query_one("#setup-chat-model-select", FuzzySelect).focus()
        for prefix in self._bases:
            if source := self._model_source(prefix):
                self._request_model_load(prefix, source)

    def _choice_value(self, prefix: str) -> str:
        return model_choice_value(
            self._model_select(prefix),
            self.query_one(f"#setup-{prefix}-model", Input),
        )

    def _model_select(self, prefix: str) -> FuzzySelect:
        return self.query_one(f"#setup-{prefix}-model-select", FuzzySelect)

    def _select(self, prefix: str, field: str) -> Select:
        return self.query_one(f"#setup-{prefix}-{field}", Select)

    def _advanced(self, prefix: str) -> AdvancedYamlEditor:
        return self.query_one(f"#setup-{prefix}-advanced", AdvancedYamlEditor)

    def _structured_values(self, prefix: str) -> dict[str, Any] | None:
        model = self._choice_value(prefix)
        if prefix == "embedding" and not model:
            return None
        return model_values_from_controls(
            self._values[prefix] or self._initial[prefix],
            model,
            self._select(prefix, "model-provider").value,
            self._select(prefix, "inference-provider").value,
            self.model_catalogs.get(prefix, {}),
            clear_base_url=False,
        )

    def _sync_yaml_from_controls(self, prefix: str) -> None:
        values = self._structured_values(prefix)
        self._values[prefix] = values
        advanced = self._advanced(prefix)
        advanced.set_text(self._yaml(values))
        advanced.set_validation("neutral")

    @on(Input.Changed, "#setup-chat-model, #setup-embedding-model")
    def custom_model_changed(self, event: Input.Changed) -> None:
        prefix = str(event.input.id).removeprefix("setup-").split("-", 1)[0]
        select = self._model_select(prefix)
        if event.value and select.value != CUSTOM:
            with self.prevent(Select.Changed):
                select.value = CUSTOM
            event.input.remove_class("hidden")
        self._sync_yaml_from_controls(prefix)

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        widget_id = str(event.select.id)
        prefix = widget_id.removeprefix("setup-").split("-", 1)[0]
        if widget_id.endswith("-model-select"):
            self.query_one(f"#setup-{prefix}-model", Input).set_class(
                event.value != CUSTOM, "hidden"
            )
            record = self.model_catalogs.get(prefix, {}).get(str(event.value))
            provider = record.model_provider if record else None
            if provider in supported_model_providers(model_kind(prefix)):
                self._select(prefix, "model-provider").value = provider
        self._sync_yaml_from_controls(prefix)
        if widget_id.endswith("-inference-provider"):
            provider = self._model_source(prefix, event.value)
            if provider is not None:
                self._request_model_load(prefix, provider)

    def _model_source(
        self, prefix: str, provider_name: object | None = None
    ) -> ProviderConfig | None:
        if provider_name is None:
            values = self._values[prefix] or self._initial[prefix]
            provider_name = values.get("inference_provider")
        if isinstance(provider_name, str):
            if provider := self.providers.get(provider_name):
                return provider
        values = self._structured_values(prefix) or self._initial[prefix]
        try:
            return self._bases[prefix].model_merge(
                dict_diff(self._initial[prefix], values)
            )
        except (ValidationError, ValueError):
            return None

    def _request_model_load(self, prefix: str, config: ProviderConfig) -> None:
        self.run_worker(
            self._load_models(prefix, config),
            group=f"setup-{prefix}-models",
            exclusive=True,
        )

    async def _load_models(self, prefix: str, config: ProviderConfig) -> None:
        catalog, warning = await discover_model_catalog(
            config,
            model_kind(prefix),
            list_models=list_provider_models,
            run_sync=asyncio.to_thread,
        )
        select = self.query_one_optional(
            f"#setup-{prefix}-model-select", FuzzySelect
        )
        custom = self.query_one_optional(f"#setup-{prefix}-model", Input)
        if select is None or custom is None:
            return
        if select.query_one_optional("FuzzySelectOverlay") is None:
            return
        self.model_catalogs[prefix] = catalog
        current = model_choice_value(select, custom)
        with self.prevent(Select.Changed):
            update_model_choices(select, custom, catalog, current)
        if warning:
            self.notify(warning, title="Model discovery", severity="warning")

    @on(TextArea.Changed, ".advanced-yaml-editor")
    def yaml_changed(self, event: TextArea.Changed) -> None:
        prefix = str(event.text_area.id).removeprefix("setup-").split("-", 1)[0]
        self._advanced(prefix).set_validation("neutral")

    def _parse(
        self, prefix: str
    ) -> tuple[bool, ChatModelConfig | EmbModelConfig | None]:
        advanced = self._advanced(prefix)
        editor = advanced.query_one(TextArea)
        base = self._bases[prefix]
        try:
            config = parse_setup_model_yaml(
                editor.text,
                type(base),
                base,
                self._initial[prefix],
                self.providers,
            )
        except SetupError as exc:
            advanced.set_validation("invalid", str(exc))
            self.query_one(
                "#setup-model-tabs", TabbedContent
            ).active = f"setup-{prefix}-tab"
            return False, None
        advanced.set_validation("valid")
        return True, config

    def submit(self) -> None:
        chat_ok, chat = self._parse("chat")
        embedding_ok, embedding = self._parse("embedding")
        if not chat_ok or not embedding_ok:
            return
        assert isinstance(chat, ChatModelConfig)
        assert embedding is None or isinstance(embedding, EmbModelConfig)
        self.post_message(self.Submitted(SetupModelSelection(chat, embedding)))
