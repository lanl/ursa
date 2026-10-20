"""Setup-only model selection, independent of the runtime ``/models`` UI."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

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
from ursa.cli.tui.widgets import AdvancedYamlEditor, FuzzySelect, HelpLabel
from ursa.util.inference_providers import (
    ProviderModel,
    list_provider_models,
    sort_provider_models,
    supported_model_providers,
)

NONE = "__ursa_setup_none__"
CUSTOM = "__ursa_setup_custom__"


@dataclass(frozen=True)
class SetupModelSelection:
    chat: ChatModelConfig
    embedding: EmbModelConfig | None


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
        embedding_values = (
            self._complete_values(embedding) if embedding else None
        )
        embedding_base = embedding or self._new_embedding_config(chat)
        self._bases = {"chat": chat, "embedding": embedding_base}
        self._values: dict[str, dict[str, Any] | None] = {
            "chat": self._complete_values(chat),
            "embedding": embedding_values,
        }
        self._initial = {
            prefix: self._complete_values(base)
            for prefix, base in self._bases.items()
        }

    @staticmethod
    def _complete_values(config: ModelConfig) -> dict[str, Any]:
        configured = type(config).model_validate(config.editable_values())
        return configured.model_dump(mode="json")

    @classmethod
    def _new_embedding_config(cls, chat: ChatModelConfig) -> EmbModelConfig:
        chat_values = cls._complete_values(chat)
        if chat.inference_provider is not None:
            return EmbModelConfig(
                model="", inference_provider=chat.inference_provider
            )
        return EmbModelConfig(
            model="",
            model_provider=chat.model_provider,
            base_url=chat_values["base_url"],
            api_key=deepcopy(chat.api_key),
            ssl_verify=chat.ssl_verify,
        )

    @staticmethod
    def _yaml(values: dict[str, Any] | None) -> str:
        return (
            "null\n"
            if values is None
            else yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
        )

    @staticmethod
    def _help(label: str, field: str, prefix: str) -> HelpLabel:
        descriptions = {
            "model": "The model identifier exposed by the provider.",
            "model-provider": "The LangChain integration used by the model.",
            "inference-provider": (
                "The named inference provider supplying endpoint and auth."
            ),
        }
        return HelpLabel(
            label,
            descriptions[field],
            id=f"setup-{prefix}-{field}-label",
        )

    @staticmethod
    def _choice_options(values: Sequence[str]) -> list[tuple[str, str]]:
        return [("None", NONE), *zip(values, values), ("Other…", CUSTOM)]

    @classmethod
    def _model_choice(
        cls, prefix: str, current: str
    ) -> tuple[FuzzySelect, Input]:
        choices = (current,) if current else ()
        listed = current in choices
        selected = current if listed else (CUSTOM if current else NONE)
        select = FuzzySelect(
            cls._choice_options(choices),
            value=selected,
            allow_blank=False,
            id=f"setup-{prefix}-model-select",
        )
        custom = Input(
            value="" if listed else current,
            placeholder="Required"
            if prefix == "chat"
            else "Optional — leave blank",
            id=f"setup-{prefix}-model",
            classes="hidden" if selected != CUSTOM else "",
        )
        return select, custom

    def _fields(self, prefix: str) -> ComposeResult:
        current = self._values[prefix] or self._initial[prefix]
        model = str(current.get("model") or "")
        yield self._help("Model", "model", prefix)
        model_select, custom_model = self._model_choice(prefix, model)
        yield model_select
        yield custom_model
        yield self._help("Model provider", "model-provider", prefix)
        model_providers = supported_model_providers(self._model_type(prefix))
        selected_model_provider = current.get("model_provider") or NONE
        provider_options = [
            ("None", NONE),
            *zip(model_providers, model_providers),
        ]
        if selected_model_provider not in {
            value for _, value in provider_options
        }:
            provider_options.append((
                f"Not found: {selected_model_provider}",
                selected_model_provider,
            ))
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
            markup=False,
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
        select = self._model_select(prefix)
        if select.value == NONE:
            return ""
        if select.value != CUSTOM:
            return str(select.value)
        return self.query_one(f"#setup-{prefix}-model", Input).value.strip()

    def _model_select(self, prefix: str) -> FuzzySelect:
        return self.query_one(f"#setup-{prefix}-model-select", FuzzySelect)

    def _select(self, prefix: str, field: str) -> Select:
        return self.query_one(f"#setup-{prefix}-{field}", Select)

    def _advanced(self, prefix: str) -> AdvancedYamlEditor:
        return self.query_one(f"#setup-{prefix}-advanced", AdvancedYamlEditor)

    @staticmethod
    def _model_type(prefix: str) -> Literal["chat", "embedding"]:
        return "embedding" if prefix == "embedding" else "chat"

    def _structured_values(self, prefix: str) -> dict[str, Any] | None:
        model = self._choice_value(prefix)
        if prefix == "embedding" and not model:
            return None
        values = deepcopy(self._values[prefix] or self._initial[prefix])
        model_provider_value = self._select(prefix, "model-provider").value
        model_provider = (
            str(model_provider_value)
            if isinstance(model_provider_value, str)
            and model_provider_value != NONE
            else ""
        )
        advertised = self.model_catalogs.get(prefix, {}).get(model)
        if (
            not model_provider
            and advertised is not None
            and advertised.model_provider is not None
        ):
            model_provider = advertised.model_provider
        provider = self._select(prefix, "inference-provider").value
        values["model"] = model
        values["model_provider"] = model_provider or None
        values["inference_provider"] = provider if provider != NONE else None
        if provider != NONE:
            values["base_url"] = None
        return values

    def _sync_yaml_from_controls(self, prefix: str) -> None:
        values = self._structured_values(prefix)
        self._values[prefix] = values
        advanced = self._advanced(prefix)
        advanced.set_text(self._yaml(values))
        advanced.set_validation("neutral")

    @on(Input.Changed)
    def custom_model_changed(self, event: Input.Changed) -> None:
        if event.input.id is not None:
            prefix = event.input.id.removeprefix("setup-").split("-", 1)[0]
            select = self._model_select(prefix)
            if event.value and select.value != CUSTOM:
                with self.prevent(Select.Changed):
                    select.value = CUSTOM
                event.input.remove_class("hidden")
            self._sync_yaml_from_controls(prefix)

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if event.select.id is None:
            return
        prefix = event.select.id.removeprefix("setup-").split("-", 1)[0]
        if event.select.id.endswith("-model-select"):
            self.query_one(f"#setup-{prefix}-model", Input).set_class(
                event.value != CUSTOM, "hidden"
            )
            record = self.model_catalogs.get(prefix, {}).get(str(event.value))
            if (
                record is not None
                and record.model_provider
                in supported_model_providers(self._model_type(prefix))
            ):
                self._select(
                    prefix, "model-provider"
                ).value = record.model_provider
        self._sync_yaml_from_controls(prefix)
        if event.select.id.endswith("-inference-provider"):
            provider = self._model_source(prefix, event.value)
            if provider is not None:
                self._request_model_load(prefix, provider)

    def _model_source(
        self, prefix: str, provider_name: object | None = None
    ) -> ModelConfig | InferenceProviderConfig | None:
        if provider_name is None:
            values = self._values[prefix] or self._initial[prefix]
            provider_name = values.get("inference_provider")
        if isinstance(provider_name, str) and provider_name in self.providers:
            return self.providers[provider_name]
        values = self._structured_values(prefix)
        if values is None:
            values = deepcopy(self._initial[prefix])
        try:
            return self._bases[prefix].model_merge(
                dict_diff(self._initial[prefix], values)
            )
        except (ValidationError, ValueError):
            return None

    def _request_model_load(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig,
    ) -> None:
        self.run_worker(
            self._load_models(prefix, config),
            group=f"setup-{prefix}-models",
            exclusive=True,
        )

    async def _load_models(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig,
    ) -> None:
        try:
            models = await asyncio.to_thread(list_provider_models, config)
        except Exception as exc:  # noqa: BLE001
            if not self.is_mounted:
                return
            self.model_catalogs[prefix] = {}
            current = self._choice_value(prefix)
            self._set_model_options(prefix, {}, current)
            self.notify(
                f"Unable to list models: {exc}",
                title="Model discovery",
                severity="warning",
            )
            return
        if not self.is_mounted:
            return
        ordered = sort_provider_models(models, self._model_type(prefix))
        catalog = {model.name: model for model in ordered}
        self.model_catalogs[prefix] = catalog
        self._set_model_options(prefix, catalog, self._choice_value(prefix))

    def _set_model_options(
        self,
        prefix: str,
        catalog: Mapping[str, ProviderModel],
        current: str,
    ) -> None:
        options = self._choice_options(tuple(catalog))
        if current and current not in catalog:
            options.insert(-1, (f"Not found: {current}", current))
        select = self._model_select(prefix)
        with self.prevent(Select.Changed):
            select.set_options(options)
            select.value = current or NONE
        self.query_one(f"#setup-{prefix}-model", Input).set_class(
            select.value != CUSTOM, "hidden"
        )

    @on(TextArea.Changed)
    def yaml_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id is None:
            return
        prefix = event.text_area.id.removeprefix("setup-").split("-", 1)[0]
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
