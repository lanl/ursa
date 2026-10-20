"""Setup-only model selection editor.

The running TUI's ``/models`` screen has runtime concerns that the setup
wizard does not share.  Keeping this editor local to setup lets the wizard
stage configuration without changing that screen's appearance or behavior.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import (
    Input,
    Select,
    TabbedContent,
    TabPane,
    TextArea,
)

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
    """Model configuration staged by the setup wizard."""

    chat: ChatModelConfig
    embedding: EmbModelConfig | None


class SetupModelEditor(Vertical):
    """Inline setup editor independent of the runtime ``/models`` UI."""

    class Submitted(Message):
        """A locally valid model selection ready for network validation."""

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
        self.chat = chat
        self.embedding = embedding
        self._syncing = False
        self.model_catalogs: dict[str, dict[str, ProviderModel]] = {}
        self._model_load_generation = {"chat": 0, "embedding": 0}
        self._values: dict[str, dict[str, Any] | None] = {
            "chat": self._complete_values(chat),
            "embedding": (
                self._complete_values(embedding)
                if embedding is not None
                else None
            ),
        }
        self._embedding_base = embedding or self._new_embedding_config(chat)
        self._embedding_template = self._complete_values(self._embedding_base)
        self._initial_values = {
            "chat": deepcopy(self._values["chat"]),
            "embedding": deepcopy(
                self._values["embedding"] or self._embedding_template
            ),
        }

    @staticmethod
    def _complete_values(config: ModelConfig) -> dict[str, Any]:
        """Return a full config block while dropping resolved provider data."""
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
        if values is None:
            return "null\n"
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

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
        return [
            ("None", NONE),
            *((value, value) for value in values),
            ("Other…", CUSTOM),
        ]

    @classmethod
    def _model_choice(
        cls, prefix: str, values: Sequence[str], current: str
    ) -> tuple[FuzzySelect, Input]:
        choices = tuple(dict.fromkeys(value for value in values if value))
        listed = current in choices
        selected = current if listed else (CUSTOM if current else NONE)
        select = FuzzySelect(
            cls._choice_options(choices),
            value=selected,
            allow_blank=False,
            id=f"setup-{prefix}-model-select",
            classes="setup-model-choice",
        )
        custom = Input(
            value="" if listed else current,
            placeholder=(
                "Required" if prefix == "chat" else "Optional — leave blank"
            ),
            id=f"setup-{prefix}-model",
            classes=(
                "setup-model-custom" + (" hidden" if selected != CUSTOM else "")
            ),
        )
        return select, custom

    def _fields(
        self,
        prefix: str,
        values: dict[str, Any] | None,
    ) -> ComposeResult:
        current = values or self._embedding_template
        yield self._help("Model", "model", prefix)
        model_select, custom_model = self._model_choice(
            prefix,
            (str(current.get("model") or ""),),
            str(current.get("model") or ""),
        )
        yield model_select
        yield custom_model
        yield self._help("Model provider", "model-provider", prefix)
        model_type = "embedding" if prefix == "embedding" else "chat"
        model_providers = supported_model_providers(model_type)
        selected_model_provider = current.get("model_provider") or NONE
        provider_options = [
            ("None", NONE),
            *((provider, provider) for provider in model_providers),
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
            classes="setup-model-provider",
        )
        yield self._help("Inference provider", "inference-provider", prefix)
        yield Select(
            [
                ("None (direct model config)", NONE),
                *((name, name) for name in self.providers),
            ],
            value=current.get("inference_provider") or NONE,
            allow_blank=False,
            id=f"setup-{prefix}-inference-provider",
            classes="setup-inference-provider",
        )
        yield AdvancedYamlEditor(
            f"setup-{prefix}",
            self._yaml(values),
            editor_classes="setup-model-yaml",
            error_classes="setup-model-yaml-error",
            markup=False,
        )

    def compose(self) -> ComposeResult:
        with TabbedContent(id="setup-model-tabs"):
            with TabPane("Language model", id="setup-chat-tab"):
                yield from self._fields("chat", self._values["chat"])
            with TabPane("Embedding model", id="setup-embedding-tab"):
                yield from self._fields("embedding", self._values["embedding"])

    def on_mount(self) -> None:
        self.query_one("#setup-chat-model-select", FuzzySelect).focus()
        chat_generation = self._next_model_load_generation("chat")
        embedding_generation = self._next_model_load_generation("embedding")
        self.run_worker(
            self._load_initial_models(chat_generation, embedding_generation),
            group="setup-model-discovery",
            exclusive=True,
        )

    def _choice_value(self, prefix: str) -> str:
        select = self.query_one(f"#setup-{prefix}-model-select", FuzzySelect)
        if select.value == NONE:
            return ""
        if select.value != CUSTOM:
            return str(select.value)
        return self.query_one(f"#setup-{prefix}-model", Input).value.strip()

    def _structured_values(self, prefix: str) -> dict[str, Any] | None:
        model = self._choice_value(prefix)
        if prefix == "embedding" and not model:
            return None
        current = self._values[prefix]
        values = deepcopy(
            current if current is not None else self._embedding_template
        )
        model_provider_value = self.query_one(
            f"#setup-{prefix}-model-provider", Select
        ).value
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
        provider = self.query_one(
            f"#setup-{prefix}-inference-provider", Select
        ).value
        values["model"] = model
        values["model_provider"] = model_provider or None
        values["inference_provider"] = provider if provider != NONE else None
        if provider != NONE:
            values["base_url"] = None
        return values

    def _sync_yaml_from_controls(self, prefix: str) -> None:
        if self._syncing:
            return
        values = self._structured_values(prefix)
        self._values[prefix] = values
        advanced = self.query_one(
            f"#setup-{prefix}-advanced", AdvancedYamlEditor
        )
        advanced.set_text(self._yaml(values))
        advanced.set_validation("neutral")

    @on(Input.Changed, ".setup-model-custom")
    def custom_model_changed(self, event: Input.Changed) -> None:
        if event.input.id is not None:
            prefix = event.input.id.removeprefix("setup-").split("-", 1)[0]
            select = self.query_one(
                f"#setup-{prefix}-model-select", FuzzySelect
            )
            if event.value and select.value != CUSTOM:
                with self.prevent(Select.Changed):
                    select.value = CUSTOM
                event.input.remove_class("hidden")
            self._sync_yaml_from_controls(prefix)

    @on(Select.Changed, ".setup-model-choice")
    def model_changed(self, event: Select.Changed) -> None:
        if event.select.id is None:
            return
        prefix = event.select.id.removeprefix("setup-").split("-", 1)[0]
        custom = self.query_one(f"#setup-{prefix}-model", Input)
        custom.set_class(event.value != CUSTOM, "hidden")
        record = self.model_catalogs.get(prefix, {}).get(str(event.value))
        if record is not None and record.model_provider is not None:
            model_type = "embedding" if prefix == "embedding" else "chat"
            if record.model_provider in supported_model_providers(model_type):
                self.query_one(
                    f"#setup-{prefix}-model-provider", Select
                ).value = record.model_provider
        self._sync_yaml_from_controls(prefix)

    @on(Select.Changed, ".setup-model-provider")
    def model_provider_changed(self, event: Select.Changed) -> None:
        if event.select.id is not None:
            prefix = event.select.id.removeprefix("setup-").split("-", 1)[0]
            self._sync_yaml_from_controls(prefix)

    @on(Select.Changed, ".setup-inference-provider")
    def provider_changed(self, event: Select.Changed) -> None:
        if event.select.id is not None:
            prefix = event.select.id.removeprefix("setup-").split("-", 1)[0]
            self._sync_yaml_from_controls(prefix)
            provider = (
                self.providers.get(str(event.value))
                if isinstance(event.value, str) and event.value != NONE
                else self._direct_model_config(prefix)
            )
            if provider is not None:
                self._request_model_load(prefix, provider)

    def _direct_model_config(self, prefix: str) -> ModelConfig | None:
        values = self._structured_values(prefix)
        if values is None:
            values = deepcopy(self._embedding_template)
        base = self.chat if prefix == "chat" else self._embedding_base
        try:
            return base.model_merge(
                dict_diff(self._initial_values[prefix], values)
            )
        except (ValidationError, ValueError):
            return None

    def _next_model_load_generation(self, prefix: str) -> int:
        self._model_load_generation[prefix] += 1
        return self._model_load_generation[prefix]

    async def _load_initial_models(
        self, chat_generation: int, embedding_generation: int
    ) -> None:
        await asyncio.gather(
            self._load_models(
                "chat", self._model_source("chat"), chat_generation
            ),
            self._load_models(
                "embedding",
                self._model_source("embedding"),
                embedding_generation,
            ),
        )

    def _model_source(
        self, prefix: str
    ) -> ModelConfig | InferenceProviderConfig | None:
        values = self._values[prefix] or self._embedding_template
        provider_name = values.get("inference_provider")
        if isinstance(provider_name, str) and provider_name in self.providers:
            return self.providers[provider_name]
        return self._direct_model_config(prefix)

    def _request_model_load(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig,
    ) -> None:
        generation = self._next_model_load_generation(prefix)
        self.run_worker(
            self._load_models(prefix, config, generation),
            group=f"setup-{prefix}-models",
            exclusive=True,
        )

    async def _load_models(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig | None,
        generation: int,
    ) -> None:
        if config is None:
            return
        try:
            models = await asyncio.to_thread(list_provider_models, config)
        except Exception as exc:  # noqa: BLE001
            if not self._model_load_current(prefix, generation):
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
        if not self._model_load_current(prefix, generation):
            return
        ordered = sort_provider_models(
            models, "embedding" if prefix == "embedding" else "chat"
        )
        catalog = {model.name: model for model in ordered}
        self.model_catalogs[prefix] = catalog
        self._set_model_options(prefix, catalog, self._choice_value(prefix))

    def _model_load_current(self, prefix: str, generation: int) -> bool:
        return (
            generation == self._model_load_generation[prefix]
            and self.is_mounted
            and self.query_one_optional(
                f"#setup-{prefix}-model-select", FuzzySelect
            )
            is not None
        )

    def _set_model_options(
        self,
        prefix: str,
        catalog: Mapping[str, ProviderModel],
        current: str,
    ) -> None:
        options = self._choice_options(tuple(catalog))
        if current and current not in catalog:
            options.insert(-1, (f"Not found: {current}", current))
        select = self.query_one(f"#setup-{prefix}-model-select", FuzzySelect)
        was_syncing = self._syncing
        self._syncing = True
        try:
            with self.prevent(Select.Changed):
                select.set_options(options)
                select.value = current or NONE
            self.query_one(f"#setup-{prefix}-model", Input).set_class(
                select.value != CUSTOM, "hidden"
            )
        finally:
            self._syncing = was_syncing

    @on(TextArea.Changed, ".setup-model-yaml")
    def yaml_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id is None:
            return
        prefix = event.text_area.id.removeprefix("setup-").split("-", 1)[0]
        self.query_one(
            f"#setup-{prefix}-advanced", AdvancedYamlEditor
        ).set_validation("neutral")

    def _parse(
        self, prefix: str
    ) -> tuple[bool, ChatModelConfig | EmbModelConfig | None]:
        advanced = self.query_one(
            f"#setup-{prefix}-advanced", AdvancedYamlEditor
        )
        editor = advanced.query_one(TextArea)
        try:
            base = self.chat if prefix == "chat" else self._embedding_base
            config = parse_setup_model_yaml(
                editor.text,
                ChatModelConfig if prefix == "chat" else EmbModelConfig,
                base,
                self._initial_values[prefix],
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
        """Post a selection when both complete config blocks are valid."""
        chat_ok, chat = self._parse("chat")
        embedding_ok, embedding = self._parse("embedding")
        if not chat_ok or not embedding_ok:
            return
        assert isinstance(chat, ChatModelConfig)
        assert embedding is None or isinstance(embedding, EmbModelConfig)
        self.post_message(self.Submitted(SetupModelSelection(chat, embedding)))
