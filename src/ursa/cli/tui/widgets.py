# ruff: noqa: TID251

"""Reusable widgets and modal screens for the Textual CLI."""

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import islice
from math import ceil
from pathlib import Path

import yaml
from pydantic import SecretStr, ValidationError
from rich.cells import cell_len, chop_cells
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Markdown,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.widgets._select import SelectCurrent, SelectOverlay
from textual.widgets.option_list import Option

from ursa.agents.base import URSA_VERSION
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    ModelConfig,
)
from ursa.cli.runtime import HITL
from ursa.cli.tui.agent_info import AgentDetails, ToolDetails, load_agent_tools
from ursa.cli.tui.helpers import _fuzzy_score
from ursa.cli.tui.tips import random_tip
from ursa.util.inference_providers import (
    ProviderModel,
    list_provider_models,
    sort_provider_models,
    supported_model_providers,
)


class PromptArea(TextArea):
    """A multiline editor whose bare Enter submits the current prompt."""

    BINDINGS = [
        Binding(
            "enter", "submit_prompt", "Submit prompt", show=False, priority=True
        ),
        Binding(
            "shift+enter",
            "insert_newline",
            "Insert newline",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+j",
            "insert_newline",
            "Insert newline",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+c", "clear_prompt", "Clear prompt", show=False, priority=True
        ),
        Binding(
            "up",
            "history_up",
            "Cursor or prompt history up",
            show=False,
            priority=True,
        ),
        Binding(
            "down",
            "history_down",
            "Cursor or prompt history down",
            show=False,
            priority=True,
        ),
        Binding(
            "alt+left,meta+left,alt+b",
            "cursor_word_left",
            "Cursor word left",
            show=False,
            priority=True,
        ),
        Binding(
            "alt+right,meta+right,alt+f",
            "cursor_word_right",
            "Cursor word right",
            show=False,
            priority=True,
        ),
        Binding(
            "@",
            "file_macro",
            "Choose workspace path",
            show=False,
            priority=True,
        ),
        Binding("#", "agent_macro", "Choose agent", show=False, priority=True),
        Binding(
            "/",
            "command_macro",
            "Open command picker",
            show=False,
            priority=True,
        ),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class MacroTyped(Message):
        """A macro trigger inserted by a real keyboard event."""

        def __init__(self, trigger: str, location: tuple[int, int]) -> None:
            super().__init__()
            self.trigger = trigger
            self.location = location

    def __init__(self) -> None:
        super().__init__(
            language="markdown",
            soft_wrap=True,
            tab_behavior="indent",
            placeholder="Ask URSA…  (@ files, # agents)",
            id="prompt",
        )
        self.prompt_history: list[str] = []
        self._history_index: int | None = None

    def on_mount(self) -> None:
        key = (
            "Shift+Enter"
            if self.app.preferred_newline_key == "shift+enter"
            else "Ctrl+J"
        )
        self.placeholder = f"Ask URSA…  (@ files, # agents, {key} newline)"

    def _remember(self, text: str) -> None:
        if text and (
            not self.prompt_history or self.prompt_history[-1] != text
        ):
            self.prompt_history.append(text)
        self._history_index = None

    def _load_history(self, index: int) -> None:
        self._history_index = index
        self.load_text(self.prompt_history[index])
        self.move_cursor((
            len(self.document.lines) - 1,
            len(self.document.lines[-1]),
        ))

    def action_submit_prompt(self) -> None:
        text = self.text.strip()
        if text:
            self._remember(text)
            self.post_message(self.Submitted(text))

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def action_clear_prompt(self) -> None:
        self._remember(self.text)
        self._history_index = len(self.prompt_history)
        self.load_text("")

    def action_history_up(self) -> None:
        if self.prompt_history and (
            not self.text or self.cursor_location[0] == 0
        ):
            index = (
                len(self.prompt_history)
                if self._history_index is None
                else self._history_index
            )
            self._load_history(max(0, index - 1))
            return
        self.action_cursor_up()

    def action_history_down(self) -> None:
        if self._history_index is not None:
            next_index = self._history_index + 1
            if next_index < len(self.prompt_history):
                self._load_history(next_index)
            else:
                self._history_index = None
                self.load_text("")
            return
        self.action_cursor_down()

    def _insert_macro(self, trigger: str) -> None:
        location = self.cursor_location
        self.insert(trigger)
        self.post_message(self.MacroTyped(trigger, location))

    def action_file_macro(self) -> None:
        self._insert_macro("@")

    def action_agent_macro(self) -> None:
        self._insert_macro("#")

    def action_command_macro(self) -> None:
        self._insert_macro("/")


class HotlistScreen(ModalScreen[str | None]):
    """Fuzzy-searchable picker overlaid above the prompt."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel picker", priority=True),
        Binding("up", "previous_choice", "Previous choice", priority=True),
        Binding("down", "next_choice", "Next choice", priority=True),
        Binding("enter", "select_choice", "Select choice", priority=True),
    ]

    def __init__(self, title: str, candidates: Sequence[str]) -> None:
        super().__init__()
        self.picker_title = title
        self.candidates = list(candidates)
        self.matches = list(candidates)

    def compose(self) -> ComposeResult:
        with Vertical(id="hotlist"):
            with Horizontal(id="hotlist-header"):
                yield Static(self.picker_title, id="hotlist-title")
                yield Static("Esc to Exit", id="hotlist-exit-hint")
            yield Input(placeholder="fzf search…", id="hotlist-query")
            yield OptionList(
                *(
                    Option(candidate, id=str(index))
                    for index, candidate in enumerate(self.matches)
                ),
                id="hotlist-options",
            )

    def on_mount(self) -> None:
        # Mounting can race app teardown, leaving children absent; a bare
        # query here would crash the app from inside the Mount dispatch.
        inputs = self.query(Input)
        if not inputs:
            return
        inputs.first().focus()
        self._highlight_first()

    def action_previous_choice(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_next_choice(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_select_choice(self) -> None:
        options = self.query_one(OptionList)
        if options.option_count:
            options.action_select()

    @on(Input.Changed)
    def filter_options(self, event: Input.Changed) -> None:
        ranked = []
        for index, candidate in enumerate(self.candidates):
            score = _fuzzy_score(event.value, candidate)
            if score is not None:
                ranked.append((-score, index, candidate))
        ranked.sort()
        self.matches = [candidate for _, _, candidate in ranked]
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options(self.matches)
        self._highlight_first()

    def _highlight_first(self) -> None:
        options = self.query(OptionList)
        if not options:
            return
        option_list = options.first()
        option_list.highlighted = 0 if option_list.option_count else None

    @on(OptionList.OptionSelected)
    def select_option(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_cancel(self) -> None:
        self.dismiss(None)


class FuzzySelectOverlay(SelectOverlay):
    """Select overlay whose type-to-search uses fuzzy matching."""

    def __init__(self, type_to_search: bool = True) -> None:
        super().__init__(type_to_search)
        self._source_options: list[Option] = []

    def set_source_options(self, options: Sequence[Option]) -> None:
        self._source_options = list(options)
        self.reset_search()

    def reset_search(self) -> None:
        self._search_query = ""
        self._show_matches()

    def _show_matches(self) -> None:
        ranked: list[tuple[int, int, Option]] = []
        for index, option in enumerate(self._source_options):
            prompt = option.prompt
            candidate = (
                prompt.plain if isinstance(prompt, Text) else str(prompt)
            )
            score = _fuzzy_score(self._search_query, candidate)
            if score is not None:
                ranked.append((-score, index, option))
        ranked.sort()
        self.clear_options()
        self.add_options(option for _, _, option in ranked)
        self.highlighted = 0 if self.option_count else None
        self.border_title = None
        self.query_ancestor(FuzzySelect).show_search_query(self._search_query)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "backspace":
            event.stop()
            event.prevent_default()
            self._search_query = self._search_query[:-1]
            self._show_matches()
        elif event.character is not None and event.is_printable:
            event.stop()
            event.prevent_default()
            self._search_query += event.character
            self._show_matches()

    def watch_has_focus(self, value: bool) -> None:
        if not value:
            self.reset_search()
        OptionList.watch_has_focus(self, value)

    def _find_search_match(self, query: str) -> int | None:
        matches: list[tuple[int, int]] = []
        for index, option in enumerate(self._options):
            prompt = option.prompt
            candidate = (
                prompt.plain if isinstance(prompt, Text) else str(prompt)
            )
            score = _fuzzy_score(query, candidate)
            if score is not None:
                matches.append((-score, index))
        return min(matches)[1] if matches else None

    def action_select(self) -> None:
        if self.highlighted is None:
            return
        option = self.get_option_at_index(self.highlighted)
        if not option.disabled and option.id is not None:
            self.post_message(self.UpdateSelection(int(option.id)))


class FuzzySelect(Select):
    """A Select with fuzzy type-to-search behavior."""

    def compose(self) -> ComposeResult:
        yield SelectCurrent(self.prompt)
        yield FuzzySelectOverlay(type_to_search=self._type_to_search).data_bind(
            compact=Select.compact
        )

    def _setup_options_renderables(self) -> None:
        options = [
            Option(prompt, id=str(index))
            for index, (prompt, _value) in enumerate(self._options)
        ]
        self.query_one(FuzzySelectOverlay).set_source_options(options)

    def show_search_query(self, query: str) -> None:
        current = self.query_one_optional(SelectCurrent)
        if current is None:
            return
        if query:
            current.update(query)
            return
        label = next(
            (prompt for prompt, value in self._options if value == self.value),
            self.NULL,
        )
        current.update(label)

    def _watch_expanded(self, expanded: bool) -> None:
        super()._watch_expanded(expanded)
        if not expanded:
            overlay = self.query_one_optional(FuzzySelectOverlay)
            if overlay is not None:
                overlay.reset_search()


class ThemeScreen(HotlistScreen):
    """Theme picker that previews highlighted themes over the current app."""

    def __init__(
        self,
        candidates: Sequence[str],
        initial_theme: str,
    ) -> None:
        super().__init__("Themes", candidates)
        self.initial_theme = initial_theme

    @on(OptionList.OptionHighlighted)
    def preview_theme(self, event: OptionList.OptionHighlighted) -> None:
        self.app.theme = str(event.option.prompt)

    def action_cancel(self) -> None:
        self.app.theme = self.initial_theme
        self.dismiss(None)


@dataclass(frozen=True)
class ModelSelection:
    chat: ChatModelConfig
    embedding: EmbModelConfig | None


class ModelFieldLabel(Horizontal):
    """Compact field label with an accented help affordance."""

    def __init__(self, label: str, help_text: str, *, id: str) -> None:
        super().__init__(id=id, classes="model-field-label")
        self.label = label
        self.tooltip = help_text

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="model-field-label-text")
        yield Static("[", classes="model-field-help-bracket")
        yield Static("?", classes="model-field-help-mark")
        yield Static("]", classes="model-field-help-bracket")


class ModelScreen(ModalScreen[ModelSelection | None]):
    """Configure chat and embedding model providers."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+enter", "apply", "Apply", priority=True),
    ]
    CUSTOM_VALUE = "__ursa_custom__"
    NONE_VALUE = "__ursa_none__"
    YAML_VALIDATION_DELAY = 0.8
    STRUCTURED_FIELDS = ("model", "model_provider", "inference_provider")

    @staticmethod
    def _validation_error_text(error: ValidationError) -> str:
        """Render every Pydantic error compactly in the bounded error panel."""
        details = error.errors(include_url=False, include_context=False)
        lines = [
            f"{len(details)} validation "
            f"{'error' if len(details) == 1 else 'errors'} for {error.title}"
        ]
        for detail in details:
            location = detail.get("loc", ())
            field = str(location[0]) if location else "configuration"
            lines.append(f"{field}: {detail['msg']}")
        return "\n".join(lines)

    FIELD_HELP = {
        "model": (
            "The model identifier exposed by the provider, such as gpt-5.4 "
            "or text-embedding-3-large. Not all models listed are valid chat "
            "or embedding models."
        ),
        "model-provider": (
            "The LangChain model integration used to create the client, such "
            "as openai, anthropic, google_genai, or ollama."
        ),
        "inference-provider": (
            "A named URSA inference provider supplying the endpoint, API key, "
            "and TLS settings for this model. Update your config files to add "
            "additional providers."
        ),
    }

    def __init__(
        self,
        providers: Mapping[str, InferenceProviderConfig],
        chat: ChatModelConfig,
        embedding: EmbModelConfig | None,
    ) -> None:
        super().__init__()
        self.providers = dict(providers)
        self.chat = chat
        self.embedding_was_configured = embedding is not None
        if embedding is None:
            direct_settings = (
                {
                    "base_url": chat.base_url,
                    "api_key": deepcopy(chat.api_key),
                    "ssl_verify": chat.ssl_verify,
                }
                if chat.inference_provider is None
                else {}
            )
            embedding = EmbModelConfig(
                model="",
                model_provider=chat.model_provider,
                inference_provider=chat.inference_provider,
                **direct_settings,
            )
            if embedding.inference_provider is not None:
                embedding = embedding.resolve_inference_provider(self.providers)
        self.embedding = embedding
        self.drafts: dict[str, ModelConfig] = {
            "chat": self.chat,
            "embedding": self.embedding,
        }
        self._yaml_values = {
            prefix: self._configured_values(config)
            for prefix, config in self.drafts.items()
        }
        self.model_catalogs: dict[str, dict[str, ProviderModel]] = {}
        self._model_load_generation = {"chat": 0, "embedding": 0}
        self._yaml_timers: dict[str, Timer] = {}
        self._syncing_controls = False

    @classmethod
    def _choice_options(cls, values: Sequence[str]) -> list[tuple[str, str]]:
        return [
            ("None", cls.NONE_VALUE),
            *((value, value) for value in values),
            ("Other…", cls.CUSTOM_VALUE),
        ]

    @classmethod
    def _editable_choice(
        cls,
        prefix: str,
        field: str,
        values: Sequence[str],
        current: str,
    ) -> tuple[Select, Input]:
        choices = tuple(dict.fromkeys(value for value in values if value))
        listed = current in choices
        selected = current if listed else cls.NONE_VALUE
        if current and not listed:
            selected = cls.CUSTOM_VALUE
        select = FuzzySelect(
            cls._choice_options(choices),
            value=selected,
            allow_blank=False,
            id=f"{prefix}-{field}",
            classes="model-editable-choice",
        )
        custom = Input(
            value="" if listed else current,
            id=f"{prefix}-{field}-custom",
            classes="model-custom-choice" + (" hidden" if listed else ""),
        )
        return select, custom

    def _model_fields(
        self,
        prefix: str,
        config: ModelConfig,
    ) -> ComposeResult:
        options = [
            ("None (direct model config)", self.NONE_VALUE),
            *(
                (
                    f"{name} ({getattr(provider, 'base_url', None) or 'default'})",
                    name,
                )
                for name, provider in sorted(self.providers.items())
            ),
        ]
        selected_provider = config.inference_provider
        if (
            selected_provider is None
            and prefix == "embedding"
            and not self.embedding_was_configured
        ):
            selected_provider = next(iter(self.providers), None)
        selected_provider = selected_provider or self.NONE_VALUE
        yield self._field_label("Model", "model", prefix)
        model_select, custom_model = self._editable_choice(
            prefix, "model-name", (config.model,), config.model
        )
        yield model_select
        yield custom_model
        yield self._field_label("Model provider", "model-provider", prefix)
        model_providers = supported_model_providers(
            "embedding" if prefix == "embedding" else "chat"
        )
        selected_model_provider = config.model_provider or self.NONE_VALUE
        yield Select(
            [
                ("None", self.NONE_VALUE),
                *((provider, provider) for provider in model_providers),
            ],
            value=selected_model_provider,
            allow_blank=False,
            id=f"{prefix}-model-provider",
        )
        yield self._field_label(
            "Inference provider", "inference-provider", prefix
        )
        yield Select(
            options,
            value=selected_provider,
            allow_blank=False,
            id=f"{prefix}-inference-provider",
            classes="inference-provider-choice",
        )

    @classmethod
    def _field_label(
        cls, label: str, field: str, prefix: str
    ) -> ModelFieldLabel:
        return ModelFieldLabel(
            label,
            cls.FIELD_HELP[field],
            id=f"{prefix}-{field}-label",
        )

    def compose(self) -> ComposeResult:
        with Vertical(classes="settings-dialog"):
            yield Static("Models", classes="settings-title")
            with TabbedContent():
                with TabPane("Chat", id="chat-model-tab"):
                    yield from self._model_fields(
                        "chat",
                        self.chat,
                    )
                    yield from self._advanced_editor("chat", self.chat)
                with TabPane("Embedding", id="embedding-model-tab"):
                    yield from self._model_fields(
                        "embedding",
                        self.embedding,
                    )
                    yield from self._advanced_editor(
                        "embedding", self.embedding
                    )
            with Horizontal(classes="settings-actions"):
                yield Button("Cancel", id="model-cancel")
                yield Button("Apply", id="model-apply", variant="primary")

    def _advanced_editor(
        self, prefix: str, config: ModelConfig
    ) -> ComposeResult:
        with Collapsible(
            title="Advanced",
            collapsed=True,
            id=f"{prefix}-advanced",
            classes="model-advanced",
        ):
            yield TextArea(
                self._dump_yaml(config),
                language="yaml",
                show_line_numbers=True,
                tab_behavior="indent",
                id=f"{prefix}-config-yaml",
                classes="model-yaml-editor",
            )
            yield Static(
                "",
                id=f"{prefix}-yaml-error",
                classes="model-yaml-error hidden",
            )

    @staticmethod
    def _configured_values(config: ModelConfig) -> dict:
        """Return editable values without materializing inherited defaults."""
        dumped = config.model_dump(
            mode="json",
            exclude_unset=True,
            context={"include_defaults": False},
        )
        dumped.setdefault("model", config.model)
        return dumped

    @classmethod
    def _dump_yaml(cls, config: ModelConfig) -> str:
        return cls._dump_yaml_values(cls._configured_values(config))

    @staticmethod
    def _dump_yaml_values(values: Mapping) -> str:
        return yaml.safe_dump(
            dict(values),
            sort_keys=False,
            allow_unicode=True,
        )

    def _yaml_text(self, prefix: str) -> str:
        return self._dump_yaml_values(self._yaml_values[prefix])

    def _update_yaml_values(self, prefix: str, config: ModelConfig) -> None:
        """Patch structured fields without reordering the YAML mapping."""
        configured = self._configured_values(config)
        values = self._yaml_values[prefix]
        if config.inference_provider is not None:
            values.pop("base_url", None)
        for field in self.STRUCTURED_FIELDS:
            if field in configured:
                values[field] = configured[field]
            else:
                values.pop(field, None)

    def on_mount(self) -> None:
        self.query_one("#chat-model-name", Select).focus()
        chat_generation = self._next_model_load_generation("chat")
        embedding_generation = self._next_model_load_generation("embedding")
        self.run_worker(
            self._load_initial_models(chat_generation, embedding_generation),
            group="model-discovery",
            exclusive=True,
        )

    def on_unmount(self) -> None:
        for timer in self._yaml_timers.values():
            timer.stop()
        self._yaml_timers.clear()

    def _next_model_load_generation(self, prefix: str) -> int:
        self._model_load_generation[prefix] += 1
        return self._model_load_generation[prefix]

    async def _load_initial_models(
        self, chat_generation: int, embedding_generation: int
    ) -> None:
        await self._load_models("chat", self.chat, chat_generation)
        await self._load_models(
            "embedding", self.embedding, embedding_generation
        )

    def _request_model_load(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig,
    ):
        generation = self._next_model_load_generation(prefix)
        return self.run_worker(
            self._load_models(prefix, config, generation),
            group=f"{prefix}-models",
            exclusive=True,
        )

    async def _load_models(
        self,
        prefix: str,
        config: ModelConfig | InferenceProviderConfig,
        generation: int,
    ) -> None:
        try:
            models = await asyncio.to_thread(list_provider_models, config)
        except Exception as exc:  # noqa: BLE001
            if generation != self._model_load_generation[prefix]:
                return
            self.model_catalogs[prefix] = {}
            current = self._choice_value(prefix, "model-name")
            self._set_model_options(prefix, {}, current)
            self.notify(
                f"Unable to list models: {exc}",
                title="Model discovery",
                severity="warning",
            )
            return
        if generation != self._model_load_generation[prefix]:
            return
        models = sort_provider_models(
            models, "embedding" if prefix == "embedding" else "chat"
        )
        catalog = {model.name: model for model in models}
        self.model_catalogs[prefix] = catalog
        current = self._choice_value(prefix, "model-name")
        self._set_model_options(prefix, catalog, current)

    def _set_model_options(
        self,
        prefix: str,
        catalog: Mapping[str, ProviderModel],
        current: str,
    ) -> None:
        """Render a catalog while retaining an unavailable current model."""
        options = self._choice_options(tuple(catalog))
        if current and current not in catalog:
            options.insert(-1, (f"Not found: {current}", current))
        select = self.query_one(f"#{prefix}-model-name", Select)
        was_syncing = self._syncing_controls
        self._syncing_controls = True
        try:
            with self.prevent(Select.Changed):
                select.set_options(options)
                select.value = current or self.NONE_VALUE
            custom = self.query_one(f"#{prefix}-model-name-custom", Input)
            custom.set_class(select.value != self.CUSTOM_VALUE, "hidden")
        finally:
            self._syncing_controls = was_syncing

    def _choice_value(self, prefix: str, field: str) -> str:
        select = self.query_one(f"#{prefix}-{field}", Select)
        if select.value == self.NONE_VALUE:
            return ""
        if select.value != self.CUSTOM_VALUE:
            return str(select.value)
        return self.query_one(f"#{prefix}-{field}-custom", Input).value.strip()

    @on(Select.Changed, ".model-editable-choice")
    def show_custom_choice(self, event: Select.Changed) -> None:
        if event.select.id is None:
            return
        custom = self.query_one(f"#{event.select.id}-custom", Input)
        custom.set_class(event.value != self.CUSTOM_VALUE, "hidden")

        prefix = event.select.id.removesuffix("-model-name")
        record = self.model_catalogs.get(prefix, {}).get(str(event.value))
        if record is None or record.model_provider is None:
            self._structured_controls_changed(prefix)
            return
        provider = record.model_provider
        provider_select = self.query_one(f"#{prefix}-model-provider", Select)
        model_type = "embedding" if prefix == "embedding" else "chat"
        if provider in supported_model_providers(model_type):
            provider_select.value = provider
        self._structured_controls_changed(prefix)

    @on(Select.Changed, ".inference-provider-choice")
    def update_models_for_inference_provider(
        self, event: Select.Changed
    ) -> None:
        if event.select.id is None or not isinstance(event.value, str):
            return
        prefix = event.select.id.removesuffix("-inference-provider")
        self._structured_controls_changed(prefix)
        config = self.providers.get(event.value, self.drafts[prefix])
        self._request_model_load(prefix, config)

    @on(Select.Changed, "#chat-model-provider, #embedding-model-provider")
    def model_provider_changed(self, event: Select.Changed) -> None:
        if event.select.id is not None:
            self._structured_controls_changed(
                event.select.id.removesuffix("-model-provider")
            )

    @on(Input.Changed, ".model-custom-choice")
    def custom_model_changed(self, event: Input.Changed) -> None:
        if event.input.id is not None:
            self._structured_controls_changed(
                event.input.id.removesuffix("-model-name-custom")
            )

    def _structured_controls_changed(self, prefix: str) -> None:
        if self._syncing_controls or not self.is_mounted:
            return
        original = self.drafts[prefix]
        try:
            updated = self._settings(prefix, original)
        except ValueError:
            return
        self.drafts[prefix] = updated
        self._update_yaml_values(prefix, updated)
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        text = self._yaml_text(prefix)
        if editor.text != text:
            self._syncing_controls = True
            try:
                with self.prevent(TextArea.Changed):
                    editor.text = text
            finally:
                self._syncing_controls = False
        self._validate_yaml(prefix, update_controls=False)

    @on(TextArea.Changed, ".model-yaml-editor")
    def yaml_changed(self, event: TextArea.Changed) -> None:
        if self._syncing_controls or event.text_area.id is None:
            return
        prefix = event.text_area.id.removesuffix("-config-yaml")
        if event.text_area.text == self._yaml_text(prefix):
            self._set_yaml_state(prefix, "valid")
            return
        self._set_yaml_state(prefix, "neutral")
        timer = self._yaml_timers.pop(prefix, None)
        if timer is not None:
            timer.stop()
        self._yaml_timers[prefix] = self.set_timer(
            self.YAML_VALIDATION_DELAY,
            lambda: self._validate_yaml(prefix, update_controls=True),
        )

    def _validate_yaml(
        self, prefix: str, *, update_controls: bool
    ) -> ModelConfig | None:
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        config_type = ChatModelConfig if prefix == "chat" else EmbModelConfig
        try:
            values = yaml.safe_load(editor.text)
            if not isinstance(values, dict):
                raise ValueError("Configuration must be a YAML mapping")
            validation_values = deepcopy(values)
            current_api_key = self.drafts[prefix].api_key
            if isinstance(current_api_key, SecretStr) and values.get(
                "api_key"
            ) == str(current_api_key):
                validation_values["api_key"] = deepcopy(current_api_key)
            config = config_type.model_validate(validation_values)
            if not config.model.strip() and (prefix == "chat" or config.model):
                model_type = "Chat" if prefix == "chat" else "Embedding"
                raise ValueError(f"{model_type} model must not be blank")
            if (
                config.inference_provider is not None
                and config.inference_provider not in self.providers
            ):
                raise ValueError(
                    f"Unknown inference_provider '{config.inference_provider}'"
                )
        except (yaml.YAMLError, ValueError) as exc:
            error = (
                self._validation_error_text(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            self._set_yaml_state(prefix, "invalid", error)
            return None
        self.drafts[prefix] = config
        self._yaml_values[prefix] = deepcopy(values)
        self._set_yaml_state(prefix, "valid")
        if update_controls:
            self._update_controls_from_config(prefix, config)
        return config

    def _set_yaml_state(self, prefix: str, state: str, error: str = "") -> None:
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        editor.remove_class("yaml-valid", "yaml-invalid")
        if state != "neutral":
            editor.add_class(f"yaml-{state}")
        message = self.query_one(f"#{prefix}-yaml-error", Static)
        message.update(Text(error))
        message.set_class(not error, "hidden")

    def _update_controls_from_config(
        self, prefix: str, config: ModelConfig
    ) -> None:
        inference_select = self.query_one(
            f"#{prefix}-inference-provider", Select
        )
        previous_inference_provider = inference_select.value
        self._syncing_controls = True
        try:
            with self.prevent(Select.Changed):
                catalog = self.model_catalogs.get(prefix, {})
                self._set_model_options(prefix, catalog, config.model)

                model_provider = config.model_provider or self.NONE_VALUE
                provider_select = self.query_one(
                    f"#{prefix}-model-provider", Select
                )
                model_type = "embedding" if prefix == "embedding" else "chat"
                options = [
                    ("None", self.NONE_VALUE),
                    *(
                        (provider, provider)
                        for provider in supported_model_providers(model_type)
                    ),
                ]
                available = {value for _, value in options}
                if model_provider not in available:
                    options.append((
                        f"Not found: {model_provider}",
                        model_provider,
                    ))
                provider_select.set_options(options)
                provider_select.value = model_provider
                inference_provider = (
                    config.inference_provider or self.NONE_VALUE
                )
                inference_select.value = inference_provider
        finally:
            self._syncing_controls = False
        if inference_provider != previous_inference_provider:
            provider_config = self.providers.get(
                str(inference_provider), config
            )
            self._request_model_load(prefix, provider_config)

    def _settings(self, prefix: str, original: ModelConfig) -> ModelConfig:
        model_name = self._choice_value(prefix, "model-name")
        model_provider_value = self.query_one(
            f"#{prefix}-model-provider", Select
        ).value
        model_provider = (
            str(model_provider_value)
            if isinstance(model_provider_value, str)
            and model_provider_value != self.NONE_VALUE
            else ""
        )
        advertised = self.model_catalogs.get(prefix, {}).get(model_name)
        if (
            not model_provider
            and advertised is not None
            and advertised.model_provider is not None
        ):
            model_provider = advertised.model_provider
        provider_value = self.query_one(
            f"#{prefix}-inference-provider", Select
        ).value
        inference_provider = (
            provider_value
            if isinstance(provider_value, str)
            and provider_value != self.NONE_VALUE
            else ""
        )
        dumped = original.model_dump(mode="python")
        configured = {
            name: deepcopy(value)
            for name, value in dumped.items()
            if name in original.model_fields_set
        }
        for field in ("model", "model_provider", "inference_provider"):
            configured.pop(field, None)
        if inference_provider:
            # A named provider owns its endpoint. This mirrors ModelConfig's
            # merge semantics when switching away from a direct endpoint.
            configured.pop("base_url", None)
        updates = {
            "model": model_name,
            "inference_provider": inference_provider or None,
        }
        if (
            "model_provider" in original.model_fields_set
            or model_provider != original.model_provider
        ):
            updates["model_provider"] = model_provider or None
        return type(original).model_validate({**configured, **updates})

    @on(Button.Pressed, "#model-apply")
    def apply(self) -> None:
        self.action_apply()

    def action_apply(self) -> None:
        try:
            for prefix in ("chat", "embedding"):
                editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
                draft = self.drafts[prefix]
                # If YAML has not diverged from the draft, fold in any control
                # event still queued in Textual before doing final validation.
                if editor.text == self._yaml_text(prefix):
                    configured = self._settings(prefix, draft)
                    self.drafts[prefix] = configured
                    self._update_yaml_values(prefix, configured)
                    editor.text = self._yaml_text(prefix)
        except ValueError as exc:
            self.notify(
                str(exc),
                title="Model not changed",
                severity="error",
                timeout=10,
                markup=False,
            )
            return

        chat = self._validate_yaml("chat", update_controls=False)
        embedding = self._validate_yaml("embedding", update_controls=False)
        if chat is None or embedding is None:
            error = next(
                (
                    str(message.content)
                    for message in self.query(".model-yaml-error")
                    if str(message.content)
                ),
                "Invalid YAML configuration",
            )
            self.notify(
                error,
                title="Model not changed",
                severity="error",
                timeout=10,
                markup=False,
            )
            return
        if not chat.model:
            self.notify("Chat model is required", severity="error")
            return
        assert isinstance(chat, ChatModelConfig)
        if not embedding.model:
            embedding = None
        assert embedding is None or isinstance(embedding, EmbModelConfig)
        self.dismiss(ModelSelection(chat, embedding))

    @on(Button.Pressed, "#model-cancel")
    def cancel_button(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        expanded = list(self.query("FuzzySelect.-expanded"))
        if expanded:
            for select in expanded:
                select.expanded = False
            return
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is None:
            return
        ancestors = set(event.widget.ancestors_with_self)
        for select in self.query("FuzzySelect.-expanded"):
            if select not in ancestors:
                select.expanded = False


class InformationScreen(ModalScreen[None]):
    """Scrollable command output displayed without leaving the application."""

    BINDINGS = [
        Binding("escape,q", "close", "Close", priority=True),
        Binding("up", "scroll_up", "Scroll up"),
        Binding("down", "scroll_down", "Scroll down"),
        Binding("home", "scroll_home", "Scroll to top"),
        Binding("end", "scroll_end", "Scroll to bottom"),
        Binding("pageup", "page_up", "Page up"),
        Binding("pagedown", "page_down", "Page down"),
    ]

    def __init__(
        self,
        title: str,
        content: str,
        *,
        config_yaml: str | None = None,
    ) -> None:
        super().__init__()
        self.screen_title = title
        self.content = content
        self.config_yaml = config_yaml

    def compose(self) -> ComposeResult:
        with Vertical(id="information"):
            yield Static(self.screen_title, id="information-title")
            if self.config_yaml is None:
                yield VerticalScroll(
                    Markdown(self.content), id="information-body"
                )
            else:
                with TabbedContent(id="status-tabs"):
                    with TabPane("Status", id="status-summary-tab"):
                        yield VerticalScroll(
                            Markdown(self.content), id="information-body"
                        )
                    with TabPane("Config", id="status-config-tab"):
                        yield Static(
                            "Read only — select text to copy",
                            id="status-config-readonly",
                        )
                        yield TextArea(
                            self.config_yaml,
                            language="yaml",
                            read_only=True,
                            show_line_numbers=True,
                            id="status-config-yaml",
                        )

    def action_close(self) -> None:
        self.dismiss(None)

    def _scroll_view(self) -> VerticalScroll:
        return self.query_one("#information-body", VerticalScroll)

    def action_scroll_up(self) -> None:
        self._scroll_view().action_scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll_view().action_scroll_down()

    def action_scroll_home(self) -> None:
        self._scroll_view().action_scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll_view().action_scroll_end()

    def action_page_up(self) -> None:
        self._scroll_view().action_page_up()

    def action_page_down(self) -> None:
        self._scroll_view().action_page_down()


def _markdown_cell(value: object) -> str:
    """Escape a value for a compact Markdown table cell."""
    return str(value).replace("`", "\\`").replace("|", "\\|").replace("\n", " ")


def _tool_markdown(tool: ToolDetails) -> str:
    """Render expanded tool metadata inspired by the legacy Rich report."""
    return "\n\n".join(
        filter(
            None,
            (
                tool.description,
                "\n".join([
                    "| Setting | Value |",
                    "|---|---|",
                    *(
                        [
                            "| Source | "
                            f"`MCP: {_markdown_cell(tool.mcp_server)}` |"
                        ]
                        if tool.mcp_server
                        else []
                    ),
                    f"| Class | `{_markdown_cell(tool.class_name)}` |",
                    f"| Args schema | `{_markdown_cell(tool.schema_name)}` |",
                    "| Return directly | "
                    f"`{_markdown_cell(tool.return_direct)}` |",
                ]),
                (
                    "\n".join([
                        "### Arguments",
                        "",
                        "| Name | Type | Required | Description |",
                        "|---|---|---|---|",
                        *(
                            "| "
                            f"`{_markdown_cell(argument.name)}` | "
                            f"`{_markdown_cell(argument.type_name)}` | "
                            f"{'yes' if argument.required else 'no'} | "
                            f"{_markdown_cell(argument.description)} |"
                            for argument in tool.arguments
                        ),
                    ])
                    if tool.arguments
                    else ""
                ),
            ),
        )
    )


class AgentToolDetails(Collapsible):
    """A tool card which builds its Markdown only when first expanded."""

    def __init__(self, tool: ToolDetails) -> None:
        title = (
            f"{tool.name} (mcp: {tool.mcp_server})"
            if tool.mcp_server
            else tool.name
        )
        super().__init__(
            title=title,
            collapsed=True,
            classes="agent-tool",
        )
        self.tool = tool
        self._details_mounted = False

    async def on_collapsible_expanded(
        self, event: Collapsible.Expanded
    ) -> None:
        if event.collapsible is not self or self._details_mounted:
            return
        self._details_mounted = True
        try:
            await self.query_one(Collapsible.Contents).mount(
                Markdown(_tool_markdown(self.tool))
            )
        except BaseException:
            self._details_mounted = False
            raise


class AgentsScreen(InformationScreen):
    """Tabbed descriptions and configured-tool details for all agents."""

    def __init__(self, agents: tuple[AgentDetails, ...], hitl: HITL) -> None:
        super().__init__("Agents", "")
        self.agents = agents
        self.hitl = hitl
        self._tool_loads_started: set[str] = set()
        self._tool_loading_frames: dict[int, int] = {}
        self._tool_loading_timers: dict[int, Timer] = {}
        self._tool_panes_pending_render: set[int] = set(range(1, len(agents)))
        self._tool_activations_started: set[int] = set()

    def compose(self) -> ComposeResult:
        with Vertical(id="information"):
            yield Static(self.screen_title, id="information-title")
            with TabbedContent(id="agents-tabs"):
                for index, agent in enumerate(self.agents):
                    with TabPane(f"#{agent.name}", id=f"agent-tab-{index}"):
                        with VerticalScroll(classes="agent-details"):
                            yield Markdown(agent.description)
                            if agent.config:
                                yield Markdown(
                                    "\n".join([
                                        "### Configuration",
                                        "",
                                        "| Option | Value |",
                                        "|---|---|",
                                        *(
                                            f"| `{_markdown_cell(key)}` | "
                                            f"`{_markdown_cell(value)}` |"
                                            for key, value in agent.config
                                        ),
                                    ])
                                )
                            yield Static(
                                "Configured tools", classes="agent-tools-title"
                            )
                            with Vertical(
                                id=f"agent-tools-{index}",
                                classes="agent-tools",
                            ):
                                if index == 0:
                                    yield from self._tool_widgets(agent)
                                else:
                                    yield Static(
                                        "Select this tab to display its tools.",
                                        classes="agent-tools-empty",
                                    )

    @staticmethod
    def _tool_widgets(agent: AgentDetails):
        if not agent.tools_loaded and not agent.tools:
            yield Static(
                "Tools have not yet been loaded.",
                classes="agent-tools-empty",
            )
        elif agent.tool_error:
            yield Static(
                Text("Unable to load tools: " + agent.tool_error),
                classes="agent-tools-error",
            )
            return
        elif not agent.tools:
            yield Static("No configured tools.", classes="agent-tools-empty")
        for tool in agent.tools:
            yield AgentToolDetails(tool)

    @on(TabbedContent.TabActivated, "#agents-tabs")
    def _load_active_agent_tools(
        self, event: TabbedContent.TabActivated
    ) -> None:
        if event.pane.id is None:
            return
        index = int(event.pane.id.rsplit("-", 1)[-1])
        agent = self.agents[index]
        if index in self._tool_activations_started or (
            agent.tools_loaded and index not in self._tool_panes_pending_render
        ):
            return
        self._tool_activations_started.add(index)
        self.run_worker(
            self._activate_agent_tools(index),
            group=f"agent-tools-activation-{index}",
        )

    async def _activate_agent_tools(self, index: int) -> None:
        agent = self.agents[index]
        try:
            if index in self._tool_panes_pending_render:
                if not await self._render_agent_tools_safely(index, agent):
                    return
                self._tool_panes_pending_render.discard(index)
                # Rendering may replace this snapshot with a terminal error.
                agent = self.agents[index]
            if agent.tools_loaded or agent.name in self._tool_loads_started:
                return
            self._tool_loads_started.add(agent.name)
            container = self.query_one(f"#agent-tools-{index}", Vertical)
            if not agent.tools:
                await container.remove_children()
            loading = Static("Fetching tools.", classes="agent-tools-loading")
            await container.mount(
                loading,
                before=container.children[0] if container.children else None,
            )
            self._tool_loading_frames[index] = 1
            self._tool_loading_timers[index] = self.set_interval(
                0.3, lambda: self._advance_tool_loading(index)
            )
            container.scroll_visible(animate=False, top=True, immediate=True)
            await self._hydrate_tools(index)
        finally:
            self._tool_activations_started.discard(index)

    def _advance_tool_loading(self, index: int) -> None:
        loading = self.query(f"#agent-tools-{index} .agent-tools-loading")
        frame = self._tool_loading_frames.get(index)
        if not loading or frame is None:
            return
        loading.first(Static).update(f"Fetching tools{'.' * (frame + 1)}")
        self._tool_loading_frames[index] = (frame + 1) % 3

    def _stop_tool_loading(self, index: int) -> None:
        timer = self._tool_loading_timers.pop(index, None)
        if timer is not None:
            timer.stop()
        self._tool_loading_frames.pop(index, None)

    async def _hydrate_tools(self, index: int) -> None:
        agent = self.agents[index]
        try:
            tools = await load_agent_tools(self.hitl, agent.name)
            updated = replace(
                agent,
                tools=tools,
                tools_loaded=True,
                tool_error="",
            )
        except asyncio.CancelledError:
            self._tool_loads_started.discard(agent.name)
            self._stop_tool_loading(index)
            raise
        except Exception as exc:  # keep the browser usable on provider failure
            updated = replace(
                agent,
                tools=(),
                tools_loaded=True,
                tool_error=f"{type(exc).__name__}: {exc}",
            )
        agents = list(self.agents)
        agents[index] = updated
        self.agents = tuple(agents)
        if not self.is_mounted:
            self._stop_tool_loading(index)
            return
        tabs = self.query_one("#agents-tabs", TabbedContent)
        if tabs.active != f"agent-tab-{index}":
            # Rendering a large collection of Markdown tool cards is UI-thread
            # work. Do not freeze whichever tab the user switched to merely to
            # update an invisible pane.
            self._tool_panes_pending_render.add(index)
            self._stop_tool_loading(index)
            return
        try:
            rendered = await self._render_agent_tools_safely(index, updated)
        finally:
            self._stop_tool_loading(index)
        if not rendered:
            self._tool_panes_pending_render.add(index)

    async def _render_agent_tools(
        self, index: int, agent: AgentDetails
    ) -> bool:
        container = self.query_one(f"#agent-tools-{index}", Vertical)
        widgets = iter(self._tool_widgets(agent))
        tab_id = f"agent-tab-{index}"
        if self.query_one("#agents-tabs", TabbedContent).active != tab_id:
            return False
        first_batch = list(islice(widgets, 1))
        if self.query_one("#agents-tabs", TabbedContent).active != tab_id:
            return False
        with self.app.batch_update():
            await container.remove_children()
            await container.mount(*first_batch)
        while True:
            if self.query_one("#agents-tabs", TabbedContent).active != tab_id:
                return False
            batch = list(islice(widgets, 1))
            if not batch:
                break
            with self.app.batch_update():
                await container.mount(*batch)
            # Let queued key and tab events preempt a large tool collection.
            await asyncio.sleep(0)
        if self.query_one("#agents-tabs", TabbedContent).active != tab_id:
            return False
        container.scroll_visible(animate=False, top=True, immediate=True)
        return True

    async def _render_agent_tools_safely(
        self, index: int, agent: AgentDetails
    ) -> bool:
        try:
            return await self._render_agent_tools(index, agent)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed = replace(
                agent,
                tools=(),
                tools_loaded=True,
                tool_error=f"{type(exc).__name__}: {exc}",
            )
            agents = list(self.agents)
            agents[index] = failed
            self.agents = tuple(agents)
            self._tool_panes_pending_render.add(index)
            container = self.query_one(f"#agent-tools-{index}", Vertical)
            await container.remove_children()
            await container.mount(*self._tool_widgets(failed))
            self._tool_panes_pending_render.discard(index)
            return True

    def _scroll_view(self) -> VerticalScroll:
        tabs = self.query_one(TabbedContent)
        return self.query_one(f"#{tabs.active} .agent-details", VerticalScroll)


class WelcomeBanner(Vertical):
    """URSA logo, active configuration snapshot, and a concise usage tip."""

    LOGO = r"""  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/"""

    def __init__(self, hitl: HITL) -> None:
        super().__init__(id="welcome")
        self.hitl = hitl
        workspace = Path(self.hitl.workspace).resolve()
        try:
            relative = workspace.relative_to(Path.home())
            self.workspace_text = str(Path("~") / relative)
        except ValueError:
            self.workspace_text = str(workspace)
        self.version_text = f"v{URSA_VERSION}"
        self.tip = ""

    @staticmethod
    def _fit_middle(text: str, width: int) -> str:
        if width <= 0 or cell_len(text) <= width:
            return text
        if width == 1:
            return "…"
        available = width - 1
        left = available // 3
        right = available - left
        prefix = chop_cells(text, left)[0]
        suffix = chop_cells(text[::-1], right)[0][::-1]
        return f"{prefix}…{suffix}"

    def _fit_metadata(self) -> None:
        version = self.query_one("#welcome-version", Static)
        workspace_row = self.query_one("#welcome-workspace-row")
        workspace = self.query_one("#welcome-workspace", Static)
        version.update(
            Text(
                self._fit_middle(
                    self.version_text, version.content_region.width
                )
            )
        )
        row_width = workspace_row.content_region.width
        inline = (
            cell_len("Workspace") + 2 + cell_len(self.workspace_text)
            <= row_width
        )
        workspace_row.set_class(inline, "workspace-inline")
        workspace_row.set_class(not inline, "workspace-stacked")
        workspace_width = row_width - 11 if inline else row_width
        workspace.update(
            Text(self._fit_middle(self.workspace_text, workspace_width))
        )

    def on_mount(self) -> None:
        self.tip = random_tip(
            self.app,
            (type(self.app), PromptArea, HotlistScreen),
        )
        self.query_one("#welcome-tip", Static).update(f"Tip: {self.tip}")
        self._fit_metadata()

    def on_resize(self) -> None:
        self._fit_metadata()

    def _config_snapshot(self) -> Text:
        embedding = self.hitl.config.emb_model
        return Text(
            "\n".join([
                f"LLM        {self.hitl.config.llm_model.pretty_repr()}",
                f"Embedding  {embedding.pretty_repr() if embedding else 'none'}",
                f"Group      {getattr(self.hitl, 'group', None) or 'default'}",
            ])
        )

    def refresh_config(self) -> None:
        """Refresh the displayed runtime configuration snapshot."""
        self.query_one("#welcome-config-values", Static).update(
            self._config_snapshot()
        )

    def compose(self) -> ComposeResult:
        with Horizontal(id="welcome-top"):
            with Vertical(id="welcome-logo"):
                with Vertical(id="welcome-logo-stack"):
                    yield Static(self.LOGO, id="welcome-logo-art")
                    yield Static(self.version_text, id="welcome-version")
            with Vertical(id="welcome-config"):
                with Vertical(id="welcome-workspace-row"):
                    yield Static("Workspace", id="welcome-workspace-label")
                    yield Static(
                        Text(self.workspace_text),
                        id="welcome-workspace",
                    )
                yield Static(
                    self._config_snapshot(), id="welcome-config-values"
                )
        yield Static(
            f"Tip: {self.tip}",
            id="welcome-tip",
        )


class MessageCard(Static):
    def __init__(self, role: str, content: str) -> None:
        super().__init__(classes=f"message-card {role}")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "assistant":
            yield Static("URSA", classes="message-role")
        yield Markdown(self.content, classes="message-body")

    @on(Markdown.TableOfContentsUpdated)
    def remove_trailing_markdown_margin(
        self, event: Markdown.TableOfContentsUpdated
    ) -> None:
        blocks = list(event.markdown.children)
        if blocks:
            blocks[-1].styles.margin = 0


class ToolMessage(Horizontal):
    """A neutral transcript entry for application-level activity."""

    def __init__(self, content: str) -> None:
        super().__init__(classes="tool-message")
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static("●", classes="tool-message-mark")
        yield Static(Text(self.content), classes="tool-message-body")


class ActivityIndicator(Horizontal):
    """Animated, event-driven status for one conversation turn."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        super().__init__(classes="activity")
        self._frame = 0
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Static(self.FRAMES[0], classes="activity-spinner")
        yield Static("Thinking…", classes="activity-text")
        yield Static("", classes="activity-done-mark")

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.08, self._advance)

    def _advance(self) -> None:
        self.query_one(".activity-spinner", Static).update(
            self.FRAMES[self._frame]
        )
        self._frame = (self._frame + 1) % len(self.FRAMES)

    def update_message(self, message: str) -> None:
        message = " ".join(str(message).split())
        if message:
            self.query_one(".activity-text", Static).update(
                Text(message[-500:])
            )

    def finish(self, *, elapsed: float, tokens: int) -> None:
        if self._timer is not None:
            self._timer.pause()
        self.query_one(".activity-spinner", Static).update("")
        if elapsed <= 30:
            self.query_one(".activity-text", Static).update("")
            self.query_one(".activity-done-mark", Static).update("")
            self.remove_class("done")
            self.add_class("hidden")
            return
        seconds = ceil(elapsed)
        if seconds < 60:
            duration = f"{seconds}s"
        else:
            minutes, seconds = divmod(seconds, 60)
            duration = f"{minutes}m {seconds:02d}s"
        self.query_one(".activity-text", Static).update(
            f"Done in {duration} and {tokens:,} tokens"
        )
        self.query_one(".activity-done-mark", Static).update("✓")
        self.remove_class("hidden")
        self.add_class("done")
