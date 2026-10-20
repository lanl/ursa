"""Provider, credential, and MCP entry widgets for guided setup."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, suppress
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast
from urllib.parse import urlparse

import yaml
from pydantic import SecretStr, ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Input,
    Select,
    Static,
    Tab,
    Tabs,
    TextArea,
)

from ursa.cli.config import InferenceProviderConfig, interpolate_mcp_server
from ursa.cli.setup_config import (
    SetupError,
    reject_literal_secrets,
    validate_setup_name,
    validate_setup_url,
)
from ursa.cli.tui.widgets import AdvancedYamlEditor, FuzzySelect, HelpLabel
from ursa.util import get_keyring
from ursa.util.mcp import (
    ServerParameters,
    parse_server_parameters,
    serialize_server_parameters,
    transport,
)
from ursa.util.secrets import (
    KEYRING_SERVICE,
    SecretReference,
    SecretTemplate,
)

CUSTOM_PROVIDER_URL = "__ursa_custom_provider_url__"
PROVIDER_DEFAULT_URL = "__ursa_provider_default_url__"
MCP_TOKEN_NONE = "__ursa_mcp_token_none__"
API_KEY_ENV = "__ursa_api_key_env__"
SPINNER = "|"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEYRING_LOCK = Lock()

PROVIDER_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "litellm": "http://localhost:4000",
    "ollama": "http://localhost:11434",
}

ENTRY_HELP = {
    "provider-name": "The unique name used in URSA configuration.",
    "provider-url": "Choose a known endpoint or Other for a custom URL.",
    "mcp-name": "The unique name used in URSA configuration.",
    "mcp-transport": "How URSA connects to this MCP server.",
    "mcp-url": "The complete remote MCP URL.",
    "mcp-command": "The executable that starts a local MCP server.",
    "mcp-args": "Arguments passed to the local MCP executable.",
}


def _quiet(callback: Callable[..., Any], *args: Any) -> Any:
    with (
        _KEYRING_LOCK,
        open(os.devnull, "w", encoding="utf-8") as stderr,
        redirect_stderr(stderr),
        suppress(Exception),
    ):
        return callback(*args)
    return None


def _auth(config: ServerParameters) -> tuple[str, SecretTemplate | None]:
    headers = getattr(config, "headers", None) or {}
    for header, value in headers.items():
        if isinstance(value, SecretTemplate):
            return str(header), value
    return "Authorization", None


def _yaml_mapping(text: str) -> dict[str, Any]:
    try:
        values = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SetupError(f"Advanced YAML is invalid: {exc}") from None
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise SetupError("Advanced YAML must be a mapping.")
    return values


def _url_options() -> list[tuple[str, str]]:
    return [
        ("Provider default", PROVIDER_DEFAULT_URL),
        *(
            (f"{name} ({url})", url)
            for name, url in PROVIDER_DEFAULT_URLS.items()
        ),
        ("Other", CUSTOM_PROVIDER_URL),
    ]


@dataclass(frozen=True)
class CredentialValue:
    """A credential reference plus an optional unsaved keyring value."""

    reference: SecretReference | None = None
    pending: str | None = None
    username: str | None = None


class ApiKeyEntry(Vertical):
    def __init__(self, prefix: str, subject: str) -> None:
        super().__init__(id=f"{prefix}-api-key", classes="api-key-entry")
        self.prefix, self.subject, self.entry_name = prefix, subject, ""
        self.known: list[str] = []
        self.loaded: str | None = None
        self.generation = 0

    def _options(self, name: str) -> list[tuple[str, str]]:
        values = dict.fromkeys((*self.known, name))
        return [("None", MCP_TOKEN_NONE), ("ENV", API_KEY_ENV)] + [
            (item, item) for item in values if item
        ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            HelpLabel(
                "Keyring",
                "Choose a keyring entry, ENV, or None.",
                id=f"{self.prefix}-source-label",
                classes="credential-source-label",
            ),
            HelpLabel(
                "API Key",
                "Existing values load automatically and stay hidden unless Show is set.",
                id=f"{self.prefix}-value-label",
                classes="credential-value-label",
            ),
            classes="credential-label-row",
        )
        yield Horizontal(
            FuzzySelect(
                [("None", MCP_TOKEN_NONE)],
                value=MCP_TOKEN_NONE,
                allow_blank=False,
                id=f"{self.prefix}-source",
                classes="credential-source",
            ),
            Horizontal(
                Input(password=True, id=f"{self.prefix}-key"),
                Checkbox("Show", id=f"{self.prefix}-show"),
                id=f"{self.prefix}-fields",
                classes="credential-value api-key-row",
            ),
            id=f"{self.prefix}-row",
            classes="credential-row",
        )

    def configure(
        self,
        name: str,
        reference: SecretReference | None,
        known: list[str],
        no_key: bool = False,
    ) -> None:
        self.entry_name, self.known = name, known
        keyring_name = reference.keyring if reference else None
        if reference and reference.env:
            source = API_KEY_ENV
        elif no_key and not reference:
            source = MCP_TOKEN_NONE
        else:
            source = name if keyring_name in (None, True) else keyring_name
        select = self.query_one(f"#{self.prefix}-source", Select)
        with self.prevent(Select.Changed):
            select.set_options(self._options(name))
            select.value = source
        initial = (
            reference.env or ""
            if source == API_KEY_ENV and reference is not None
            else ""
        )
        self._source(source, initial)

    def _source(self, source: object, initial: str = "") -> None:
        none, env = source == MCP_TOKEN_NONE, source == API_KEY_ENV
        key = self.query_one(f"#{self.prefix}-key", Input)
        self.generation += 1
        self.loaded = None
        key.value = initial
        self.query_one(f"#{self.prefix}-fields").set_class(none, "hidden")
        show = self.query_one(f"#{self.prefix}-show", Checkbox)
        show.set_class(none or env, "hidden")
        key.password = not env and not show.value
        label = self.query_one(f"#{self.prefix}-value-label", HelpLabel)
        label.set_label(
            "Environmental Variable with API Key" if env else "API Key"
        )
        key.placeholder = (
            "Environment variable name"
            if env
            else f"Enter the API Key for the {self.subject}"
        )
        if not none and not env:
            key.placeholder = SPINNER
            self.run_worker(
                self._load(str(source), self.generation),
                group=f"{self.prefix}-secret",
                exclusive=True,
            )

    def rename(self, name: str) -> None:
        if not name:
            return
        select = self.query_one(f"#{self.prefix}-source", Select)
        source, old = select.value, self.entry_name
        self.entry_name = name
        with self.prevent(Select.Changed):
            select.set_options(self._options(name))
            select.value = name if source == old else source
        if source == old:
            self.loaded = None

    async def _load(self, username: str, generation: int) -> None:
        credential_store = get_keyring()
        secret = await asyncio.to_thread(
            _quiet,
            credential_store.get_password,
            KEYRING_SERVICE,
            username,
        )
        if generation != self.generation:
            return
        self.loaded = secret
        key = self.query_one(f"#{self.prefix}-key", Input)
        key.value = secret or ""
        key.placeholder = f"Enter the API Key for the {self.subject}"

    def values(self) -> CredentialValue:
        source = self.query_one(f"#{self.prefix}-source", Select).value
        value = self.query_one(f"#{self.prefix}-key", Input).value.strip()
        if source == MCP_TOKEN_NONE:
            return CredentialValue()
        if source == API_KEY_ENV:
            if not _ENV_NAME.fullmatch(value):
                raise SetupError("Enter a valid environment variable name.")
            return CredentialValue(SecretReference(env=value))
        username = str(source)
        reference = SecretReference(
            keyring=username if username != self.entry_name else True
        )
        pending = value if value and value != self.loaded else None
        return CredentialValue(reference, pending, username)

    @on(Select.Changed)
    def source_changed(self, event: Select.Changed) -> None:
        if event.select.id == f"{self.prefix}-source":
            self._source(event.value)

    @on(Checkbox.Changed)
    def show_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == f"{self.prefix}-show":
            key = self.query_one(f"#{self.prefix}-key", Input)
            key.password = not event.value


@dataclass(frozen=True)
class ProviderCandidate:
    draft_id: str
    original_name: str | None
    name: str
    stored: InferenceProviderConfig
    runtime: InferenceProviderConfig
    credential: CredentialValue


@dataclass(frozen=True)
class McpCandidate:
    draft_id: str
    original_name: str | None
    name: str
    stored: ServerParameters
    runtime: ServerParameters
    credential: CredentialValue


class EntryDraftEditor(Vertical):
    """A named draft whose form and complete YAML stay in sync."""

    kind = ""

    class Changed(Message):
        def __init__(self, editor: EntryDraftEditor) -> None:
            super().__init__()
            self.editor = editor

    def __init__(
        self,
        draft_id: str,
        original_name: str | None,
        config: InferenceProviderConfig | ServerParameters | None,
        known_keys: list[str],
    ) -> None:
        super().__init__(id=f"{draft_id}-editor", classes="entry-draft-editor")
        self.draft_id = self.prefix = draft_id
        self.original_name = original_name
        self.initial_name = self.draft_name = original_name or ""
        self.known_keys, self._initial = known_keys, config
        self._syncing = False
        self.header, self.template = "Authorization", "Bearer %s"
        if config is None:
            self._values: dict[str, Any] = {}
        elif self.kind == "provider":
            self._values = config.model_dump(mode="json", exclude_none=True)
            self._values.pop("ssl_verify", None)
        else:
            self._values = serialize_server_parameters(
                cast(ServerParameters, config),
                exclude_defaults=False,
                exclude_none=True,
                exclude_unset=False,
            )
            self.header, reference = _auth(cast(ServerParameters, config))
            if reference is not None:
                self.template = reference.template

    def _input(self, field: str) -> Input:
        return self.query_one(f"#{self.prefix}-{field}", Input)

    def _help(self, field: str, label: str) -> HelpLabel:
        return HelpLabel(
            label,
            ENTRY_HELP[f"{self.kind}-{field}"],
            id=f"{self.prefix}-{field}-label",
        )

    def compose(self) -> ComposeResult:
        yield self._help("name", "Name")
        yield Input(
            value=self.initial_name,
            max_length=15,
            id=f"{self.prefix}-name",
            classes="entry-name",
        )
        if self.kind == "provider":
            yield from self._provider_fields()
        else:
            yield from self._mcp_fields()
        yield AdvancedYamlEditor(
            self.prefix,
            yaml.safe_dump(self._values, sort_keys=False, allow_unicode=True),
            editor_classes=f"{self.kind}-draft-yaml",
            markup=False,
        )
        yield Static("", classes="draft-message form-message", markup=False)

    def _provider_fields(self) -> ComposeResult:
        base = self._values.get("base_url")
        if self._initial is None:
            selected = CUSTOM_PROVIDER_URL
        elif not base:
            selected = PROVIDER_DEFAULT_URL
        elif base in PROVIDER_DEFAULT_URLS.values():
            selected = base
        else:
            selected = CUSTOM_PROVIDER_URL
        yield self._help("url", "Base URL")
        yield Horizontal(
            FuzzySelect(
                _url_options(),
                value=selected,
                allow_blank=False,
                id=f"{self.prefix}-url",
                classes="provider-url",
            ),
            Input(
                value=(base or "") if selected == CUSTOM_PROVIDER_URL else "",
                id=f"{self.prefix}-url-custom",
                classes="provider-url-custom"
                + ("" if selected == CUSTOM_PROVIDER_URL else " hidden"),
            ),
            classes="provider-url-row"
            + (" custom-url" if selected == CUSTOM_PROVIDER_URL else ""),
        )
        yield ApiKeyEntry(f"{self.prefix}-key", "provider")

    def _mcp_fields(self) -> ComposeResult:
        kind = self._values.get("transport", "streamable_http")
        yield self._help("transport", "Transport")
        yield Select(
            [
                ("Remote HTTP", "streamable_http"),
                ("Remote SSE", "sse"),
                ("Local command", "stdio"),
            ],
            value=kind,
            allow_blank=False,
            id=f"{self.prefix}-transport",
        )
        yield Vertical(
            self._help("url", "Server URL"),
            Input(value=self._values.get("url", ""), id=f"{self.prefix}-url"),
            ApiKeyEntry(f"{self.prefix}-token", "MCP server"),
            classes="mcp-remote-fields" if kind != "stdio" else "hidden",
        )
        yield Vertical(
            self._help("command", "Command"),
            Input(self._values.get("command", ""), id=f"{self.prefix}-command"),
            self._help("args", "Arguments"),
            Input(
                shlex.join(self._values.get("args", [])),
                id=f"{self.prefix}-args",
            ),
            classes="mcp-local-fields" + (" hidden" if kind != "stdio" else ""),
        )

    def on_mount(self) -> None:
        reference = None
        no_key = self._initial is None
        if self._initial is not None:
            if self.kind == "provider":
                reference = getattr(self._initial, "api_key", None)
                base = getattr(self._initial, "base_url", None)
                no_key = bool(base) and urlparse(base).hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
            else:
                reference = _auth(cast(ServerParameters, self._initial))[1]
                no_key = reference is None
        self.credential.configure(
            self.initial_name,
            reference if isinstance(reference, SecretReference) else None,
            self.known_keys,
            no_key,
        )
        self._validate_yaml(update_controls=False)

    @property
    def credential(self) -> ApiKeyEntry:
        suffix = "key" if self.kind == "provider" else "token"
        return self.query_one(f"#{self.prefix}-{suffix}-api-key", ApiKeyEntry)

    @property
    def yaml_editor(self) -> AdvancedYamlEditor:
        return self.query_one(f"#{self.prefix}-advanced", AdvancedYamlEditor)

    def raw_name(self) -> str:
        field = self.query_one_optional(f"#{self.prefix}-name", Input)
        return (field.value if field is not None else self.draft_name).strip()

    def set_status(self, text: str, failed: bool = False) -> None:
        widget = self.query_one(".draft-message", Static)
        widget.update(text)
        widget.set_class(failed, "error")

    def _form_values(self) -> tuple[dict[str, Any], CredentialValue]:
        values = _yaml_mapping(
            self.query_one(f"#{self.prefix}-config-yaml", TextArea).text
        )
        if self.kind == "provider":
            return self._provider_form_values(values)
        return self._mcp_form_values(values)

    def _provider_form_values(
        self, values: dict[str, Any]
    ) -> tuple[dict[str, Any], CredentialValue]:
        values.pop("ssl_verify", None)
        selected = self.query_one(f"#{self.prefix}-url", Select).value
        raw = self._input("url-custom").value
        base = (
            None
            if selected == PROVIDER_DEFAULT_URL
            else validate_setup_url(
                raw if selected == CUSTOM_PROVIDER_URL else str(selected),
                "Base URL",
            )
        )
        if base is None:
            values.pop("base_url", None)
        else:
            values["base_url"] = base
        credential = self.credential.values()
        if credential.reference is None:
            values.pop("api_key", None)
        else:
            values["api_key"] = credential.reference.model_dump(
                mode="json", exclude_none=True
            )
        return values, credential

    def _mcp_form_values(
        self, values: dict[str, Any]
    ) -> tuple[dict[str, Any], CredentialValue]:
        kind = str(self.query_one(f"#{self.prefix}-transport", Select).value)
        values["transport"] = kind
        if kind == "stdio":
            values.update(
                command=self._input("command").value.strip(),
                args=shlex.split(self._input("args").value),
            )
            values.pop("url", None)
            values.pop("headers", None)
            return values, CredentialValue()
        values["url"] = validate_setup_url(self._input("url").value, "MCP URL")
        values.pop("command", None)
        values.pop("args", None)
        credential = self.credential.values()
        headers = dict(values.get("headers") or {})
        if credential.reference is None:
            headers.pop(self.header, None)
        else:
            headers[self.header] = {
                **credential.reference.model_dump(exclude_none=True),
                "template": self.template,
            }
        if headers:
            values["headers"] = headers
        else:
            values.pop("headers", None)
        return values, credential

    def _parse(self, values: dict[str, Any]) -> Any:
        if self.kind == "mcp":
            reject_literal_secrets(values)
            return parse_server_parameters(values)
        parsed = InferenceProviderConfig.model_validate(values)
        if isinstance(parsed.api_key, SecretStr):
            raise SetupError("API keys must use an ENV or keyring reference.")
        return parsed

    def _write_yaml_from_form(self) -> None:
        try:
            values, _ = self._form_values()
            self._parse(values)
        except (SetupError, ValidationError, ValueError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self.yaml_editor.set_text(
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
        )
        self.yaml_editor.set_validation("valid")

    def _validate_yaml(self, *, update_controls: bool = True) -> None:
        try:
            values = _yaml_mapping(
                self.query_one(f"#{self.prefix}-config-yaml", TextArea).text
            )
            values.pop("ssl_verify", None)
            parsed = self._parse(values)
        except (SetupError, ValidationError, ValueError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self.yaml_editor.set_validation("valid")
        if update_controls:
            self._syncing = True
            try:
                self._sync_controls(parsed)
            finally:
                self._syncing = False

    def _sync_controls(self, parsed: Any) -> None:
        if self.kind == "provider":
            base = parsed.base_url
            select = self.query_one(f"#{self.prefix}-url", Select)
            select.value = (
                PROVIDER_DEFAULT_URL
                if not base
                else base
                if base in PROVIDER_DEFAULT_URLS.values()
                else CUSTOM_PROVIDER_URL
            )
            if select.value == CUSTOM_PROVIDER_URL:
                self._input("url-custom").value = base
            self._toggle_fields(select.value)
            reference = (
                parsed.api_key
                if isinstance(parsed.api_key, SecretReference)
                else None
            )
            self.credential.configure(
                self.raw_name(), reference, self.known_keys, reference is None
            )
            return
        kind = transport(parsed)
        self.query_one(f"#{self.prefix}-transport", Select).value = kind
        if kind == "stdio":
            self._input("command").value = getattr(parsed, "command", "")
            self._input("args").value = shlex.join(getattr(parsed, "args", []))
        else:
            self._input("url").value = getattr(parsed, "url", "")
            self.header, reference = _auth(parsed)
            if reference is not None:
                self.template = reference.template
            self.credential.configure(
                self.raw_name(), reference, self.known_keys, reference is None
            )
        self._toggle_fields(kind)

    def _toggle_fields(self, value: object) -> None:
        if self.kind == "provider":
            custom = value == CUSTOM_PROVIDER_URL
            self.query_one(f"#{self.prefix}-url-custom").set_class(
                not custom, "hidden"
            )
            self.query_one(".provider-url-row").set_class(custom, "custom-url")
            return
        local = value == "stdio"
        self.query_one(".mcp-local-fields").set_class(not local, "hidden")
        self.query_one(".mcp-remote-fields").set_class(local, "hidden")

    def candidate(self) -> ProviderCandidate | McpCandidate:
        label = (
            "Provider name" if self.kind == "provider" else "MCP server name"
        )
        name = validate_setup_name(self.raw_name(), label)
        values, credential = self._form_values()
        stored = self._parse(values)
        identity = (self.draft_id, self.original_name, name)
        if self.kind == "provider":
            runtime_values = stored.model_dump(mode="python")
            if credential.pending:
                runtime_values["api_key"] = SecretStr(credential.pending)
            elif isinstance(stored.api_key, SecretReference):
                runtime_values["api_key"] = _quiet(stored.api_key.resolve, name)
            runtime = InferenceProviderConfig.model_validate(
                runtime_values
            ).resolve(name)
            return ProviderCandidate(*identity, stored, runtime, credential)
        runtime = interpolate_mcp_server(stored)
        headers = dict(getattr(runtime, "headers", None) or {})
        if credential.pending:
            headers[self.header] = self.template % credential.pending
        if headers:
            runtime = runtime.model_copy(update={"headers": headers})
        return McpCandidate(*identity, stored, runtime, credential)

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        if self._syncing:
            return
        if event.input.id == f"{self.prefix}-name":
            self.draft_name = event.value.strip()
            self.credential.rename(self.draft_name)
        else:
            self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if self._syncing:
            return
        expected = "url" if self.kind == "provider" else "transport"
        if event.select.id == f"{self.prefix}-{expected}":
            self._toggle_fields(event.value)
        self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(TextArea.Changed, ".advanced-yaml-editor")
    def yaml_changed(self) -> None:
        if self._syncing:
            return
        self.yaml_editor.set_validation("neutral")
        self.yaml_editor.schedule_validation(0.5, self._validate_yaml)
        self.post_message(self.Changed(self))


class ProviderDraftEditor(EntryDraftEditor):
    """A provider configuration draft."""

    kind = "provider"


class McpDraftEditor(EntryDraftEditor):
    """An MCP server configuration draft."""

    kind = "mcp"


class DynamicEntryTabs(Vertical):
    """Shared tab/add/remove shell for provider and MCP configuration."""

    def __init__(
        self,
        kind: str,
        entries: Mapping[str, Any],
        known_keys: list[str],
    ) -> None:
        super().__init__(id=f"{kind}-entries", classes="dynamic-entry-tabs")
        self.kind = kind
        self.known_keys = known_keys
        self._counter = 0
        self._editors: dict[str, EntryDraftEditor] = {}
        for name, config in entries.items():
            self._make_editor(name, config)
        if kind == "provider" and not self._editors:
            self._make_editor(None, None)

    def _make_editor(self, name: str | None, config: Any) -> EntryDraftEditor:
        self._counter += 1
        draft_id = f"{self.kind}-draft-{self._counter}"
        editor_type = (
            ProviderDraftEditor if self.kind == "provider" else McpDraftEditor
        )
        editor = editor_type(draft_id, name, config, self.known_keys)
        self._editors[draft_id] = editor
        return editor

    def _button(self, label: str, action: str, **kwargs: Any) -> Button:
        return Button(
            label,
            id=f"{self.kind}-{action}",
            classes=f"entry-{action}",
            compact=True,
            **kwargs,
        )

    def compose(self) -> ComposeResult:
        tabs = [
            Tab(self._label(editor), id=f"{draft_id}-tab")
            for draft_id, editor in self._editors.items()
        ]
        yield Horizontal(
            Tabs(*tabs, id=f"{self.kind}-tabs", classes="entry-tabs"),
            self._button("+", "add"),
            self._button("Remove", "remove", variant="error"),
            classes="entry-tab-row",
        )
        empty_id = f"{self.kind}-empty"
        empty = Static(
            "No MCP servers configured.", id=empty_id, classes="entry-empty"
        )
        yield ContentSwitcher(
            empty,
            *self._editors.values(),
            initial=(
                f"{next(iter(self._editors))}-editor"
                if self._editors
                else empty_id
            ),
            id=f"{self.kind}-drafts",
            classes="entry-drafts",
        )

    def on_mount(self) -> None:
        self._sync_remove()

    @staticmethod
    def _label(editor: EntryDraftEditor, status: str = "") -> str:
        return f"{status}{editor.raw_name() or 'New'}"

    def editors(self) -> list[EntryDraftEditor]:
        return list(self._editors.values())

    def raw_names(self) -> set[str]:
        return {
            name for editor in self.editors() if (name := editor.raw_name())
        }

    def active_editor(self) -> EntryDraftEditor | None:
        active = self.query_one(f"#{self.kind}-tabs", Tabs).active
        return (
            self._editors.get(active.removesuffix("-tab")) if active else None
        )

    def select(self, draft_id: str) -> None:
        if draft_id not in self._editors:
            return
        self.query_one(f"#{self.kind}-tabs", Tabs).active = f"{draft_id}-tab"
        switcher = self.query_one(f"#{self.kind}-drafts", ContentSwitcher)
        switcher.current = f"{draft_id}-editor"

    def select_name(self, name: str | None) -> None:
        target = next(
            (e.draft_id for e in self.editors() if e.original_name == name),
            next(iter(self._editors), None),
        )
        if target:
            self.select(target)

    def set_tab_status(self, draft_id: str, status: str) -> None:
        editor = self._editors[draft_id]
        tab = self.query_one(f"#{draft_id}-tab", Tab)
        tab.label = self._label(editor, status)

    def set_draft_status(
        self,
        draft_id: str,
        message: str,
        *,
        marker: str = "",
        failed: bool = False,
    ) -> None:
        """Update a draft's inline message and tab marker together."""
        editor = self._editors[draft_id]
        editor.set_status(message, failed)
        self.set_tab_status(draft_id, marker)

    def focus_error(self, draft_id: str) -> None:
        self.select(draft_id)
        name = self._editors[draft_id].query_one(".entry-name", Input)
        self.call_after_refresh(name.focus)

    def _sync_remove(self) -> None:
        self.query_one(f"#{self.kind}-remove", Button).disabled = (
            not self._editors
            or self.kind == "provider"
            and len(self._editors) == 1
        )

    async def _add(self) -> None:
        editor = self._make_editor(None, None)
        drafts = self.query_one(f"#{self.kind}-drafts", ContentSwitcher)
        await drafts.mount(editor)
        await self.query_one(f"#{self.kind}-tabs", Tabs).add_tab(
            Tab(self._label(editor), id=f"{editor.draft_id}-tab")
        )
        self.select(editor.draft_id)
        self._sync_remove()
        editor.query_one(".entry-name", Input).focus()

    async def _remove(self) -> None:
        editor = self.active_editor()
        if editor is None or (
            self.kind == "provider" and len(self._editors) == 1
        ):
            return
        draft_id = editor.draft_id
        index = list(self._editors).index(draft_id)
        del self._editors[draft_id]
        await self.query_one(f"#{self.kind}-tabs", Tabs).remove_tab(
            f"{draft_id}-tab"
        )
        await editor.remove()
        if self._editors:
            remaining = list(self._editors)
            self.select(remaining[min(index, len(remaining) - 1)])
        else:
            self.query_one(
                f"#{self.kind}-drafts", ContentSwitcher
            ).current = f"{self.kind}-empty"
        self._sync_remove()

    def candidates(
        self, reserved_names: set[str]
    ) -> list[ProviderCandidate | McpCandidate] | None:
        results: list[ProviderCandidate | McpCandidate] = []
        failures: list[str] = []
        names: set[str] = set()
        for editor in self.editors():
            self.set_tab_status(editor.draft_id, "")
            editor.set_status("")
            try:
                candidate = editor.candidate()
                if candidate.name in reserved_names:
                    raise SetupError(
                        f"The name '{candidate.name}' is already configured."
                    )
                if candidate.name in names:
                    raise SetupError(
                        f"The name '{candidate.name}' is duplicated on this page."
                    )
                names.add(candidate.name)
                results.append(candidate)
            except Exception as exc:
                message = str(exc)
                editor.set_status(message, True)
                self.set_tab_status(editor.draft_id, "✗ ")
                failures.append(editor.draft_id)
        if failures:
            self.focus_error(failures[0])
            return None
        return results

    @on(Tabs.TabActivated)
    def tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tabs.id != f"{self.kind}-tabs":
            return
        draft_id = (event.tab.id or "").removesuffix("-tab")
        if draft_id in self._editors:
            drafts = self.query_one(f"#{self.kind}-drafts", ContentSwitcher)
            drafts.current = f"{draft_id}-editor"

    @on(EntryDraftEditor.Changed)
    def editor_changed(self, event: EntryDraftEditor.Changed) -> None:
        event.stop()
        self.set_tab_status(event.editor.draft_id, "")
        event.editor.set_status("")

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        action = (event.button.id or "").removeprefix(f"{self.kind}-")
        if action not in {"add", "remove"}:
            return
        event.stop()
        await (self._add() if action == "add" else self._remove())
