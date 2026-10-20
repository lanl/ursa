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
from typing import Any
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
        options = [("None", MCP_TOKEN_NONE), ("ENV", API_KEY_ENV)]
        options += [
            (item, item) for item in dict.fromkeys((*known, name)) if item
        ]
        select = self.query_one(f"#{self.prefix}-source", Select)
        with self.prevent(Select.Changed):
            select.set_options(options)
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
        options = [("None", MCP_TOKEN_NONE), ("ENV", API_KEY_ENV)]
        options += [
            (item, item) for item in dict.fromkeys((*self.known, name)) if item
        ]
        with self.prevent(Select.Changed):
            select.set_options(options)
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
            self.query_one(
                f"#{self.prefix}-key", Input
            ).password = not event.value


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


class ProviderDraftEditor(Vertical):
    """A mounted provider draft whose form and complete YAML stay in sync."""

    class Changed(Message):
        def __init__(self, editor: ProviderDraftEditor) -> None:
            super().__init__()
            self.editor = editor

    def __init__(
        self,
        draft_id: str,
        original_name: str | None,
        config: InferenceProviderConfig | None,
        known_keys: list[str],
    ) -> None:
        super().__init__(id=f"{draft_id}-editor", classes="entry-draft-editor")
        self.draft_id = draft_id
        self.original_name = original_name
        self.initial_name = original_name or ""
        self.draft_name = self.initial_name
        self.known_keys = known_keys
        self._syncing = False
        self._initial = config
        self._values = (
            config.model_dump(mode="json", exclude_none=True)
            if config is not None
            else {}
        )
        self._values.pop("ssl_verify", None)

    @property
    def prefix(self) -> str:
        return self.draft_id

    def compose(self) -> ComposeResult:
        config = self._initial
        base = config.base_url if config is not None else None
        if config is None:
            selected = CUSTOM_PROVIDER_URL
        elif not base:
            selected = PROVIDER_DEFAULT_URL
        elif base in PROVIDER_DEFAULT_URLS.values():
            selected = base
        else:
            selected = CUSTOM_PROVIDER_URL
        yield HelpLabel(
            "Name",
            ENTRY_HELP["provider-name"],
            id=f"{self.prefix}-name-label",
        )
        yield Input(
            value=self.initial_name,
            max_length=15,
            id=f"{self.prefix}-name",
            classes="entry-name",
        )
        yield HelpLabel(
            "Base URL",
            ENTRY_HELP["provider-url"],
            id=f"{self.prefix}-url-label",
        )
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
                classes=(
                    "provider-url-custom"
                    if selected == CUSTOM_PROVIDER_URL
                    else "provider-url-custom hidden"
                ),
            ),
            classes=(
                "provider-url-row custom-url"
                if selected == CUSTOM_PROVIDER_URL
                else "provider-url-row"
            ),
        )
        yield ApiKeyEntry(f"{self.prefix}-key", "provider")
        yield AdvancedYamlEditor(
            self.prefix,
            yaml.safe_dump(self._values, sort_keys=False, allow_unicode=True),
            editor_classes="provider-draft-yaml",
            markup=False,
        )
        yield Static("", classes="draft-message form-message", markup=False)

    def on_mount(self) -> None:
        reference = self._initial.api_key if self._initial is not None else None
        base = self._initial.base_url if self._initial is not None else None
        local = bool(base) and urlparse(base).hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        self.credential.configure(
            self.initial_name,
            reference if isinstance(reference, SecretReference) else None,
            self.known_keys,
            local or self._initial is None,
        )
        self._validate_yaml(update_controls=False)

    @property
    def credential(self) -> ApiKeyEntry:
        return self.query_one(f"#{self.prefix}-key-api-key", ApiKeyEntry)

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

    def _url_value(self) -> str | None:
        selected = self.query_one(f"#{self.prefix}-url", Select).value
        if selected == PROVIDER_DEFAULT_URL:
            return None
        raw = self.query_one(f"#{self.prefix}-url-custom", Input).value
        return validate_setup_url(
            raw if selected == CUSTOM_PROVIDER_URL else str(selected),
            "Base URL",
        )

    def _form_values(self) -> tuple[dict[str, Any], CredentialValue]:
        values = _yaml_mapping(
            self.query_one(f"#{self.prefix}-config-yaml", TextArea).text
        )
        values.pop("ssl_verify", None)
        base = self._url_value()
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

    def _write_yaml_from_form(self) -> None:
        try:
            values, _ = self._form_values()
            InferenceProviderConfig.model_validate(values)
        except (SetupError, ValidationError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self._values = values
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
            parsed = InferenceProviderConfig.model_validate(values)
            if isinstance(parsed.api_key, SecretStr):
                raise SetupError(
                    "API keys must use an ENV or keyring reference."
                )
        except (SetupError, ValidationError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self._values = values
        self.yaml_editor.set_validation("valid")
        if not update_controls:
            return
        self._syncing = True
        try:
            base = parsed.base_url
            select = self.query_one(f"#{self.prefix}-url", Select)
            if not base:
                select.value = PROVIDER_DEFAULT_URL
            elif base in PROVIDER_DEFAULT_URLS.values():
                select.value = base
            else:
                select.value = CUSTOM_PROVIDER_URL
                self.query_one(f"#{self.prefix}-url-custom", Input).value = base
            self._toggle_url(select.value)
            reference = (
                parsed.api_key
                if isinstance(parsed.api_key, SecretReference)
                else None
            )
            self.credential.configure(
                self.raw_name(), reference, self.known_keys, reference is None
            )
        finally:
            self._syncing = False

    def _toggle_url(self, value: object) -> None:
        custom = value == CUSTOM_PROVIDER_URL
        self.query_one(f"#{self.prefix}-url-custom").set_class(
            not custom, "hidden"
        )
        self.query_one(".provider-url-row").set_class(custom, "custom-url")

    def candidate(self) -> ProviderCandidate:
        name = validate_setup_name(self.raw_name(), "Provider name")
        values, credential = self._form_values()
        stored = InferenceProviderConfig.model_validate(values)
        if isinstance(stored.api_key, SecretStr):
            raise SetupError("API keys must use an ENV or keyring reference.")
        runtime_values = stored.model_dump(mode="python")
        if credential.pending:
            runtime_values["api_key"] = SecretStr(credential.pending)
        elif isinstance(stored.api_key, SecretReference):
            runtime_values["api_key"] = _quiet(stored.api_key.resolve, name)
        runtime = InferenceProviderConfig.model_validate(
            runtime_values
        ).resolve(name)
        return ProviderCandidate(
            self.draft_id,
            self.original_name,
            name,
            stored,
            runtime,
            credential,
        )

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        if self._syncing:
            return
        if event.input.id == f"{self.prefix}-name":
            self.draft_name = event.value.strip()
            self.credential.rename(self.draft_name)
            self.post_message(self.Changed(self))
            return
        self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if self._syncing:
            return
        if event.select.id == f"{self.prefix}-url":
            self._toggle_url(event.value)
        self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(TextArea.Changed, ".provider-draft-yaml")
    def yaml_changed(self) -> None:
        if self._syncing:
            return
        self.yaml_editor.set_validation("neutral")
        self.yaml_editor.schedule_validation(0.5, self._validate_yaml)
        self.post_message(self.Changed(self))


class McpDraftEditor(Vertical):
    """A mounted MCP draft whose form and complete YAML stay in sync."""

    class Changed(Message):
        def __init__(self, editor: McpDraftEditor) -> None:
            super().__init__()
            self.editor = editor

    def __init__(
        self,
        draft_id: str,
        original_name: str | None,
        config: ServerParameters | None,
        known_keys: list[str],
    ) -> None:
        super().__init__(id=f"{draft_id}-editor", classes="entry-draft-editor")
        self.draft_id = draft_id
        self.original_name = original_name
        self.initial_name = original_name or ""
        self.draft_name = self.initial_name
        self.known_keys = known_keys
        self._initial = config
        self._syncing = False
        self.header = "Authorization"
        self.template = "Bearer %s"
        if config is None:
            self._values: dict[str, Any] = {}
        else:
            self._values = serialize_server_parameters(
                config,
                exclude_defaults=False,
                exclude_none=True,
                exclude_unset=False,
            )
            self.header, reference = _auth(config)
            if reference is not None:
                self.template = reference.template

    @property
    def prefix(self) -> str:
        return self.draft_id

    def compose(self) -> ComposeResult:
        values = self._values
        kind = values.get("transport", "streamable_http")
        yield HelpLabel(
            "Name", ENTRY_HELP["mcp-name"], id=f"{self.prefix}-name-label"
        )
        yield Input(
            value=self.initial_name,
            max_length=15,
            id=f"{self.prefix}-name",
            classes="entry-name",
        )
        yield HelpLabel(
            "Transport",
            ENTRY_HELP["mcp-transport"],
            id=f"{self.prefix}-transport-label",
        )
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
            HelpLabel(
                "Server URL",
                ENTRY_HELP["mcp-url"],
                id=f"{self.prefix}-url-label",
            ),
            Input(value=values.get("url", ""), id=f"{self.prefix}-url"),
            ApiKeyEntry(f"{self.prefix}-token", "MCP server"),
            classes=("mcp-remote-fields" if kind != "stdio" else "hidden"),
        )
        yield Vertical(
            HelpLabel(
                "Command",
                ENTRY_HELP["mcp-command"],
                id=f"{self.prefix}-command-label",
            ),
            Input(
                value=values.get("command", ""),
                id=f"{self.prefix}-command",
            ),
            HelpLabel(
                "Arguments",
                ENTRY_HELP["mcp-args"],
                id=f"{self.prefix}-args-label",
            ),
            Input(
                value=shlex.join(values.get("args", [])),
                id=f"{self.prefix}-args",
            ),
            classes=(
                "mcp-local-fields hidden"
                if kind != "stdio"
                else "mcp-local-fields"
            ),
        )
        yield AdvancedYamlEditor(
            self.prefix,
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
            editor_classes="mcp-draft-yaml",
            markup=False,
        )
        yield Static("", classes="draft-message form-message", markup=False)

    def on_mount(self) -> None:
        reference = None
        if self._initial is not None:
            _, reference = _auth(self._initial)
        self.credential.configure(
            self.initial_name,
            reference,
            self.known_keys,
            reference is None,
        )
        self._validate_yaml(update_controls=False)

    @property
    def credential(self) -> ApiKeyEntry:
        return self.query_one(f"#{self.prefix}-token-api-key", ApiKeyEntry)

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
        kind = str(self.query_one(f"#{self.prefix}-transport", Select).value)
        values["transport"] = kind
        if kind == "stdio":
            values["command"] = self.query_one(
                f"#{self.prefix}-command", Input
            ).value.strip()
            values["args"] = shlex.split(
                self.query_one(f"#{self.prefix}-args", Input).value
            )
            values.pop("url", None)
            values.pop("headers", None)
            return values, CredentialValue()
        values["url"] = validate_setup_url(
            self.query_one(f"#{self.prefix}-url", Input).value, "MCP URL"
        )
        values.pop("command", None)
        values.pop("args", None)
        credential = self.credential.values()
        headers = values.get("headers")
        headers = dict(headers) if isinstance(headers, dict) else {}
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

    def _write_yaml_from_form(self) -> None:
        try:
            values, _ = self._form_values()
            reject_literal_secrets(values)
            parse_server_parameters(values)
        except (SetupError, ValidationError, ValueError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self._values = values
        self.yaml_editor.set_text(
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
        )
        self.yaml_editor.set_validation("valid")

    def _validate_yaml(self, *, update_controls: bool = True) -> None:
        try:
            values = _yaml_mapping(
                self.query_one(f"#{self.prefix}-config-yaml", TextArea).text
            )
            reject_literal_secrets(values)
            parsed = parse_server_parameters(values)
        except (SetupError, ValidationError, ValueError) as exc:
            self.yaml_editor.set_validation("invalid", str(exc))
            return
        self._values = values
        self.yaml_editor.set_validation("valid")
        if not update_controls:
            return
        self._syncing = True
        try:
            kind = transport(parsed)
            self.query_one(f"#{self.prefix}-transport", Select).value = kind
            if kind == "stdio":
                self.query_one(
                    f"#{self.prefix}-command", Input
                ).value = getattr(parsed, "command", "")
                self.query_one(
                    f"#{self.prefix}-args", Input
                ).value = shlex.join(getattr(parsed, "args", []))
            else:
                self.query_one(f"#{self.prefix}-url", Input).value = getattr(
                    parsed, "url", ""
                )
                self.header, reference = _auth(parsed)
                if reference is not None:
                    self.template = reference.template
                self.credential.configure(
                    self.raw_name(),
                    reference,
                    self.known_keys,
                    reference is None,
                )
            self._toggle_transport(kind)
        finally:
            self._syncing = False

    def _toggle_transport(self, value: object) -> None:
        local = value == "stdio"
        self.query_one(".mcp-local-fields").set_class(not local, "hidden")
        self.query_one(".mcp-remote-fields").set_class(local, "hidden")

    def candidate(self) -> McpCandidate:
        name = validate_setup_name(self.raw_name(), "MCP server name")
        values, credential = self._form_values()
        reject_literal_secrets(values)
        stored = parse_server_parameters(values)
        runtime = interpolate_mcp_server(stored)
        headers = dict(getattr(runtime, "headers", None) or {})
        if credential.pending:
            headers[self.header] = self.template % credential.pending
        if headers:
            runtime = runtime.model_copy(update={"headers": headers})
        return McpCandidate(
            self.draft_id,
            self.original_name,
            name,
            stored,
            runtime,
            credential,
        )

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        if self._syncing:
            return
        if event.input.id == f"{self.prefix}-name":
            self.draft_name = event.value.strip()
            self.credential.rename(self.draft_name)
            self.post_message(self.Changed(self))
            return
        self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if self._syncing:
            return
        if event.select.id == f"{self.prefix}-transport":
            self._toggle_transport(event.value)
        self._write_yaml_from_form()
        self.post_message(self.Changed(self))

    @on(TextArea.Changed, ".mcp-draft-yaml")
    def yaml_changed(self) -> None:
        if self._syncing:
            return
        self.yaml_editor.set_validation("neutral")
        self.yaml_editor.schedule_validation(0.5, self._validate_yaml)
        self.post_message(self.Changed(self))


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
        self._editors: dict[str, ProviderDraftEditor | McpDraftEditor] = {}
        for name, config in entries.items():
            self._make_editor(name, config)
        if kind == "provider" and not self._editors:
            self._make_editor(None, None)

    def _make_editor(
        self, name: str | None, config: Any
    ) -> ProviderDraftEditor | McpDraftEditor:
        self._counter += 1
        draft_id = f"{self.kind}-draft-{self._counter}"
        editor: ProviderDraftEditor | McpDraftEditor
        if self.kind == "provider":
            editor = ProviderDraftEditor(
                draft_id, name, config, self.known_keys
            )
        else:
            editor = McpDraftEditor(draft_id, name, config, self.known_keys)
        self._editors[draft_id] = editor
        return editor

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Tabs(
                *(
                    Tab(self._label(editor), id=f"{draft_id}-tab")
                    for draft_id, editor in self._editors.items()
                ),
                id=f"{self.kind}-tabs",
                classes="entry-tabs",
            ),
            Button(
                "+",
                id=f"{self.kind}-add",
                classes="entry-add",
                compact=True,
            ),
            Button(
                "Remove",
                id=f"{self.kind}-remove",
                variant="error",
                classes="entry-remove",
                compact=True,
            ),
            classes="entry-tab-row",
        )
        children: list[Any] = [
            Static(
                "No MCP servers configured.",
                id=f"{self.kind}-empty",
                classes="entry-empty",
            )
        ]
        children.extend(self._editors.values())
        current = (
            f"{next(iter(self._editors))}-editor"
            if self._editors
            else f"{self.kind}-empty"
        )
        yield ContentSwitcher(
            *children,
            initial=current,
            id=f"{self.kind}-drafts",
            classes="entry-drafts",
        )

    def on_mount(self) -> None:
        self._sync_remove()

    @staticmethod
    def _label(
        editor: ProviderDraftEditor | McpDraftEditor, status: str = ""
    ) -> str:
        return f"{status}{editor.raw_name() or 'New'}"

    def editors(self) -> list[ProviderDraftEditor | McpDraftEditor]:
        return list(self._editors.values())

    def raw_names(self) -> set[str]:
        return {
            editor.raw_name() for editor in self.editors() if editor.raw_name()
        }

    def active_editor(self) -> ProviderDraftEditor | McpDraftEditor | None:
        active = self.query_one(f"#{self.kind}-tabs", Tabs).active
        if not active:
            return None
        return self._editors.get(active.removesuffix("-tab"))

    def select(self, draft_id: str) -> None:
        if draft_id not in self._editors:
            return
        self.query_one(f"#{self.kind}-tabs", Tabs).active = f"{draft_id}-tab"
        self.query_one(
            f"#{self.kind}-drafts", ContentSwitcher
        ).current = f"{draft_id}-editor"

    def select_name(self, name: str | None) -> None:
        target = next(
            (
                editor.draft_id
                for editor in self.editors()
                if editor.original_name == name
            ),
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
        self._editors[draft_id].set_status(message, failed)
        self.set_tab_status(draft_id, marker)

    def focus_error(self, draft_id: str) -> None:
        self.select(draft_id)
        self.call_after_refresh(
            self._editors[draft_id].query_one(".entry-name", Input).focus
        )

    def _sync_remove(self) -> None:
        count = len(self._editors)
        self.query_one(f"#{self.kind}-remove", Button).disabled = (
            count == 0 or self.kind == "provider" and count == 1
        )

    async def _add(self) -> None:
        editor = self._make_editor(None, None)
        switcher = self.query_one(f"#{self.kind}-drafts", ContentSwitcher)
        await switcher.mount(editor)
        await self.query_one(f"#{self.kind}-tabs", Tabs).add_tab(
            Tab(self._label(editor), id=f"{editor.draft_id}-tab")
        )
        self.select(editor.draft_id)
        self._sync_remove()
        editor.query_one(".entry-name", Input).focus()

    async def _remove(self) -> None:
        editor = self.active_editor()
        if (
            editor is None
            or self.kind == "provider"
            and len(self._editors) == 1
        ):
            return
        draft_id = editor.draft_id
        ids = list(self._editors)
        index = ids.index(draft_id)
        self._editors.pop(draft_id)
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
        names: dict[str, str] = {}
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
                names[candidate.name] = candidate.draft_id
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
            self.query_one(
                f"#{self.kind}-drafts", ContentSwitcher
            ).current = f"{draft_id}-editor"

    @on(ProviderDraftEditor.Changed)
    @on(McpDraftEditor.Changed)
    def editor_changed(
        self, event: ProviderDraftEditor.Changed | McpDraftEditor.Changed
    ) -> None:
        event.stop()
        self.set_tab_status(event.editor.draft_id, "")
        event.editor.set_status("")

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"{self.kind}-add":
            event.stop()
            await self._add()
        elif event.button.id == f"{self.kind}-remove":
            event.stop()
            await self._remove()
