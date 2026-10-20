"""Guided first-run configuration for URSA."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, suppress
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import yaml
from keyring.errors import PasswordDeleteError
from pydantic import SecretStr, TypeAdapter, ValidationError
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Footer,
    Input,
    Select,
    Static,
    TextArea,
)

from ursa.cli.config import (
    ChatModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
    deep_interp_env,
    load_config_file,
    save_config_file,
)
from ursa.cli.tui.widgets import (
    AdvancedYamlEditor,
    FuzzySelect,
    HelpLabel,
    ModelEditor,
    ModelSelection,
)
from ursa.util import get_keyring
from ursa.util.crossplatform import default_user_config_path
from ursa.util.health import (
    check_config_health,
    probe_chat_config,
    probe_embedding_model,
    probe_inference_provider,
    probe_mcp_server,
    run_probe_with_deadline,
)
from ursa.util.http import inject_truststore_into_ssl
from ursa.util.mcp import (
    ServerParameters,
    transport,
)
from ursa.util.secrets import (
    KEYRING_SERVICE,
    SecretReference,
    SecretTemplate,
    keyring_usernames,
)

CUSTOM_PROVIDER_URL = "__ursa_custom_provider_url__"
PROVIDER_DEFAULT_URL = "__ursa_provider_default_url__"
MCP_TOKEN_NONE = "__ursa_mcp_token_none__"
API_KEY_ENV = "__ursa_api_key_env__"
SPINNER = "|"
VALIDATING = "?"
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROVIDER_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "litellm": "http://localhost:4000",
    "ollama": "http://localhost:11434",
}
KEYRING_LOCK = Lock()
SERVER_PARAMETERS = TypeAdapter(ServerParameters)


class SetupError(ValueError):
    pass


def _quiet(callback: Callable[..., Any], *args: Any) -> Any:
    with (
        KEYRING_LOCK,
        open(os.devnull, "w", encoding="utf-8") as stderr,
        redirect_stderr(stderr),
        suppress(Exception),
    ):
        return callback(*args)
    return None


def _name(value: str, noun: str) -> str:
    value = value.strip()
    if not NAME.fullmatch(value):
        raise SetupError(
            f"{noun} must start with a letter and contain only letters, numbers, '-' or '_'."
        )
    return value


def _url(value: str, noun: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SetupError(f"{noun} must be a complete HTTP URL.")
    return value


def _friendly(exc: Exception, mcp: bool = False) -> str:
    messages, current = [], exc
    while current:
        text = str(current).strip()
        if text and text not in messages:
            messages.append(text)
        current = current.__cause__ or current.__context__
    joined = " ".join(messages).lower()
    if any(word in joined for word in ("401", "403", "unauthorized")):
        return (
            "The MCP server rejected the auth key."
            if mcp
            else "The API key was rejected. Check its value and permissions."
        )
    if "404" in joined or "not found" in joined:
        return "The endpoint was not found. Check the URL."
    if "timed out" in joined:
        return "The connection timed out. Check that the service is reachable."
    if any(word in joined for word in ("connection refused", "connecterror")):
        return "URSA could not connect. Check the address, network, and VPN."
    return messages[0] if messages else type(exc).__name__


def _server(name: str, value: Any) -> ServerParameters:
    """Parse an MCP entry with the same canonical type used by UrsaConfig."""
    try:
        return SERVER_PARAMETERS.validate_python(deepcopy(value))
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(f"MCP server '{name}' is invalid: {exc}") from None


def _server_values(config: ServerParameters) -> dict[str, Any]:
    return {
        "transport": transport(config),
        **config.model_dump(mode="json", exclude_unset=True),
    }


def _auth(config: ServerParameters) -> tuple[str, SecretTemplate | None]:
    headers = getattr(config, "headers", None) or {}
    for header, value in headers.items():
        reference = SecretTemplate.maybe_validate(value)
        if isinstance(reference, SecretTemplate):
            return str(header), reference
    return "Authorization", None


@dataclass(frozen=True)
class CredentialValue:
    """A credential reference plus an optional unsaved keyring value."""

    reference: SecretReference | None = None
    pending: str | None = None
    username: str | None = None


def _load_config(path: Path) -> tuple[UrsaConfig, bool]:
    """Load setup data once into the application's canonical config model."""
    try:
        raw = (
            deepcopy(load_config_file(path, interpolate=False))
            if path.is_file()
            else {}
        )
        providers = raw.get("inference_providers", {})
        if not isinstance(providers, dict):
            raise ValueError("Existing inference_providers must be a mapping.")
        for value in providers.values():
            if isinstance(value, dict):
                value.pop("ssl_verify", None)
        config = UrsaConfig.model_validate(raw)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise SetupError(f"Could not load configuration: {exc}") from None

    # UrsaConfig supplies an OpenAI default for normal CLI use. Setup only
    # presents providers the user actually selected.
    explicit = {
        str(name): config.inference_providers[str(name)] for name in providers
    }
    config = config.model_copy(update={"inference_providers": explicit})
    if overlap := sorted(set(explicit) & set(config.mcp_servers)):
        raise SetupError(f"Names used by both provider and MCP: {overlap}")
    return config, "llm_model" in raw


def _save(path: Path, config: UrsaConfig, pending: Mapping[str, str]) -> Path:
    if not config.inference_providers or not config.llm_model.model:
        raise SetupError("A provider and language model are required.")
    UrsaConfig.model_validate(config.editable_values())
    credential_store = get_keyring()
    previous = {
        username: credential_store.get_password(KEYRING_SERVICE, username)
        for username in pending
    }
    written: list[str] = []
    try:
        for username, secret in pending.items():
            credential_store.set_password(KEYRING_SERVICE, username, secret)
            written.append(username)
        runtime = UrsaConfig.model_validate(
            deep_interp_env(config.editable_values())
        )
        implicit = (
            runtime.inference_providers.keys()
            - config.inference_providers.keys()
        )
        for name in implicit:
            runtime.inference_providers.pop(name)
        failures = check_config_health(runtime)
        if failures:
            failed = ", ".join(".".join(path) for path in failures)
            raise SetupError(f"Configuration checks failed: {failed}")
        return save_config_file(path, config)
    except BaseException as exc:
        for username in reversed(written):
            try:
                if previous[username] is None:
                    credential_store.delete_password(KEYRING_SERVICE, username)
                else:
                    credential_store.set_password(
                        KEYRING_SERVICE, username, previous[username]
                    )
            except PasswordDeleteError:
                pass
            except Exception as rollback_error:
                exc.add_note(
                    f"Could not restore keyring entry '{username}': "
                    f"{rollback_error}"
                )
        raise


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
        initial = reference.env if source == API_KEY_ENV and reference else ""
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
            if not ENV_NAME.fullmatch(value):
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


SetupAction = Enum(
    "SetupAction",
    {"EXIT": "exit", "TERMINAL": "terminal", "DASHBOARD": "dashboard"},
)


class SetupApp(App[SetupAction]):
    CSS_PATH = "setup.tcss"
    MCP_STRUCTURED = ("transport", "url", "command", "args")
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("ctrl+q", "cancel", "Quit", priority=True),
    ]
    HELP = {
        "provider-name": "The unique name used in URSA configuration.",
        "provider-url": "Choose a known endpoint or Other for a custom URL.",
        "mcp-name": "The unique name used in URSA configuration.",
        "mcp-transport": "How URSA connects to this MCP server.",
        "mcp-url": "The complete remote MCP URL.",
        "mcp-command": "The executable that starts a local MCP server.",
        "mcp-args": "Arguments passed to the local MCP executable.",
    }
    DONE = {
        "setup-launch-terminal": SetupAction.TERMINAL,
        "setup-launch-dashboard": SetupAction.DASHBOARD,
        "setup-exit": SetupAction.EXIT,
    }

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = (path or default_user_config_path()).expanduser()
        self.config, self.llm_configured = _load_config(self.path)
        self.page, self.history = "welcome", []
        self.editing = {"provider": None, "mcp": None}
        self.model_ok = {"llm": None, "embedding": None}
        self.pending: dict[str, str] = {}
        self.mcp_auth = ("Authorization", "Bearer %s")

    @classmethod
    def _field(cls, label: str, key: str, widget: Any) -> tuple[Any, Any]:
        return HelpLabel(label, cls.HELP[key], id=f"{key}-label"), widget

    @staticmethod
    def _buttons(*specs: tuple) -> Horizontal:
        return Horizontal(
            *(
                Button(label, id=id, variant=variant, disabled=disabled)
                for label, id, variant, disabled in specs
            ),
            classes="setup-actions",
        )

    def _page(
        self,
        name: str,
        title: str,
        *content: Any,
        copy: str = "",
        actions: tuple = (),
    ) -> VerticalScroll:
        children = [Static(title, classes="page-title")]
        if copy:
            children.append(Static(copy, classes="page-copy"))
        children.extend(content)
        if actions:
            children.append(self._buttons(*actions))
        return VerticalScroll(
            *children, id=f"{name}-page", classes="setup-page"
        )

    def _entry_row(self, kind: str) -> Horizontal:
        noun = "provider" if kind == "provider" else "MCP server"
        return Horizontal(
            Select(
                [],
                allow_blank=True,
                prompt=f"No {noun} configured",
                id=f"{kind}-configured",
            ),
            Button("Add", id=f"{kind}-add"),
            Button("Remove", id=f"{kind}-remove", variant="error"),
            classes="manage-actions",
        )

    def compose(self) -> ComposeResult:
        provider_fields = (
            self._entry_row("provider"),
            *self._field(
                "Name",
                "provider-name",
                Input(max_length=15, id="provider-name"),
            ),
            *self._field(
                "Base URL",
                "provider-url",
                Horizontal(
                    FuzzySelect(
                        self._url_options(),
                        value=PROVIDER_DEFAULT_URLS["openai"],
                        allow_blank=False,
                        id="provider-url",
                    ),
                    Input(id="provider-url-custom", classes="hidden"),
                    id="provider-url-row",
                ),
            ),
            ApiKeyEntry("provider-key", "provider"),
            Static("", id="provider-error", classes="form-message"),
        )
        mcp_fields = (
            self._entry_row("mcp"),
            *self._field(
                "Name", "mcp-name", Input(max_length=15, id="mcp-name")
            ),
            *self._field(
                "Transport",
                "mcp-transport",
                Select(
                    [
                        ("Remote HTTP", "streamable_http"),
                        ("Remote SSE", "sse"),
                        ("Local command", "stdio"),
                    ],
                    value="streamable_http",
                    allow_blank=False,
                    id="mcp-transport",
                ),
            ),
            Vertical(
                *self._field("Server URL", "mcp-url", Input(id="mcp-url")),
                ApiKeyEntry("mcp-token", "MCP server"),
                id="mcp-remote-fields",
            ),
            Vertical(
                *self._field("Command", "mcp-command", Input(id="mcp-command")),
                *self._field("Arguments", "mcp-args", Input(id="mcp-args")),
                id="mcp-local-fields",
                classes="hidden",
            ),
            AdvancedYamlEditor(
                "mcp", "{}\n", editor_classes="mcp-yaml-editor", markup=False
            ),
            Static("○ Not tested", id="mcp-validation-status", markup=False),
        )
        pages = (
            self._page(
                "welcome",
                "Welcome to URSA",
                Static(f"Configuration: {self.path}", classes="path-box"),
                copy=(
                    "URSA found your existing user configuration. This guide will preserve unrelated settings while you review each setup step."
                    if self.path.is_file()
                    else "This guide will connect URSA to a model provider, choose your models, and optionally add MCP tools. Nothing is written until the final Save button."
                ),
                actions=(("Begin setup", "welcome-next", "primary", False),),
            ),
            self._page(
                "provider",
                "Connect an inference provider",
                *provider_fields,
                copy="URSA verifies the API key by asking the provider for its model list. The key is hidden and is not written to YAML.",
                actions=(
                    ("Back", "provider-back", "default", False),
                    ("Continue to models", "provider-next", "default", True),
                    (
                        "Validate and save",
                        "provider-save",
                        "primary",
                        False,
                    ),
                ),
            ),
            self._page(
                "models",
                "Choose default models",
                Vertical(id="model-editor-host"),
                copy="Choose the language model URSA will use. Embeddings are optional and may use a different inference provider.",
            ),
            self._page(
                "mcp",
                "Configure an MCP server",
                *mcp_fields,
                copy="MCP servers add tools and data sources to URSA. Choose a remote server URL or a local command.",
                actions=(
                    ("Back", "mcp-back", "default", False),
                    ("No MCP, review setup", "mcp-next", "default", False),
                    ("Validate and save", "mcp-save", "primary", False),
                ),
            ),
            self._page(
                "review",
                "Write configuration",
                Static("", id="review-summary", classes="review-box"),
                Static("", id="review-error", classes="form-message"),
                actions=(
                    ("Back", "review-back", "default", False),
                    ("Write configuration", "review-save", "success", True),
                ),
                copy="API keys and tokens will go to your URSA keyring. The YAML file stores only keyring references.",
            ),
            self._page(
                "done",
                "URSA is ready",
                Static("", id="done-copy"),
                actions=tuple(
                    (label, id, variant, False)
                    for id, label, variant in (
                        (
                            "setup-launch-terminal",
                            "Launch URSA - Terminal",
                            "success",
                        ),
                        (
                            "setup-launch-dashboard",
                            "Launch URSA - Dashboard",
                            "success",
                        ),
                        ("setup-exit", "Exit", "default"),
                    )
                ),
            ),
        )
        yield Vertical(
            ContentSwitcher(*pages, initial="welcome-page", id="setup-pages"),
            id="setup-main",
        )
        yield Footer()

    @staticmethod
    def _url_options():
        return [
            ("Provider default", PROVIDER_DEFAULT_URL),
            *(
                (f"{name} ({url})", url)
                for name, url in PROVIDER_DEFAULT_URLS.items()
            ),
            ("Other", CUSTOM_PROVIDER_URL),
        ]

    def _show(self, page: str, remember: bool = True) -> None:
        previous = self.page
        if remember and page != self.page:
            self.history.append(self.page)
        self.page = page
        self.query_one("#setup-pages", ContentSwitcher).current = f"{page}-page"
        if page == "models" and page != previous:
            self.run_worker(
                self._prepare_models(), group="model-editor", exclusive=True
            )
        elif page == "review":
            self._review()

    def action_back(self) -> None:
        if self.history:
            self._show(self.history.pop(), False)

    def action_cancel(self) -> None:
        self.exit(SetupAction.EXIT)

    def _status(self, selector: str, text: str, failed: bool = False) -> None:
        widget = self.query_one(selector, Static)
        widget.update(text)
        widget.set_class(failed, "error")

    def _keys(self) -> list[str]:
        config = {
            "inference_providers": self.config.inference_providers,
            "mcp_servers": self.config.mcp_servers,
        }
        return keyring_usernames(config)

    def _discard_unused_pending(self, usernames) -> None:
        active = set(self._keys())
        for username in set(usernames) - active:
            self.pending.pop(username, None)

    def _items(self, kind: str) -> dict[str, Any]:
        return (
            self.config.inference_providers
            if kind == "provider"
            else self.config.mcp_servers
        )

    def _sync_entries(self, kind: str, selected: str | None) -> None:
        items = self._items(kind)
        selector = self.query_one(f"#{kind}-configured", Select)
        with self.prevent(Select.Changed):
            selector.set_options((name, name) for name in items)
            if selected in items:
                selector.value = selected
            else:
                selector.clear()
        self.query_one(f"#{kind}-remove", Button).disabled = (
            selected is None or kind == "provider" and len(items) == 1
        )
        if kind == "provider":
            self.query_one("#provider-next", Button).disabled = not items

    def _provider(self, name: str) -> InferenceProviderConfig:
        values = self.config.inference_providers[name].model_dump(mode="python")
        config = InferenceProviderConfig.model_validate(
            deep_interp_env(values)
        ).resolve(name)
        reference = config.api_key
        if isinstance(reference, SecretReference):
            username = reference.keyring_username(name)
            secret = self.pending.get(username) if username else None
            resolved = (
                SecretStr(secret) if secret else _quiet(reference.resolve, name)
            )
            config = config.model_copy(update={"api_key": resolved})
        model = next(
            (
                candidate
                for candidate in (self.config.llm_model, self.config.emb_model)
                if candidate is not None
                and candidate.inference_provider == name
            ),
            None,
        )
        if not getattr(config, "model_provider", None) and model is not None:
            config = config.model_copy(
                update={"model_provider": model.model_provider}
            )
        return config

    def _runtime_mcp(
        self, name: str, config: ServerParameters
    ) -> ServerParameters:
        result = _server(name, deep_interp_env(_server_values(config)))
        header, reference = _auth(result)
        if reference:
            username = reference.keyring_username(name)
            secret = self.pending.get(username) if username else None
            if secret:
                headers = dict(getattr(result, "headers", None) or {})
                headers[header] = reference.template % secret
                result = result.model_copy(update={"headers": headers})
        return result

    async def _probe_mcp(self, name: str, config: ServerParameters) -> None:
        await asyncio.to_thread(
            run_probe_with_deadline,
            probe_mcp_server,
            name,
            self._runtime_mcp(name, config),
            timeout=15,
        )

    def _begin_provider(self, name: str | None = None) -> None:
        self.editing["provider"] = name
        values = self.config.inference_providers.get(name or "")
        self._show("provider")
        self._sync_entries("provider", name)
        self.query_one("#provider-name", Input).value = name or "openai"
        base = values.base_url if values else None
        select = self.query_one("#provider-url", Select)
        select.set_options(self._url_options())
        if name is None:
            select.value = PROVIDER_DEFAULT_URLS["openai"]
        elif not base:
            select.value = PROVIDER_DEFAULT_URL
        elif base in PROVIDER_DEFAULT_URLS.values():
            select.value = base
        else:
            select.value = CUSTOM_PROVIDER_URL
        custom = self.query_one("#provider-url-custom", Input)
        custom.value = base if select.value == CUSTOM_PROVIDER_URL else ""
        self._toggle_url(select.value)
        local = bool(base) and urlparse(base).hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        reference = values.api_key if values else None
        self.query_one("#provider-key-api-key", ApiKeyEntry).configure(
            name or "openai",
            reference if isinstance(reference, SecretReference) else None,
            self._keys(),
            local,
        )

    async def _save_provider(self) -> None:
        try:
            name = _name(self.query_one("#provider-name", Input).value, "Name")
            original = self.editing["provider"]
            if (
                name in self.config.mcp_servers
                or name in self.config.inference_providers
                and name != original
            ):
                raise SetupError(f"The name '{name}' is already configured.")
            selected = self.query_one("#provider-url", Select).value
            raw_url = self.query_one("#provider-url-custom", Input).value
            base = (
                None
                if selected == PROVIDER_DEFAULT_URL
                else _url(
                    raw_url
                    if selected == CUSTOM_PROVIDER_URL
                    else str(selected),
                    "Base URL",
                )
            )
            credential = self.query_one(
                "#provider-key-api-key", ApiKeyEntry
            ).values()
            existing = self.config.inference_providers.get(original or "")
            values = (
                existing.model_dump(mode="python", exclude_unset=True)
                if existing
                else {}
            )
            values.pop("ssl_verify", None)
            values.update({} if base is None else {"base_url": base})
            if base is None:
                values.pop("base_url", None)
            if credential.reference:
                values["api_key"] = credential.reference
            else:
                values.pop("api_key", None)
            stored = InferenceProviderConfig.model_validate(values)
            typed = InferenceProviderConfig.model_validate({
                **values,
                **(
                    {"api_key": SecretStr(credential.pending)}
                    if credential.pending
                    else {}
                ),
            }).resolve(name)
            self._status("#provider-error", f"{VALIDATING} Validating {name}")
            await asyncio.to_thread(
                run_probe_with_deadline,
                probe_inference_provider,
                typed,
                True,
            )
        except Exception as exc:
            self._status("#provider-error", _friendly(exc), True)
            return
        old_users = set(
            keyring_usernames({"inference_providers": {original: existing}})
            if original and existing
            else ()
        )
        self.config.inference_providers[name] = stored
        if original and original != name:
            self.config.inference_providers.pop(original)
            for model in (self.config.llm_model, self.config.emb_model):
                if model and model.inference_provider == original:
                    model.inference_provider = name
        self.config.__pydantic_fields_set__.add("inference_providers")
        self._discard_unused_pending(old_users - {credential.username})
        if credential.pending and credential.username:
            self.pending[credential.username] = credential.pending
        self.editing["provider"] = name
        self._sync_entries("provider", name)
        self._status("#provider-error", "✓ Connected")

    async def _prepare_models(self) -> None:
        providers = {
            name: self._provider(name)
            for name in self.config.inference_providers
        }
        chat = (
            self.config.llm_model
            if self.llm_configured
            else ChatModelConfig(
                model="", inference_provider=next(iter(providers), None)
            )
        )
        host = self.query_one("#model-editor-host", Vertical)
        await host.remove_children()
        if self.page == "models":
            editor = ModelEditor(
                providers,
                chat,
                self.config.emb_model,
                apply_label="Validate and continue",
                cancel_label="Back",
            )
            await host.mount(editor)

    @on(ModelEditor.Submitted)
    def model_editor_submitted(self, event: ModelEditor.Submitted) -> None:
        event.stop()
        self._models_selected(event.selection)

    def _models_selected(self, selection: ModelSelection | None) -> None:
        if selection:
            self.config = self.config.model_copy(
                update={
                    "llm_model": selection.chat,
                    "emb_model": selection.embedding,
                }
            )
            self.llm_configured = True
            self.run_worker(
                self._validate_models(), group="model-check", exclusive=True
            )
        else:
            self.action_back()

    async def _validate_models(self) -> None:
        providers = {
            name: self._provider(name)
            for name in self.config.inference_providers
        }

        async def check(prefix: str) -> bool:
            model = (
                self.config.llm_model
                if prefix == "llm"
                else self.config.emb_model
            )
            if model is None:
                self.model_ok[prefix] = True
                return True
            try:
                await asyncio.to_thread(_probe_model, model, providers)
            except Exception as exc:
                self.model_ok[prefix] = False
                message = _friendly(exc)
                self.notify(message, severity="error", markup=False)
                return False
            else:
                self.model_ok[prefix] = True
                return True

        valid = all(await asyncio.gather(check("llm"), check("embedding")))
        if valid:
            self._begin_mcp(next(iter(self.config.mcp_servers), None))

    def _begin_mcp(self, name: str | None = None) -> None:
        self.editing["mcp"] = name
        configured = self.config.mcp_servers.get(name or "")
        values = (
            _server_values(configured)
            if configured is not None
            else {"transport": "streamable_http", "url": ""}
        )
        self._show("mcp")
        self._sync_entries("mcp", name)
        self.query_one("#mcp-name", Input).value = name or ""
        transport_name = values.get("transport", "streamable_http")
        self.query_one("#mcp-transport", Select).value = transport_name
        self.query_one("#mcp-url", Input).value = values.get("url", "")
        self.query_one("#mcp-command", Input).value = values.get("command", "")
        self.query_one("#mcp-args", Input).value = shlex.join(
            values.get("args", [])
        )
        header, reference = (
            _auth(configured) if configured else ("Authorization", None)
        )
        self.mcp_auth = (header, getattr(reference, "template", "Bearer %s"))
        if reference and isinstance(headers := values.get("headers"), dict):
            headers.pop(header, None)
        if not values.get("headers"):
            values.pop("headers", None)
        for field in self.MCP_STRUCTURED:
            values.pop(field, None)
        advanced = self.query_one("#mcp-advanced", AdvancedYamlEditor)
        advanced.set_text(
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
        )
        advanced.set_validation("neutral")
        self.query_one("#mcp-token-api-key", ApiKeyEntry).configure(
            name or "", reference, self._keys(), reference is None
        )
        self._toggle_transport(transport_name)
        self._status("#mcp-validation-status", "○ Not tested")

    def _mcp_advanced_values(self) -> dict[str, Any] | None:
        editor = self.query_one("#mcp-advanced", AdvancedYamlEditor)
        try:
            text = self.query_one("#mcp-config-yaml", TextArea).text
            values = yaml.safe_load(text)
            values = {} if values is None else values
            if not isinstance(values, dict):
                raise SetupError("Advanced YAML must be a mapping.")
            if structured := set(values).intersection(self.MCP_STRUCTURED):
                raise SetupError(f"Use the form for: {sorted(structured)}")
            kind = str(self.query_one("#mcp-transport", Select).value)
            required = (
                {"command": "setup"}
                if kind == "stdio"
                else {"url": "https://localhost"}
            )
            parsed = _server(
                "server", {"transport": kind, **required, **values}
            )
            unknown = set(values) - set(type(parsed).model_fields)
            if unknown:
                raise SetupError(f"Unsupported settings: {sorted(unknown)}")
        except (SetupError, yaml.YAMLError) as exc:
            editor.set_validation("invalid", str(exc))
            return None
        editor.set_validation("valid")
        return values

    def _mcp_values(self):
        name = _name(self.query_one("#mcp-name", Input).value, "Name")
        config = self._mcp_advanced_values()
        if config is None:
            raise SetupError("Fix the advanced MCP configuration.")
        transport = str(self.query_one("#mcp-transport", Select).value)
        config["transport"] = transport
        if transport == "stdio":
            config["command"] = self.query_one("#mcp-command", Input).value
            config["args"] = shlex.split(
                self.query_one("#mcp-args", Input).value
            )
            config.pop("url", None)
            config.pop("headers", None)
            return name, _server(name, config), CredentialValue()
        config["url"] = _url(self.query_one("#mcp-url", Input).value, "MCP URL")
        config.pop("command", None)
        config.pop("args", None)
        credential = self.query_one("#mcp-token-api-key", ApiKeyEntry).values()
        header, template = self.mcp_auth
        headers = config.get("headers")
        if not isinstance(headers, dict):
            headers = {}
            config["headers"] = headers
        if credential.reference:
            managed = SecretTemplate.model_validate({
                **credential.reference.model_dump(exclude_none=True),
                "template": template,
            })
            # The credential control owns its selected header; advanced YAML
            # may add other headers but must not silently override it.
            headers = {**headers, header: managed}
            config["headers"] = headers
        if not headers:
            config.pop("headers")
        return name, _server(name, config), credential

    async def _save_mcp(self) -> None:
        credential = CredentialValue()
        previous = None
        old_users: set[str] = set()
        try:
            name, config, credential = self._mcp_values()
            original = self.editing["mcp"]
            existing = self.config.mcp_servers.get(original or "")
            if original and existing:
                old_users = set(
                    keyring_usernames({"mcp_servers": {original: existing}})
                )
            if (
                name in self.config.inference_providers
                or name in self.config.mcp_servers
                and name != original
            ):
                raise SetupError(f"The name '{name}' is already configured.")
            if credential.pending and credential.username:
                previous = self.pending.get(credential.username)
                self.pending[credential.username] = credential.pending
            self._status("#mcp-validation-status", VALIDATING)
            await self._probe_mcp(name, config)
        except Exception as exc:
            if credential.pending and credential.username:
                if previous is None:
                    self.pending.pop(credential.username, None)
                else:
                    self.pending[credential.username] = previous
            message = _friendly(exc, True)
            self._status("#mcp-validation-status", f"✗ {message}", True)
            self.notify(message, severity="error", markup=False)
            return
        if original and original != name:
            self.config.mcp_servers.pop(original)
        self.config.mcp_servers[name] = config
        self.config.__pydantic_fields_set__.add("mcp_servers")
        self._discard_unused_pending(old_users - {credential.username})
        self.editing["mcp"] = name
        self._sync_entries("mcp", name)
        self._status("#mcp-validation-status", "✓ Connected")

    def _review(self) -> None:
        llm = self.config.llm_model
        emb = self.config.emb_model
        self.query_one("#review-summary", Static).update(
            f"Providers: {', '.join(self.config.inference_providers)}\n"
            f"Language model: {llm.model if llm else 'none'}\n"
            f"Embedding model: {emb.model if emb else 'none'}\n"
            f"MCP servers: {', '.join(self.config.mcp_servers) or 'none'}"
        )
        valid = (
            bool(self.config.inference_providers)
            and self.model_ok["llm"] is True
        )
        self.query_one("#review-save", Button).disabled = not valid

    async def _save_config(self) -> None:
        self.query_one("#review-save", Button).disabled = True
        self._status("#review-error", "")
        try:
            path = await asyncio.to_thread(
                _save, self.path, self.config, self.pending
            )
        except Exception as exc:
            self._review()
            self._status("#review-error", f"Could not save: {exc}", True)
            return
        self.query_one("#done-copy", Static).update(f"Saved to {path}")
        self._show("done")

    def _toggle_url(self, value: object) -> None:
        custom = value == CUSTOM_PROVIDER_URL
        self.query("#provider-url-custom").set_class(not custom, "hidden")
        self.query("#provider-url-row").set_class(custom, "custom-url")

    def _toggle_transport(self, value: object) -> None:
        local = value == "stdio"
        self.query("#mcp-local-fields").set_class(not local, "hidden")
        self.query("#mcp-remote-fields").set_class(local, "hidden")

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if event.select.id in {"provider-configured", "mcp-configured"}:
            kind = event.select.id.removesuffix("-configured")
            if isinstance(event.value, str):
                if kind == "provider":
                    self._begin_provider(event.value)
                else:
                    self._begin_mcp(event.value)
        elif event.select.id == "provider-url":
            self._toggle_url(event.value)
            host = urlparse(str(event.value)).hostname
            if not self.editing["provider"] and host in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                self.query_one(
                    "#provider-key-source", Select
                ).value = MCP_TOKEN_NONE
        elif event.select.id == "mcp-transport":
            self._toggle_transport(event.value)
            self._mcp_advanced_values()

    @on(TextArea.Changed, ".mcp-yaml-editor")
    def mcp_yaml_changed(self) -> None:
        editor = self.query_one("#mcp-advanced", AdvancedYamlEditor)
        editor.set_validation("neutral")
        editor.schedule_validation(0.5, self._mcp_advanced_values)

    @on(Input.Changed, "#provider-name, #mcp-name")
    def name_changed(self, event: Input.Changed) -> None:
        prefix = (
            "provider-key" if event.input.id == "provider-name" else "mcp-token"
        )
        self.query_one(f"#{prefix}-api-key", ApiKeyEntry).rename(
            event.value.strip()
        )

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        id = event.button.id
        if id in self.DONE:
            self.exit(self.DONE[id])
        elif id == "welcome-next":
            self._begin_provider(
                next(iter(self.config.inference_providers), None)
            )
        elif id and id.endswith("-back"):
            self.action_back()
        elif id == "provider-save":
            await self._save_provider()
        elif id in {"provider-add", "mcp-add"}:
            self._begin_provider() if id.startswith(
                "provider"
            ) else self._begin_mcp()
        elif id in {"provider-remove", "mcp-remove"}:
            kind = id.removesuffix("-remove")
            items = self._items(kind)
            name = self.editing[kind]
            if name and (kind == "mcp" or len(items) > 1):
                section = (
                    "inference_providers"
                    if kind == "provider"
                    else "mcp_servers"
                )
                removed_users = keyring_usernames({
                    section: {name: items[name]}
                })
                items.pop(name)
                self._discard_unused_pending(removed_users)
                selected = next(iter(items), None)
                if kind == "provider":
                    self._begin_provider(selected)
                else:
                    self._begin_mcp(selected)
        elif id == "provider-next":
            self._show("models")
        elif id == "mcp-save":
            await self._save_mcp()
        elif id == "mcp-next":
            self._show("review")
        elif id == "review-save":
            await self._save_config()


def _probe_model(config, providers: dict[str, InferenceProviderConfig]) -> None:
    resolved = config.resolve_inference_provider(providers)
    if isinstance(config, ChatModelConfig):
        run_probe_with_deadline(probe_chat_config, resolved, "chat")
    else:
        run_probe_with_deadline(probe_embedding_model, resolved)


def run_setup(path: Path | None = None) -> SetupAction:
    inject_truststore_into_ssl()
    app = SetupApp(path)
    return app.run() or SetupAction.EXIT
