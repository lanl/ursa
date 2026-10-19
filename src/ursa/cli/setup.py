"""Guided first-run configuration for URSA."""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
import shlex
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import SecretStr, ValidationError
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
    TabbedContent,
    TabPane,
    TextArea,
)

from ursa.cli.auth import KEYRING_SERVICE, keyring_usernames
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    UrsaConfig,
    editable_model_values,
    load_config_file,
)
from ursa.cli.installer_config import InstallerConfigError, load_installer_json
from ursa.cli.tui.widgets import (
    AdvancedYamlEditor,
    FuzzySelect,
    HelpLabel,
    ModelEditor,
    ModelSelection,
)
from ursa.util import get_keyring
from ursa.util.http import inject_truststore_into_ssl
from ursa.util.inference_providers import list_provider_models
from ursa.util.mcp import (
    _serialize_server_config,
    start_mcp_client,
    validate_server_parameters,
)
from ursa.util.secrets import SecretReference, SecretTemplate

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


class SetupError(ValueError):
    pass


def load_setup_defaults(path: Path) -> dict[str, Any]:
    try:
        return load_installer_json(path)
    except InstallerConfigError as exc:
        raise SetupError(f"Could not load setup defaults: {exc}") from None


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


def _secret(value: Any) -> SecretReference | None:
    reference = SecretReference.maybe_validate(value)
    return reference if isinstance(reference, SecretReference) else None


def _secret_template(value: Any) -> SecretTemplate | None:
    reference = SecretTemplate.maybe_validate(value)
    return reference if isinstance(reference, SecretTemplate) else None


def _serializable(value: Any) -> Any:
    """Convert typed secret references at a YAML/config boundary."""
    if isinstance(value, SecretReference):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_serializable(item) for item in value)
    return deepcopy(value)


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


def _mcp(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SetupError(f"MCP server '{name}' must be a mapping.")
    try:
        parsed = validate_server_parameters(deepcopy(value))
    except (TypeError, ValueError, ValidationError) as exc:
        raise SetupError(f"MCP server '{name}' is invalid: {exc}") from None
    allowed = {"transport", *type(parsed).model_fields}
    if unknown := sorted(set(value) - allowed):
        raise SetupError(f"Unknown MCP configuration: {', '.join(unknown)}")
    serialized = _serialize_server_config(
        parsed, exclude_defaults=False, exclude_none=True
    )
    result = {
        key: deepcopy(serialized[key])
        for key in ("transport", *value)
        if key in serialized
    }
    if headers := getattr(parsed, "headers", None):
        result["headers"] = deepcopy(headers)
    return result


def _auth(config: Mapping[str, Any]) -> tuple[str, SecretTemplate | None]:
    headers = config.get("headers") or {}
    for header, value in headers.items():
        if reference := _secret_template(value):
            return str(header), reference
    return "Authorization", None


@dataclass(frozen=True)
class CredentialValue:
    """A credential reference plus an optional unsaved keyring value."""

    reference: SecretReference | None = None
    pending: str | None = None
    username: str | None = None


@dataclass
class _State:
    path: Path
    base: dict[str, Any]
    providers: OrderedDict[str, dict[str, Any]] = field(
        default_factory=OrderedDict
    )
    mcps: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)
    models: dict[str, ChatModelConfig | EmbModelConfig | None] = field(
        default_factory=dict
    )
    pending: dict[str, str] = field(default_factory=dict)
    verified: dict[str, dict[str, bool | None]] = field(
        default_factory=lambda: {"provider": {}, "mcp": {}}
    )
    errors: dict[str, dict[str, str]] = field(
        default_factory=lambda: {"provider": {}, "mcp": {}}
    )

    @classmethod
    def load(cls, path: Path) -> _State:
        path = path.expanduser()
        try:
            raw = load_config_file(path) if path.is_file() else {}
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SetupError(f"Could not load configuration: {exc}") from None
        state = cls(path, deepcopy(raw))
        sections = {
            "inference_providers": state.providers,
            "mcp_servers": state.mcps,
        }
        for section, target in sections.items():
            values = raw.get(section, {})
            if not isinstance(values, dict):
                raise SetupError(f"Existing {section} must be a mapping.")
            for name, value in values.items():
                name = str(name)
                target[name] = (
                    _mcp(name, value)
                    if section == "mcp_servers"
                    else cls._provider(name, value)
                )
                kind = "mcp" if section == "mcp_servers" else "provider"
                state.verified[kind][name] = None
        for key, name, model in (
            ("llm_model", "llm", ChatModelConfig),
            ("emb_model", "embedding", EmbModelConfig),
        ):
            if key in raw and raw[key] is not None:
                try:
                    state.models[name] = model.model_validate(raw[key])
                except ValidationError as exc:
                    raise SetupError(
                        f"Existing {key} is invalid: {exc}"
                    ) from None
        if overlap := sorted(set(state.providers) & set(state.mcps)):
            raise SetupError(f"Names used by both provider and MCP: {overlap}")
        return state

    @staticmethod
    def _provider(name: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise SetupError(f"Inference provider '{name}' must be a mapping.")
        parsed = InferenceProviderConfig.model_validate(value)
        result = deepcopy(value)
        result.pop("ssl_verify", None)
        if isinstance(parsed.api_key, SecretReference):
            result["api_key"] = parsed.api_key
        return result

    def defaults(self, payload: Mapping[str, Any]) -> None:
        seen = set()
        for item in payload.get("inference_providers", []):
            assert isinstance(item, dict) and set(item) == {"name", "base_url"}
            name = _name(item["name"], "Provider name")
            base = _url(item["base_url"], "Provider URL")
            if name in seen:
                raise SetupError(f"Provider default '{name}' is duplicated.")
            seen.add(name)
            PROVIDER_DEFAULT_URLS[name] = base
            if name in self.mcps:
                raise SetupError(f"'{name}' is already used by an MCP server.")
            local = (urlparse(base).hostname or "") in {
                "localhost",
                "127.0.0.1",
                "::1",
            }
            self.providers.setdefault(
                name,
                {
                    "base_url": base,
                    **(
                        {}
                        if local
                        else {"api_key": SecretReference(keyring=True)}
                    ),
                },
            )
            self.verified["provider"].setdefault(name, None)
        for name, value in payload.get("mcp_servers", {}).items():
            name = _name(name, "MCP server name")
            if name in self.providers:
                raise SetupError(f"'{name}' is already used by a provider.")
            if name in self.mcps:
                continue
            value = deepcopy(value)
            for header, secret in (value.get("headers") or {}).items():
                if not isinstance(secret, dict):
                    if re.search(
                        "auth|token|secret|password|api.key", header, re.I
                    ):
                        raise SetupError("Installer defaults contain a secret.")
                elif _secret_template(secret):
                    raise SetupError("Installer defaults contain a secret.")
                elif "template" in secret:
                    value["headers"][header] = SecretTemplate.model_validate({
                        **secret,
                        "keyring": True,
                    })
            self.mcps[name] = _mcp(name, value)
            self.verified["mcp"][name] = None

    def provider(self, name: str):
        values = deepcopy(self.providers[name])
        reference = _secret(values.get("api_key"))
        username = reference.keyring_username(name) if reference else None
        secret = self.pending.get(username) if username else None
        if secret:
            values["api_key"] = SecretStr(secret)
        if not values.get("model_provider"):
            model = next(
                (
                    model
                    for model in self.models.values()
                    if model and model.inference_provider == name
                ),
                None,
            )
            if model and model.model_provider:
                values["model_provider"] = model.model_provider
        config = InferenceProviderConfig.model_validate(values).resolve(name)
        if isinstance(config.api_key, SecretReference):
            resolved = _quiet(config.api_key.resolve, name)
            config = config.model_copy(update={"api_key": resolved})
        return config

    def output(self) -> dict[str, Any]:
        if not self.providers or not self.models.get("llm"):
            raise SetupError("A provider and language model are required.")
        config = deepcopy(self.base)
        config["inference_providers"] = _serializable(self.providers)
        config["mcp_servers"] = _serializable(self.mcps)
        config["llm_model"] = editable_model_values(self.models["llm"])
        embedding = self.models.get("embedding")
        if embedding:
            config["emb_model"] = editable_model_values(embedding)
        else:
            config.pop("emb_model", None)
        UrsaConfig.model_validate(config)
        return config


def _save(state: _State) -> Path:
    path = state.path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        config = state.output()
        credential_store = get_keyring()
        for username, secret in state.pending.items():
            credential_store.set_password(KEYRING_SERVICE, username, secret)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return path


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


class ConfiguredTabs(TabbedContent):
    def __init__(self, prefix: str, *, id: str) -> None:
        super().__init__("", id=id)
        self.prefix, self.names = prefix, {}
        self.compose_add_child(TabPane("", Static(""), id=f"{prefix}-pane-0"))

    async def sync(self, items: Mapping[str, Any], render: Callable) -> None:
        values, panes = (
            list(items.items()) or [("", {})],
            list(self.query(TabPane)),
        )
        while len(panes) < len(values):
            await self.add_pane(
                TabPane("", Static(""), id=f"{self.prefix}-pane-{len(panes)}")
            )
            panes = list(self.query(TabPane))
        while len(panes) > len(values):
            await self.remove_pane(panes.pop().id)
        self.names = {}
        for index, (name, value) in enumerate(values):
            title, summary = render(name, value)
            pane_id = f"{self.prefix}-pane-{index}"
            self.get_tab(pane_id).label = title
            self.get_pane(pane_id).query_one(Static).update(summary)
            if name:
                self.names[pane_id] = name

    def selected(self) -> str | None:
        return self.names.get(self.active)


SetupAction = Enum(
    "SetupAction",
    {"EXIT": "exit", "TERMINAL": "terminal", "DASHBOARD": "dashboard"},
)


class SetupApp(App[SetupAction]):
    CSS_PATH = "setup.tcss"
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

    def __init__(
        self, path: Path | None = None, defaults_json: Path | None = None
    ) -> None:
        super().__init__()
        self.state = _State.load(
            path or Path.home() / ".config/ursa/config.yaml"
        )
        if defaults_json:
            self.state.defaults(load_setup_defaults(defaults_json))
        self.page, self.history, self.visited = "welcome", [], {"welcome"}
        self.editing = {"provider": None, "mcp": None}
        self.checking = {"provider": set(), "mcp": set(), "model": set()}
        self.model_ok = {"llm": None, "embedding": None}
        self.locks = {"provider": asyncio.Lock(), "mcp": asyncio.Lock()}

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

    def _list_page(self, kind: Literal["provider", "mcp"]):
        provider = kind == "provider"
        plural = "providers" if provider else "mcps"
        return self._page(
            plural,
            "Inference providers" if provider else "Add an MCP server?",
            ConfiguredTabs(kind, id=f"{plural}-tabs"),
            Horizontal(
                Button("Edit", id=f"{kind}-edit"),
                Button("Remove", id=f"{kind}-remove", variant="error"),
                Button("Check again", id=f"{plural}-retry"),
                classes="manage-actions",
            ),
            Static("", id=f"{plural}-error", classes="form-message"),
            copy=(
                "URSA checks each connection automatically. Select a tab to review or edit that provider, or add another service."
                if provider
                else "MCP is optional. You can finish setup without one or configure as many servers as you need."
            ),
            actions=(
                ("Back", f"{plural}-back", "default", False),
                (
                    "Add provider" if provider else "Yes, add one",
                    f"{kind}-add",
                    "default",
                    False,
                ),
                (
                    "Continue to models" if provider else "No, review setup",
                    f"{plural}-next",
                    "primary",
                    True,
                ),
            ),
        )

    def compose(self) -> ComposeResult:
        provider_fields = (
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
                "mcp",
                markup=False,
            ),
            Static("○ Not tested", id="mcp-validation-status", markup=False),
        )
        pages = (
            self._page(
                "welcome",
                "Welcome to URSA",
                Static(f"Configuration: {self.state.path}", classes="path-box"),
                copy=(
                    "URSA found your existing user configuration. This guide will preserve unrelated settings while you review each setup step."
                    if self.state.path.is_file()
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
                    (
                        "Validate and continue",
                        "provider-save",
                        "primary",
                        False,
                    ),
                ),
            ),
            self._list_page("provider"),
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
                    ("Validate and save", "mcp-save", "primary", False),
                ),
            ),
            self._list_page("mcp"),
            self._page(
                "review",
                "Review and save",
                Static("", id="review-summary", classes="review-box"),
                Horizontal(
                    Button("Edit providers", id="review-providers"),
                    Button("Edit models", id="review-models"),
                    Button("Edit MCP", id="review-mcps"),
                    classes="manage-actions",
                ),
                Static("", id="review-error", classes="form-message"),
                copy="API keys and tokens will go to your URSA keyring. The YAML file stores only keyring references.",
                actions=(
                    ("Back", "review-back", "default", False),
                    ("Save configuration", "review-save", "success", True),
                ),
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
        yield Horizontal(
            Vertical(
                Static("URSA\nSETUP", id="setup-banner"),
                Static("", id="setup-progress", markup=False),
                id="progress-panel",
            ),
            Vertical(
                ContentSwitcher(
                    *pages, initial="welcome-page", id="setup-pages"
                ),
                id="setup-main",
            ),
            id="setup-shell",
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

    def on_ready(self) -> None:
        self._progress()

    def _show(self, page: str, remember: bool = True) -> None:
        previous = self.page
        if remember and page != self.page:
            self.history.append(self.page)
        self.page = page
        self.visited.add(page)
        self.query_one("#setup-pages", ContentSwitcher).current = f"{page}-page"
        self._progress()
        if page in {"providers", "mcps"} and page != previous:
            kind = "provider" if page == "providers" else "mcp"
            self.run_worker(
                self._enter_list(kind), group=f"{kind}-list", exclusive=True
            )
        elif page == "models" and page != previous:
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

    def _mark(self, kind: str, name: str) -> str:
        if name in self.checking[kind]:
            return VALIDATING
        value = self.state.verified[kind].get(name)
        return "✓" if value else "✗" if value is False else "○"

    def _model_status(self, prefix: str, label: str) -> str:
        model = self.state.models.get(prefix)
        name = model.model if model else "None"
        name = name if len(name) <= 20 else name[:17] + "..."
        return f"{self._model_mark(prefix)} {label}: {name}"

    def _model_mark(self, prefix: str) -> str:
        if prefix in self.checking["model"]:
            return VALIDATING
        value = self.model_ok[prefix]
        return "✓" if value else "✗" if value is False else "○"

    def _update_model_status(self, prefix: str) -> None:
        editor = self.query_one_optional(ModelEditor)
        if editor is None:
            return
        model = self.state.models.get(prefix)
        editor.set_tab_status(
            "chat" if prefix == "llm" else "embedding",
            self._model_mark(prefix) if model is not None else None,
            model.model if model is not None else "",
        )

    def _progress(self) -> None:
        groups = {
            "providers": {"provider", "providers"},
            "models": {"models"},
            "mcps": {"mcp", "mcps"},
            "review": {"review", "done"},
        }

        def step(name: str, complete: bool) -> str:
            return (
                ">"
                if self.page in groups.get(name, {name})
                else "✓"
                if complete
                else "○"
            )

        providers_complete = bool(self.state.providers) and all(
            self.state.verified["provider"].get(name) is True
            for name in self.state.providers
        )
        lines = [
            f"{step('welcome', 'provider' in self.visited or 'providers' in self.visited)}  Welcome",
            f"{step('providers', providers_complete)}  Providers",
            *(
                f"   ├─ {self._mark('provider', name)} {name}"
                for name in self.state.providers
            ),
        ]
        models_complete = self.model_ok["llm"] is True and (
            self.state.models.get("embedding") is None
            or self.model_ok["embedding"] is True
        )
        lines.append(f"{step('models', models_complete)}  Models")
        for prefix, label in (("llm", "LLM"), ("embedding", "Emb")):
            lines.append(f"   ├─ {self._model_status(prefix, label)}")
        mcps_complete = all(
            self.state.verified["mcp"].get(name) is True
            for name in self.state.mcps
        ) and ("review" in self.visited or "done" in self.visited)
        lines += [
            f"{step('mcps', mcps_complete)}  MCP servers",
            *(
                f"   ├─ {self._mark('mcp', name)} {name}"
                for name in self.state.mcps
            ),
            f"{step('review', self.page == 'done')}  Save",
        ]
        self.query_one("#setup-progress", Static).update("\n".join(lines))

    def _status(self, selector: str, text: str, failed: bool = False) -> None:
        widget = self.query_one(selector, Static)
        widget.update(text)
        widget.set_class(failed, "error")

    def _keys(self) -> list[str]:
        config = {
            "inference_providers": self.state.providers,
            "mcp_servers": self.state.mcps,
        }
        return keyring_usernames(config)

    def _runtime_mcp(self, name: str, config: dict[str, Any]):
        result = deepcopy(config)
        header, reference = _auth(result)
        if reference:
            username = reference.keyring_username(name)
            secret = self.state.pending.get(username) if username else None
            if secret:
                result["headers"][header] = reference.template % secret
        return result

    async def _probe_mcp(self, name: str, config: dict[str, Any]) -> None:
        client = start_mcp_client({name: self._runtime_mcp(name, config)})
        await asyncio.wait_for(client.get_tools(server_name=name), timeout=15)

    async def _check(self, kind: str, name: str) -> bool:
        items = self.state.providers if kind == "provider" else self.state.mcps
        self.checking[kind].add(name)
        self.state.verified[kind][name] = None
        self._progress()
        await self._tabs(kind)
        try:
            if kind == "provider":
                models = await asyncio.to_thread(
                    list_provider_models, self.state.provider(name)
                )
                if not models:
                    raise SetupError("The provider returned no models.")
            else:
                await self._probe_mcp(name, items[name])
        except Exception as exc:
            self.state.verified[kind][name] = False
            self.state.errors[kind][name] = _friendly(exc, kind == "mcp")
            return False
        else:
            self.state.verified[kind][name] = True
            self.state.errors[kind].pop(name, None)
            return True
        finally:
            self.checking[kind].discard(name)
            self._progress()
            await self._tabs(kind)

    async def _verify(self, kind: str, force: bool = False) -> bool:
        items = self.state.providers if kind == "provider" else self.state.mcps
        async with self.locks[kind]:
            names = [
                name
                for name in items
                if force or self.state.verified[kind].get(name) is not True
            ]
            await asyncio.gather(*(self._check(kind, name) for name in names))
        failed = [
            name
            for name in items
            if self.state.verified[kind].get(name) is False
        ]
        complete = (bool(items) or kind == "mcp") and not failed
        plural = "providers" if kind == "provider" else "mcps"
        self.query_one(f"#{plural}-next", Button).disabled = not complete
        if failed:
            if kind == "provider":
                heading = (
                    "Fix or remove the providers that failed verification:"
                    if len(failed) <= 3
                    else "3+ providers failed verification, the first three are:"
                )
                details = "\n".join(
                    f"- {name}: {self.state.errors[kind][name]}"
                    for name in failed[:3]
                )
            else:
                heading = (
                    "Fix or remove the MCP servers that failed verification:"
                )
                details = "; ".join(
                    f"{name}: {self.state.errors[kind][name]}"
                    for name in failed
                )
            self._status(f"#{plural}-error", f"{heading}\n{details}", True)
        else:
            self._status(f"#{plural}-error", "")
        return complete

    async def _enter_list(self, kind: str) -> None:
        page = "providers" if kind == "provider" else "mcps"
        if self.page == page:
            await self._tabs(kind)
        if self.page == page:
            await self._verify(kind)

    def _render_tab(self, kind: str, name: str, value: dict[str, Any]):
        if not name:
            return "None configured", ""
        title = f"{self._mark(kind, name)} {name}"
        if kind == "provider":
            summary = f"Base URL: {value.get('base_url') or 'default'}"
        else:
            summary = yaml.safe_dump(
                _serializable(value), sort_keys=False
            ).strip()
        return title, summary

    async def _tabs(self, kind: str) -> None:
        plural = "providers" if kind == "provider" else "mcps"
        items = self.state.providers if kind == "provider" else self.state.mcps
        await self.query_one(f"#{plural}-tabs", ConfiguredTabs).sync(
            items, lambda name, value: self._render_tab(kind, name, value)
        )

    def _begin_provider(self, name: str | None = None) -> None:
        self.editing["provider"] = name
        values = self.state.providers.get(name or "", {})
        self._show("provider")
        self.query_one("#provider-name", Input).value = name or "openai"
        base = values.get("base_url")
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
        self.query_one("#provider-key-api-key", ApiKeyEntry).configure(
            name or "openai",
            _secret(values.get("api_key")),
            self._keys(),
            local,
        )

    async def _save_provider(self) -> None:
        try:
            name = _name(self.query_one("#provider-name", Input).value, "Name")
            original = self.editing["provider"]
            if (
                name in self.state.mcps
                or name in self.state.providers
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
            values = deepcopy(self.state.providers.get(original or "", {}))
            values.pop("ssl_verify", None)
            values.update({} if base is None else {"base_url": base})
            if base is None:
                values.pop("base_url", None)
            if credential.reference:
                values["api_key"] = credential.reference
            else:
                values.pop("api_key", None)
            typed = InferenceProviderConfig.model_validate({
                **values,
                **(
                    {"api_key": SecretStr(credential.pending)}
                    if credential.pending
                    else {}
                ),
            }).resolve(name)
            self._status("#provider-error", VALIDATING)
            models = await asyncio.to_thread(list_provider_models, typed)
            if not models:
                raise SetupError("The provider returned no models.")
        except Exception as exc:
            self._status("#provider-error", _friendly(exc), True)
            return
        if original and original != name:
            self.state.providers.pop(original)
            self.state.verified["provider"].pop(original, None)
        self.state.providers[name] = values
        self.state.verified["provider"][name] = True
        if credential.pending and credential.username:
            self.state.pending[credential.username] = credential.pending
        self._show("providers")

    async def _prepare_models(self) -> None:
        providers = {
            name: self.state.provider(name) for name in self.state.providers
        }
        chat = self.state.models.get("llm") or ChatModelConfig(
            model="", inference_provider=next(iter(providers), None)
        )
        host = self.query_one("#model-editor-host", Vertical)
        await host.remove_children()
        if self.page == "models":
            editor = ModelEditor(
                providers,
                chat,
                self.state.models.get("embedding"),
                apply_label="Validate and continue",
                cancel_label="Back",
            )
            await host.mount(editor)
            self._update_model_status("llm")
            self._update_model_status("embedding")

    @on(ModelEditor.Submitted)
    def model_editor_submitted(self, event: ModelEditor.Submitted) -> None:
        event.stop()
        self._models_selected(event.selection)

    def _models_selected(self, selection: ModelSelection | None) -> None:
        if selection:
            self.state.models = {
                "llm": selection.chat,
                "embedding": selection.embedding,
            }
            self.run_worker(
                self._validate_models(), group="model-check", exclusive=True
            )
        else:
            self.action_back()

    async def _validate_models(self) -> None:
        providers = {
            name: self.state.provider(name) for name in self.state.providers
        }

        async def check(prefix: str) -> bool:
            model = self.state.models.get(prefix)
            if model is None:
                self.model_ok[prefix] = True
                self._update_model_status(prefix)
                return True
            self.checking["model"].add(prefix)
            self._update_model_status(prefix)
            self._progress()
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
            finally:
                self.checking["model"].discard(prefix)
                self._update_model_status(prefix)

        valid = all(await asyncio.gather(check("llm"), check("embedding")))
        self._progress()
        if valid:
            self._show("mcps")

    def _begin_mcp(self, name: str | None = None) -> None:
        self.editing["mcp"] = name
        config = deepcopy(
            self.state.mcps.get(
                name or "", {"transport": "streamable_http", "url": ""}
            )
        )
        self._show("mcp")
        self.query_one("#mcp-name", Input).value = name or ""
        transport = config.get("transport", "streamable_http")
        self.query_one("#mcp-transport", Select).value = transport
        self.query_one("#mcp-url", Input).value = config.get("url", "")
        self.query_one("#mcp-command", Input).value = config.get("command", "")
        self.query_one("#mcp-args", Input).value = shlex.join(
            config.get("args", [])
        )
        _, reference = _auth(config)
        self.query_one("#mcp-token-api-key", ApiKeyEntry).configure(
            name or "", reference, self._keys(), reference is None
        )
        editor = self.query_one("#mcp-advanced", AdvancedYamlEditor)
        editor.set_text(yaml.safe_dump(_serializable(config), sort_keys=False))
        self._toggle_transport(transport)
        self._status("#mcp-validation-status", "○ Not tested")

    def _mcp_values(self):
        name = _name(self.query_one("#mcp-name", Input).value, "Name")
        editor = self.query_one("#mcp-advanced", AdvancedYamlEditor)
        config = yaml.safe_load(editor.query_one(TextArea).text) or {}
        if not isinstance(config, dict):
            raise SetupError("MCP configuration must be a mapping.")
        transport = str(self.query_one("#mcp-transport", Select).value)
        config["transport"] = transport
        if transport == "stdio":
            config["command"] = self.query_one("#mcp-command", Input).value
            config["args"] = shlex.split(
                self.query_one("#mcp-args", Input).value
            )
            config.pop("url", None)
            config.pop("headers", None)
            return name, _mcp(name, config), CredentialValue()
        config["url"] = _url(self.query_one("#mcp-url", Input).value, "MCP URL")
        config.pop("command", None)
        config.pop("args", None)
        credential = self.query_one("#mcp-token-api-key", ApiKeyEntry).values()
        header, old = _auth(config)
        headers = config.setdefault("headers", {})
        if old:
            headers.pop(header, None)
        if credential.reference:
            headers[header] = SecretTemplate.model_validate({
                **credential.reference.model_dump(exclude_none=True),
                "template": old.template if old else "Bearer %s",
            })
        if not headers:
            config.pop("headers")
        return name, _mcp(name, config), credential

    async def _save_mcp(self) -> None:
        credential = CredentialValue()
        previous = None
        try:
            name, config, credential = self._mcp_values()
            original = self.editing["mcp"]
            if (
                name in self.state.providers
                or name in self.state.mcps
                and name != original
            ):
                raise SetupError(f"The name '{name}' is already configured.")
            if credential.pending and credential.username:
                previous = self.state.pending.get(credential.username)
                self.state.pending[credential.username] = credential.pending
            self._status("#mcp-validation-status", VALIDATING)
            await self._probe_mcp(name, config)
        except Exception as exc:
            if credential.pending and credential.username:
                if previous is None:
                    self.state.pending.pop(credential.username, None)
                else:
                    self.state.pending[credential.username] = previous
            message = _friendly(exc, True)
            self._status("#mcp-validation-status", f"✗ {message}", True)
            self.notify(message, severity="error", markup=False)
            return
        if original and original != name:
            self.state.mcps.pop(original)
            self.state.verified["mcp"].pop(original, None)
        self.state.mcps[name] = config
        self.state.verified["mcp"][name] = True
        self._show("mcps")

    def _review(self) -> None:
        llm = self.state.models.get("llm")
        emb = self.state.models.get("embedding")
        self.query_one("#review-summary", Static).update(
            f"Providers: {', '.join(self.state.providers)}\n"
            f"Language model: {llm.model if llm else 'none'}\n"
            f"Embedding model: {emb.model if emb else 'none'}\n"
            f"MCP servers: {', '.join(self.state.mcps) or 'none'}"
        )
        valid = (
            bool(self.state.providers)
            and all(self.state.verified["provider"].values())
            and self.model_ok["llm"] is True
            and all(
                value is True for value in self.state.verified["mcp"].values()
            )
        )
        self.query_one("#review-save", Button).disabled = not valid

    async def _save_config(self) -> None:
        try:
            path = await asyncio.to_thread(_save, self.state)
        except Exception as exc:
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
        if event.select.id == "provider-url":
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

    @on(Input.Changed, "#provider-name, #mcp-name")
    def name_changed(self, event: Input.Changed) -> None:
        prefix = (
            "provider-key" if event.input.id == "provider-name" else "mcp-token"
        )
        self.query_one(f"#{prefix}-api-key", ApiKeyEntry).rename(
            event.value.strip()
        )

    @on(TextArea.Changed, "#mcp-config-yaml")
    def mcp_yaml_changed(self, event: TextArea.Changed) -> None:
        editor = self.query_one_optional("#mcp-advanced", AdvancedYamlEditor)
        if editor is None:
            return
        try:
            value = _mcp("draft", yaml.safe_load(event.text_area.text))
        except Exception as exc:
            editor.set_validation("invalid", str(exc))
        else:
            editor.set_validation("valid")
            transport = value.get("transport", "streamable_http")
            self.query_one("#mcp-transport", Select).value = transport
            self.query_one("#mcp-url", Input).value = value.get("url", "")
            self.query_one("#mcp-command", Input).value = value.get(
                "command", ""
            )
            self.query_one("#mcp-args", Input).value = shlex.join(
                value.get("args", [])
            )
            _, reference = _auth(value)
            name = self.query_one("#mcp-name", Input).value.strip()
            self.query_one("#mcp-token-api-key", ApiKeyEntry).configure(
                name, reference, self._keys(), reference is None
            )

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        id = event.button.id
        if id in self.DONE:
            self.exit(self.DONE[id])
        elif id == "welcome-next":
            self._show(
                "providers"
            ) if self.state.providers else self._begin_provider()
        elif id and id.endswith("-back"):
            self.action_back()
        elif id == "provider-save":
            await self._save_provider()
        elif id in {"provider-add", "mcp-add"}:
            self._begin_provider() if id.startswith(
                "provider"
            ) else self._begin_mcp()
        elif id in {"provider-edit", "mcp-edit"}:
            kind = id.removesuffix("-edit")
            plural = "providers" if kind == "provider" else "mcps"
            name = self.query_one(f"#{plural}-tabs", ConfiguredTabs).selected()
            self._begin_provider(
                name
            ) if kind == "provider" else self._begin_mcp(name)
        elif id in {"provider-remove", "mcp-remove"}:
            kind = id.removesuffix("-remove")
            items = (
                self.state.providers if kind == "provider" else self.state.mcps
            )
            plural = "providers" if kind == "provider" else "mcps"
            name = self.query_one(f"#{plural}-tabs", ConfiguredTabs).selected()
            if name and (kind == "mcp" or len(items) > 1):
                items.pop(name)
                self.state.verified[kind].pop(name, None)
                await self._tabs(kind)
        elif id in {"providers-retry", "mcps-retry"}:
            await self._verify(
                "provider" if id.startswith("providers") else "mcp", True
            )
        elif id == "providers-next" and await self._verify("provider"):
            self._show("models")
        elif id == "mcp-save":
            await self._save_mcp()
        elif id == "mcps-next" and await self._verify("mcp"):
            self._show("review")
        elif id in {"review-providers", "review-models", "review-mcps"}:
            self._show(id.removeprefix("review-"))
        elif id == "review-save":
            await self._save_config()


def _probe_model(config, providers: dict[str, InferenceProviderConfig]) -> None:
    resolved = config.resolve_inference_provider(providers)
    result = (
        resolved.init_chat_model().invoke("Reply with only OK.")
        if isinstance(config, ChatModelConfig)
        else resolved.init_embedding().embed_query("URSA setup validation")
    )
    if not result:
        raise SetupError("The model returned no response.")


def run_setup(
    path: Path | None = None, defaults_json: Path | None = None
) -> SetupAction:
    inject_truststore_into_ssl()
    app = SetupApp(
        path or Path.home() / ".config/ursa/config.yaml", defaults_json
    )
    return app.run() or SetupAction.EXIT
