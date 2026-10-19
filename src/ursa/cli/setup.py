"""Guided first-run configuration for URSA."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import keyring
import yaml
from pydantic import SecretStr, ValidationError
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    ContentSwitcher,
    Footer,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from ursa.cli.auth import KEYRING_SERVICE
from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
    ModelConfig,
    UrsaConfig,
    load_config_file,
)
from ursa.cli.tui.widgets import FuzzySelect
from ursa.util.http import inject_truststore_into_ssl
from ursa.util.inference_providers import (
    ProviderModel,
    list_provider_models,
    model_listing_provider,
    sort_provider_models,
    supported_model_providers,
)
from ursa.util.mcp import start_mcp_client, validate_server_parameters
from ursa.util.secrets import SecretReference, SecretTemplate

ProviderKind = Literal["openai", "anthropic", "litellm", "ollama"]

CUSTOM_MODEL = "__ursa_custom_model__"
CUSTOM_PROVIDER_URL = "__ursa_custom_provider_url__"
MCP_TOKEN_NEW = "__ursa_mcp_token_new__"
MCP_TOKEN_NONE = "__ursa_mcp_token_none__"
NO_MODEL = "__ursa_no_model__"
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
PROVIDER_DEFAULT_URLS: dict[ProviderKind, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "litellm": "http://localhost:4000",
    "ollama": "http://localhost:11434",
}
_KEYRING_STDERR_LOCK = threading.Lock()


class SetupError(ValueError):
    """An error that can be corrected inside the setup wizard."""


def _quiet_keyring_call(callback: Callable[..., Any], *args: Any) -> Any:
    """Run a keyring operation without letting its backend corrupt the TUI."""
    with _KEYRING_STDERR_LOCK:
        with open(os.devnull, "w", encoding="utf-8") as stderr:
            with redirect_stderr(stderr):
                return callback(*args)


def _resolve_provider_secret(
    config: InferenceProviderConfig,
) -> InferenceProviderConfig:
    reference = config.api_key
    if not isinstance(reference, SecretReference):
        return config
    secret = _quiet_keyring_call(reference.resolve)
    return config.model_copy(update={"api_key": secret})


def _list_setup_provider_models(
    config: InferenceProviderConfig,
) -> list[ProviderModel]:
    return list_provider_models(_resolve_provider_secret(config))


def default_setup_config_path() -> Path:
    """Return the portable user config path managed by the wizard."""
    return Path.home() / ".config" / "ursa" / "config.yaml"


def _secret_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and (
        value.get("env") is not None
        or value.get("keyring") not in (None, False)
    ):
        return deepcopy(value)
    return None


def _friendly_validation_error(exc: Exception) -> str:
    """Turn common provider failures into useful setup guidance."""
    current: BaseException | None = exc
    messages: list[str] = []
    status_code = None
    while current is not None:
        text = str(current).strip()
        if text and text not in messages:
            messages.append(text)
        status_code = status_code or getattr(current, "status_code", None)
        current = current.__cause__ or current.__context__
    joined = " ".join(messages).lower()
    if status_code in {401, 403} or any(
        marker in joined
        for marker in ("unauthorized", "authentication", "invalid api key")
    ):
        return (
            "The API key was rejected. Check that it was copied completely and "
            "has permission to list models."
        )
    if status_code == 404 or "not found" in joined:
        return (
            "The model endpoint was not found. Check the base URL; OpenAI-style "
            "endpoints usually end in /v1."
        )
    if any(
        marker in joined
        for marker in (
            "connection refused",
            "connecterror",
            "name or service not known",
            "nodename nor servname",
        )
    ):
        return (
            "URSA could not connect to that base URL. Check the address, VPN, "
            "proxy, and network connection."
        )
    detail = messages[0] if messages else type(exc).__name__
    return f"Provider validation failed: {detail}"


def _friendly_mcp_error(exc: Exception) -> str:
    """Turn common MCP connection failures into setup guidance."""
    current: BaseException | None = exc
    messages: list[str] = []
    status_code = None
    while current is not None:
        detail = str(current).strip()
        if detail and detail not in messages:
            messages.append(detail)
        status_code = status_code or getattr(current, "status_code", None)
        current = current.__cause__ or current.__context__
    joined = " ".join(messages).lower()
    if isinstance(exc, TimeoutError) or "timed out" in joined:
        return (
            "The MCP server did not respond in time. Check that it is running "
            "and reachable."
        )
    if status_code in {401, 403} or any(
        marker in joined for marker in ("unauthorized", "forbidden")
    ):
        return (
            "The MCP server rejected the auth key. Check the key, header name, "
            "and template."
        )
    if status_code == 404 or "not found" in joined:
        return "The MCP endpoint was not found. Check the server URL."
    if any(
        marker in joined
        for marker in (
            "connection refused",
            "connecterror",
            "name or service not known",
            "nodename nor servname",
        )
    ):
        return (
            "URSA could not connect to the MCP server. Check the address, "
            "network, VPN, and whether the server is running."
        )
    detail = messages[0] if messages else type(exc).__name__
    return f"MCP validation failed: {detail}"


def _validate_name(name: str, noun: str) -> str:
    name = name.strip()
    if not name:
        raise SetupError(f"{noun} name is required.")
    if not NAME_PATTERN.fullmatch(name):
        raise SetupError(
            f"{noun} names must start with a letter and contain only letters, "
            "numbers, '-' or '_'."
        )
    return name


def _validate_url(value: str, noun: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SetupError(f"{noun} must be a complete http:// or https:// URL.")
    return value


def probe_chat_model(
    config: ChatModelConfig,
    providers: dict[str, InferenceProviderConfig],
) -> None:
    """Instantiate a chat model and require one minimal response."""
    resolved = config.resolve_inference_provider(providers)
    model = resolved.init_chat_model()
    response = model.invoke("Reply with only the word OK.")
    if response is None:
        raise SetupError("The language model returned no response.")


def probe_embedding_model(
    config: EmbModelConfig,
    providers: dict[str, InferenceProviderConfig],
) -> None:
    """Instantiate an embedding model and require one non-empty vector."""
    resolved = config.resolve_inference_provider(providers)
    model = resolved.init_embedding()
    vector = model.embed_query("URSA setup validation")
    if not vector:
        raise SetupError("The embedding model returned an empty vector.")


@dataclass
class ProviderDraft:
    """Provider values retained while the user moves between setup pages."""

    name: str
    kind: ProviderKind
    base_url: str
    api_key_reference: dict[str, Any] | None = None
    pending_secret: str | None = field(default=None, repr=False)
    keyring_username: str | None = None
    ssl_verify: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, value: Any) -> ProviderDraft:
        raw = dict(value) if isinstance(value, dict) else {}
        config = InferenceProviderConfig.model_validate(raw)
        configured_kind = (config.model_extra or {}).get("model_provider")
        inferred = model_listing_provider(config)
        kind = (
            configured_kind
            if configured_kind in PROVIDER_DEFAULT_URLS
            else inferred
        )
        if kind not in PROVIDER_DEFAULT_URLS:
            kind = "openai"
        pending_secret = None
        if isinstance(config.api_key, SecretStr):
            pending_secret = config.api_key.get_secret_value()
        reference = _secret_mapping(raw.get("api_key"))
        keyring_value = reference.get("keyring") if reference else None
        keyring_username = (
            name
            if keyring_value is True
            else keyring_value
            if isinstance(keyring_value, str)
            else None
        )
        return cls(
            name=name,
            kind=kind,
            base_url=config.base_url or PROVIDER_DEFAULT_URLS[kind],
            api_key_reference=reference,
            pending_secret=pending_secret,
            keyring_username=keyring_username,
            ssl_verify=True,
            extra={
                key: deepcopy(value)
                for key, value in (config.model_extra or {}).items()
                if key != "model_provider"
            },
        )

    def validation_config(
        self, entered_secret: str = ""
    ) -> InferenceProviderConfig:
        secret = entered_secret.strip() or self.pending_secret
        api_key: Any = SecretStr(secret) if secret else self.api_key_reference
        if api_key is None:
            raise SetupError("API key is required.")
        return InferenceProviderConfig(
            base_url=self.base_url,
            api_key=api_key,
            ssl_verify=self.ssl_verify,
            model_provider=self.kind,
        )

    def as_config(self) -> dict[str, Any]:
        reference = deepcopy(self.api_key_reference)
        if self.pending_secret is not None:
            username = self.keyring_username or self.name
            reference = {"keyring": True if username == self.name else username}
        if reference is None:
            raise SetupError(
                f"Inference provider '{self.name}' has no API key."
            )
        result: dict[str, Any] = {
            **deepcopy(self.extra),
            "model_provider": self.kind,
            "base_url": self.base_url,
            "api_key": reference,
        }
        return result


@dataclass
class MCPDraft:
    """MCP server values retained by the setup wizard."""

    name: str
    transport: Literal["streamable_http", "sse", "stdio"]
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    auth: bool = False
    header: str = "Authorization"
    template: str = "Bearer %s"
    secret_reference: dict[str, Any] | None = None
    pending_secret: str | None = field(default=None, repr=False)
    keyring_username: str | None = None
    additional_headers: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, value: Any) -> MCPDraft:
        raw = dict(value) if isinstance(value, dict) else {}
        transport = raw.get("transport", "streamable_http")
        if transport == "streamable-http":
            transport = "streamable_http"
        if transport not in {"streamable_http", "sse", "stdio"}:
            transport = "streamable_http"
        headers = (
            raw.get("headers") if isinstance(raw.get("headers"), dict) else {}
        )
        header = "Authorization"
        template = "Bearer %s"
        reference = None
        keyring_username = None
        additional_headers = deepcopy(headers)
        for candidate, header_value in headers.items():
            candidate_reference = _secret_mapping(header_value)
            if candidate_reference is None:
                continue
            header = candidate
            template = str(header_value.get("template", template))
            reference = candidate_reference
            additional_headers.pop(candidate, None)
            keyring_value = reference.get("keyring")
            keyring_username = (
                name
                if keyring_value is True
                else keyring_value
                if isinstance(keyring_value, str)
                else None
            )
            break
        return cls(
            name=name,
            transport=transport,
            url=raw.get("url"),
            command=raw.get("command"),
            args=[str(arg) for arg in raw.get("args", [])],
            auth=reference is not None,
            header=header,
            template=template,
            secret_reference=reference,
            keyring_username=keyring_username,
            additional_headers=additional_headers,
            extra={
                key: deepcopy(value)
                for key, value in raw.items()
                if key
                not in {
                    "transport",
                    "url",
                    "command",
                    "args",
                    "headers",
                }
            },
        )

    def as_config(self) -> dict[str, Any]:
        if self.transport == "stdio":
            result: dict[str, Any] = {
                **deepcopy(self.extra),
                "transport": "stdio",
                "command": self.command,
            }
            if self.args:
                result["args"] = self.args
            return result
        result = {
            **deepcopy(self.extra),
            "transport": self.transport,
            "url": self.url,
        }
        headers = deepcopy(self.additional_headers)
        if self.auth:
            reference = deepcopy(self.secret_reference)
            if self.pending_secret is not None:
                username = self.keyring_username or self.name
                reference = {
                    "keyring": True if username == self.name else username
                }
            if reference is None:
                raise SetupError(f"MCP server '{self.name}' has no auth token.")
            headers[self.header] = {**reference, "template": self.template}
        if headers:
            result["headers"] = headers
        return result

    def validation_config(self) -> dict[str, Any]:
        """Build a runtime config that can use an unsaved auth key."""
        config = self.as_config()
        if self.auth and self.pending_secret is not None:
            config.setdefault("headers", {})[self.header] = (
                self.template % self.pending_secret
            )
        elif self.auth and self.secret_reference is not None:
            reference = SecretTemplate.model_validate({
                **self.secret_reference,
                "template": self.template,
            })
            value = _quiet_keyring_call(
                reference.get_secret_value,
                self.keyring_username or self.name,
            )
            config.setdefault("headers", {})[self.header] = value
        return config


async def probe_mcp_server(server: MCPDraft, timeout: float = 15) -> int:
    """Connect to an MCP server and return its advertised tool count."""
    client = start_mcp_client({server.name: server.validation_config()})
    tools = await asyncio.wait_for(
        client.get_tools(server_name=server.name), timeout=timeout
    )
    return len(tools)


@dataclass
class SetupState:
    """Serializable state for one guided setup session."""

    path: Path
    base_config: dict[str, Any] = field(default_factory=dict)
    providers: OrderedDict[str, ProviderDraft] = field(
        default_factory=OrderedDict
    )
    provider_models: dict[str, list[ProviderModel]] = field(
        default_factory=dict
    )
    provider_verified: dict[str, bool | None] = field(default_factory=dict)
    provider_errors: dict[str, str] = field(default_factory=dict)
    llm_model: ChatModelConfig | None = None
    embedding_model: EmbModelConfig | None = None
    mcp_servers: OrderedDict[str, MCPDraft] = field(default_factory=OrderedDict)
    mcp_verified: dict[str, bool | None] = field(default_factory=dict)
    mcp_errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SetupState:
        path = path.expanduser()
        raw = load_config_file(path) if path.is_file() else {}
        state = cls(path=path, base_config=deepcopy(raw))
        for name, value in (raw.get("inference_providers") or {}).items():
            state.providers[str(name)] = ProviderDraft.from_config(
                str(name), value
            )
            state.provider_verified[str(name)] = None
        llm = raw.get("llm_model")
        if isinstance(llm, dict):
            try:
                state.llm_model = ChatModelConfig.model_validate(llm)
            except ValidationError:
                state.llm_model = None
        embedding = raw.get("emb_model")
        if isinstance(embedding, dict):
            try:
                state.embedding_model = EmbModelConfig.model_validate(embedding)
            except ValidationError:
                state.embedding_model = None
        for name, value in (raw.get("mcp_servers") or {}).items():
            state.mcp_servers[str(name)] = MCPDraft.from_config(
                str(name), value
            )
            state.mcp_verified[str(name)] = None
        return state

    @staticmethod
    def _model_values(
        model: ChatModelConfig | EmbModelConfig,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "model": model.model,
            "inference_provider": model.inference_provider,
        }
        if model.model_provider is not None:
            values["model_provider"] = model.model_provider
        return values

    def _merged_model_values(
        self,
        field_name: str,
        model: ChatModelConfig | EmbModelConfig,
    ) -> dict[str, Any]:
        existing = self.base_config.get(field_name)
        values = deepcopy(existing) if isinstance(existing, dict) else {}
        for name in (
            "model",
            "model_provider",
            "inference_provider",
            "base_url",
            "api_key",
            "api_key_env",
        ):
            values.pop(name, None)
        values.update(self._model_values(model))
        return values

    def configuration(self) -> dict[str, Any]:
        if not self.providers:
            raise SetupError("Add at least one inference provider.")
        if self.llm_model is None:
            raise SetupError("Choose a default language model.")
        config = deepcopy(self.base_config)
        config["inference_providers"] = {
            name: provider.as_config()
            for name, provider in self.providers.items()
        }
        config["llm_model"] = self._merged_model_values(
            "llm_model", self.llm_model
        )
        if self.embedding_model is None:
            config.pop("emb_model", None)
        else:
            config["emb_model"] = self._merged_model_values(
                "emb_model", self.embedding_model
            )
        config["mcp_servers"] = {
            name: server.as_config()
            for name, server in self.mcp_servers.items()
        }
        try:
            UrsaConfig.model_validate(config)
        except ValidationError as exc:
            detail = exc.errors(include_url=False, include_context=False)[0]
            location = ".".join(str(part) for part in detail.get("loc", ()))
            raise SetupError(
                f"Configuration is invalid at {location or 'config'}: "
                f"{detail['msg']}"
            ) from None
        return config

    def secrets(self) -> list[tuple[str, str]]:
        values = [
            (
                provider.keyring_username or name,
                provider.pending_secret,
            )
            for name, provider in self.providers.items()
            if provider.pending_secret is not None
        ]
        values.extend(
            (server.keyring_username or name, server.pending_secret)
            for name, server in self.mcp_servers.items()
            if server.pending_secret is not None
        )
        return [(name, secret) for name, secret in values if secret]

    def runtime_providers(self) -> dict[str, InferenceProviderConfig]:
        """Return provider configs that can use pending, unsaved secrets."""
        return {
            name: _resolve_provider_secret(
                provider.validation_config().resolve(name)
            )
            for name, provider in self.providers.items()
        }


def save_setup_state(state: SetupState) -> Path:
    """Validate and atomically save config after keyring writes succeed."""
    config = state.configuration()
    path = state.path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    for username, secret in state.secrets():
        _quiet_keyring_call(
            keyring.set_password, KEYRING_SERVICE, username, secret
        )

    content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_file():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


class ApiKeyEntry(Vertical):
    """Reusable masked API-key input with a reveal control."""

    def __init__(
        self,
        *,
        input_id: str,
        show_id: str,
        subject: str,
        id: str,
    ) -> None:
        super().__init__(id=id, classes="api-key-entry")
        self._input_id = input_id
        self._show_id = show_id
        self._placeholder = f"Enter the API Key for the {subject}"

    def compose(self) -> ComposeResult:
        yield Label("API Key")
        with Horizontal(classes="api-key-row"):
            yield Input(
                password=True,
                placeholder=self._placeholder,
                id=self._input_id,
            )
            yield Checkbox("Show", value=False, id=self._show_id)

    def reset(self, secret: str = "") -> None:
        """Reset the control to its masked state with an optional secret."""
        field = self.query_one(f"#{self._input_id}", Input)
        field.value = secret
        field.password = True
        field.placeholder = self._placeholder
        self.query_one(f"#{self._show_id}", Checkbox).value = False

    @on(Checkbox.Changed)
    def show_key_changed(self, event: Checkbox.Changed) -> None:
        self.query_one(f"#{self._input_id}", Input).password = not event.value


class ExpandingFuzzySelect(FuzzySelect):
    """Fuzzy selector that temporarily claims its entire horizontal row."""

    def _watch_expanded(self, expanded: bool) -> None:
        super()._watch_expanded(expanded)
        if not self.is_mounted:
            return
        parent = self.parent
        while parent is not None and not isinstance(parent, Horizontal):
            parent = parent.parent
        if isinstance(parent, Horizontal):
            parent.set_class(expanded, "selecting")


class ProviderUrlSelect(ExpandingFuzzySelect):
    """Provider URL selector."""


class MCPTokenSelect(ExpandingFuzzySelect):
    """MCP API-token source selector."""


class SetupApp(App[bool]):
    """Small, linear TUI for creating a working URSA configuration."""

    TITLE = "URSA setup"
    CSS_PATH = "setup.tcss"
    MODEL_SPINNER_FRAMES = ("|", "/", "-", "\\")
    MODEL_YAML_VALIDATION_DELAY = 0.8
    MCP_YAML_VALIDATION_DELAY = 0.8
    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("ctrl+q", "cancel", "Quit setup", priority=True),
    ]

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.state = SetupState.load(path or default_setup_config_path())
        self.current_page = "welcome"
        self.history: list[str] = []
        self.visited = {"welcome"}
        self.editing_provider: str | None = None
        self.editing_mcp: str | None = None
        self._provider_tab_names: dict[str, str] = {}
        self._provider_validation_lock = asyncio.Lock()
        self._providers_validating: set[str] = set()
        self._provider_list_spinner_frame = 0
        self._provider_list_spinner_timer = None
        self._provider_checking = False
        self._provider_spinner_frame = 0
        self._provider_spinner_timer = None
        self._mcp_validation_lock = asyncio.Lock()
        self._loaded_api_keys: dict[str, str | None] = {
            "provider": None,
            "mcp": None,
        }
        self._api_key_load_generation = {"provider": 0, "mcp": 0}
        self.model_validation: dict[str, bool | None] = {
            "llm": None,
            "embedding": None,
        }
        self._model_validation_loading = {"llm": False, "embedding": False}
        self._model_validation_text = {"llm": "", "embedding": ""}
        self._model_spinner_frame = 0
        self._model_spinner_timer = None
        self._model_yaml_values: dict[str, dict[str, Any]] = {
            "llm": {},
            "embedding": {},
        }
        self._model_yaml_initialized: set[str] = set()
        self._model_yaml_timers: dict[str, Timer] = {}
        self._syncing_model_controls = False
        self._mcp_validation_loading = False
        self._mcp_validation_text = ""
        self._mcp_spinner_frame = 0
        self._mcp_spinner_timer = None
        self._mcp_yaml_values: dict[str, Any] = {}
        self._mcp_yaml_timer: Timer | None = None
        self._syncing_mcp_controls = False
        self._mcp_active_token_source: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="setup-shell"):
            with Vertical(id="progress-panel"):
                yield Static("URSA\nSETUP", id="setup-banner")
                yield Static("", id="setup-progress", markup=False)
            with Vertical(id="setup-main"):
                with ContentSwitcher(initial="welcome-page", id="setup-pages"):
                    yield self._welcome_page()
                    yield self._provider_page()
                    yield self._models_page()
                    yield self._providers_page()
                    yield self._mcp_page()
                    yield self._mcps_page()
                    yield self._review_page()
                    yield self._done_page()
        yield Footer()

    @staticmethod
    def _actions(*buttons: Button) -> Horizontal:
        return Horizontal(*buttons, classes="setup-actions")

    @staticmethod
    def _model_advanced_editor(prefix: str) -> Collapsible:
        return Collapsible(
            TextArea(
                "",
                language="yaml",
                show_line_numbers=True,
                tab_behavior="indent",
                id=f"{prefix}-config-yaml",
                classes="setup-model-yaml-editor",
            ),
            Static(
                "",
                id=f"{prefix}-yaml-error",
                classes="setup-model-yaml-error hidden",
                markup=False,
            ),
            title="Advanced",
            collapsed=True,
            id=f"{prefix}-advanced",
            classes="setup-model-advanced",
        )

    @staticmethod
    def _provider_url_options() -> list[tuple[str, str]]:
        return [
            *(
                (f"{name} ({url})", url)
                for name, url in PROVIDER_DEFAULT_URLS.items()
            ),
            ("Other", CUSTOM_PROVIDER_URL),
        ]

    def _known_keyring_usernames(self) -> list[str]:
        names: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                keyring_name = value.get("keyring")
                if isinstance(keyring_name, str):
                    names.add(keyring_name)
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        collect(self.state.base_config)
        for name, provider in self.state.providers.items():
            reference = provider.api_key_reference or {}
            keyring_name = reference.get("keyring")
            if provider.pending_secret is not None or keyring_name is True:
                names.add(provider.keyring_username or name)
            elif isinstance(keyring_name, str):
                names.add(keyring_name)
        for name, server in self.state.mcp_servers.items():
            reference = server.secret_reference or {}
            keyring_name = reference.get("keyring")
            if server.pending_secret is not None or keyring_name is True:
                names.add(server.keyring_username or name)
            elif isinstance(keyring_name, str):
                names.add(keyring_name)
        return sorted(names)

    def _mcp_token_options(
        self, selected: str | None = None
    ) -> list[tuple[str, str]]:
        names = set(self._known_keyring_usernames())
        if selected not in {None, MCP_TOKEN_NONE, MCP_TOKEN_NEW}:
            names.add(selected)
        return [
            *((name, name) for name in sorted(names)),
            ("None", MCP_TOKEN_NONE),
            ("New entry", MCP_TOKEN_NEW),
        ]

    def _welcome_page(self) -> VerticalScroll:
        existing = self.state.path.is_file()
        text = (
            "URSA found your existing user configuration. This guide will keep "
            "unrelated settings and create a backup when you save."
            if existing
            else "This guide will connect URSA to a model provider, choose your "
            "models, and optionally add MCP tools."
        )
        return VerticalScroll(
            Static("Welcome to URSA", classes="page-title"),
            Static(text, classes="page-copy"),
            Static(f"Configuration: {self.state.path}", classes="path-box"),
            Static(
                "You can go back at any time. Nothing is written until the final "
                "Save button.",
                classes="page-copy",
            ),
            self._actions(
                Button("Begin setup", variant="primary", id="welcome-next")
            ),
            id="welcome-page",
            classes="setup-page",
        )

    def _provider_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Connect an inference provider", classes="page-title"),
            Static(
                "URSA verifies the API key by asking the provider for its model "
                "list. The key is hidden and is not written to YAML.",
                classes="page-copy",
            ),
            Label("Name"),
            Input(
                value="openai",
                placeholder="short name for the inference provider",
                id="provider-name",
                max_length=128,
            ),
            Label("Base URL"),
            Horizontal(
                ProviderUrlSelect(
                    self._provider_url_options(),
                    value=PROVIDER_DEFAULT_URLS["openai"],
                    allow_blank=False,
                    id="provider-url",
                ),
                Input(
                    placeholder="https://models.example/v1",
                    id="provider-url-custom",
                    classes="hidden",
                ),
                id="provider-url-row",
            ),
            ApiKeyEntry(
                input_id="provider-key",
                show_id="provider-show-key",
                subject="provider",
                id="provider-api-key",
            ),
            Static(
                "", id="provider-error", classes="form-message", markup=False
            ),
            self._actions(
                Button("Back", id="provider-back"),
                Button(
                    "Validate and continue",
                    variant="primary",
                    id="provider-save",
                ),
            ),
            id="provider-page",
            classes="setup-page",
        )

    def _models_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Choose default models", classes="page-title"),
            Static(
                "Choose the language model URSA will use. Embeddings are optional "
                "and can be added later.",
                classes="page-copy",
            ),
            Label("Inference provider"),
            Select(
                [("Add and validate a provider first", NO_MODEL)],
                value=NO_MODEL,
                allow_blank=False,
                id="model-provider",
            ),
            Label("Language model"),
            Select(
                [("Loading models…", NO_MODEL)],
                value=NO_MODEL,
                allow_blank=False,
                id="llm-choice",
            ),
            Input(
                placeholder="Model ID",
                id="llm-custom",
                classes="hidden",
            ),
            self._model_advanced_editor("llm"),
            Horizontal(
                Static(
                    "○ Not tested",
                    id="llm-validation-status",
                    classes="model-validation-status",
                    markup=False,
                ),
                classes="model-validation",
            ),
            Label("Embedding model (optional)"),
            Select(
                [("None", NO_MODEL)],
                value=NO_MODEL,
                allow_blank=False,
                id="embedding-choice",
            ),
            Input(
                placeholder="Embedding model ID",
                id="embedding-custom",
                classes="hidden",
            ),
            self._model_advanced_editor("embedding"),
            Horizontal(
                Static(
                    "○ Not tested",
                    id="embedding-validation-status",
                    classes="model-validation-status",
                    markup=False,
                ),
                classes="model-validation",
            ),
            Static("", id="models-error", classes="form-message", markup=False),
            self._actions(
                Button("Back", id="models-back"),
                Button("Continue", variant="primary", id="models-save"),
            ),
            id="models-page",
            classes="setup-page",
        )

    def _providers_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Inference providers", classes="page-title"),
            Static(
                "URSA checks each connection automatically. Select a tab to "
                "review or edit that provider, or add another service.",
                classes="page-copy",
            ),
            TabbedContent(id="providers-tabs"),
            Horizontal(
                Button("Edit", id="provider-edit"),
                Button("Remove", id="provider-remove", variant="error"),
                Button("Check again", id="providers-retry"),
                classes="manage-actions",
            ),
            Horizontal(
                Static(
                    "",
                    id="providers-validation-status",
                    classes="model-validation-status",
                    markup=False,
                ),
                classes="model-validation",
            ),
            Static(
                "", id="providers-error", classes="form-message", markup=False
            ),
            self._actions(
                Button("Back", id="providers-back"),
                Button("Add provider", id="provider-add"),
                Button(
                    "Continue to models",
                    variant="primary",
                    id="providers-next",
                    disabled=True,
                ),
            ),
            id="providers-page",
            classes="setup-page",
        )

    def _mcp_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Configure an MCP server", classes="page-title"),
            Static(
                "MCP servers add tools and data sources to URSA. Choose a remote "
                "server URL or a local command.",
                classes="page-copy",
            ),
            Label("Name"),
            Input(placeholder="e.g. project_tools", id="mcp-name"),
            Label("Transport"),
            Select(
                [
                    ("Remote HTTP (recommended)", "streamable_http"),
                    ("Remote SSE", "sse"),
                    ("Local command", "stdio"),
                ],
                value="streamable_http",
                allow_blank=False,
                id="mcp-transport",
            ),
            Vertical(
                Label("Server URL"),
                Input(placeholder="https://example.com/mcp", id="mcp-url"),
                Horizontal(
                    Vertical(
                        Label("API Token"),
                        MCPTokenSelect(
                            self._mcp_token_options(),
                            value=MCP_TOKEN_NONE,
                            allow_blank=False,
                            id="mcp-token-source",
                        ),
                        id="mcp-token-selector",
                    ),
                    Vertical(
                        ApiKeyEntry(
                            input_id="mcp-token",
                            show_id="mcp-show-key",
                            subject="MCP server",
                            id="mcp-api-key",
                        ),
                        id="mcp-auth-fields",
                        classes="hidden",
                    ),
                    id="mcp-token-row",
                ),
                id="mcp-remote-fields",
            ),
            Vertical(
                Label("Command"),
                Input(placeholder="e.g. uvx", id="mcp-command"),
                Label("Arguments"),
                Input(
                    placeholder="e.g. mcp-server-fetch",
                    id="mcp-args",
                ),
                id="mcp-local-fields",
                classes="hidden",
            ),
            Collapsible(
                TextArea(
                    "",
                    language="yaml",
                    show_line_numbers=True,
                    tab_behavior="indent",
                    id="mcp-config-yaml",
                    classes="setup-mcp-yaml-editor",
                ),
                Static(
                    "",
                    id="mcp-yaml-error",
                    classes="setup-mcp-yaml-error hidden",
                    markup=False,
                ),
                title="Advanced",
                collapsed=True,
                id="mcp-advanced",
                classes="setup-mcp-advanced",
            ),
            Horizontal(
                Static(
                    "○ Not tested",
                    id="mcp-validation-status",
                    classes="model-validation-status",
                    markup=False,
                ),
                classes="model-validation",
            ),
            Static("", id="mcp-error", classes="form-message", markup=False),
            self._actions(
                Button("Back", id="mcp-back"),
                Button(
                    "Validate and save",
                    variant="primary",
                    id="mcp-save",
                ),
            ),
            id="mcp-page",
            classes="setup-page",
        )

    def _mcps_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Add an MCP server?", classes="page-title"),
            Static(
                "MCP is optional. You can finish setup without one or configure "
                "as many servers as you need.",
                classes="page-copy",
            ),
            Label("Configured MCP servers"),
            Select(
                [("No MCP servers configured", NO_MODEL)],
                value=NO_MODEL,
                allow_blank=False,
                id="mcps-list",
            ),
            Horizontal(
                Button("Edit selected", id="mcp-edit"),
                Button("Remove selected", id="mcp-remove", variant="error"),
                Button("Check again", id="mcps-retry"),
                classes="manage-actions",
            ),
            Static("", id="mcps-error", classes="form-message", markup=False),
            self._actions(
                Button("Back", id="mcps-back"),
                Button("Yes, add one", id="mcp-add"),
                Button(
                    "No, review setup",
                    variant="primary",
                    id="mcps-next",
                    disabled=True,
                ),
            ),
            id="mcps-page",
            classes="setup-page",
        )

    def _review_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("Review and save", classes="page-title"),
            Static(
                "API keys and tokens will go to your operating-system keyring. "
                "The YAML file stores only keyring references.",
                classes="page-copy",
            ),
            Static("", id="review-summary", classes="review-box", markup=False),
            Horizontal(
                Button("Edit providers", id="review-providers"),
                Button("Edit models", id="review-models"),
                Button("Edit MCP", id="review-mcps"),
                classes="manage-actions",
            ),
            Static("", id="review-error", classes="form-message", markup=False),
            self._actions(
                Button("Back", id="review-back"),
                Button(
                    "Save configuration",
                    variant="success",
                    id="review-save",
                    disabled=True,
                ),
            ),
            id="review-page",
            classes="setup-page",
        )

    def _done_page(self) -> VerticalScroll:
        return VerticalScroll(
            Static("URSA is ready", classes="page-title"),
            Static("", id="done-copy", classes="page-copy", markup=False),
            self._actions(
                Button("Finish", variant="success", id="setup-finish")
            ),
            id="done-page",
            classes="setup-page",
        )

    def on_mount(self) -> None:
        self._provider_spinner_timer = self.set_interval(
            0.12, self._advance_provider_spinner
        )
        self._provider_spinner_timer.pause()
        self._provider_list_spinner_timer = self.set_interval(
            0.12, self._advance_provider_list_spinner
        )
        self._provider_list_spinner_timer.pause()
        self._model_spinner_timer = self.set_interval(
            0.12, self._advance_model_spinner
        )
        self._model_spinner_timer.pause()
        self._mcp_spinner_timer = self.set_interval(
            0.12, self._advance_mcp_spinner
        )
        self._mcp_spinner_timer.pause()
        self._update_progress()
        self._update_provider_continue_button()
        self._update_mcp_continue_button()
        self._update_review_save_button()
        self.query_one("#welcome-next", Button).focus()

    def on_unmount(self) -> None:
        for timer in self._model_yaml_timers.values():
            timer.stop()
        self._model_yaml_timers.clear()
        if self._mcp_yaml_timer is not None:
            self._mcp_yaml_timer.stop()
            self._mcp_yaml_timer = None

    def _show(self, page: str, *, remember: bool = True) -> None:
        if page == self.current_page:
            return
        if remember:
            self.history.append(self.current_page)
        self.current_page = page
        self.visited.add(page)
        self.query_one("#setup-pages", ContentSwitcher).current = f"{page}-page"
        if page == "providers":
            self._update_provider_continue_button()
            self.call_after_refresh(self._prepare_providers_page)
        elif page == "models":
            self._refresh_model_provider_list()
        elif page == "mcps":
            self._update_mcp_continue_button()
            self.call_after_refresh(self._prepare_mcps_page)
        elif page == "review":
            self._refresh_review()
        self._update_progress()

    def action_back(self) -> None:
        if not self.history or self.current_page == "done":
            return
        page = self.history.pop()
        self._show(page, remember=False)

    def action_cancel(self) -> None:
        self.exit(False)

    def _providers_are_verified(self) -> bool:
        return bool(self.state.providers) and all(
            self.state.provider_verified.get(name) is True
            for name in self.state.providers
        )

    def _models_are_verified(self) -> bool:
        return self.model_validation["llm"] is True and (
            self.state.embedding_model is None
            or self.model_validation["embedding"] is True
        )

    def _mcp_servers_are_verified(self) -> bool:
        return all(
            self.state.mcp_verified.get(name) is True
            for name in self.state.mcp_servers
        )

    def _update_provider_continue_button(self) -> None:
        buttons = self.query("#providers-next")
        if buttons:
            buttons.first(Button).disabled = not self._providers_are_verified()

    def _update_mcp_continue_button(self) -> None:
        buttons = self.query("#mcps-next")
        if buttons:
            buttons.first(
                Button
            ).disabled = not self._mcp_servers_are_verified()

    def _update_review_save_button(self) -> None:
        buttons = self.query("#review-save")
        if buttons:
            buttons.first(Button).disabled = not (
                self._providers_are_verified()
                and self._models_are_verified()
                and self._mcp_servers_are_verified()
            )

    def _update_progress(self) -> None:
        provider_names = list(self.state.providers)
        mcp_names = list(self.state.mcp_servers)
        current_groups = {
            "provider": {"provider", "providers"},
            "models": {"models"},
            "mcp": {"mcp", "mcps"},
            "review": {"review", "done"},
        }

        def mark(name: str, complete: bool) -> str:
            active = self.current_page in current_groups.get(name, {name})
            return ">" if active else "✓" if complete else "○"

        providers_complete = bool(provider_names) and all(
            self.state.provider_verified.get(name) is True
            for name in provider_names
        )
        lines = [
            f"{mark('welcome', 'welcome' in self.visited)}  Welcome",
            f"{mark('provider', providers_complete)}  Providers",
        ]
        validation_marks = {True: "✓", False: "✗", None: "○"}
        lines.extend(
            f"   ├─ {self._provider_validation_mark(name)} {name}"
            for name in provider_names
        )
        model_complete = self.model_validation["llm"] is True and (
            self.state.embedding_model is None
            or self.model_validation["embedding"] is True
        )
        lines.append(f"{mark('models', model_complete)}  Models")
        llm_name = self._tree_model_name("llm")
        embedding_name = self._tree_model_name("embedding")
        lines.append(
            f"   ├─ {self._tree_model_mark('llm', validation_marks)} "
            f"LLM: {llm_name}"
        )
        lines.append(
            f"   ├─ {self._tree_model_mark('embedding', validation_marks)} "
            f"Emb: {embedding_name}"
        )
        mcp_complete = (
            not mcp_names
            or all(
                self.state.mcp_verified.get(name) is True for name in mcp_names
            )
        ) and "review" in self.visited
        lines.append(f"{mark('mcp', mcp_complete)}  MCP servers")
        lines.extend(
            f"   ├─ {validation_marks[self.state.mcp_verified.get(name)]} "
            f"{name}"
            for name in mcp_names
        )
        lines.append(f"{mark('review', self.current_page == 'done')}  Save")
        progress = self.query("#setup-progress")
        if progress:
            progress.first(Static).update("\n".join(lines))

    @staticmethod
    def _short_model_name(name: str) -> str:
        return name if len(name) <= 15 else name[:12] + "..."

    def _tree_model_name(self, prefix: str) -> str:
        configured = (
            self.state.llm_model
            if prefix == "llm"
            else self.state.embedding_model
        )
        fallback = "not selected" if prefix == "llm" else "not configured"
        name = configured.model if configured is not None else fallback
        if self.is_mounted and "models" in self.visited:
            selected = self.query_one(f"#{prefix}-choice", Select).value
            if selected == CUSTOM_MODEL:
                name = (
                    self.query_one(f"#{prefix}-custom", Input).value.strip()
                    or fallback
                )
            elif selected != NO_MODEL and isinstance(selected, str):
                name = selected
            elif prefix == "embedding":
                name = fallback
        return self._short_model_name(name)

    def _tree_model_mark(
        self, prefix: str, validation_marks: dict[bool | None, str]
    ) -> str:
        if self._model_validation_loading[prefix]:
            return self.MODEL_SPINNER_FRAMES[self._model_spinner_frame]
        return validation_marks[self.model_validation[prefix]]

    def _provider_validation_mark(self, name: str) -> str:
        if name in self._providers_validating:
            return self.MODEL_SPINNER_FRAMES[self._provider_list_spinner_frame]
        return {True: "✓", False: "✗", None: "○"}[
            self.state.provider_verified.get(name)
        ]

    def _advance_provider_list_spinner(self) -> None:
        if not self._providers_validating:
            if self._provider_list_spinner_timer is not None:
                self._provider_list_spinner_timer.pause()
            return
        self._provider_list_spinner_frame = (
            self._provider_list_spinner_frame + 1
        ) % len(self.MODEL_SPINNER_FRAMES)
        self._update_progress()
        self._update_provider_tabs()
        self._update_provider_connection_checking_message()

    def _update_provider_connection_checking_message(self) -> None:
        frame = self.MODEL_SPINNER_FRAMES[self._provider_list_spinner_frame]
        status = self.query("#providers-validation-status")
        if status:
            status.first(Static).update(
                f"{frame} Checking provider connection..."
            )

    def _advance_model_spinner(self) -> None:
        if not any(self._model_validation_loading.values()):
            if self._model_spinner_timer is not None:
                self._model_spinner_timer.pause()
            return
        self._model_spinner_frame = (self._model_spinner_frame + 1) % len(
            self.MODEL_SPINNER_FRAMES
        )
        self._update_progress()
        frame = self.MODEL_SPINNER_FRAMES[self._model_spinner_frame]
        for prefix, loading in self._model_validation_loading.items():
            if loading:
                self.query_one(f"#{prefix}-validation-status", Static).update(
                    f"{frame} {self._model_validation_text[prefix]}"
                )

    def _set_provider_checking(self, checking: bool) -> None:
        self._provider_checking = checking
        if checking:
            self._provider_spinner_frame = 0
            if self._provider_spinner_timer is not None:
                self._provider_spinner_timer.resume()
            self._update_provider_checking_message()
        elif self._provider_spinner_timer is not None:
            self._provider_spinner_timer.pause()

    def _advance_provider_spinner(self) -> None:
        if not self._provider_checking:
            if self._provider_spinner_timer is not None:
                self._provider_spinner_timer.pause()
            return
        self._provider_spinner_frame = (self._provider_spinner_frame + 1) % len(
            self.MODEL_SPINNER_FRAMES
        )
        self._update_provider_checking_message()

    def _update_provider_checking_message(self) -> None:
        frame = self.MODEL_SPINNER_FRAMES[self._provider_spinner_frame]
        self._set_message(
            self.query_one("#provider-error", Static),
            f"{frame} Checking the API key and loading models",
            success=True,
        )

    @staticmethod
    def _set_message(
        widget: Static, message: str, *, success: bool = False
    ) -> None:
        widget.update(message)
        widget.set_class(bool(message) and not success, "error")
        widget.set_class(bool(message) and success, "success")

    @staticmethod
    def _api_key_control_id(prefix: str) -> str:
        return f"#{prefix}-api-key"

    @staticmethod
    def _api_key_input_id(prefix: str) -> str:
        return "#provider-key" if prefix == "provider" else "#mcp-token"

    def _reset_api_key(self, prefix: str, secret: str = "") -> None:
        self._api_key_load_generation[prefix] += 1
        self._loaded_api_keys[prefix] = None
        self.query_one(self._api_key_control_id(prefix), ApiKeyEntry).reset(
            secret
        )

    def _queue_api_key_load(self, prefix: str, name: str) -> None:
        """Load a named key without blocking typing in the name field."""
        self._api_key_load_generation[prefix] += 1
        generation = self._api_key_load_generation[prefix]
        self.run_worker(
            self._load_api_key(prefix, name.strip(), generation),
            group=f"{prefix}-api-key",
            exclusive=True,
        )

    async def _load_api_key(
        self, prefix: str, name: str, generation: int
    ) -> None:
        field = self.query_one(self._api_key_input_id(prefix), Input)
        loaded_secret = self._loaded_api_keys[prefix]
        if field.value and field.value != loaded_secret:
            return
        starting_value = field.value
        if not name:
            if field.value == loaded_secret:
                field.value = ""
            self._loaded_api_keys[prefix] = None
            return
        try:
            secret = await asyncio.to_thread(
                _quiet_keyring_call,
                keyring.get_password,
                KEYRING_SERVICE,
                name,
            )
        except keyring.errors.KeyringError:
            secret = None
        if generation != self._api_key_load_generation[prefix]:
            return
        if field.value != starting_value:
            return
        field.value = secret or ""
        self._loaded_api_keys[prefix] = secret or None

    def _provider_values(
        self,
    ) -> tuple[str, ProviderKind, str, str, str]:
        name = _validate_name(
            self.query_one("#provider-name", Input).value, "Provider"
        )
        base_url = self._provider_url()
        kind = self._provider_kind(name, base_url)
        secret = self.query_one("#provider-key", Input).value.strip()
        if not secret:
            raise SetupError("API key is required.")
        collision = name in self.state.mcp_servers
        if collision:
            raise SetupError(
                "That name is already used by an MCP server. Choose a unique name "
                "so keyring credentials cannot collide."
            )
        return name, kind, base_url, secret, name

    def _provider_url(self) -> str:
        selected = self.query_one("#provider-url", ProviderUrlSelect).value
        value = (
            self.query_one("#provider-url-custom", Input).value
            if selected == CUSTOM_PROVIDER_URL
            else str(selected)
        )
        return _validate_url(value, "Base URL")

    def _provider_kind(self, name: str, base_url: str) -> ProviderKind:
        for kind, default_url in PROVIDER_DEFAULT_URLS.items():
            if base_url == default_url:
                return kind
        original = self.state.providers.get(self.editing_provider or "")
        if original is not None:
            return original.kind
        normalized_name = name.lower()
        name_kind = next(
            (kind for kind in PROVIDER_DEFAULT_URLS if kind == normalized_name),
            None,
        )
        if name_kind is not None:
            return name_kind
        inferred = model_listing_provider(
            InferenceProviderConfig(base_url=base_url)
        )
        return inferred if inferred in PROVIDER_DEFAULT_URLS else "openai"

    def _begin_provider(self, name: str | None = None) -> None:
        self.editing_provider = name
        provider = self.state.providers.get(name or "")
        kind: ProviderKind = provider.kind if provider else "openai"
        secret = provider.pending_secret or "" if provider else ""
        self._reset_api_key("provider", secret)
        selected_name = provider.name if provider else kind
        self.query_one("#provider-name", Input).value = selected_name
        url = provider.base_url if provider else PROVIDER_DEFAULT_URLS[kind]
        url_select = self.query_one("#provider-url", ProviderUrlSelect)
        if url in PROVIDER_DEFAULT_URLS.values():
            url_select.value = url
            self.query_one("#provider-url-custom", Input).value = ""
            self._toggle_provider_custom_url(url)
        else:
            url_select.value = CUSTOM_PROVIDER_URL
            self.query_one("#provider-url-custom", Input).value = url
            self._toggle_provider_custom_url(CUSTOM_PROVIDER_URL)
        self._queue_api_key_load("provider", selected_name)
        self._set_message(self.query_one("#provider-error", Static), "")
        self._show("provider")

    async def _validate_provider(self) -> None:
        message = self.query_one("#provider-error", Static)
        try:
            name, kind, url, entered_secret, keyring_username = (
                self._provider_values()
            )
            original = self.state.providers.get(self.editing_provider or "")
            draft = ProviderDraft(
                name=name,
                kind=kind,
                base_url=url,
                api_key_reference={"keyring": True},
                pending_secret=entered_secret,
                keyring_username=keyring_username,
                ssl_verify=True,
                extra=deepcopy(original.extra) if original else {},
            )
            config = draft.validation_config().resolve(name)
        except (SetupError, ValidationError) as exc:
            self._set_message(message, str(exc))
            return
        button = self.query_one("#provider-save", Button)
        button.disabled = True
        self._set_provider_checking(True)
        try:
            models = await asyncio.to_thread(
                _list_setup_provider_models, config
            )
            if not models:
                raise SetupError(
                    "The provider connected successfully but returned no models. "
                    "Check that this key can list models."
                )
        except Exception as exc:
            self._set_provider_checking(False)
            error = _friendly_validation_error(exc)
            self._set_message(message, error)
            failed_name = self.editing_provider or name
            if failed_name in self.state.providers:
                self.state.provider_verified[failed_name] = False
                self.state.provider_errors[failed_name] = error
                self._update_progress()
            button.disabled = False
            return
        self._set_provider_checking(False)
        button.disabled = False

        original_name = self.editing_provider
        if original_name and original_name != name:
            del self.state.providers[original_name]
            self.state.provider_models.pop(original_name, None)
            self.state.provider_verified.pop(original_name, None)
            self.state.provider_errors.pop(original_name, None)
            for model in (self.state.llm_model, self.state.embedding_model):
                if (
                    model is not None
                    and model.inference_provider == original_name
                ):
                    model.inference_provider = name
        self.state.providers[name] = draft
        self.state.provider_models[name] = models
        self.state.provider_verified[name] = True
        self.state.provider_errors.pop(name, None)
        self.editing_provider = None
        self.model_validation = {"llm": None, "embedding": None}
        self._set_message(
            message,
            f"Connected. Found {len(models)} models.",
            success=True,
        )
        self._show("providers")

    async def _refresh_provider_tabs(self) -> None:
        tabs = self.query_one("#providers-tabs", TabbedContent)
        await tabs.clear_panes()
        self._provider_tab_names.clear()
        if not self.state.providers:
            pane_id = "provider-pane-empty"
            await tabs.add_pane(
                TabPane(
                    "No providers configured",
                    Static("Add a provider to continue."),
                    id=pane_id,
                )
            )
            return
        for index, (name, provider) in enumerate(self.state.providers.items()):
            pane_id = f"provider-pane-{index}"
            title, summary = self._provider_tab_content(name, provider)
            await tabs.add_pane(
                TabPane(
                    title,
                    Static(summary, classes="provider-summary", markup=False),
                    id=pane_id,
                )
            )
            self._provider_tab_names[pane_id] = name

    def _provider_tab_content(
        self, name: str, provider: ProviderDraft
    ) -> tuple[str, str]:
        status = self.state.provider_verified.get(name)
        labels = {
            True: "Verified",
            False: "Verification failed",
            None: "Checking connection",
        }
        reference = provider.api_key_reference or {}
        if provider.pending_secret is not None:
            credential = "System keyring (saved on completion)"
        elif reference.get("env"):
            credential = f"Environment variable: {reference['env']}"
        elif reference.get("keyring"):
            username = reference["keyring"]
            credential = (
                f"System keyring: {name}"
                if username is True
                else f"System keyring: {username}"
            )
        else:
            credential = "Not configured"
        summary = (
            f"Status: {labels[status]}\n"
            f"Type: {provider.kind}\n"
            f"Base URL: {provider.base_url}\n"
            f"API key: {credential}"
        )
        if error := self.state.provider_errors.get(name):
            summary += f"\n\n{error}"
        return f"{self._provider_validation_mark(name)} {name}", summary

    def _update_provider_tabs(self) -> None:
        tab_containers = self.query("#providers-tabs")
        if not tab_containers:
            return
        tabs = tab_containers.first(TabbedContent)
        for pane_id, name in self._provider_tab_names.items():
            provider = self.state.providers.get(name)
            if provider is None:
                continue
            title, summary = self._provider_tab_content(name, provider)
            tabs.get_tab(pane_id).label = title
            tabs.get_pane(pane_id).query_one(
                ".provider-summary", Static
            ).update(summary)

    async def _prepare_providers_page(self) -> None:
        await self._refresh_provider_tabs()
        self._start_provider_validation()

    def _start_provider_validation(
        self, *, force: bool = False, continue_to_models: bool = False
    ) -> None:
        self.run_worker(
            self._run_provider_validation(
                force=force, continue_to_models=continue_to_models
            ),
            group="provider-validation",
        )

    async def _run_provider_validation(
        self, *, force: bool, continue_to_models: bool
    ) -> None:
        verified = await self._verify_all_providers(force=force)
        if verified and continue_to_models and self.current_page == "providers":
            self._show("models")

    def _active_provider_name(self) -> str | None:
        pane_id = self.query_one("#providers-tabs", TabbedContent).active
        return self._provider_tab_names.get(pane_id)

    async def _verify_provider(self, name: str) -> bool:
        provider = self.state.providers[name]
        self._providers_validating.add(name)
        if len(self._providers_validating) == 1:
            self._provider_list_spinner_frame = 0
            if self._provider_list_spinner_timer is not None:
                self._provider_list_spinner_timer.resume()
        self.state.provider_verified[name] = None
        self.state.provider_errors.pop(name, None)
        self._update_progress()
        self._update_provider_tabs()
        self._update_provider_connection_checking_message()
        try:
            try:
                config = provider.validation_config().resolve(name)
                models = await asyncio.to_thread(
                    _list_setup_provider_models, config
                )
                if not models:
                    raise SetupError(
                        "The provider connected but returned no models. Check "
                        "that the key can list models."
                    )
            except Exception as exc:
                self.state.provider_verified[name] = False
                self.state.provider_errors[name] = _friendly_validation_error(
                    exc
                )
                return False
            self.state.provider_models[name] = models
            self.state.provider_verified[name] = True
            return True
        finally:
            self._providers_validating.discard(name)
            if (
                not self._providers_validating
                and self._provider_list_spinner_timer is not None
            ):
                self._provider_list_spinner_timer.pause()
            self._update_progress()
            self._update_provider_tabs()

    async def _verify_all_providers(self, *, force: bool = False) -> bool:
        async with self._provider_validation_lock:
            targets = [
                name
                for name in self.state.providers
                if force or self.state.provider_verified.get(name) is None
            ]
            message = self.query_one("#providers-error", Static)
            status_widget = self.query_one(
                "#providers-validation-status", Static
            )
            button_ids = (
                "#provider-add",
                "#provider-edit",
                "#provider-remove",
                "#providers-retry",
            )
            self.query_one("#providers-next", Button).disabled = True
            if targets:
                for button_id in button_ids:
                    self.query_one(button_id, Button).disabled = True
                self._update_provider_connection_checking_message()
                self._set_message(message, "")
                try:
                    for name in targets:
                        await self._verify_provider(name)
                finally:
                    for button_id in button_ids:
                        for button in self.query(button_id):
                            if isinstance(button, Button):
                                button.disabled = False

            failed = [
                name
                for name in self.state.providers
                if self.state.provider_verified.get(name) is False
            ]
            pending = [
                name
                for name in self.state.providers
                if self.state.provider_verified.get(name) is None
            ]
            if self.state.providers and not failed and not pending:
                status_widget.update("✓ All providers verified")
                status_widget.set_class(True, "success")
                status_widget.set_class(False, "error")
                self._set_message(message, "")
                self._update_provider_continue_button()
                self._update_review_save_button()
                return True
            status_widget.update("✗ Provider check failed" if failed else "")
            status_widget.set_class(False, "success")
            status_widget.set_class(bool(failed), "error")
            if failed:
                heading = (
                    "Fix or remove the providers that failed verification:"
                    if len(failed) <= 3
                    else "3+ providers failed verification, the first three are:"
                )
                details = "\n".join(
                    f"- {name}: {self.state.provider_errors[name]}"
                    for name in failed[:3]
                )
                self._set_message(message, f"{heading}\n{details}")
            self._update_provider_continue_button()
            self._update_review_save_button()
            return False

    @staticmethod
    def _model_yaml_error_text(error: ValidationError) -> str:
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

    @staticmethod
    def _configured_model_values(config: ModelConfig) -> dict[str, Any]:
        values = config.model_dump(
            mode="json",
            exclude_unset=True,
            context={"include_defaults": False},
        )
        values.setdefault("model", config.model)
        return values

    @staticmethod
    def _dump_model_yaml(values: dict[str, Any]) -> str:
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

    def _initialize_model_yaml(self, prefix: str) -> None:
        if prefix in self._model_yaml_initialized:
            return
        configured = (
            self.state.llm_model
            if prefix == "llm"
            else self.state.embedding_model
        )
        if configured is not None:
            self._model_yaml_values[prefix] = self._configured_model_values(
                configured
            )
        self._model_yaml_initialized.add(prefix)

    def _model_name_for_yaml(self, prefix: str) -> str:
        value = self.query_one(f"#{prefix}-choice", Select).value
        if value == NO_MODEL:
            return ""
        if value == CUSTOM_MODEL:
            return self.query_one(f"#{prefix}-custom", Input).value.strip()
        return str(value)

    def _derived_model_provider(
        self, prefix: str, provider_name: str, model_name: str
    ) -> str:
        advertised = next(
            (
                model
                for model in self.state.provider_models.get(provider_name, [])
                if model.name == model_name
            ),
            None,
        )
        draft = self.state.providers[provider_name]
        fallback = "openai" if draft.kind == "litellm" else draft.kind
        return (
            advertised.model_provider
            if advertised is not None and advertised.model_provider is not None
            else fallback
        )

    def _sync_model_yaml_from_controls(self, prefix: str) -> None:
        if self._syncing_model_controls or not self.is_mounted:
            return
        provider = self.query_one("#model-provider", Select).value
        if (
            not isinstance(provider, str)
            or provider not in self.state.providers
        ):
            return
        self._initialize_model_yaml(prefix)
        values = deepcopy(self._model_yaml_values[prefix])
        model_name = self._model_name_for_yaml(prefix)
        values["model"] = model_name
        values["model_provider"] = self._derived_model_provider(
            prefix, provider, model_name
        )
        values["inference_provider"] = provider
        self._model_yaml_values[prefix] = values
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        text = self._dump_model_yaml(values)
        if editor.text != text:
            self._syncing_model_controls = True
            try:
                with self.prevent(TextArea.Changed):
                    editor.text = text
            finally:
                self._syncing_model_controls = False
        self._validate_model_yaml(prefix, update_controls=False)

    def _set_model_yaml_state(
        self, prefix: str, state: str, error: str = ""
    ) -> None:
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        editor.remove_class("yaml-valid", "yaml-invalid")
        if state != "neutral":
            editor.add_class(f"yaml-{state}")
        message = self.query_one(f"#{prefix}-yaml-error", Static)
        message.update(error)
        message.set_class(not error, "hidden")

    def _update_model_control_from_yaml(
        self, prefix: str, config: ModelConfig
    ) -> None:
        value: str
        catalog = {
            model.name
            for model in self.state.provider_models.get(
                config.inference_provider or "", []
            )
        }
        if not config.model and prefix == "embedding":
            value = NO_MODEL
        elif config.model in catalog:
            value = config.model
        else:
            value = CUSTOM_MODEL
        choice = self.query_one(f"#{prefix}-choice", Select)
        custom = self.query_one(f"#{prefix}-custom", Input)
        self._syncing_model_controls = True
        try:
            with self.prevent(Select.Changed, Input.Changed):
                choice.value = value
                custom.value = config.model if value == CUSTOM_MODEL else ""
        finally:
            self._syncing_model_controls = False
        self._toggle_custom_model(prefix, value)
        self._reset_model_validation()
        self._update_progress()

    def _validate_model_yaml(
        self, prefix: str, *, update_controls: bool
    ) -> ModelConfig | None:
        editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
        config_type = ChatModelConfig if prefix == "llm" else EmbModelConfig
        try:
            values = yaml.safe_load(editor.text)
            if not isinstance(values, dict):
                raise ValueError("Configuration must be a YAML mapping")
            config = config_type.model_validate(deepcopy(values))
            if prefix == "llm" and not config.model.strip():
                raise ValueError("Language model must not be blank")
            provider = self.query_one("#model-provider", Select).value
            if config.inference_provider != provider:
                raise ValueError(
                    "inference_provider must match the selected inference "
                    f"provider '{provider}'"
                )
            if config.ssl_verify is not True:
                raise ValueError("ssl_verify must remain true during setup")
        except (yaml.YAMLError, ValueError) as exc:
            error = (
                self._model_yaml_error_text(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            self._set_model_yaml_state(prefix, "invalid", error)
            return None
        self._model_yaml_values[prefix] = deepcopy(values)
        self._set_model_yaml_state(prefix, "valid")
        if update_controls:
            self._update_model_control_from_yaml(prefix, config)
        return config

    @on(TextArea.Changed, ".setup-model-yaml-editor")
    def model_yaml_changed(self, event: TextArea.Changed) -> None:
        if self._syncing_model_controls or event.text_area.id is None:
            return
        prefix = event.text_area.id.removesuffix("-config-yaml")
        current = self._dump_model_yaml(self._model_yaml_values[prefix])
        if event.text_area.text == current:
            self._set_model_yaml_state(prefix, "valid")
            return
        self._set_model_yaml_state(prefix, "neutral")
        timer = self._model_yaml_timers.pop(prefix, None)
        if timer is not None:
            timer.stop()
        self._model_yaml_timers[prefix] = self.set_timer(
            self.MODEL_YAML_VALIDATION_DELAY,
            lambda: self._validate_model_yaml(prefix, update_controls=True),
        )

    def _refresh_model_provider_list(self) -> None:
        options = [(name, name) for name in self.state.providers]
        if not options:
            options = [("Add and validate a provider first", NO_MODEL)]
        select = self.query_one("#model-provider", Select)
        current = (
            self.state.llm_model.inference_provider
            if self.state.llm_model is not None
            else None
        )
        select.set_options(options)
        select.value = (
            current if current in self.state.providers else options[0][1]
        )
        selected = str(select.value)
        if selected in self.state.provider_models:
            self._populate_model_choices(selected)
        else:
            self.call_after_refresh(self._load_selected_provider_models)

    async def _load_selected_provider_models(self) -> None:
        value = self.query_one("#model-provider", Select).value
        if not isinstance(value, str) or value not in self.state.providers:
            return
        message = self.query_one("#models-error", Static)
        self._set_message(message, "Loading models…", success=True)
        provider = self.state.providers[value]
        try:
            config = provider.validation_config().resolve(value)
            models = await asyncio.to_thread(
                _list_setup_provider_models, config
            )
            if not models:
                raise SetupError("This provider returned no models.")
        except Exception as exc:
            self._set_message(message, _friendly_validation_error(exc))
            return
        self.state.provider_models[value] = models
        self._set_message(message, f"Found {len(models)} models.", success=True)
        self._populate_model_choices(value)

    @staticmethod
    def _looks_like_embedding(model: ProviderModel) -> bool:
        name = model.name.casefold()
        return model.type == "embedding" or "embed" in name

    def _populate_model_choices(self, provider_name: str) -> None:
        models = self.state.provider_models.get(provider_name, [])
        chat = [
            model for model in models if not self._looks_like_embedding(model)
        ]
        embedding = [
            model for model in models if self._looks_like_embedding(model)
        ]
        chat_options = [
            (model.name, model.name)
            for model in sort_provider_models(chat or models, "chat")
        ]
        chat_options.append(("Enter a model ID…", CUSTOM_MODEL))
        embedding_options = [("None", NO_MODEL)]
        embedding_options.extend(
            (model.name, model.name)
            for model in sort_provider_models(embedding, "embedding")
        )
        embedding_options.append(("Enter a model ID…", CUSTOM_MODEL))
        llm_select = self.query_one("#llm-choice", Select)
        emb_select = self.query_one("#embedding-choice", Select)
        llm_select.set_options(chat_options)
        emb_select.set_options(embedding_options)
        configured_llm = self.state.llm_model
        llm_value = chat_options[0][1]
        if (
            configured_llm is not None
            and configured_llm.inference_provider == provider_name
        ):
            values = {value for _, value in chat_options}
            if configured_llm.model in values:
                llm_value = configured_llm.model
            else:
                llm_value = CUSTOM_MODEL
                self.query_one(
                    "#llm-custom", Input
                ).value = configured_llm.model
        llm_select.value = llm_value
        configured_emb = self.state.embedding_model
        emb_value = NO_MODEL
        if (
            configured_emb is not None
            and configured_emb.inference_provider == provider_name
        ):
            values = {value for _, value in embedding_options}
            if configured_emb.model in values:
                emb_value = configured_emb.model
            else:
                emb_value = CUSTOM_MODEL
                self.query_one(
                    "#embedding-custom", Input
                ).value = configured_emb.model
        emb_select.value = emb_value
        self._toggle_custom_model("llm", llm_value)
        self._toggle_custom_model("embedding", emb_value)
        self._sync_model_yaml_from_controls("llm")
        self._sync_model_yaml_from_controls("embedding")
        self._reset_model_validation()

    def _selected_model_name(self, prefix: str) -> str | None:
        value = self.query_one(f"#{prefix}-choice", Select).value
        if value == NO_MODEL:
            return None
        if value == CUSTOM_MODEL:
            custom = self.query_one(f"#{prefix}-custom", Input).value.strip()
            if not custom:
                raise SetupError(
                    "Enter a model ID or choose one from the list."
                )
            return custom
        return str(value)

    def _model_provider(
        self, provider_name: str, model_name: str, model_type: str
    ) -> str:
        draft = self.state.providers[provider_name]
        catalog = self.state.provider_models.get(provider_name, [])
        advertised = next(
            (model for model in catalog if model.name == model_name), None
        )
        fallback = "openai" if draft.kind == "litellm" else draft.kind
        selected = advertised.model_provider if advertised else None
        candidate = selected or fallback
        supported = supported_model_providers(
            "embedding" if model_type == "embedding" else "chat"
        )
        if candidate not in supported:
            raise SetupError(
                f"The '{candidate}' integration does not support {model_type} "
                "models in this URSA installation. Choose another model or provider."
            )
        return candidate

    def _set_model_validation_status(
        self,
        prefix: str,
        status: bool | None,
        text: str,
        *,
        loading: bool = False,
    ) -> None:
        was_loading = any(self._model_validation_loading.values())
        self.model_validation[prefix] = status
        self._model_validation_loading[prefix] = loading
        self._model_validation_text[prefix] = text
        if loading and not was_loading:
            self._model_spinner_frame = 0
            if self._model_spinner_timer is not None:
                self._model_spinner_timer.resume()
        elif not any(self._model_validation_loading.values()):
            if self._model_spinner_timer is not None:
                self._model_spinner_timer.pause()
        widget = self.query_one(f"#{prefix}-validation-status", Static)
        display = (
            f"{self.MODEL_SPINNER_FRAMES[self._model_spinner_frame]} {text}"
            if loading
            else text
        )
        widget.update(display)
        widget.set_class(status is True, "success")
        widget.set_class(status is False, "error")
        self._update_progress()
        self._update_review_save_button()

    def _reset_model_validation(self) -> None:
        if not self.is_mounted:
            return
        self._set_model_validation_status("llm", None, "○ Not tested")
        embedding = self.query_one("#embedding-choice", Select).value
        embedding_text = (
            "○ Not configured" if embedding == NO_MODEL else "○ Not tested"
        )
        self._set_model_validation_status("embedding", None, embedding_text)

    async def _probe_model(
        self,
        prefix: str,
        config: ChatModelConfig | EmbModelConfig,
        providers: dict[str, InferenceProviderConfig],
    ) -> bool:
        label = "Language model" if prefix == "llm" else "Embedding model"
        self._set_model_validation_status(
            prefix, None, "Testing...", loading=True
        )
        try:
            if isinstance(config, ChatModelConfig):
                await asyncio.to_thread(probe_chat_model, config, providers)
            else:
                await asyncio.to_thread(
                    probe_embedding_model, config, providers
                )
        except Exception as exc:
            error = _friendly_validation_error(exc)
            self._set_model_validation_status(
                prefix, False, f"✗ {label} failed"
            )
            self.notify(
                error,
                title=f"{label} validation failed",
                severity="error",
                timeout=10,
                markup=False,
            )
            return False
        self._set_model_validation_status(prefix, True, f"✓ {label} validated")
        return True

    async def _save_models(self) -> None:
        message = self.query_one("#models-error", Static)
        provider = self.query_one("#model-provider", Select).value
        try:
            if (
                not isinstance(provider, str)
                or provider not in self.state.providers
            ):
                raise SetupError("Choose a validated inference provider.")
            for prefix in ("llm", "embedding"):
                editor = self.query_one(f"#{prefix}-config-yaml", TextArea)
                if editor.text == self._dump_model_yaml(
                    self._model_yaml_values[prefix]
                ):
                    self._sync_model_yaml_from_controls(prefix)
            llm_config = self._validate_model_yaml("llm", update_controls=True)
            embedding_config = self._validate_model_yaml(
                "embedding", update_controls=True
            )
            if llm_config is None or embedding_config is None:
                raise SetupError(
                    "Fix the invalid Advanced model configuration before "
                    "continuing."
                )
            assert isinstance(llm_config, ChatModelConfig)
            assert isinstance(embedding_config, EmbModelConfig)
            llm = llm_config
            embedding = embedding_config if embedding_config.model else None
        except SetupError as exc:
            self._set_message(message, str(exc))
            return
        try:
            providers = await asyncio.to_thread(self.state.runtime_providers)
        except Exception as exc:
            self._set_message(message, _friendly_validation_error(exc))
            return
        self.state.llm_model = llm
        self.state.embedding_model = embedding
        button = self.query_one("#models-save", Button)
        button.disabled = True
        self._set_message(
            message,
            "Sending a short test to the selected models…",
            success=True,
        )
        llm_task = self._probe_model("llm", llm, providers)
        if embedding is None:
            llm_ok = await llm_task
            embedding_ok = True
            self._set_model_validation_status(
                "embedding", None, "○ Not configured"
            )
        else:
            llm_ok, embedding_ok = await asyncio.gather(
                llm_task,
                self._probe_model("embedding", embedding, providers),
            )
        button.disabled = False
        if not (llm_ok and embedding_ok):
            self._set_message(
                message,
                "One or more models could not be validated. Review the error "
                "notification and choose a working model.",
            )
            return
        self._set_message(message, "Models validated.", success=True)
        self._show("mcps")

    @staticmethod
    def _mcp_yaml_text(values: dict[str, Any]) -> str:
        return yaml.safe_dump(values, sort_keys=False, allow_unicode=True)

    def _mcp_token_source(self, server: MCPDraft | None) -> str:
        if server is None or not server.auth:
            return MCP_TOKEN_NONE
        if server.pending_secret is not None:
            return MCP_TOKEN_NEW
        reference = server.secret_reference or {}
        keyring_name = reference.get("keyring")
        if keyring_name is True:
            return server.keyring_username or server.name
        if isinstance(keyring_name, str):
            return keyring_name
        return MCP_TOKEN_NONE

    def _set_mcp_token_options(self, selected: str) -> None:
        control = self.query_one("#mcp-token-source", FuzzySelect)
        control.set_options(self._mcp_token_options(selected))
        control.value = selected

    def _pending_keyring_secret(self, username: str) -> str:
        for name, provider in self.state.providers.items():
            if (provider.keyring_username or name) == username:
                return provider.pending_secret or ""
        for name, server in self.state.mcp_servers.items():
            if (server.keyring_username or name) == username:
                return server.pending_secret or ""
        return ""

    def _activate_mcp_token_source(self, value: object, *, reset: bool) -> None:
        visible = isinstance(value, str) and value != MCP_TOKEN_NONE
        self.query_one("#mcp-auth-fields").set_class(not visible, "hidden")
        self.query_one("#mcp-token-row", Horizontal).set_class(
            visible, "has-token"
        )
        if not visible:
            self._mcp_active_token_source = MCP_TOKEN_NONE
            return
        username = (
            self.query_one("#mcp-name", Input).value.strip()
            if value == MCP_TOKEN_NEW
            else value
        )
        if reset:
            self._reset_api_key("mcp", self._pending_keyring_secret(username))
        self._mcp_active_token_source = value
        if username and not self.query_one("#mcp-token", Input).value:
            self._queue_api_key_load("mcp", username)

    @staticmethod
    def _mcp_auth_header(
        values: dict[str, Any],
    ) -> tuple[str, str]:
        headers = values.get("headers")
        if isinstance(headers, dict):
            for name, value in headers.items():
                if _secret_mapping(value) is not None:
                    template = (
                        str(value.get("template", "Bearer %s"))
                        if isinstance(value, dict)
                        else "Bearer %s"
                    )
                    return str(name), template
        return "Authorization", "Bearer %s"

    def _mcp_config_from_controls(self, base: dict[str, Any]) -> dict[str, Any]:
        values = deepcopy(base)
        transport = self.query_one("#mcp-transport", Select).value
        assert transport in {"streamable_http", "sse", "stdio"}
        values["transport"] = transport
        if transport == "stdio":
            values.pop("url", None)
            values.pop("headers", None)
            values["command"] = self.query_one(
                "#mcp-command", Input
            ).value.strip()
            try:
                args = shlex.split(self.query_one("#mcp-args", Input).value)
            except ValueError:
                args = []
            if args:
                values["args"] = args
            else:
                values.pop("args", None)
            return values

        values.pop("command", None)
        values.pop("args", None)
        values["url"] = self.query_one("#mcp-url", Input).value.strip()
        header, template = self._mcp_auth_header(values)
        headers = values.get("headers")
        headers = deepcopy(headers) if isinstance(headers, dict) else {}
        headers = {
            name: value
            for name, value in headers.items()
            if _secret_mapping(value) is None
        }
        token_source = self.query_one("#mcp-token-source", FuzzySelect).value
        if isinstance(token_source, str) and token_source != MCP_TOKEN_NONE:
            server_name = self.query_one("#mcp-name", Input).value.strip()
            reference: dict[str, Any]
            if token_source == MCP_TOKEN_NEW or token_source == server_name:
                reference = {"keyring": True}
            else:
                reference = {"keyring": token_source}
            headers[header] = {**reference, "template": template}
        if headers:
            values["headers"] = headers
        else:
            values.pop("headers", None)
        return values

    def _set_mcp_yaml_state(self, state: str, error: str = "") -> None:
        editor = self.query_one("#mcp-config-yaml", TextArea)
        editor.remove_class("yaml-valid", "yaml-invalid")
        if state != "neutral":
            editor.add_class(f"yaml-{state}")
        message = self.query_one("#mcp-yaml-error", Static)
        message.update(error)
        message.set_class(not error, "hidden")

    def _mcp_draft_from_values(
        self,
        name: str,
        values: dict[str, Any],
        *,
        require_new_secret: bool,
    ) -> MCPDraft:
        draft = MCPDraft.from_config(name, values)
        if draft.auth and draft.keyring_username is None:
            raise SetupError(
                "Choose an API Token keyring entry instead of configuring an "
                "environment secret in Advanced."
            )
        token_source = self.query_one("#mcp-token-source", FuzzySelect).value
        if token_source == MCP_TOKEN_NEW:
            secret = self.query_one("#mcp-token", Input).value.strip()
            if require_new_secret and not secret:
                raise SetupError("API token is required for a new entry.")
            draft.auth = True
            draft.pending_secret = secret or None
            draft.keyring_username = name
            draft.secret_reference = {"keyring": True}
        elif isinstance(token_source, str) and token_source != MCP_TOKEN_NONE:
            secret = self.query_one("#mcp-token", Input).value.strip()
            if require_new_secret and not secret:
                raise SetupError(
                    f"No API token was found for keyring entry '{token_source}'."
                )
            draft.auth = True
            draft.keyring_username = token_source
            draft.secret_reference = {
                "keyring": True if token_source == name else token_source
            }
            draft.pending_secret = (
                secret
                if secret and secret != self._loaded_api_keys["mcp"]
                else None
            )
        try:
            validate_server_parameters(draft.as_config())
        except (TypeError, ValueError, ValidationError) as exc:
            raise SetupError(
                f"MCP server configuration is invalid: {exc}"
            ) from None
        return draft

    def _update_mcp_controls_from_yaml(self, draft: MCPDraft) -> None:
        current_source = self.query_one("#mcp-token-source", FuzzySelect).value
        source = self._mcp_token_source(draft)
        current_name = self.query_one("#mcp-name", Input).value.strip()
        if current_source == MCP_TOKEN_NEW and source == current_name:
            source = MCP_TOKEN_NEW
        self._syncing_mcp_controls = True
        try:
            self.query_one("#mcp-transport", Select).value = draft.transport
            self.query_one("#mcp-url", Input).value = draft.url or ""
            self.query_one("#mcp-command", Input).value = draft.command or ""
            self.query_one("#mcp-args", Input).value = shlex.join(draft.args)
            self._set_mcp_token_options(source)
        finally:
            self._syncing_mcp_controls = False
        self._toggle_mcp_transport(draft.transport)
        self._activate_mcp_token_source(source, reset=source != current_source)
        self._set_mcp_validation_status(None, "○ Not tested")

    def _validate_mcp_yaml(self, *, update_controls: bool) -> MCPDraft | None:
        editor = self.query_one("#mcp-config-yaml", TextArea)
        try:
            values = yaml.safe_load(editor.text)
            if not isinstance(values, dict):
                raise SetupError("Configuration must be a YAML mapping.")
            name = self.query_one("#mcp-name", Input).value.strip() or "mcp"
            draft = self._mcp_draft_from_values(
                name, values, require_new_secret=False
            )
        except (SetupError, yaml.YAMLError) as exc:
            self._set_mcp_yaml_state("invalid", str(exc))
            return None
        self._mcp_yaml_values = deepcopy(values)
        self._set_mcp_yaml_state("valid")
        if update_controls:
            self._update_mcp_controls_from_yaml(draft)
        return draft

    def _sync_mcp_yaml_from_controls(self) -> None:
        if self._syncing_mcp_controls or not self.is_mounted:
            return
        values = self._mcp_config_from_controls(self._mcp_yaml_values)
        self._mcp_yaml_values = values
        editor = self.query_one("#mcp-config-yaml", TextArea)
        text = self._mcp_yaml_text(values)
        if editor.text != text:
            with self.prevent(TextArea.Changed):
                editor.text = text
        self._validate_mcp_yaml(update_controls=False)
        self._set_mcp_validation_status(None, "○ Not tested")

    def _begin_mcp(self, name: str | None = None) -> None:
        self.editing_mcp = name
        server = self.state.mcp_servers.get(name or "")
        if name in self.state.mcp_servers:
            self.state.mcp_verified[name] = None
            self.state.mcp_errors.pop(name, None)
        transport = server.transport if server else "streamable_http"
        token_source = self._mcp_token_source(server)
        secret = server.pending_secret or "" if server else ""
        values = (
            server.as_config()
            if server is not None
            else {"transport": "streamable_http", "url": ""}
        )
        self._syncing_mcp_controls = True
        try:
            self._reset_api_key("mcp", secret)
            self.query_one("#mcp-name", Input).value = (
                server.name if server else ""
            )
            self.query_one("#mcp-transport", Select).value = transport
            self.query_one("#mcp-url", Input).value = (
                server.url or "" if server else ""
            )
            self.query_one("#mcp-command", Input).value = (
                server.command or "" if server else ""
            )
            self.query_one("#mcp-args", Input).value = (
                shlex.join(server.args) if server else ""
            )
            self._set_mcp_token_options(token_source)
            self._mcp_yaml_values = deepcopy(values)
            with self.prevent(TextArea.Changed):
                self.query_one(
                    "#mcp-config-yaml", TextArea
                ).text = self._mcp_yaml_text(values)
        finally:
            self._syncing_mcp_controls = False
        self.query_one("#mcp-advanced", Collapsible).collapsed = True
        self._toggle_mcp_transport(transport)
        self._mcp_active_token_source = token_source
        self._activate_mcp_token_source(token_source, reset=False)
        self._set_mcp_yaml_state("valid" if server else "neutral")
        self._set_mcp_validation_status(None, "○ Not tested")
        self._set_message(self.query_one("#mcp-error", Static), "")
        self._update_progress()
        self._show("mcp")

    def _mcp_values(self) -> MCPDraft:
        name = _validate_name(
            self.query_one("#mcp-name", Input).value, "MCP server"
        )
        if name in self.state.providers:
            raise SetupError(
                "That name is already used by an inference provider. Choose a "
                "unique name so keyring credentials cannot collide."
            )
        editor = self.query_one("#mcp-config-yaml", TextArea)
        try:
            values = yaml.safe_load(editor.text)
        except yaml.YAMLError as exc:
            raise SetupError(f"Advanced YAML is invalid: {exc}") from None
        if not isinstance(values, dict):
            raise SetupError("Advanced configuration must be a YAML mapping.")
        values = self._mcp_config_from_controls(values)
        draft = self._mcp_draft_from_values(
            name, values, require_new_secret=True
        )
        self._mcp_yaml_values = deepcopy(values)
        self._set_mcp_yaml_state("valid")
        return draft

    def _set_mcp_validation_status(
        self,
        status: bool | None,
        text: str,
        *,
        loading: bool = False,
    ) -> None:
        self._mcp_validation_loading = loading
        self._mcp_validation_text = text
        if loading:
            self._mcp_spinner_frame = 0
            if self._mcp_spinner_timer is not None:
                self._mcp_spinner_timer.resume()
        elif self._mcp_spinner_timer is not None:
            self._mcp_spinner_timer.pause()
        widget = self.query_one("#mcp-validation-status", Static)
        display = (
            f"{self.MODEL_SPINNER_FRAMES[self._mcp_spinner_frame]} {text}"
            if loading
            else text
        )
        widget.update(display)
        widget.set_class(status is True, "success")
        widget.set_class(status is False, "error")

    def _advance_mcp_spinner(self) -> None:
        if not self._mcp_validation_loading:
            if self._mcp_spinner_timer is not None:
                self._mcp_spinner_timer.pause()
            return
        self._mcp_spinner_frame = (self._mcp_spinner_frame + 1) % len(
            self.MODEL_SPINNER_FRAMES
        )
        frame = self.MODEL_SPINNER_FRAMES[self._mcp_spinner_frame]
        self.query_one("#mcp-validation-status", Static).update(
            f"{frame} {self._mcp_validation_text}"
        )

    async def _save_mcp(self) -> None:
        message = self.query_one("#mcp-error", Static)
        try:
            draft = self._mcp_values()
        except SetupError as exc:
            self._set_message(message, str(exc))
            return
        button = self.query_one("#mcp-save", Button)
        button.disabled = True
        self._set_message(
            message, "Connecting to the MCP server...", success=True
        )
        self._set_mcp_validation_status(None, "Connecting...", loading=True)
        try:
            tool_count = await probe_mcp_server(draft)
        except Exception as exc:
            error = _friendly_mcp_error(exc)
            self._set_message(message, error)
            self._set_mcp_validation_status(False, "✗ Connection failed")
            if self.editing_mcp in self.state.mcp_servers:
                self.state.mcp_verified[self.editing_mcp] = False
                self.state.mcp_errors[self.editing_mcp] = error
                self._update_progress()
            self.notify(
                error,
                title="MCP server validation failed",
                severity="error",
                timeout=10,
                markup=False,
            )
            button.disabled = False
            return
        self._set_mcp_validation_status(
            True,
            f"✓ Connected · {tool_count} tool"
            + ("" if tool_count == 1 else "s"),
        )
        self._set_message(message, "MCP server validated.", success=True)
        button.disabled = False
        original_name = self.editing_mcp
        if original_name and original_name != draft.name:
            del self.state.mcp_servers[original_name]
            self.state.mcp_verified.pop(original_name, None)
            self.state.mcp_errors.pop(original_name, None)
        self.state.mcp_servers[draft.name] = draft
        self.state.mcp_verified[draft.name] = True
        self.state.mcp_errors.pop(draft.name, None)
        self.editing_mcp = None
        self._show("mcps")

    def _refresh_mcp_list(self) -> None:
        options = [(name, name) for name in self.state.mcp_servers]
        if not options:
            options = [("No MCP servers configured", NO_MODEL)]
        select = self.query_one("#mcps-list", Select)
        select.set_options(options)
        select.value = options[0][1]

    async def _verify_mcp_server(self, name: str) -> bool:
        self.state.mcp_verified[name] = None
        self.state.mcp_errors.pop(name, None)
        self._update_progress()
        try:
            await probe_mcp_server(self.state.mcp_servers[name])
        except Exception as exc:
            self.state.mcp_verified[name] = False
            self.state.mcp_errors[name] = _friendly_mcp_error(exc)
            self._update_progress()
            return False
        self.state.mcp_verified[name] = True
        self._update_progress()
        return True

    async def _verify_all_mcp_servers(self, *, force: bool = False) -> bool:
        async with self._mcp_validation_lock:
            targets = [
                name
                for name in self.state.mcp_servers
                if force or self.state.mcp_verified.get(name) is None
            ]
            message = self.query_one("#mcps-error", Static)
            buttons = (
                "#mcp-add",
                "#mcp-edit",
                "#mcp-remove",
                "#mcps-retry",
            )
            self.query_one("#mcps-next", Button).disabled = True
            if targets:
                for button_id in buttons:
                    self.query_one(button_id, Button).disabled = True
                self._set_message(
                    message,
                    "Checking configured MCP servers…",
                    success=True,
                )
                try:
                    for name in targets:
                        await self._verify_mcp_server(name)
                finally:
                    for button_id in buttons:
                        self.query_one(button_id, Button).disabled = False
            failed = [
                name
                for name in self.state.mcp_servers
                if self.state.mcp_verified.get(name) is False
            ]
            pending = [
                name
                for name in self.state.mcp_servers
                if self.state.mcp_verified.get(name) is None
            ]
            if not failed and not pending:
                self._set_message(
                    message,
                    "✓ All MCP servers verified."
                    if self.state.mcp_servers
                    else "",
                    success=True,
                )
                self._update_mcp_continue_button()
                self._update_review_save_button()
                return True
            details = "; ".join(
                f"{name}: {self.state.mcp_errors[name]}" for name in failed
            )
            self._set_message(
                message,
                "Fix or remove the MCP servers that failed validation. "
                + details,
            )
            self._update_mcp_continue_button()
            self._update_review_save_button()
            return False

    async def _prepare_mcps_page(self) -> None:
        self._refresh_mcp_list()
        await self._verify_all_mcp_servers()

    def _refresh_review(self) -> None:
        try:
            config = self.state.configuration()
            llm = config["llm_model"]["model"]
            embedding = (
                config.get("emb_model", {}).get("model") or "not configured"
            )
            summary = (
                f"Configuration file\n  {self.state.path}\n\n"
                f"Inference providers\n  "
                f"{', '.join(self.state.providers) or 'none'}\n\n"
                f"Language model\n  {llm}\n\n"
                f"Embedding model\n  {embedding}\n\n"
                f"MCP servers\n  "
                f"{', '.join(self.state.mcp_servers) or 'none'}"
            )
            self._set_message(self.query_one("#review-error", Static), "")
        except SetupError as exc:
            summary = "Setup is incomplete. Use the edit buttons below."
            self._set_message(self.query_one("#review-error", Static), str(exc))
        self.query_one("#review-summary", Static).update(summary)
        self._update_review_save_button()

    async def _save_configuration(self) -> None:
        message = self.query_one("#review-error", Static)
        failed_providers = [
            name
            for name in self.state.providers
            if self.state.provider_verified.get(name) is not True
        ]
        if failed_providers:
            self._set_message(
                message,
                "Verify every provider before saving: "
                + ", ".join(failed_providers),
            )
            return
        models_valid = self.model_validation["llm"] is True and (
            self.state.embedding_model is None
            or self.model_validation["embedding"] is True
        )
        if not models_valid:
            self._set_message(
                message,
                "Validate the selected language and embedding models before "
                "saving.",
            )
            return
        failed_mcp_servers = [
            name
            for name in self.state.mcp_servers
            if self.state.mcp_verified.get(name) is not True
        ]
        if failed_mcp_servers:
            self._set_message(
                message,
                "Validate every configured MCP server before saving: "
                + ", ".join(failed_mcp_servers),
            )
            return
        button = self.query_one("#review-save", Button)
        button.disabled = True
        self._set_message(
            message, "Saving credentials and configuration…", success=True
        )
        try:
            path = await asyncio.to_thread(save_setup_state, self.state)
        except keyring.errors.KeyringError as exc:
            self._set_message(
                message,
                "The operating-system keyring is unavailable, so the "
                "configuration was not written. Unlock or configure the keyring "
                f"and try again. ({exc})",
            )
            button.disabled = False
            return
        except Exception as exc:
            self._set_message(message, f"Could not save configuration: {exc}")
            button.disabled = False
            return
        self.query_one("#done-copy", Static).update(
            f"Configuration saved to {path}.\n\nRun `ursa` to open the terminal "
            "app or `ursa-dashboard` to use URSA in a browser."
        )
        self._show("done")

    def _toggle_custom_model(self, prefix: str, value: object) -> None:
        self.query_one(f"#{prefix}-custom", Input).set_class(
            value != CUSTOM_MODEL, "hidden"
        )

    def _toggle_provider_custom_url(self, value: object) -> None:
        custom = value == CUSTOM_PROVIDER_URL
        self.query_one("#provider-url-custom", Input).set_class(
            not custom, "hidden"
        )
        self.query_one("#provider-url-row", Horizontal).set_class(
            custom, "custom-url"
        )

    def _toggle_mcp_transport(self, value: object) -> None:
        local = value == "stdio"
        self.query_one("#mcp-local-fields", Vertical).set_class(
            not local, "hidden"
        )
        self.query_one("#mcp-remote-fields", Vertical).set_class(
            local, "hidden"
        )

    @on(Input.Changed, "#provider-name")
    def provider_name_changed(self, event: Input.Changed) -> None:
        self._queue_api_key_load("provider", event.value)

    @on(Select.Changed, "#provider-url")
    def provider_url_changed(self, event: Select.Changed) -> None:
        select = self.query_one("#provider-url", ProviderUrlSelect)
        if event.value != select.value:
            return
        self._toggle_provider_custom_url(event.value)

    @on(Select.Changed, "#model-provider")
    async def model_provider_changed(self, event: Select.Changed) -> None:
        if not self.is_mounted or event.value not in self.state.providers:
            return
        if event.value in self.state.provider_models:
            self._populate_model_choices(str(event.value))
        else:
            await self._load_selected_provider_models()

    @on(Select.Changed, "#llm-choice")
    def llm_choice_changed(self, event: Select.Changed) -> None:
        self._toggle_custom_model("llm", event.value)
        self._sync_model_yaml_from_controls("llm")
        self._reset_model_validation()

    @on(Select.Changed, "#embedding-choice")
    def embedding_choice_changed(self, event: Select.Changed) -> None:
        self._toggle_custom_model("embedding", event.value)
        self._sync_model_yaml_from_controls("embedding")
        self._reset_model_validation()

    @on(Input.Changed, "#llm-custom, #embedding-custom")
    def custom_model_changed(self, event: Input.Changed) -> None:
        if event.input.id is not None:
            self._sync_model_yaml_from_controls(
                event.input.id.removesuffix("-custom")
            )
        self._reset_model_validation()

    @on(Select.Changed, "#mcp-transport")
    def mcp_transport_changed(self, event: Select.Changed) -> None:
        if self._syncing_mcp_controls:
            return
        self._toggle_mcp_transport(event.value)
        self._sync_mcp_yaml_from_controls()

    @on(Select.Changed, "#mcp-token-source")
    def mcp_token_source_changed(self, event: Select.Changed) -> None:
        if self._syncing_mcp_controls:
            return
        control = self.query_one("#mcp-token-source", FuzzySelect)
        if event.value != control.value:
            return
        self._activate_mcp_token_source(
            event.value, reset=event.value != self._mcp_active_token_source
        )
        self._sync_mcp_yaml_from_controls()

    @on(Input.Changed, "#mcp-name")
    def mcp_name_changed(self, event: Input.Changed) -> None:
        if self._syncing_mcp_controls:
            return
        source = self.query_one("#mcp-token-source", FuzzySelect).value
        if source == MCP_TOKEN_NEW:
            self._queue_api_key_load("mcp", event.value)
        self._sync_mcp_yaml_from_controls()

    @on(Input.Changed, "#mcp-url, #mcp-command, #mcp-args")
    def mcp_connection_field_changed(self, _event: Input.Changed) -> None:
        if not self._syncing_mcp_controls:
            self._sync_mcp_yaml_from_controls()

    @on(Input.Changed, "#mcp-token")
    def mcp_token_changed(self, _event: Input.Changed) -> None:
        if not self._syncing_mcp_controls:
            self._set_mcp_validation_status(None, "○ Not tested")

    @on(TextArea.Changed, "#mcp-config-yaml")
    def mcp_yaml_changed(self, event: TextArea.Changed) -> None:
        if self._syncing_mcp_controls:
            return
        current = self._mcp_yaml_text(self._mcp_yaml_values)
        if event.text_area.text == current:
            self._set_mcp_yaml_state("valid")
            return
        self._set_mcp_yaml_state("neutral")
        if self._mcp_yaml_timer is not None:
            self._mcp_yaml_timer.stop()
        self._mcp_yaml_timer = self.set_timer(
            self.MCP_YAML_VALIDATION_DELAY,
            lambda: self._validate_mcp_yaml(update_controls=True),
        )

    @on(Button.Pressed)
    async def button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id in {
            "provider-back",
            "models-back",
            "providers-back",
            "mcp-back",
            "mcps-back",
            "review-back",
        }:
            self.action_back()
        elif button_id == "welcome-next":
            if self.state.providers:
                self._show("providers")
            else:
                self._begin_provider()
        elif button_id == "provider-save":
            await self._validate_provider()
        elif button_id == "models-save":
            await self._save_models()
        elif button_id == "provider-add":
            self._begin_provider()
        elif button_id == "provider-edit":
            value = self._active_provider_name()
            if value in self.state.providers:
                self._begin_provider(value)
        elif button_id == "provider-remove":
            value = self._active_provider_name()
            if value in self.state.providers:
                if len(self.state.providers) == 1:
                    self._set_message(
                        self.query_one("#providers-error", Static),
                        "URSA needs at least one inference provider.",
                    )
                else:
                    del self.state.providers[value]
                    self.state.provider_models.pop(value, None)
                    self.state.provider_verified.pop(value, None)
                    self.state.provider_errors.pop(value, None)
                    if self.state.llm_model and (
                        self.state.llm_model.inference_provider == value
                    ):
                        self.state.llm_model = None
                    if self.state.embedding_model and (
                        self.state.embedding_model.inference_provider == value
                    ):
                        self.state.embedding_model = None
                    self.model_validation = {
                        "llm": None,
                        "embedding": None,
                    }
                    await self._refresh_provider_tabs()
                    self._update_progress()
                    self._update_provider_continue_button()
                    self._update_review_save_button()
        elif button_id == "providers-next":
            self._start_provider_validation(continue_to_models=True)
        elif button_id == "providers-retry":
            self._start_provider_validation(force=True)
        elif button_id == "mcp-add":
            self._begin_mcp()
        elif button_id == "mcp-edit":
            value = self.query_one("#mcps-list", Select).value
            if isinstance(value, str) and value in self.state.mcp_servers:
                self._begin_mcp(value)
        elif button_id == "mcp-remove":
            value = self.query_one("#mcps-list", Select).value
            if isinstance(value, str) and value in self.state.mcp_servers:
                del self.state.mcp_servers[value]
                self.state.mcp_verified.pop(value, None)
                self.state.mcp_errors.pop(value, None)
                self._refresh_mcp_list()
                self._update_progress()
                self._update_mcp_continue_button()
                self._update_review_save_button()
        elif button_id == "mcp-save":
            await self._save_mcp()
        elif button_id == "mcps-next":
            if await self._verify_all_mcp_servers():
                self._show("review")
        elif button_id == "mcps-retry":
            await self._verify_all_mcp_servers(force=True)
        elif button_id == "review-providers":
            self._show("providers")
        elif button_id == "review-models":
            self._show("models")
        elif button_id == "review-mcps":
            self._show("mcps")
        elif button_id == "review-save":
            await self._save_configuration()
        elif button_id == "setup-finish":
            self.exit(True)


def run_setup(path: Path | None = None) -> bool:
    """Launch the guided configuration application."""
    inject_truststore_into_ssl()
    return bool(SetupApp(path or default_setup_config_path()).run())
