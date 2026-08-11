import json
import logging
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, Self

import yaml
from jsonargparse import Namespace
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from ursa.security import enforce_group_base_url_policy
from ursa.util.http import (
    build_httpx_async_client,
    build_httpx_client,
    httpx_verify_value,
)
from ursa.util.mcp import ServerParameters, _serialize_server_config

logger = logging.getLogger(__name__)

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical"
]


def _strip_blank_optional_strings(value: Any) -> str | Any | None:
    """Normalize blank strings to ``None`` after stripping whitespace.
    Non-string values are returned unchanged.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    return value or None


class InferenceProviderConfig(BaseModel):
    """Reusable provider-level inference settings for model configs."""

    model_config = ConfigDict(extra="allow")

    base_url: Annotated[
        str | None, AfterValidator(_strip_blank_optional_strings)
    ] = None
    """Base URL for model API access"""

    api_key_env: Annotated[
        str | None, AfterValidator(_strip_blank_optional_strings)
    ] = None
    """Environmental variable containing the API key for this session"""

    ssl_verify: bool = True


class ModelConfig(BaseModel):
    """Configuration manager for LangChain's `init_*` factories."""

    model_config = ConfigDict(extra="allow")

    model: str
    """Model provider and model name.
    Use the format <provider>:<model-name>
    """
    base_url: Annotated[
        str | None, AfterValidator(_strip_blank_optional_strings)
    ] = None
    """Base URL for model API access"""

    api_key_env: Annotated[
        str | None, AfterValidator(_strip_blank_optional_strings)
    ] = None
    """Environmental variable containing the API key for this session"""

    inference_provider: Annotated[
        str | None, AfterValidator(_strip_blank_optional_strings)
    ] = None
    """Optional named inference provider to inherit shared settings from."""

    ssl_verify: bool = True
    """Flag for verifying SSL certs. during API access."""

    def _model_provider(self) -> str:
        return self.model.split(":", 1)[0]

    def resolve_inference_provider(
        self, providers: dict[str, InferenceProviderConfig]
    ) -> Self:
        """Return a copy with provider defaults merged under model-specific overrides."""
        if self.inference_provider is None:
            return self

        provider_name = self.inference_provider
        provider_config = providers.get(provider_name)
        if provider_config is None:
            raise ValueError(f"Unknown inference_provider '{provider_name}'")
        assert isinstance(provider_config, InferenceProviderConfig)

        provider_values = provider_config.model_dump(
            mode="python", exclude_unset=True
        )
        model_values = self.model_dump(mode="python", exclude_unset=True)

        return type(self).model_validate({
            **provider_values,
            **model_values,
            "inference_provider": None,
        })

    @staticmethod
    def _merge_provider_kwargs(
        kwargs: dict[str, Any], key: str, extra: dict[str, Any]
    ) -> None:
        current = kwargs.get(key)
        if current is None:
            kwargs[key] = extra
            return
        if isinstance(current, dict):
            kwargs[key] = {**extra, **current}

    @property
    def kwargs(self) -> dict:
        """Return a dict suitable for init_chat_model/init_embedding_model
        Removes parameters set to `None`
        """
        if self.inference_provider is not None:
            raise ValueError(
                f"Model config references unresolved inference provider "
                f"'{self.inference_provider}'"
            )
        kwargs = {k: v for k, v in self.model_dump().items() if v is not None}
        ssl_verify = kwargs.pop("ssl_verify", True)
        model_provider = self._model_provider()
        if model_provider in {"openai", "azure_openai"}:
            kwargs["http_client"] = build_httpx_client(verify=ssl_verify)
            kwargs["http_async_client"] = build_httpx_async_client(
                verify=ssl_verify
            )
        elif model_provider == "ollama":
            self._merge_provider_kwargs(
                kwargs,
                "client_kwargs",
                {"verify": httpx_verify_value(verify=ssl_verify)},
            )
        if api_key_env := kwargs.pop("api_key_env", None):
            try:
                kwargs["api_key"] = environ[api_key_env]
            except KeyError:
                logger.error(
                    f"Env variable '{api_key_env}' for {self.model}'s API key was not set"
                )
                sys.exit(1)
        return kwargs

    @staticmethod
    def _get_model_base_url(model) -> str | None:
        for attr in ["base_url", "api_base", "openai_api_base"]:
            if base_url := getattr(model, attr, None):
                return base_url
        logger.warning(
            f"Missing base_url for {model} ({model.__class__.__name__})"
        )

    def check_instantiated_model(self, model):
        """Validates that `model` matches the configuration of `self`"""
        if (
            self.base_url is not None
            and (model_url := self._get_model_base_url(model)) is not None
            and self.base_url != model_url
        ):
            logger.error(
                f"Model base url ({model_url}) and config ({self.base_url}) do not match"
            )


class ChatModelConfig(ModelConfig):
    """Configuration for instantiating a chat model"""

    model: str = "openai:gpt-5.4"

    max_completion_tokens: int | None = None
    """Maximum tokens for LLM to output"""

    @property
    def kwargs(self) -> dict:
        kwargs = super().kwargs
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        match self._model_provider():
            case "openai" | "azure_openai":
                kwargs.setdefault("use_responses_api", True)
        return kwargs

    def init_chat_model(self) -> BaseChatModel:
        llm = init_chat_model(**self.kwargs)
        self.check_instantiated_model(llm)
        return llm


class EmbModelConfig(ModelConfig):
    """Configuration for instantiating an embeddings model"""

    def init_embedding(self) -> Embeddings:
        emb = init_embeddings(**self.kwargs)
        self.check_instantiated_model(emb)
        return emb


class UrsaConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    _temp_workspace: TemporaryDirectory | None = PrivateAttr(default=None)

    workspace: Path = Field(
        default_factory=lambda: Path("."),
    )
    """Directory to store URSA's output."""

    agent_name: str | None = None
    """Name of the agent for persistence."""

    group: str | None = "default"
    """Security group for the agent to control information flow"""

    thread_id: str | None = None
    """ Thread ID for persistence """

    use_web: bool = False
    """Enable web-search tools for ChatAgent and ExecutionAgent."""

    inference_providers: dict[str, InferenceProviderConfig] = Field(
        default_factory=dict
    )
    """Named reusable inference provider configurations."""

    llm_model: ChatModelConfig = Field(default_factory=ChatModelConfig)
    """Default LLM"""

    emb_model: EmbModelConfig | None = None
    """Default Embedding model"""

    rag_tools: list[str] = Field(default_factory=list)
    """Persisted RAG agent names to bind as tools."""

    agent_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """ Configuration options for URSA Agents """

    mcp_servers: dict[str, ServerParameters] = Field(default_factory=dict)
    """MCP Servers to connect to Ursa."""

    @field_validator("rag_tools", mode="before")
    @classmethod
    def _normalize_rag_tools(cls, value):
        from ursa.rag.persistence import normalize_rag_tool_names

        return normalize_rag_tool_names(value)

    @field_validator("agent_config", mode="before")
    @classmethod
    def _normalize_agent_config(cls, value):
        if value is None:
            logger.warning(
                "Setting agent_config to null is deprecated; treating it as an empty mapping"
            )
            return {}
        return value

    @field_validator("llm_model", "emb_model", mode="after")
    @classmethod
    def _check_inference_provider(
        cls, mc: ModelConfig | None, info: ValidationInfo
    ) -> ModelConfig | None:
        """Validate that any referenced inference provider exists.

        This field validator relies on ``inference_providers`` being declared
        earlier on ``UrsaConfig``, making it available via ``info.data`` when
        validating ``llm_model`` and ``emb_model``.
        """
        if mc is None or mc.inference_provider is None:
            return mc
        providers = info.data.get("inference_providers")
        if providers is None:
            # Let the provider catalog's validation error remain the primary
            # error when that earlier field could not be resolved.
            return mc
        if mc.inference_provider not in providers:
            raise ValueError(
                f"Unknown inference_provider '{mc.inference_provider}'"
            )
        return mc

    @classmethod
    def from_file(cls, path: Path):
        return cls.model_validate(load_config_file(path))

    @field_serializer("workspace")
    def serialize_workspace(self, workspace: Path, _info):
        return workspace.as_posix()

    @field_serializer("mcp_servers")
    def serialize_mcp_servers(
        self, mcp_servers: dict[str, ServerParameters], info
    ):
        include_defaults = bool(
            info.context and info.context.get("include_defaults")
        )
        return {
            server: _serialize_server_config(
                config,
                exclude_defaults=not include_defaults,
                exclude_none=not include_defaults,
            )
            for server, config in mcp_servers.items()
        }


def load_config_file(path: Path) -> dict[str, Any]:
    """Load raw config-file data for merging before validation."""
    loader = yaml.safe_load if path.suffix in [".yaml", ".yml"] else json.load
    with open(path, "r") as fid:
        data = loader(fid)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file '{path}' must contain a mapping at its root"
        )
    return deep_interp_env(data)


def config_path_from_namespace(cfg: Namespace) -> Path | None:
    """Return a root or subcommand-local explicit config path."""
    config_path = getattr(cfg, "config", None)
    subcommand = cfg.get("subcommand", None)
    if subcommand is not None:
        cmd_cfg = cfg.get(subcommand, None)
        cmd_config_path = (
            getattr(cmd_cfg, "config", None) if cmd_cfg is not None else None
        )
        config_path = cmd_config_path or config_path
    return config_path


def xdg_config_search_paths() -> list[Path]:
    """Return candidate XDG config paths in increasing precedence order."""
    candidates: list[Path] = []
    # XDG_CONFIG_DIRS lists its most-preferred directory first, whereas config
    # merging consumes sources from lowest to highest precedence.
    path_list_separator = ";" if Path.cwd().drive else ":"
    config_dirs = getenv("XDG_CONFIG_DIRS", "/etc/xdg").split(
        path_list_separator
    )
    for config_dir in reversed(config_dirs):
        if config_dir:
            candidates.append(
                Path(config_dir).expanduser() / "ursa" / "config.yaml"
            )

    config_home = getenv("XDG_CONFIG_HOME")
    candidates.append(
        Path(config_home).expanduser() / "ursa" / "config.yaml"
        if config_home
        else Path.home() / ".config" / "ursa" / "config.yaml"
    )
    return candidates


def config_search_paths(cfg: Namespace, level: str = "final") -> list[Path]:
    """Return isolated or cumulative config paths for a precedence level."""
    cumulative = level.endswith("+")
    level = level.removesuffix("+")
    if level not in {"user", "file", "final"}:
        raise ValueError(f"Unknown config level '{level}'")
    explicit_path = config_path_from_namespace(cfg)

    seen: set[Path] = set()
    user_paths: list[Path] = []
    for candidate in xdg_config_search_paths():
        candidate = candidate.expanduser()
        if candidate == explicit_path or candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            user_paths.append(candidate)

    file_paths = [explicit_path] if explicit_path is not None else []

    if level == "user":
        return user_paths
    if level == "file" and not cumulative:
        return file_paths
    return [*user_paths, *file_paths]


def merge_ursa_config(
    cfg: Namespace,
    level: str = "final",
    overrides: Namespace | dict[str, Any] | None = None,
) -> UrsaConfig:
    """Merge sparse configuration sources, then validate the result once."""
    merged: dict[str, Any] = {}
    for config_path in config_search_paths(cfg, level):
        merged = deep_merge_dicts(merged, load_config_file(config_path))

    if level.removesuffix("+") == "final":
        if overrides is None:
            raise ValueError("Final config merging requires sparse overrides")
        override_values = (
            overrides.as_dict()
            if isinstance(overrides, Namespace)
            else overrides
        )
        merged = deep_merge_dicts(
            merged,
            {
                key: value
                for key, value in override_values.items()
                if key in UrsaConfig.model_fields
            },
        )

    return UrsaConfig.model_validate(merged)


@dataclass
class MCPServerConfig:
    """MCP Server Options"""

    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "localhost"
    """Host to bind for network transports (ignored for stdio)"""
    port: int = 8000
    """Port to bind for network transports (ignored for stdio)"""
    log_level: LoggingLevel = "info"


def dict_diff(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return the subset of candidate entries that differ from the reference."""
    missing = object()
    diff: dict[str, Any] = {}
    for key, value in candidate.items():
        ref_value = reference.get(key, missing)
        if isinstance(value, dict) and isinstance(ref_value, dict):
            nested = dict_diff(ref_value, value)
            if nested:
                diff[key] = nested
        elif isinstance(value, list) and isinstance(ref_value, list):
            if value != ref_value:
                diff[key] = value
        elif isinstance(value, tuple) and isinstance(ref_value, tuple):
            if value != ref_value:
                diff[key] = value
        elif ref_value is missing or value != ref_value:
            diff[key] = value
    return diff


def deep_merge_dicts(
    base: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge updates into base without mutating inputs."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)  # type: ignore[index]
        else:
            merged[key] = value
    return merged


ENV_SUB_REGEX = re.compile(r"\${(?P<env>\w+)(?::(?P<default>.+))?}")


def deep_interp_env(x: dict[str, Any] | str | Any):
    """Interpolate all environment variables in stored keys"""
    if isinstance(x, dict):
        return {k: deep_interp_env(v) for k, v in x.items()}
    elif isinstance(x, str):
        return interpolate_env(x)
    else:
        return x


def resolve_ursa_config(config: UrsaConfig) -> UrsaConfig:
    """Resolve and group-policy-check config after validation and merging."""
    # Copy public fields independently while retaining the same private
    # TemporaryDirectory owner. Deep-copying the owner duplicates its cleanup
    # finalizer and can remove the workspace while a resolved config still uses
    # it. Copying the nested model objects also preserves their fields-set state,
    # which provider resolution uses to distinguish defaults from overrides.
    resolved = config.model_copy(
        update={
            name: deepcopy(getattr(config, name))
            for name in type(config).model_fields
        }
    )

    if resolved.llm_model is not None:
        resolved.llm_model = resolved.llm_model.resolve_inference_provider(
            resolved.inference_providers
        )
        enforce_group_base_url_policy(
            resolved.llm_model.base_url, resolved.group
        )
    if resolved.emb_model is not None:
        resolved.emb_model = resolved.emb_model.resolve_inference_provider(
            resolved.inference_providers
        )
        enforce_group_base_url_policy(
            resolved.emb_model.base_url, resolved.group
        )

    if str(resolved.workspace) == "tmp":
        if resolved._temp_workspace is not None:
            resolved.workspace = Path(resolved._temp_workspace.name)
        elif not resolved.workspace.exists():
            resolved._temp_workspace = TemporaryDirectory(prefix="ursa")
            resolved.workspace = Path(resolved._temp_workspace.name)

    if resolved.use_web:
        for agent_name in ["chat", "execute", "deep_review", "prompt"]:
            resolved.agent_config.setdefault(agent_name, {}).setdefault(
                "use_web", True
            )

    return resolved


def interpolate_env(value: str) -> str:
    """
    Interpolate environment variables in a string

    Supported patterns:
        ${VAR}
            Replaced with the value of VAR if set, otherwise an empty string.

        ${VAR:DEFAULT}
            Replaced with the value of VAR if set; otherwise replaced with
            DEFAULT.

    Args:
        value: The input string containing zero or more environment
            variable expressions.

    Returns:
        The input string with all supported environment variable
        expressions expanded.
    """

    def interpolate_env(m: re.Match[str]) -> str:
        groups = m.groupdict("")
        return environ.get(groups["env"], default=groups["default"])

    return ENV_SUB_REGEX.sub(interpolate_env, value)
