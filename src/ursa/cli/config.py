import json
import logging
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from os import environ
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

    ssl_verify: bool | None = None
    """Flag for verifying SSL certs. during API access.

    ``None`` inherits the inference-provider value, or enables verification
    when no provider value is configured.
    """

    def _model_provider(self) -> str:
        return self.model.split(":", 1)[0]

    def resolve_inference_provider(
        self, providers: dict[str, InferenceProviderConfig]
    ) -> Self:
        """Return a copy with provider defaults merged under model-specific overrides."""
        if self.inference_provider is None:
            if self.ssl_verify is None:
                return self.model_copy(update={"ssl_verify": True})
            return self

        provider_name = self.inference_provider
        provider_config = providers.get(provider_name)
        if provider_config is None:
            raise ValueError(f"Unknown inference_provider '{provider_name}'")
        assert isinstance(provider_config, InferenceProviderConfig)

        values = {
            **provider_config.model_dump(mode="python", exclude_unset=True),
            **self.model_dump(mode="python", exclude_none=True),
            "inference_provider": None,
        }
        values.setdefault("ssl_verify", True)
        return type(self).model_validate(values)

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
        kwargs = {k: v for k, v in self.model_dump().items() if v is not None}
        ssl_verify = kwargs.pop("ssl_verify", True)
        kwargs.pop("inference_provider", None)
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

    def init_chat_model(
        self,
        inference_providers: dict[str, InferenceProviderConfig] | None = None,
    ) -> BaseChatModel:
        resolved = self.resolve_inference_provider(inference_providers or {})
        llm = init_chat_model(**resolved.kwargs)
        resolved.check_instantiated_model(llm)
        return llm


class EmbModelConfig(ModelConfig):
    """Configuration for instantiating an embeddings model"""

    def init_embedding(
        self,
        inference_providers: dict[str, InferenceProviderConfig] | None = None,
    ) -> Embeddings:
        resolved = self.resolve_inference_provider(inference_providers or {})
        emb = init_embeddings(**resolved.kwargs)
        resolved.check_instantiated_model(emb)
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

    llm_model: ChatModelConfig = Field(
        default_factory=lambda: ChatModelConfig(
            model="openai:gpt-5.4",
        )
    )
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
        if mc.inference_provider not in info.data["inference_providers"]:
            raise ValueError(
                f"Unknown inference_provider '{mc.inference_provider}'"
            )
        return mc

    def model_post_init(self, __context):
        """Handle temporary workspace creation post validation."""

    def update(self, other: "UrsaConfig") -> "UrsaConfig":
        """Merge non-default values from another config into this config."""
        defaults = type(self)().model_dump(mode="python")
        updates = dict_diff(defaults, other.model_dump(mode="python"))
        merged = deep_merge_dicts(self.model_dump(mode="python"), updates)
        updated = type(self).model_validate(merged)

        for field_name in type(self).model_fields:
            setattr(self, field_name, getattr(updated, field_name))

        if other._temp_workspace and other.workspace == self.workspace:
            self._temp_workspace = other._temp_workspace
        elif (
            self._temp_workspace
            and Path(self._temp_workspace.name) != self.workspace
        ):
            self._temp_workspace = None
        return self

    @classmethod
    def from_namespace(cls, cfg: Namespace):
        """Instantiate from a jsonargparse namespace."""
        return cls.model_validate(cfg.as_dict(), extra="ignore")

    @classmethod
    def from_file(cls, path: Path):
        return cls.model_validate(load_config_file(path))

    @classmethod
    def resolve_config(cls, config: "UrsaConfig") -> "UrsaConfig":
        return resolve_ursa_config(config)

    @field_serializer("workspace")
    def serialize_workspace(self, workspace: Path, _info):
        return workspace.as_posix()

    @field_serializer("mcp_servers")
    def serialize_mcp_servers(
        self, mcp_servers: dict[str, ServerParameters], _info
    ):
        return {
            server: _serialize_server_config(config)
            for server, config in mcp_servers.items()
        }


def load_config_file(path: Path) -> dict[str, Any]:
    """Load and validate raw config-file data before config merging."""
    loader = yaml.safe_load if path.suffix in [".yaml", ".yml"] else json.load
    with open(path, "r") as fid:
        data = loader(fid)

    return deep_interp_env(data)


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
    """Resolve derived config state after validation and merge steps."""
    resolved = config.model_copy(deep=True)

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

    if str(resolved.workspace) == "tmp" and not resolved.workspace.exists():
        temp_workspace = TemporaryDirectory(prefix="ursa")
        resolved.workspace = Path(temp_workspace.name)
        resolved._temp_workspace = temp_workspace

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
