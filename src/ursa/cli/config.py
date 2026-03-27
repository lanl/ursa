import json
import re
from copy import deepcopy
from dataclasses import dataclass
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import httpx
import yaml
from jsonargparse import Namespace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer

from ursa.util.mcp import ServerParameters, _serialize_server_config

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical"
]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    """Model provider and model name.
    Use the format <provider>:<model-name>
    """
    base_url: str | None = None
    """Base URL for model API access"""

    api_key_env: str | None = None
    """Environmental variable containing the API key for this session"""

    max_completion_tokens: int | None = None
    """Maximum tokens for LLM to output"""

    ssl_verify: bool = True
    """Flag for verifying SSL certs. during API access"""

    @property
    def kwargs(self) -> dict:
        """Return a dict suitable for init_chat_model/init_embedding_model
        Removes parameters set to `None`
        """
        kwargs = {k: v for k, v in self.model_dump().items() if v is not None}
        if kwargs.pop("ssl_verify", None) is False:
            kwargs["http_client"] = httpx.Client(verify=False)
        if api_key_env := kwargs.pop("api_key_env", None):
            kwargs["api_key"] = environ.get(api_key_env, None)
        return kwargs


class ModelsDefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    profile: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    profile: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choices: list[str] = Field(default_factory=list)
    default: str | None = None
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    defaults: ModelsDefaultsConfig = Field(default_factory=ModelsDefaultsConfig)
    agents: dict[str, ModelsAgentConfig] = Field(default_factory=dict)


class UrsaConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    _temp_workspace: TemporaryDirectory | None = PrivateAttr(default=None)

    workspace: Path = Field(
        default_factory=lambda: Path("ursa_workspace"),
    )
    """Directory to store URSA's output."""

    thread_id: str | None = None
    """ Thread ID for persistence """

    llm_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="openai:gpt-5.2",
            max_completion_tokens=5000,
        )
    )
    """Default LLM"""

    models: ModelsConfig | None = None
    """Optional richer model-selection config."""

    emb_model: ModelConfig | None = None

    agent_config: dict[str, dict[str, Any]] | None = None
    """ Configuration options for URSA Agents """

    mcp_servers: dict[str, ServerParameters] = Field(default_factory=dict)
    """MCP Servers to connect to Ursa."""

    def model_post_init(self, __context):
        """Handle temporary workspace creation and derived model config."""
        if str(self.workspace) == "tmp" and not self.workspace.exists():
            temp_workspace = TemporaryDirectory(prefix="ursa")
            self.workspace = Path(temp_workspace.name)
            self._temp_workspace = temp_workspace

        if self.models is not None:
            self.llm_model = resolve_llm_model_config(
                self.models,
                base_llm_model=self.llm_model,
            )

    @classmethod
    def from_namespace(cls, cfg: Namespace):
        """Instantiate from a jsonargparse namespace."""
        return cls.model_validate(cfg.as_dict(), extra="ignore")

    @classmethod
    def from_file(cls, path: Path):
        loader = (
            yaml.safe_load if path.suffix in [".yaml", ".yml"] else json.load
        )
        with open(path, "r") as fid:
            data = loader(fid)

        data = deep_interp_env(data)

        return cls.model_validate(data)

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


def get_default_models(models_cfg: ModelsConfig | None) -> tuple[str, ...]:
    fallback = (
        "openai:gpt-5",
        "openai:gpt-5-mini",
        "openai:o3",
        "openai:o3-mini",
    )
    if models_cfg and models_cfg.choices:
        return tuple(models_cfg.choices)
    return fallback


def get_default_model(models_cfg: ModelsConfig | None) -> str | None:
    if models_cfg is None:
        return None
    return models_cfg.default


def resolve_model_choice_from_models_cfg(
    model_choice: str, models_cfg: ModelsConfig | None
) -> ModelConfig:
    """
    Resolve a model choice like:
      - openai:gpt-5.2
      - my_endpoint:openai/gpt-oss-120b

    using the richer `models:` config block and return a concrete ModelConfig.
    """
    if ":" in model_choice:
        alias, pure_model = model_choice.split(":", 1)
    else:
        alias, pure_model = "openai", model_choice

    providers = models_cfg.providers if models_cfg else {}
    prov = providers.get(alias, {})

    model_provider = prov.get("model_provider", alias)

    model_kwargs: dict[str, Any] = {"model": f"{model_provider}:{pure_model}"}

    if prov.get("base_url"):
        model_kwargs["base_url"] = prov["base_url"]
    if prov.get("api_key_env"):
        model_kwargs["api_key_env"] = prov["api_key_env"]

    return ModelConfig.model_validate(model_kwargs, extra="allow")


def resolve_llm_model_config(
    models_cfg: ModelsConfig | None,
    base_llm_model: ModelConfig | None = None,
    agent_name: str | None = None,
) -> ModelConfig:
    """
    Resolve the richer `models:` block into a single concrete ModelConfig.

    Merge order:
      1) base_llm_model or default fallback
      2) models.defaults.params
      3) models.profiles[defaults.profile]
      4) models.agents[agent_name].profile
      5) models.agents[agent_name].params

    Model selection:
      - agent-specific model if present
      - models.defaults.model if present
      - models.default if present
      - base_llm_model.model if present
      - fallback openai:gpt-5.2
    """
    if base_llm_model is None:
        base_llm_model = ModelConfig(
            model="openai:gpt-5.2",
            max_completion_tokens=5000,
        )

    merged_kwargs = base_llm_model.model_dump(exclude_none=True)

    if not models_cfg:
        return ModelConfig.model_validate(merged_kwargs, extra="allow")

    merged_kwargs = deep_merge_dicts(
        merged_kwargs, models_cfg.defaults.params or {}
    )

    default_profile_name = models_cfg.defaults.profile
    if default_profile_name and default_profile_name in models_cfg.profiles:
        merged_kwargs = deep_merge_dicts(
            merged_kwargs, models_cfg.profiles[default_profile_name] or {}
        )

    agent_cfg = None
    if agent_name and agent_name in models_cfg.agents:
        agent_cfg = models_cfg.agents[agent_name]

    if agent_cfg and agent_cfg.profile:
        if agent_cfg.profile in models_cfg.profiles:
            merged_kwargs = deep_merge_dicts(
                merged_kwargs,
                models_cfg.profiles[agent_cfg.profile] or {},
            )

    if agent_cfg:
        merged_kwargs = deep_merge_dicts(merged_kwargs, agent_cfg.params or {})

    chosen_model = None
    if agent_cfg and agent_cfg.model:
        chosen_model = agent_cfg.model
    elif models_cfg.defaults.model:
        chosen_model = models_cfg.defaults.model
    elif models_cfg.default:
        chosen_model = models_cfg.default
    elif merged_kwargs.get("model"):
        chosen_model = merged_kwargs["model"]
    else:
        chosen_model = "openai:gpt-5.2"

    resolved_model_cfg = resolve_model_choice_from_models_cfg(
        chosen_model, models_cfg
    )

    final_kwargs = deep_merge_dicts(
        resolved_model_cfg.model_dump(exclude_none=True),
        {k: v for k, v in merged_kwargs.items() if k != "model"},
    )

    return ModelConfig.model_validate(final_kwargs, extra="allow")


ENV_SUB_REGEX = re.compile(r"\${(?P<env>\w+)(?::(?P<default>.+))?}")


def deep_interp_env(x: dict[str, Any] | str | Any):
    """Interpolate all environment variables in stored keys"""
    if isinstance(x, dict):
        return {k: deep_interp_env(v) for k, v in x.items()}
    elif isinstance(x, str):
        return interpolate_env(x)
    else:
        return x


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


def get_config_planning_mode(cfg: dict[str, Any] | Any) -> str | None:
    config_mode = None

    if isinstance(cfg, dict):
        planning_cfg = cfg.get("planning")
        if isinstance(planning_cfg, dict):
            config_mode = planning_cfg.get("mode")
        if not config_mode:
            config_mode = cfg.get("planning_mode")
        return config_mode

    planning_cfg = getattr(cfg, "planning", None)
    if isinstance(planning_cfg, dict):
        config_mode = planning_cfg.get("mode")
    if not config_mode:
        config_mode = getattr(cfg, "planning_mode", None)
    return config_mode


def get_models_cfg(cfg: dict[str, Any] | UrsaConfig) -> ModelsConfig | None:
    if isinstance(cfg, dict):
        raw = cfg.get("models")
        if not raw:
            return None
        return ModelsConfig.model_validate(raw)
    return cfg.models
