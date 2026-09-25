from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain.chat_models import BaseChatModel, init_chat_model
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from ursa.cli.config import (
    InferenceProviderConfig,
    ModelConfig,
    deep_interp_env,
)
from ursa.security import (
    enforce_group_base_url_policy,
    group_environments_dir,
    validate_group_name,
)


@dataclass(frozen=True)
class EnvironmentMemberConfig:
    """Configuration for one agent or nested environment member.

    YAML fields:
      name: stable member name used in prompts/tool names/persistence
      role: human-readable role or specialty
      agent: Python class path or URSA agent class name, e.g. ExecutionAgent
      model: optional ModelConfig-compatible mapping for this member
      config: kwargs passed to the agent/environment constructor
      prompt: optional extra role/system guidance included in delegated tasks
      reviewer: whether this member participates in symposium review phases
    """

    name: str
    role: str = "Team member"
    agent: str = "ExecutionAgent"
    model: ModelConfig | None = None
    config: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    reviewer: bool = True

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        inference_providers: Mapping[str, InferenceProviderConfig]
        | None = None,
        group: str = "default",
    ) -> EnvironmentMemberConfig:
        raw = dict(data)
        model = raw.get("model")
        if isinstance(model, Mapping):
            model_config = ModelConfig.model_validate(model)
            model_config = model_config.resolve_inference_provider(
                dict(inference_providers or {})
            )
            enforce_group_base_url_policy(model_config.base_url, group)
            raw["model"] = model_config
        return cls(**raw)


def _inference_providers(
    data: Mapping[str, Any],
) -> dict[str, InferenceProviderConfig]:
    return {
        name: InferenceProviderConfig.model_validate(config)
        for name, config in data.items()
    }


@dataclass(frozen=True)
class AgentTeamConfig:
    """YAML-loadable configuration for an Agent Team environment."""

    name: str
    group: str = "default"
    description: str | None = None
    inference_providers: dict[str, InferenceProviderConfig] = field(
        default_factory=dict
    )
    pi: EnvironmentMemberConfig = field(
        default_factory=lambda: EnvironmentMemberConfig(
            name="pi", role="Principal investigator", agent="ExecutionAgent"
        )
    )
    members: list[EnvironmentMemberConfig] = field(default_factory=list)
    workspace: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AgentTeamConfig:
        raw = dict(data)
        providers = _inference_providers(raw.get("inference_providers") or {})
        raw["inference_providers"] = providers
        group = str(raw.get("group") or "default")
        if "pi" in raw and isinstance(raw["pi"], Mapping):
            raw["pi"] = EnvironmentMemberConfig.from_mapping(
                raw["pi"], providers, group
            )
        if "members" in raw:
            raw["members"] = [
                EnvironmentMemberConfig.from_mapping(member, providers, group)
                for member in raw["members"]
            ]
        return cls(**raw)


@dataclass(frozen=True)
class AgentSymposiumConfig:
    """YAML-loadable configuration for an Agent Symposium environment."""

    name: str
    group: str = "default"
    description: str | None = None
    inference_providers: dict[str, InferenceProviderConfig] = field(
        default_factory=dict
    )
    organizer: EnvironmentMemberConfig = field(
        default_factory=lambda: EnvironmentMemberConfig(
            name="organizer", role="Symposium organizer", agent="ChatAgent"
        )
    )
    members: list[EnvironmentMemberConfig] = field(default_factory=list)
    workspace: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    revision_rounds: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AgentSymposiumConfig:
        raw = dict(data)
        providers = _inference_providers(raw.get("inference_providers") or {})
        raw["inference_providers"] = providers
        group = str(raw.get("group") or "default")
        if "organizer" in raw and isinstance(raw["organizer"], Mapping):
            raw["organizer"] = EnvironmentMemberConfig.from_mapping(
                raw["organizer"], providers, group
            )
        if "members" in raw:
            raw["members"] = [
                EnvironmentMemberConfig.from_mapping(member, providers, group)
                for member in raw["members"]
            ]
        return cls(**raw)


class AgentEloConfig(BaseModel):
    """Validated configuration for a new or resumed Elo environment."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True
    )

    name: str
    group: str = "default"
    description: str | None = None
    inference_providers: dict[str, InferenceProviderConfig] = Field(
        default_factory=dict
    )
    members: list[EnvironmentMemberConfig] = Field(default_factory=list)
    workspace: str | None = None
    defaults: dict[str, Any] = Field(default_factory=dict)
    initial_rating: float = Field(default=1500.0, allow_inf_nan=False)
    k_factor: float = Field(default=32.0, gt=0, allow_inf_nan=False)
    deaths_per_round: int = Field(default=1, ge=0)
    seed: int | None = None
    generations: int = Field(default=1, ge=1)
    restart_from_json: str | None = None
    member_timeout_seconds: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    judge_prompt: str | None = None

    @field_validator("workspace", "restart_from_json", mode="before")
    @classmethod
    def _path_to_string(cls, value: Any) -> Any:
        return str(value) if isinstance(value, Path) else value

    @model_validator(mode="before")
    @classmethod
    def _resolve_members(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw = dict(value)
        group = raw.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError("Group name must be a string")
        raw["group"] = validate_group_name(group)
        provider_data = raw.get("inference_providers") or {}
        if not isinstance(provider_data, Mapping):
            raise ValueError("inference_providers must be a mapping")
        providers = _inference_providers(provider_data)
        raw["inference_providers"] = providers
        members = raw.get("members", [])
        if isinstance(members, (list, tuple)):
            resolved = []
            for member in members:
                if isinstance(member, Mapping):
                    member = dict(member)
                    model = member.get("model")
                elif isinstance(member, EnvironmentMemberConfig):
                    model = member.model
                else:
                    resolved.append(member)
                    continue
                if model is not None:
                    if not isinstance(model, ModelConfig):
                        model = ModelConfig.model_validate(model)
                    if not model._inference_provider_resolved:
                        model = model.resolve_inference_provider(providers)
                    enforce_group_base_url_policy(model.base_url, raw["group"])
                    if isinstance(member, dict):
                        member["model"] = model
                    else:
                        member = replace(member, model=model)
                resolved.append(member)
            raw["members"] = resolved
        return raw

    @model_validator(mode="after")
    def _validate_population(self) -> Self:
        seen: set[str] = set()
        for member in self.members:
            self.validate_member_config(member)
            if member.name in seen:
                raise ValueError(
                    f"Elo member name {member.name!r} is duplicated. "
                    "Each member must have a unique name."
                )
            seen.add(member.name)
        if self.restart_from_json is None:
            self.validate_population_size(len(self.members))
        return self

    @staticmethod
    def validate_population_size(population_size: int) -> None:
        if population_size < 2:
            raise ValueError(
                "AgentEloEnvironment requires at least two active members."
            )
        if population_size % 2:
            raise ValueError(
                "AgentEloEnvironment requires an even number of active members. "
                f"Received {population_size}."
            )

    @staticmethod
    def validate_member_config(member: EnvironmentMemberConfig) -> None:
        """Validate member settings for both new and restored populations."""
        if "workspace" in (member.config or {}):
            raise ValueError(
                f"Member {member.name!r} sets config.workspace. "
                "AgentEloEnvironment manages member workspaces automatically. "
                "Set the top-level workspace instead."
            )
        if "agent_name" in (member.config or {}):
            raise ValueError(
                f"Member {member.name!r} sets config.agent_name. "
                "AgentEloEnvironment generates persistent agent identities "
                "automatically. Remove config.agent_name and use the "
                "member's name field instead."
            )

    @model_serializer(mode="wrap")
    def _serialize_resolved_models(self, handler):
        # Resolved endpoints are self-contained. Retaining the provider reference
        # would make the saved model invalid when read from YAML or dashboard JSON.
        data = handler(self)
        for member in data.get("members", []):
            model = member.get("model")
            if isinstance(model, dict):
                model.pop("inference_provider", None)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Compatibility entry point for callers loading a mapping."""
        return cls.model_validate(data)

    @classmethod
    def from_source(
        cls,
        config: Self | Mapping[str, Any] | str | Path | None = None,
        **overrides: Any,
    ) -> Self:
        """Merge constructor overrides before validating the final configuration.

        None overrides retain the supplied configuration, as in the environment
        constructor. Existing model instances are preserved during merging.
        """
        if isinstance(config, (str, Path)):
            raw = load_yaml_mapping(config)
        elif isinstance(config, cls):
            raw = {name: getattr(config, name) for name in cls.model_fields}
        elif isinstance(config, Mapping):
            raw = dict(config)
        elif config is None:
            raw = {"name": "agent_elo"}
        else:
            raise TypeError(
                "Elo config must be a config object, mapping, or YAML path"
            )
        raw.update({
            key: value for key, value in overrides.items() if value is not None
        })
        return cls.model_validate(raw)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with URSA-style environment interpolation."""
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML file {p} must contain a top-level mapping.")
    return deep_interp_env(dict(data))


def load_team_config(path: str | Path) -> AgentTeamConfig:
    return AgentTeamConfig.from_mapping(load_yaml_mapping(path))


def load_symposium_config(path: str | Path) -> AgentSymposiumConfig:
    return AgentSymposiumConfig.from_mapping(load_yaml_mapping(path))


def team_cache_dir(group: str, name: str) -> Path:
    """Return the persistent configuration directory for a named team."""
    return group_environments_dir(group) / "agent_teams" / name


def symposium_cache_dir(group: str, name: str) -> Path:
    """Return the persistent configuration directory for a named symposium."""
    return group_environments_dir(group) / "agent_symposia" / name


def save_team_config(
    config: AgentTeamConfig, path: str | Path | None = None
) -> Path:
    """Persist a team configuration under ~/.cache/ursa/<group>/environments by default."""
    target = (
        Path(path).expanduser()
        if path
        else team_cache_dir(config.group, config.name) / "team.yaml"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(_dataclass_to_plain(config), sort_keys=False),
        encoding="utf-8",
    )
    return target


def save_symposium_config(
    config: AgentSymposiumConfig, path: str | Path | None = None
) -> Path:
    """Persist a symposium configuration under ~/.cache/ursa/<group>/environments by default."""
    target = (
        Path(path).expanduser()
        if path
        else symposium_cache_dir(config.group, config.name) / "symposium.yaml"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(_dataclass_to_plain(config), sort_keys=False),
        encoding="utf-8",
    )
    return target


def _dataclass_to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _dataclass_to_plain(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, list):
        return [_dataclass_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _dataclass_to_plain(v) for k, v in value.items()}
    return value


def load_object(path_or_name: str) -> Any:
    """Load a class by URSA short name or full module path.

    Short names are resolved first against ``ursa.agents`` and then against
    ``ursa.environments``. This lets YAML use concise names such as
    ``ExecutionAgent`` or ``AgentTeamEnvironment`` while still allowing fully
    qualified custom class paths.
    """
    if "." not in path_or_name:
        for module_name in ("ursa.agents", "ursa.environments"):
            module = importlib.import_module(module_name)
            try:
                return getattr(module, path_or_name)
            except AttributeError:
                continue
        raise AttributeError(
            f"Could not resolve {path_or_name!r} in ursa.agents or "
            "ursa.environments. Use a full Python import path for custom classes."
        )
    module_name, attr_name = path_or_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def make_llm(
    default_llm: BaseChatModel,
    model_config: ModelConfig | Mapping[str, Any] | None,
) -> BaseChatModel:
    """Return a member-specific model if configured, otherwise the default."""
    if model_config is None:
        return default_llm
    if isinstance(model_config, Mapping):
        model_config = ModelConfig.model_validate(model_config)
    return init_chat_model(**model_config.kwargs)


def load_elo_config(
    path: str | Path,
) -> AgentEloConfig:
    return AgentEloConfig.from_mapping(load_yaml_mapping(path))


def elo_cache_dir(
    group: str,
    name: str,
) -> Path:
    """Return the persistent configuration directory for a named Elo environment."""
    return group_environments_dir(group) / "agent_elo" / name


def save_elo_config(
    config: AgentEloConfig,
    path: str | Path | None = None,
) -> Path:
    """Persist an Elo environment configuration."""
    target = (
        Path(path).expanduser()
        if path
        else elo_cache_dir(
            config.group,
            config.name,
        )
        / "elo.yaml"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        yaml.safe_dump(
            _dataclass_to_plain(config),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return target
