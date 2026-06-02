from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from langchain.chat_models import BaseChatModel, init_chat_model

from ursa.cli.config import ModelConfig, deep_interp_env
from ursa.security import group_environments_dir


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
    """

    name: str
    role: str = "Team member"
    agent: str = "ExecutionAgent"
    model: ModelConfig | None = None
    config: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EnvironmentMemberConfig":
        raw = dict(data)
        model = raw.get("model")
        if isinstance(model, Mapping):
            raw["model"] = ModelConfig.model_validate(model)
        return cls(**raw)


@dataclass(frozen=True)
class AgentTeamConfig:
    """YAML-loadable configuration for an Agent Team environment."""

    name: str
    group: str = "default"
    description: str | None = None
    pi: EnvironmentMemberConfig = field(
        default_factory=lambda: EnvironmentMemberConfig(
            name="pi", role="Principal investigator", agent="ExecutionAgent"
        )
    )
    members: list[EnvironmentMemberConfig] = field(default_factory=list)
    workspace: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentTeamConfig":
        raw = dict(data)
        if "pi" in raw and isinstance(raw["pi"], Mapping):
            raw["pi"] = EnvironmentMemberConfig.from_mapping(raw["pi"])
        if "members" in raw:
            raw["members"] = [
                EnvironmentMemberConfig.from_mapping(member)
                for member in raw["members"]
            ]
        return cls(**raw)


@dataclass(frozen=True)
class AgentSymposiumConfig:
    """YAML-loadable configuration for an Agent Symposium environment."""

    name: str
    group: str = "default"
    description: str | None = None
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
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentSymposiumConfig":
        raw = dict(data)
        if "organizer" in raw and isinstance(raw["organizer"], Mapping):
            raw["organizer"] = EnvironmentMemberConfig.from_mapping(
                raw["organizer"]
            )
        if "members" in raw:
            raw["members"] = [
                EnvironmentMemberConfig.from_mapping(member)
                for member in raw["members"]
            ]
        return cls(**raw)


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
    return group_environments_dir(group) / "agent_symposiums" / name


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
