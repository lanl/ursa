"""Reusable configuration and resolution for one Harbor invocation.

Campaign design, output naming, and scheduling belong to the caller. Resolution
does not read credentials, launch a process, or modify the input configuration.
"""

from __future__ import annotations

import json
import string
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    model_validator,
)

from ursa.cli.config import InferenceProviderConfig, UrsaConfig
from ursa.util.secrets import SecretReference

if TYPE_CHECKING:
    from harbor.models.trial.config import AgentConfig as HarborAgentConfig

URSA_AGENT = "ursa.integrations.harbor:UrsaHarborAgent"
NonEmptyString = Annotated[str, Field(min_length=1)]
EnvironmentVariable = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class AgentKwargs(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


class AgentConfig(BaseModel):
    """Harness settings, independent of a benchmark or provider catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    harbor_agent: NonEmptyString
    kwargs: AgentKwargs = Field(default_factory=AgentKwargs)
    provider_kwargs: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict
    )
    env: dict[EnvironmentVariable, str] = Field(default_factory=dict)
    setup_timeout_multiplier: Annotated[float, Field(gt=0)] = 1.2
    reasoning_kwarg: NonEmptyString | None = None
    api_key_env: EnvironmentVariable = "URSA_HARBOR_API_KEY"
    native_config: dict[str, JsonValue] | str = Field(default_factory=dict)
    model_template: str | None = None
    api_format: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        reserved = {
            "config",
            "config_file",
            "config_only",
            "base_url",
            "model_api_key_name",
            "model_api_format",
        }
        for options in [
            self.kwargs.model_dump(),
            *self.provider_kwargs.values(),
        ]:
            if conflict := reserved.intersection(options):
                raise ValueError(
                    f"Use native/provider configuration for {sorted(conflict)}"
                )
            if self.reasoning_kwarg in options:
                raise ValueError("reasoning setting is also defined in kwargs")
        if self.model_template is not None:
            for _, name, spec, conversion in string.Formatter().parse(
                self.model_template
            ):
                if name is not None and (
                    name not in {"provider", "model_provider", "model"}
                    or spec
                    or conversion
                ):
                    raise ValueError("Invalid model_template placeholder")
        return self


class ModelConfig(BaseModel):
    """One model selection; names may contain slashes or provider tags."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: NonEmptyString
    provider: NonEmptyString


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = (
            tomllib.loads(text)
            if path.suffix == ".toml"
            else json.loads(text)
            if path.suffix == ".json"
            else yaml.safe_load(text)
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration {path} must be a mapping")
    return data


def load_agent_config(path: Path) -> AgentConfig:
    """Resolve a native-config filename relative to this agent file."""
    try:
        agent = AgentConfig.model_validate(_read_mapping(path))
        if isinstance(agent.native_config, str):
            native = (path.resolve().parent / agent.native_config).resolve()
            agent = agent.model_copy(update={"native_config": str(native)})
        return agent
    except ValueError as exc:
        raise ValueError(f"Invalid agent configuration {path}: {exc}") from exc


def load_provider_config(path: Path, name: str) -> InferenceProviderConfig:
    try:
        return InferenceProviderConfig.model_validate(
            _read_mapping(path)
        ).resolve(name)
    except ValueError as exc:
        raise ValueError(
            f"Invalid provider configuration {path}: {exc}"
        ) from exc


@dataclass(frozen=True)
class ResolvedRun:
    """Inspectable Harbor settings plus a lazily resolved credential."""

    task: Path
    agent: HarborAgentConfig
    setup_timeout_multiplier: float
    _credential: SecretReference | SecretStr | None = field(repr=False)
    _provider: str = field(repr=False)

    def environment(self) -> dict[str, str]:
        """Materialize only the selected credential immediately before launch."""
        if self._credential is None:
            return {}
        secret = (
            self._credential.get_secret_value(self._provider)
            if isinstance(self._credential, SecretReference)
            else self._credential.get_secret_value()
        )
        if not secret:
            raise ValueError(
                f"Empty credential for provider {self._provider!r}"
            )
        return {self.agent.model_api_key_name: secret}

    def cli_args(self) -> list[str]:
        """Arguments for ``harbor run``, excluding the provider credential."""
        args = [
            "--path",
            str(self.task),
            "--agent",
            self.agent.name or self.agent.import_path,
            "--model",
            self.agent.model_name,
            "--agent-setup-timeout-multiplier",
            str(self.setup_timeout_multiplier),
        ]
        for flag, value in (
            ("--base-url", self.agent.base_url),
            ("--api-key-name", self.agent.model_api_key_name),
            ("--api-format", self.agent.model_api_format),
        ):
            if value is not None:
                args.extend((flag, str(value)))
        for key, value in self.agent.kwargs.items():
            rendered = value if isinstance(value, str) else json.dumps(value)
            args.extend(("--agent-kwarg", f"{key}={rendered}"))
        for key, value in self.agent.env.items():
            args.extend(("--agent-env", f"{key}={value}"))
        for host in self.agent.extra_allowed_hosts:
            args.extend(("--allow-agent-host", host))
        return args


def resolve_run(
    task: Path,
    agent: AgentConfig,
    model: ModelConfig,
    provider: InferenceProviderConfig,
    *,
    reasoning: str | None = None,
    base_url: str | None = None,
    base_dir: Path | None = None,
) -> ResolvedRun:
    """Resolve one run using Harbor's options/capabilities and native config.

    ``base_url`` overrides the provider endpoint, e.g. for a timing relay.
    Native mappings are passed directly to Harbor, which owns format-specific
    materialization in the task environment. File inputs may be YAML/JSON/TOML.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig as HarborAgentConfig
    from harbor.trial.network_policy import inference_endpoint_host

    root = base_dir or Path.cwd()
    native = (
        _read_mapping(root / agent.native_config)
        if isinstance(agent.native_config, str)
        else deepcopy(agent.native_config)
    )
    options = agent.kwargs.model_dump(exclude_none=True)
    options.update(deepcopy(agent.provider_kwargs.get(model.provider, {})))
    endpoint = base_url if base_url is not None else provider.base_url
    if endpoint is not None:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) URL")
    model_provider = (provider.model_extra or {}).get(
        "model_provider", model.provider
    )
    native_config = None
    if agent.harbor_agent == URSA_AGENT:
        connection = provider.model_dump(
            mode="json", exclude={"api_key", "base_url"}, exclude_none=True
        )
        if provider.api_key is not None:
            connection["api_key"] = {"env": agent.api_key_env}
        native.setdefault("inference_providers", {})[model.provider] = (
            connection
        )
        if reasoning is not None:
            native.setdefault("llm_model", {})["reasoning_effort"] = reasoning
        # Validate native settings without resolving secrets or model clients.
        # Validate a copy: normalizing the layer would insert default models
        # and providers, turning implicit defaults into explicit requirements.
        native_config = UrsaConfig.model_validate(deepcopy(native))
        template = agent.model_template or "{provider}/{model}"
    else:
        template = agent.model_template or "{model_provider}/{model}"
    if native:
        options["config"] = native
    harbor_agent = HarborAgentConfig(
        name=agent.harbor_agent,
        model_name=template.format(
            provider=model.provider,
            model_provider=model_provider,
            model=model.name,
        ),
        base_url=endpoint,
        model_api_format=agent.api_format,
        model_api_key_name=(
            agent.api_key_env if provider.api_key is not None else None
        ),
        kwargs=options,
        env=dict(agent.env),
    )
    harness = AgentFactory.get_agent_class_from_config(harbor_agent)
    if reasoning is not None and agent.harbor_agent != URSA_AGENT:
        key = agent.reasoning_kwarg
        if key is None:
            if not harness.capabilities.reasoning_effort:
                raise ValueError(
                    f"{agent.harbor_agent} does not support reasoning"
                )
            key = "reasoning_effort"
        if key in options:
            raise ValueError(f"Reasoning conflicts with kwargs.{key}")
        options[key] = reasoning
        harbor_agent.kwargs = options
    if native and not harness.capabilities.native_config:
        raise ValueError(
            f"{agent.harbor_agent} does not support native configuration"
        )
    harness.parse_options(
        {**options, **({"base_url": endpoint} if endpoint else {})}, agent.env
    )
    if native_config is not None and native_config.emb_model is not None:
        if endpoint is not None:
            native_config.inference_providers[
                model.provider
            ].base_url = endpoint
        embedding = native_config.emb_model.resolve_inference_provider(
            native_config.inference_providers
        )
        if embedding_host := inference_endpoint_host(embedding.base_url):
            # Harbor replaces automatic inference discovery when an explicit
            # list is supplied, so retain the chat endpoint alongside embeddings.
            chat_host = inference_endpoint_host(
                endpoint
                or harness.MODEL_CONNECTION.default_base_url(
                    model_provider, agent.api_format
                )
            )
            if chat_host is None:
                raise ValueError(
                    "Set base_url to allow chat and embedding endpoints"
                )
            harbor_agent.extra_allowed_hosts = list(
                dict.fromkeys((chat_host, embedding_host))
            )
    return ResolvedRun(
        task=(root / task).resolve(),
        agent=harbor_agent,
        setup_timeout_multiplier=agent.setup_timeout_multiplier,
        _credential=provider.api_key,
        _provider=model.provider,
    )
