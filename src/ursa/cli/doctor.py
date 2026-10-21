# ruff: noqa: TID251

"""Health checks and rendering for ``ursa self doctor``."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import IntEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr, ValidationError
from rich.console import Console
from rich.text import Text
from yaml import YAMLError

from ursa.cli.config import UrsaConfig
from ursa.util.health import (
    PROBE_TIMEOUT_SECONDS,
    probe_chat_config,
    probe_embedding_model,
    probe_inference_provider,
    probe_mcp_server,
    run_probe_with_deadline,
)
from ursa.util.secrets import SecretReference, SecretTemplate

_FAILURES = (Exception, SystemExit, asyncio.CancelledError)


class CheckStatus(IntEnum):
    OK, WARN, ERROR = range(3)

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def indicator(self) -> str:
        return ("✅", "⚠️", "❌")[self]

    @property
    def style(self) -> str:
        return ("green", "yellow", "red")[self]


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        locations = {
            ".".join(map(str, error["loc"])) or "configuration"
            for error in exc.errors(include_input=False)
        }
        return "Configuration validation failed at " + ", ".join(
            sorted(locations)
        )
    try:
        return str(exc).strip() or type(exc).__name__
    except BaseException:
        return type(exc).__name__


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str | BaseException | None = None
    checks: tuple[CheckResult, ...] = ()
    secrets: InitVar[Sequence[str]] = ()

    def __post_init__(self, secrets: Sequence[str]) -> None:
        message = (
            _exception_message(self.message)
            if isinstance(self.message, BaseException)
            else self.message
        )
        if isinstance(message, str):
            for secret in sorted(set(secrets), key=len, reverse=True):
                if secret:
                    message = message.replace(secret, "[REDACTED]")
        object.__setattr__(self, "message", message)

    @classmethod
    def group(
        cls,
        name: str,
        checks: Sequence[CheckResult],
        message: str | None = None,
    ) -> CheckResult:
        children = tuple(checks)
        return cls(
            name,
            max((child.status for child in children), default=CheckStatus.OK),
            message,
            children,
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"name": self.name, "status": self.status.label}
        if self.message is not None:
            value["message"] = self.message
        if self.checks:
            value["checks"] = [check.as_dict() for check in self.checks]
        return value

    def leaves(self) -> list[CheckResult]:
        return (
            [self]
            if not self.checks
            else sum((check.leaves() for check in self.checks), [])
        )


@dataclass(frozen=True)
class DoctorContext:
    config: UrsaConfig | None
    config_error: BaseException | None = None
    credentials: dict[tuple[str, str], str | BaseException] = field(
        default_factory=dict, compare=False, repr=False
    )


def _unavailable(name: str) -> CheckResult:
    return CheckResult(
        name, CheckStatus.WARN, "Skipped: URSA configuration is unavailable"
    )


def _attempt(
    name: str,
    probe: Callable[..., Any],
    *args: Any,
    secrets: Sequence[str] = (),
) -> CheckResult:
    try:
        message = run_probe_with_deadline(
            probe, *args, timeout=PROBE_TIMEOUT_SECONDS
        )
        return CheckResult(name, CheckStatus.OK, message, secrets=secrets)
    except _FAILURES as exc:
        return CheckResult(name, CheckStatus.ERROR, exc, secrets=secrets)


def _resolve_secret(
    context: DoctorContext,
    reference: SecretReference,
    default_username: str | None,
) -> str:
    if reference.env is not None:
        source = ("env", reference.env)
    else:
        username = reference.keyring_username(default_username)
        if username is None:
            raise ValueError("keyring=true requires a default username")
        source = ("keyring", username)
    cached = context.credentials.get(source)
    if isinstance(cached, BaseException):
        raise cached
    if cached is not None:
        return cached
    try:
        value = SecretReference.get_secret_value(reference, default_username)
        if value is None:
            raise ValueError(
                f"Secret source {source[0]} '{source[1]}' is not set"
            )
    except _FAILURES as exc:
        context.credentials[source] = exc
        raise
    context.credentials[source] = value
    return value


def _resolve_api_key(
    context: DoctorContext, config: Any, username: str | None
) -> tuple[Any, tuple[str, ...]]:
    key = getattr(config, "api_key", None)
    if isinstance(key, SecretReference):
        value = _resolve_secret(context, key, username)
        return config.model_copy(update={"api_key": SecretStr(value)}), (value,)
    return (
        (config, (key.get_secret_value(),))
        if isinstance(key, SecretStr)
        else (config, ())
    )


def _resolve_mcp_server(
    context: DoctorContext, name: str, server: Any
) -> tuple[Any, tuple[str, ...]]:
    if not isinstance(server, BaseModel):
        return server, ()
    environment = getattr(server, "env", None)
    secrets = (
        [value for value in environment.values() if isinstance(value, str)]
        if isinstance(environment, Mapping)
        else []
    )
    headers = getattr(server, "headers", None)
    if not isinstance(headers, Mapping):
        return server, tuple(secrets)
    resolved = dict(headers)
    for header, value in headers.items():
        reference = SecretTemplate.maybe_validate(value)
        if isinstance(reference, SecretTemplate):
            secret = _resolve_secret(context, reference, name)
            resolved[header] = rendered = reference.template % secret
            secrets.extend((secret, rendered))
        elif isinstance(value, str):
            secrets.append(value)
    return server.model_copy(update={"headers": resolved}), tuple(secrets)


def check_configuration(context: DoctorContext) -> CheckResult:
    if context.config is not None:
        return CheckResult(
            "Configuration", CheckStatus.OK, "Merged and resolved successfully"
        )
    error = context.config_error or "Configuration is unavailable"
    if isinstance(error, (JSONDecodeError, YAMLError)):
        error = (
            f"Configuration file could not be parsed ({type(error).__name__})"
        )
    return CheckResult("Configuration", CheckStatus.ERROR, error)


def _check_entries(
    context: DoctorContext,
    title: str,
    entries: Mapping[str, Any],
    resolve: Callable[[DoctorContext, str, Any], tuple[Any, tuple[str, ...]]],
    probe: Callable[..., Any],
    empty: tuple[CheckStatus, str],
    *,
    named: bool = False,
) -> CheckResult:
    if not entries:
        return CheckResult(title, *empty)
    checks = []
    for name, value in entries.items():
        try:
            value, secrets = resolve(context, name, value)
            args = (name, value) if named else (value,)
            result = _attempt(name, probe, *args, secrets=secrets)
        except _FAILURES as exc:
            result = CheckResult(name, CheckStatus.ERROR, exc)
        checks.append(result)
    return CheckResult.group(title, checks)


def check_inference_providers(context: DoctorContext) -> CheckResult:
    if context.config is None:
        return _unavailable("Inference providers")
    referenced = {
        model.inference_provider
        for field_name in ("llm_model", "emb_model")
        if (model := getattr(context.config, field_name, None)) is not None
        and model.inference_provider is not None
    }
    providers = {
        name: provider
        for name, provider in context.config.inference_providers.items()
        if name != "openai" or name in referenced
    }
    return _check_entries(
        context,
        "Inference providers",
        providers,
        lambda ctx, name, value: _resolve_api_key(ctx, value, name),
        probe_inference_provider,
        (CheckStatus.WARN, "No inference providers are configured"),
    )


def _check_model(
    context: DoctorContext,
    section: str,
    config: Any,
    probes: Sequence[tuple[str | None, Callable[..., str], tuple[Any, ...]]],
) -> CheckResult:
    label = config.pretty_repr(short=True)
    try:
        config, secrets = _resolve_api_key(
            context, config, config.inference_provider or config.model_provider
        )
        checks = [
            _attempt(name or label, probe, config, *args, secrets=secrets)
            for name, probe, args in probes
        ]
        result = (
            CheckResult.group(label, checks)
            if any(name is not None for name, _, _ in probes)
            else checks[0]
        )
    except _FAILURES as exc:
        result = CheckResult(label, CheckStatus.ERROR, exc)
    return CheckResult.group(section, [result])


def check_llm_models(context: DoctorContext) -> CheckResult:
    if context.config is None:
        return _unavailable("LLM model")
    return _check_model(
        context,
        "LLM model",
        context.config.llm_model,
        tuple(
            (name, probe_chat_config, (capability,))
            for name, capability in (
                ("Chat invocation", "chat"),
                ("Tool calling", "tool"),
                ("Structured output", "structured"),
            )
        ),
    )


def check_embedding_models(context: DoctorContext) -> CheckResult:
    if context.config is None:
        return _unavailable("Embedding model")
    if context.config.emb_model is None:
        return CheckResult(
            "Embedding model",
            CheckStatus.OK,
            "No embedding model is configured",
        )
    return _check_model(
        context,
        "Embedding model",
        context.config.emb_model,
        ((None, probe_embedding_model, ()),),
    )


def check_mcp_servers(context: DoctorContext) -> CheckResult:
    if context.config is None:
        return _unavailable("MCP servers")
    return _check_entries(
        context,
        "MCP servers",
        context.config.mcp_servers,
        _resolve_mcp_server,
        probe_mcp_server,
        (CheckStatus.OK, "No MCP servers are configured"),
        named=True,
    )


def build_report(
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
) -> CheckResult:
    context = DoctorContext(config, config_error)
    results = []
    for check in (
        check_configuration,
        check_inference_providers,
        check_llm_models,
        check_embedding_models,
        check_mcp_servers,
    ):
        try:
            result = check(context)
        except _FAILURES as exc:
            result = CheckResult(check.__name__, CheckStatus.ERROR, exc)
        results.append(result)
    return CheckResult.group("URSA Doctor", results)


def build_report_from_path(path: Path) -> CheckResult:
    try:
        config = UrsaConfig.from_file(path).resolve()
    except _FAILURES as exc:
        return build_report(config_error=exc)
    return build_report(config)


def report_counts(report: CheckResult) -> Counter[CheckStatus]:
    return Counter(result.status for result in report.leaves())


def render_report(report: CheckResult) -> Text:
    rendered = Text()

    def append(result: CheckResult, depth: int = 0) -> None:
        if rendered:
            rendered.append("\n")
        rendered.append("  " * depth)
        rendered.append(
            f"{result.status.indicator} {result.name}",
            f"bold {result.status.style}",
        )
        if result.message:
            rendered.append(f"  {result.message}", "dim")
        for child in result.checks:
            append(child, depth + 1)

    append(report)
    counts = report_counts(report)
    rendered.append("\n")
    rendered.append(
        "  |  ".join(
            f"{counts[status]} {label}"
            for status, label in (
                (CheckStatus.OK, "ok"),
                (CheckStatus.WARN, "warn"),
                (CheckStatus.ERROR, "error(s)"),
            )
        ),
        f"bold {report.status.style}",
    )
    return rendered


def run_doctor(
    args=None,
    *,
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
) -> int:
    report = build_report(config, config_error)
    if bool(getattr(args, "json", False)):
        print(json.dumps(report.as_dict(), indent=2, default=str))  # noqa: T201
    else:
        try:
            Console().print(render_report(report))
        except Exception:
            print(json.dumps(report.as_dict(), indent=2))  # noqa: T201
    return int(report.status is CheckStatus.ERROR)
