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
    run_probe_with_deadline,
)
from ursa.util.health import probe_mcp_server as _probe_mcp_server
from ursa.util.secrets import SecretReference, SecretTemplate

_FAILURES = (Exception, SystemExit, asyncio.CancelledError)


class CheckStatus(IntEnum):
    """Health-check outcomes, ordered from best to worst."""

    OK = 0
    WARN = 1
    ERROR = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def indicator(self) -> str:
        return ("✅", "⚠️", "❌")[self]

    @property
    def style(self) -> str:
        return ("green", "yellow", "red")[self]


@dataclass(frozen=True)
class CheckResult:
    """A check and its optional child checks."""

    name: str
    status: CheckStatus
    message: str | BaseException | None = None
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)
    secrets: InitVar[Sequence[str]] = ()

    def __post_init__(self, secrets: Sequence[str]) -> None:
        if isinstance(self.message, BaseException):
            if isinstance(self.message, ValidationError):
                locations = {
                    ".".join(map(str, error["loc"])) or "configuration"
                    for error in self.message.errors(include_input=False)
                }
                message = "Configuration validation failed at " + ", ".join(
                    sorted(locations)
                )
            else:
                try:
                    message = str(self.message).strip()
                except BaseException:
                    message = ""
                message = message or type(self.message).__name__
        else:
            message = self.message
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
        status = max(
            (child.status for child in children), default=CheckStatus.OK
        )
        return cls(name, status, message, children)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": self.name,
            "status": self.status.label,
        }
        if self.message is not None:
            value["message"] = self.message
        if self.checks:
            value["checks"] = [child.as_dict() for child in self.checks]
        return value

    def leaves(self) -> list[CheckResult]:
        return (
            [self]
            if not self.checks
            else [leaf for child in self.checks for leaf in child.leaves()]
        )


@dataclass(frozen=True)
class DoctorContext:
    """Inputs that cannot be recovered by an individual health check."""

    config: UrsaConfig | None
    config_error: BaseException | None = None
    credentials: dict[tuple[str, str], str | BaseException] = field(
        default_factory=dict, compare=False, repr=False
    )


def _unavailable(name: str) -> CheckResult:
    return CheckResult(
        name,
        CheckStatus.WARN,
        "Skipped: URSA configuration is unavailable",
    )


def _attempt(
    name: str,
    probe: Callable[..., str],
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
    context: DoctorContext,
    config: Any,
    default_username: str | None,
) -> tuple[Any, tuple[str, ...]]:
    api_key = getattr(config, "api_key", None)
    if isinstance(api_key, SecretReference):
        value = _resolve_secret(context, api_key, default_username)
        return config.model_copy(update={"api_key": SecretStr(value)}), (value,)
    if isinstance(api_key, SecretStr):
        return config, (api_key.get_secret_value(),)
    return config, ()


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
        if not isinstance(reference, SecretTemplate):
            if isinstance(value, str):
                secrets.append(value)
            continue
        secret = _resolve_secret(context, reference, name)
        rendered = reference.template % secret
        resolved[header] = rendered
        secrets.extend((secret, rendered))
    return server.model_copy(update={"headers": resolved}), tuple(secrets)


def check_configuration(context: DoctorContext) -> CheckResult:
    """Report whether the normal configuration pipeline resolved."""
    if context.config is None:
        error = context.config_error or "Configuration is unavailable"
        if isinstance(error, (JSONDecodeError, YAMLError)):
            error = f"Configuration file could not be parsed ({type(error).__name__})"
        return CheckResult(
            "Configuration",
            CheckStatus.ERROR,
            error,
        )
    return CheckResult(
        "Configuration", CheckStatus.OK, "Merged and resolved successfully"
    )


def check_inference_providers(context: DoctorContext) -> CheckResult:
    """Authenticate to every configured inference provider."""
    if context.config is None:
        return _unavailable("Inference providers")

    referenced = {
        model.inference_provider
        for field_name in ("llm_model", "emb_model")
        if (model := getattr(context.config, field_name, None)) is not None
        and model.inference_provider is not None
    }
    checks = []
    for name, provider in context.config.inference_providers.items():
        if name == "openai" and name not in referenced:
            continue
        try:
            provider, secrets = _resolve_api_key(context, provider, name)
        except _FAILURES as exc:
            checks.append(CheckResult(name, CheckStatus.ERROR, exc))
            continue
        checks.append(
            _attempt(
                name,
                probe_inference_provider,
                provider,
                secrets=secrets,
            )
        )
    return (
        CheckResult.group("Inference providers", checks)
        if checks
        else CheckResult(
            "Inference providers",
            CheckStatus.WARN,
            "No inference providers are configured",
        )
    )


def _model_label(config: Any) -> str:
    endpoint = (
        getattr(config, "inference_provider", None)
        or getattr(config, "model_provider", None)
        or getattr(config, "base_url", None)
        or "direct configuration"
    )
    return f"{config.model} ({endpoint})"


def _check_model(
    context: DoctorContext,
    section: str,
    config: Any,
    probes: Sequence[tuple[str | None, Callable[..., str], tuple[Any, ...]]],
) -> CheckResult:
    """Resolve one model config and run its configured capability probes."""
    label = _model_label(config)
    try:
        config, secrets = _resolve_api_key(
            context,
            config,
            config.inference_provider or config.model_provider,
        )
    except _FAILURES as exc:
        model = CheckResult(label, CheckStatus.ERROR, exc)
    else:
        checks = [
            _attempt(name or label, probe, config, *args, secrets=secrets)
            for name, probe, args in probes
        ]
        model = (
            CheckResult.group(label, checks)
            if any(name is not None for name, _, _ in probes)
            else checks[0]
        )
    return CheckResult.group(section, [model])


def check_llm_models(context: DoctorContext) -> CheckResult:
    """Exercise chat, tool calling, and structured output."""
    if context.config is None:
        return _unavailable("LLM model")
    return _check_model(
        context,
        "LLM model",
        context.config.llm_model,
        (
            ("Chat invocation", probe_chat_config, ("chat",)),
            ("Forced tool use", probe_chat_config, ("tool",)),
            ("Structured output", probe_chat_config, ("structured",)),
        ),
    )


def check_embedding_models(context: DoctorContext) -> CheckResult:
    """Call the configured embedding model."""
    if context.config is None:
        return _unavailable("Embedding model")
    config = context.config.emb_model
    if config is None:
        return CheckResult(
            "Embedding model",
            CheckStatus.OK,
            "No embedding model is configured",
        )
    return _check_model(
        context,
        "Embedding model",
        config,
        ((None, probe_embedding_model, ()),),
    )


def check_mcp_servers(context: DoctorContext) -> CheckResult:
    """Connect to every configured MCP server and enumerate its tools."""
    if context.config is None:
        return _unavailable("MCP servers")
    servers = context.config.mcp_servers
    if not servers:
        return CheckResult(
            "MCP servers",
            CheckStatus.OK,
            "No MCP servers are configured",
        )

    checks = []
    for name, server in servers.items():
        try:
            server, secrets = _resolve_mcp_server(context, name, server)
        except _FAILURES as exc:
            checks.append(CheckResult(name, CheckStatus.ERROR, exc))
            continue
        checks.append(
            _attempt(
                name,
                _probe_mcp_server,
                name,
                server,
                secrets=secrets,
            )
        )
    return CheckResult.group("MCP servers", checks)


def run_checks(context: DoctorContext) -> CheckResult:
    """Run checks in display order, containing failures per section."""
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
            result = CheckResult(
                getattr(check, "__name__", type(check).__name__),
                CheckStatus.ERROR,
                exc,
            )
        results.append(result)
    return CheckResult.group("URSA Doctor", results)


def build_report(
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
) -> CheckResult:
    """Build a contained doctor report suitable for any presentation layer."""
    try:
        return run_checks(DoctorContext(config, config_error))
    except _FAILURES as exc:
        return CheckResult.group(
            "URSA Doctor",
            [CheckResult("Doctor runtime", CheckStatus.ERROR, exc)],
        )


def build_report_from_path(path: Path) -> CheckResult:
    """Load and resolve a config file before building its doctor report."""
    try:
        config = UrsaConfig.from_file(path).resolve()
    except _FAILURES as exc:
        return build_report(config_error=exc)
    return build_report(config)


def report_counts(report: CheckResult) -> Counter[CheckStatus]:
    """Count final health outcomes without counting their parent groups."""
    return Counter(leaf.status for leaf in report.leaves())


def render_report(report: CheckResult) -> Text:
    """Render a doctor report for Rich consoles or Textual widgets."""
    rendered = Text()

    def append_result(result: CheckResult, depth: int = 0) -> None:
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
            append_result(child, depth + 1)

    append_result(report)
    counts = report_counts(report)
    rendered.append("\n")
    rendered.append(
        f"{counts[CheckStatus.OK]} ok  |  "
        f"{counts[CheckStatus.WARN]} warn  |  "
        f"{counts[CheckStatus.ERROR]} error(s)",
        f"bold {report.status.style}",
    )
    return rendered


def _print_report(console: Console, report: CheckResult) -> None:
    console.print(render_report(report))


def run_doctor(
    args=None,
    *,
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
) -> int:
    """Run health checks, print the selected format, and return a status."""
    report = build_report(config, config_error)

    if bool(getattr(args, "json", False)):
        print(json.dumps(report.as_dict(), indent=2, default=str))  # noqa: T201
    else:
        try:
            _print_report(Console(), report)
        except Exception:
            print(json.dumps(report.as_dict(), indent=2))  # noqa: T201
    return int(report.status is CheckStatus.ERROR)
