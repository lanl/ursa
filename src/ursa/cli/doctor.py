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
from typing import Any

from pydantic import BaseModel, SecretStr, ValidationError
from rich.console import Console
from rich.text import Text
from yaml import YAMLError

from ursa.cli.config import UrsaConfig
from ursa.util.health import (
    probe_chat_model,
    probe_chat_structured_output,
    probe_chat_tool_use,
    probe_embedding_model,
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


DoctorCheck = Callable[[DoctorContext], CheckResult]


def _unavailable(name: str) -> CheckResult:
    return CheckResult(
        name,
        CheckStatus.WARN,
        "Skipped: URSA configuration is unavailable",
    )


def _attempt(
    name: str,
    action: Callable[[], str],
    *,
    secrets: Sequence[str] = (),
) -> CheckResult:
    try:
        return CheckResult(name, CheckStatus.OK, action(), secrets=secrets)
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
    headers = getattr(server, "headers", None)
    if not isinstance(server, BaseModel) or not isinstance(headers, Mapping):
        return server, ()
    resolved = dict(headers)
    secrets: list[str] = []
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

    from ursa.util.inference_providers import list_provider_models

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
                lambda provider=provider: (
                    "Connected; authentication accepted; provider advertised "
                    f"{len(list_provider_models(provider))} model(s)"
                ),
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


def check_llm_models(context: DoctorContext) -> CheckResult:
    """Exercise chat, tool calling, and structured output."""
    if context.config is None:
        return _unavailable("LLM model")
    config = context.config.llm_model
    secrets: tuple[str, ...] = ()
    try:
        config, secrets = _resolve_api_key(
            context,
            config,
            config.inference_provider or config.model_provider,
        )
        model = config.init_chat_model()
    except _FAILURES as exc:
        result = CheckResult(
            _model_label(config), CheckStatus.ERROR, exc, secrets=secrets
        )
    else:
        result = CheckResult.group(
            _model_label(config),
            [
                _attempt(
                    "Chat invocation",
                    lambda: probe_chat_model(model),
                    secrets=secrets,
                ),
                _attempt(
                    "Forced tool use",
                    lambda: probe_chat_tool_use(model),
                    secrets=secrets,
                ),
                _attempt(
                    "Structured output",
                    lambda: probe_chat_structured_output(model),
                    secrets=secrets,
                ),
            ],
        )
    return CheckResult.group("LLM model", [result])


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
    try:
        config, secrets = _resolve_api_key(
            context,
            config,
            config.inference_provider or config.model_provider,
        )
    except _FAILURES as exc:
        result = CheckResult(_model_label(config), CheckStatus.ERROR, exc)
    else:
        result = _attempt(
            _model_label(config),
            lambda: probe_embedding_model(config),
            secrets=secrets,
        )
    return CheckResult.group("Embedding model", [result])


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

    async def probe_all() -> list[CheckResult]:
        async def probe(name: str, server: Any) -> CheckResult:
            secrets: tuple[str, ...] = ()
            try:
                server, secrets = _resolve_mcp_server(context, name, server)
                message = await _probe_mcp_server(name, server)
                return CheckResult(
                    name, CheckStatus.OK, message, secrets=secrets
                )
            except _FAILURES as exc:
                return CheckResult(
                    name,
                    CheckStatus.ERROR,
                    exc,
                    secrets=secrets,
                )

        return list(
            await asyncio.gather(
                *(probe(name, server) for name, server in servers.items())
            )
        )

    return CheckResult.group("MCP servers", asyncio.run(probe_all()))


BUILTIN_CHECKS: tuple[DoctorCheck, ...] = (
    check_configuration,
    check_inference_providers,
    check_llm_models,
    check_embedding_models,
    check_mcp_servers,
)


def run_checks(
    context: DoctorContext,
    checks: Sequence[DoctorCheck] | None = None,
) -> CheckResult:
    """Run checks in display order, containing failures per section."""
    results = []
    for check in BUILTIN_CHECKS if checks is None else checks:
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


def _print_result(
    console: Console, result: CheckResult, depth: int = 0
) -> None:
    line = Text("  " * depth)
    line.append(
        f"{result.status.indicator} {result.name}",
        f"bold {result.status.style}",
    )
    if result.message:
        line.append(f"  {result.message}", "dim")
    console.print(line)
    for child in result.checks:
        _print_result(console, child, depth + 1)


def _print_report(console: Console, report: CheckResult) -> None:
    _print_result(console, report)
    counts = Counter(leaf.status for leaf in report.leaves())
    console.print(
        f"{counts[CheckStatus.OK]} ok  |  "
        f"{counts[CheckStatus.WARN]} warn  |  "
        f"{counts[CheckStatus.ERROR]} error(s)",
        style=f"bold {report.status.style}",
    )


def run_doctor(
    args=None,
    *,
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
) -> int:
    """Run health checks, print the selected format, and return a status."""
    try:
        report = run_checks(DoctorContext(config, config_error))
    except _FAILURES as exc:
        report = CheckResult.group(
            "URSA Doctor",
            [CheckResult("Doctor runtime", CheckStatus.ERROR, exc)],
        )

    if bool(getattr(args, "json", False)):
        print(json.dumps(report.as_dict(), indent=2, default=str))  # noqa: T201
    else:
        try:
            _print_report(Console(), report)
        except Exception:
            print(json.dumps(report.as_dict(), indent=2))  # noqa: T201
    return int(report.status is CheckStatus.ERROR)
