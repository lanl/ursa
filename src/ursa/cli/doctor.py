# ruff: noqa: TID251

"""Composable health checks for ``ursa self doctor``."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib.metadata import EntryPoint, entry_points
from inspect import isawaitable
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Thread
from time import perf_counter
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, SecretStr
from rich.console import Console
from rich.text import Text

from ursa.cli.config import UrsaConfig
from ursa.util.secrets import SecretReference, SecretTemplate

HEALTHCHECK_ENTRYPOINT_GROUP = "ursa-healthcheck"
CHECK_DEADLINE_SECONDS = 60
_CHECK_FAILURES = (Exception, SystemExit, asyncio.CancelledError)


class CheckStatus(StrEnum):
    """Possible health-check outcomes, ordered from best to worst."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"


_STATUS_RANK = {
    CheckStatus.OK: 0,
    CheckStatus.WARN: 1,
    CheckStatus.ERROR: 2,
}


@dataclass(frozen=True)
class CheckResult:
    """One heading or subcheck in a doctor report."""

    name: str
    status: CheckStatus
    message: str | None = None
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)
    duration_ms: int | None = None

    @classmethod
    def group(
        cls,
        name: str,
        checks: Sequence[CheckResult],
        message: str | None = None,
    ) -> CheckResult:
        children = tuple(checks)
        status = max(
            (child.status for child in children),
            key=_STATUS_RANK.get,
            default=CheckStatus.OK,
        )
        return cls(name, status, message, children)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
        }
        if self.message is not None:
            result["message"] = self.message
        if self.checks:
            result["checks"] = [check.as_dict() for check in self.checks]
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


@dataclass(frozen=True)
class DoctorContext:
    """Inputs shared by built-in and third-party checks."""

    config: UrsaConfig | None
    config_error: BaseException | None = None
    config_paths: tuple[Path, ...] = ()
    credential_errors: Mapping[int, BaseException] = field(default_factory=dict)
    mcp_credential_errors: Mapping[str, BaseException] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _PathUrsa:
    path_entry: str
    executable: Path


DoctorCheck = Callable[[DoctorContext], CheckResult]
ProgressCallback = Callable[[int, int, CheckResult], None]

_SPINNER_NAME = "dots"  # ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏


def _configuration_error(heading: str) -> CheckResult:
    return CheckResult(
        heading,
        CheckStatus.WARN,
        "Skipped: URSA configuration is unavailable",
    )


def _exception_message(exc: BaseException, secrets: Sequence[str] = ()) -> str:
    try:
        message = str(exc).strip() or type(exc).__name__
    except BaseException:
        message = type(exc).__name__
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def _error_result(
    name: str,
    exc: BaseException,
    secrets: Sequence[str] = (),
) -> CheckResult:
    return CheckResult(
        name,
        CheckStatus.ERROR,
        _exception_message(exc, secrets),
    )


def check_configuration(context: DoctorContext) -> CheckResult:
    """Parse each config file and report final resolution separately."""
    from ursa.cli.config import load_config_file

    checks = []

    if context.config_paths:
        for path in context.config_paths:
            try:
                load_config_file(path)
            except _CHECK_FAILURES as exc:
                checks.append(_error_result(str(path), exc))
            else:
                checks.append(
                    CheckResult(
                        str(path),
                        CheckStatus.OK,
                        "Parsed configuration file",
                    )
                )
    else:
        checks.append(
            CheckResult(
                "Configuration files",
                CheckStatus.OK,
                "No configuration files discovered",
            )
        )

    if context.config is None:
        final_status = CheckStatus.ERROR
        final_message = _exception_message(
            context.config_error or RuntimeError("configuration is unavailable")
        )
    else:
        final_status = CheckStatus.OK
        final_message = "Merged and resolved successfully"
    checks.append(
        CheckResult("Final resolved configuration", final_status, final_message)
    )
    return CheckResult.group("Configuration", checks)


def _secret_value(config: Any) -> str | None:
    reference = getattr(config, "api_key", None)
    if reference is None:
        return None
    return reference.get_secret_value()


def _keyring_username(
    reference: SecretReference, default_username: str | None = None
) -> str:
    username = (
        default_username if reference.keyring is True else reference.keyring
    )
    if not isinstance(username, str) or not username:
        raise ValueError("keyring=true requires a default username")
    return username


def _prime_keyring_credentials(context: DoctorContext) -> DoctorContext:
    """Resolve keyring secrets before concurrent checks and terminal output."""
    if context.config is None:
        return context

    cache: dict[str, SecretStr | BaseException] = {}
    credential_errors: dict[int, BaseException] = {}
    mcp_error_messages: dict[str, list[str]] = {}

    def resolve(
        reference: SecretReference, default_username: str | None = None
    ) -> SecretStr | BaseException:
        try:
            username = _keyring_username(reference, default_username)
        except _CHECK_FAILURES as exc:
            return exc
        if username not in cache:
            try:
                secret = reference.resolve(default_username)
                if secret is None:
                    raise ValueError(
                        f"No secret found in the system keyring for "
                        f"'{username}'"
                    )
                cache[username] = secret
            except _CHECK_FAILURES as exc:
                cache[username] = exc
        return cache[username]

    def prime_api_key(config: Any) -> None:
        reference = getattr(config, "api_key", None)
        if not isinstance(reference, SecretReference) or reference.keyring in (
            None,
            False,
        ):
            return
        outcome = resolve(reference)
        if isinstance(outcome, BaseException):
            credential_errors[id(config)] = outcome
        else:
            config.api_key = outcome

    for provider in context.config.inference_providers.values():
        prime_api_key(provider)
    prime_api_key(context.config.llm_model)
    if context.config.emb_model is not None:
        prime_api_key(context.config.emb_model)

    for server_name, server in context.config.mcp_servers.items():
        headers = getattr(server, "headers", None)
        if not isinstance(headers, dict):
            continue
        for header_name, value in tuple(headers.items()):
            reference = SecretTemplate.maybe_validate(value)
            if not isinstance(
                reference, SecretTemplate
            ) or reference.keyring in (None, False):
                continue
            outcome = resolve(reference, server_name)
            if isinstance(outcome, BaseException):
                mcp_error_messages.setdefault(server_name, []).append(
                    f"{header_name}: {_exception_message(outcome)}"
                )
            else:
                headers[header_name] = (
                    reference.template % outcome.get_secret_value()
                )

    mcp_credential_errors = {
        name: RuntimeError("; ".join(messages))
        for name, messages in mcp_error_messages.items()
    }
    return replace(
        context,
        credential_errors=credential_errors,
        mcp_credential_errors=mcp_credential_errors,
    )


def _run_probe(
    context: DoctorContext,
    name: str,
    config: Any,
    probe: Callable[[], str],
) -> CheckResult:
    if credential_error := context.credential_errors.get(id(config)):
        return _error_result(name, credential_error)
    secret = None
    try:
        secret = _secret_value(config)
        return CheckResult(name, CheckStatus.OK, probe())
    except _CHECK_FAILURES as exc:
        return _error_result(name, exc, (secret,) if secret else ())


def check_self_status(_context: DoctorContext) -> CheckResult:
    """Report every field from ``ursa self status``."""
    from ursa.cli.self import status_details

    checks = []
    for name, value in status_details().items():
        status = CheckStatus.OK
        if value == "unknown" or value.startswith("unavailable"):
            status = CheckStatus.WARN
        checks.append(CheckResult(name, status, value))
    return CheckResult.group("URSA self status", checks)


def _check_uv_receipt() -> CheckResult:
    from ursa.cli.self import _format_requirement, _read_requirement

    receipt = Path(sys.prefix) / "uv-receipt.toml"
    if not receipt.exists():
        return CheckResult(
            "uv tool receipt",
            CheckStatus.OK,
            "Not a uv tool installation",
        )
    try:
        requirement = _format_requirement(_read_requirement(receipt))
    except _CHECK_FAILURES as exc:
        return _error_result("uv tool receipt", exc)
    return CheckResult(
        "uv tool receipt",
        CheckStatus.OK,
        f"Valid receipt for {requirement}",
    )


def _check_dependencies() -> CheckResult:
    uv = shutil.which("uv")
    if uv is None:
        return CheckResult(
            "Python dependencies",
            CheckStatus.WARN,
            "Skipped: uv was not found on PATH",
        )
    try:
        result = subprocess.run(
            [uv, "pip", "check", "--python", sys.executable, "--no-progress"],
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, CHECK_DEADLINE_SECONDS - 5),
        )
    except _CHECK_FAILURES as exc:
        return _error_result("Python dependencies", exc)
    detail = "\n".join(
        output.strip()
        for output in (result.stdout, result.stderr)
        if output.strip()
    )
    if result.returncode:
        return CheckResult(
            "Python dependencies",
            CheckStatus.ERROR,
            detail or f"uv pip check exited with status {result.returncode}",
        )
    return CheckResult(
        "Python dependencies",
        CheckStatus.OK,
        "Installed dependencies are compatible",
    )


def _check_writable(name: str, path: Path) -> CheckResult:
    probe_directory = path.expanduser()
    while (
        not probe_directory.exists()
        and probe_directory != probe_directory.parent
    ):
        probe_directory = probe_directory.parent
    if not probe_directory.is_dir():
        return CheckResult(
            name,
            CheckStatus.ERROR,
            f"No writable parent directory for {path}",
        )
    try:
        with NamedTemporaryFile(prefix=".ursa-doctor-", dir=probe_directory):
            pass
    except _CHECK_FAILURES as exc:
        return _error_result(name, exc)
    return CheckResult(name, CheckStatus.OK, f"Writable or creatable: {path}")


def check_installation_integrity(context: DoctorContext) -> CheckResult:
    """Validate installation metadata, dependencies, and runtime storage."""
    from ursa.security import URSA_CACHE_DIR

    checks = [_check_uv_receipt(), _check_dependencies()]
    if context.config is None:
        checks.append(
            CheckResult(
                "Workspace",
                CheckStatus.WARN,
                "Skipped: URSA configuration is unavailable",
            )
        )
    else:
        checks.append(_check_writable("Workspace", context.config.workspace))
    checks.append(_check_writable("URSA cache", URSA_CACHE_DIR))
    return CheckResult.group("Installation integrity", checks)


def _ursa_executables_on_path() -> list[_PathUrsa]:
    found: list[_PathUrsa] = []
    seen: set[str] = set()
    for raw_directory in os.environ.get("PATH", "").split(os.pathsep):
        directory = raw_directory or os.curdir
        executable = shutil.which("ursa", path=directory)
        if executable is None:
            continue
        path = Path(executable).absolute()
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            found.append(
                _PathUrsa(
                    path_entry=(
                        raw_directory
                        if raw_directory
                        else "<empty> (current directory)"
                    ),
                    executable=path,
                )
            )
    return found


def _running_ursa_path() -> Path | None:
    invoked = Path(sys.argv[0])
    if invoked.stem.casefold() != "ursa":
        return None
    resolved = invoked if invoked.is_absolute() else shutil.which(str(invoked))
    return Path(resolved).absolute() if resolved else None


def _same_executable(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def check_ursa_on_path(_context: DoctorContext) -> CheckResult:
    """List URSA executables in PATH order and identify shadowing entries."""
    executables = _ursa_executables_on_path()
    if not executables:
        return CheckResult(
            "URSA on PATH",
            CheckStatus.ERROR,
            "No URSA executable was found on PATH",
        )

    running = _running_ursa_path()
    if running is None:
        current_index = 0
    else:
        current_index = next(
            (
                index
                for index, entry in enumerate(executables)
                if _same_executable(entry.executable, running)
            ),
            None,
        )
    if current_index is None:
        checks = tuple(
            CheckResult(
                f"PATH {entry.path_entry} · not current",
                CheckStatus.WARN,
                str(entry.executable),
            )
            for entry in executables
        )
        return CheckResult(
            "URSA on PATH",
            CheckStatus.ERROR,
            f"Running URSA is not on PATH: {running}",
            checks,
        )

    checks = []
    for index, entry in enumerate(executables):
        if index < current_index:
            label = "SHADOWS CURRENT"
            status = CheckStatus.ERROR
        elif index == current_index:
            label = "current"
            status = CheckStatus.OK
        else:
            label = "additional copy"
            status = CheckStatus.WARN
        checks.append(
            CheckResult(
                f"PATH {entry.path_entry} · {label}",
                status,
                str(entry.executable),
            )
        )

    additional = len(executables) - 1
    message = None
    if additional:
        message = f"Found {len(executables)} URSA executables"
        message += (
            f"; {current_index} before current"
            if current_index
            else f"; {additional} additional"
        )
    return CheckResult.group("URSA on PATH", checks, message)


def check_inference_providers(context: DoctorContext) -> CheckResult:
    """Validate credentials and connectivity for every configured provider."""
    if context.config is None:
        return _configuration_error("Inference providers")

    from ursa.util.inference_providers import list_provider_models

    checks = []
    for name, provider in context.config.inference_providers.items():

        def probe(provider=provider):
            models = list_provider_models(provider)
            return (
                "Connected; authentication accepted; "
                f"provider advertised {len(models)} model(s)"
            )

        checks.append(_run_probe(context, name, provider, probe))
    if not checks:
        return CheckResult(
            "Inference providers",
            CheckStatus.WARN,
            "No inference providers are configured",
        )
    return CheckResult.group("Inference providers", checks)


def _model_label(config: Any) -> str:
    endpoint = (
        getattr(config, "inference_provider", None)
        or getattr(config, "model_provider", None)
        or getattr(config, "base_url", None)
        or "direct configuration"
    )
    return f"{config.model} ({endpoint})"


@tool
def _ursa_doctor_echo(value: str) -> str:
    """Return a value unchanged for an URSA health check."""
    return value


class _DoctorStructuredResponse(BaseModel):
    """Typed response expected from the structured-output probe."""

    status: str


ModelProbe = Callable[[Any], str]


def _run_model_probe(
    context: DoctorContext, config: Any, name: str, probe: ModelProbe
) -> CheckResult:
    return _run_probe(
        context, name, config, lambda: probe(config.init_chat_model())
    )


def _probe_chat(model: Any) -> str:
    response = model.invoke(
        "URSA health check: reply with a short acknowledgement."
    )
    if response is None:
        raise ValueError("chat invocation returned no response")
    return "Model returned a response"


def _probe_forced_tool_use(model: Any) -> str:
    tool_model = model.bind_tools(
        [_ursa_doctor_echo], tool_choice=_ursa_doctor_echo.name
    )
    response = tool_model.invoke(
        "Call ursa_doctor_echo with the value 'ursa'. Do not answer directly."
    )
    tool_calls = getattr(response, "tool_calls", None) or []
    if not any(
        call.get("name") == _ursa_doctor_echo.name
        for call in tool_calls
        if isinstance(call, Mapping)
    ):
        raise ValueError("model did not return the requested tool call")
    return "Model returned the explicitly required tool call"


def _probe_structured_output(model: Any) -> str:
    from ursa.util.structured_output import invoke_structured

    structured = invoke_structured(
        model,
        _DoctorStructuredResponse,
        "Return a structured response whose status field is exactly 'ok'.",
        methods=(None,),
        context="URSA doctor structured-output probe",
    )
    if (
        not isinstance(structured, _DoctorStructuredResponse)
        or structured.status != "ok"
    ):
        raise ValueError("model did not return the expected status='ok'")
    return "Model returned and validated the typed response"


def check_llm_models(context: DoctorContext) -> CheckResult:
    """Check chat, forced tool use, and structured output for every LLM."""
    if context.config is None:
        return _configuration_error("LLM models")
    model = context.config.llm_model
    probes = tuple(
        _run_model_probe(context, model, name, probe)
        for name, probe in (
            ("Chat invocation", _probe_chat),
            ("Forced tool use", _probe_forced_tool_use),
            ("Structured output", _probe_structured_output),
        )
    )
    model_result = CheckResult.group(_model_label(model), probes)
    return CheckResult.group("LLM models", [model_result])


def _probe_embedding_model(config: Any) -> str:
    embedding = config.init_embedding().embed_query("URSA health check")
    if not isinstance(embedding, Sequence) or not embedding:
        raise ValueError("embedding invocation returned no vector")
    return f"Embedding invocation returned {len(embedding)} dimensions"


def check_embedding_models(context: DoctorContext) -> CheckResult:
    """Call every configured embedding model."""
    if context.config is None:
        return _configuration_error("Embedding models")
    if context.config.emb_model is None:
        return CheckResult(
            "Embedding models",
            CheckStatus.WARN,
            "No embedding model is configured",
        )
    config = context.config.emb_model
    result = _run_probe(
        context,
        _model_label(config),
        config,
        lambda: _probe_embedding_model(config),
    )
    return CheckResult.group("Embedding models", [result])


async def _probe_mcp_server(
    name: str,
    server: Any,
    credential_error: BaseException | None = None,
) -> CheckResult:
    from ursa.util.mcp import start_mcp_client

    if credential_error is not None:
        return _error_result(name, credential_error)
    try:
        client = start_mcp_client({name: server})
        tools = await client.get_tools(server_name=name)
        return CheckResult(
            name,
            CheckStatus.OK,
            f"Connected; server advertised {len(tools)} tool(s)",
        )
    except _CHECK_FAILURES as exc:
        return _error_result(name, exc)


async def _probe_mcp_servers(
    servers: Mapping[str, Any],
    credential_errors: Mapping[str, BaseException],
) -> list[CheckResult]:
    return list(
        await asyncio.gather(
            *(
                _probe_mcp_server(name, server, credential_errors.get(name))
                for name, server in servers.items()
            )
        )
    )


def check_mcp_servers(context: DoctorContext) -> CheckResult:
    """Connect to every configured MCP server and enumerate its tools."""
    if context.config is None:
        return _configuration_error("MCP servers")
    if not context.config.mcp_servers:
        return CheckResult(
            "MCP servers", CheckStatus.WARN, "No MCP servers are configured"
        )
    checks = asyncio.run(
        _probe_mcp_servers(
            context.config.mcp_servers, context.mcp_credential_errors
        )
    )
    return CheckResult.group("MCP servers", checks)


def _additional_entry_points() -> list[EntryPoint]:
    return sorted(
        entry_points(group=HEALTHCHECK_ENTRYPOINT_GROUP),
        key=lambda item: item.name,
    )


def check_additional_healthchecks(context: DoctorContext) -> CheckResult:
    """Load and run checks from the ``ursa-healthcheck`` entry-point group."""
    checks = []
    for entry_point in _additional_entry_points():
        try:
            plugin = entry_point.load()
            if not callable(plugin):
                raise TypeError("entry point did not load a callable")
            value = plugin(context)
            if isawaitable(value):
                value = asyncio.run(value)
            if not isinstance(value, CheckResult):
                raise TypeError("health check must return CheckResult")
            result = replace(value, name=entry_point.name)
            checks.append(result)
        except _CHECK_FAILURES as exc:
            checks.append(_error_result(entry_point.name, exc))
    if not checks:
        return CheckResult(
            "Additional health checks",
            CheckStatus.OK,
            f"No {HEALTHCHECK_ENTRYPOINT_GROUP} entry points are installed",
        )
    return CheckResult.group("Additional health checks", checks)


BUILTIN_CHECKS: tuple[DoctorCheck, ...] = (
    check_configuration,
    check_self_status,
    check_installation_integrity,
    check_ursa_on_path,
    check_inference_providers,
    check_llm_models,
    check_embedding_models,
    check_mcp_servers,
    check_additional_healthchecks,
)

_BUILTIN_CHECK_NAMES = {
    check_configuration: "Configuration",
    check_self_status: "URSA self status",
    check_installation_integrity: "Installation integrity",
    check_ursa_on_path: "URSA on PATH",
    check_inference_providers: "Inference providers",
    check_llm_models: "LLM models",
    check_embedding_models: "Embedding models",
    check_mcp_servers: "MCP servers",
    check_additional_healthchecks: "Additional health checks",
}


def _check_name(check: DoctorCheck) -> str:
    return _BUILTIN_CHECK_NAMES.get(
        check, getattr(check, "__name__", type(check).__name__)
    )


async def _run_check(context: DoctorContext, check: DoctorCheck) -> CheckResult:
    """Run one heading in a detachable daemon thread."""
    started = perf_counter()
    name = _check_name(check)
    loop = asyncio.get_running_loop()
    completed = loop.create_future()

    def finish(outcome: tuple[bool, Any]) -> None:
        if not completed.done():
            completed.set_result(outcome)

    def worker() -> None:
        try:
            outcome = True, check(context)
        except BaseException as exc:
            outcome = False, exc
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(finish, outcome)

    Thread(target=worker, name=f"ursa-doctor-{name}", daemon=True).start()

    try:
        try:
            async with asyncio.timeout(CHECK_DEADLINE_SECONDS):
                succeeded, outcome = await completed
        except TimeoutError:
            result = CheckResult(
                name,
                CheckStatus.ERROR,
                f"Timed out after {CHECK_DEADLINE_SECONDS} seconds",
            )
        else:
            if not succeeded:
                if isinstance(outcome, (KeyboardInterrupt, GeneratorExit)):
                    raise outcome
                result = _error_result(name, outcome)
            else:
                result = outcome
        if not isinstance(result, CheckResult):
            raise TypeError(
                "doctor heading functions must return CheckResult, got "
                f"{type(result).__name__}"
            )
    except asyncio.CancelledError:
        raise
    except _CHECK_FAILURES as exc:
        result = _error_result(name, exc)
    duration_ms = round((perf_counter() - started) * 1000)
    return replace(result, duration_ms=duration_ms)


async def run_checks_async(
    context: DoctorContext,
    checks: Sequence[DoctorCheck] = BUILTIN_CHECKS,
    *,
    progress: ProgressCallback | None = None,
) -> CheckResult:
    """Run heading functions concurrently, preserving their display order."""
    completed = 0
    total = len(checks)

    async def run_and_report(check: DoctorCheck) -> CheckResult:
        nonlocal completed
        result = await _run_check(context, check)
        completed += 1
        if progress is not None:
            try:
                progress(completed, total, result)
            except Exception:
                pass
        return result

    results = await asyncio.gather(*(run_and_report(check) for check in checks))
    return CheckResult.group("URSA Doctor", results)


def run_checks(
    context: DoctorContext,
    checks: Sequence[DoctorCheck] = BUILTIN_CHECKS,
    *,
    progress: ProgressCallback | None = None,
) -> CheckResult:
    """Synchronously run the concurrent doctor check scheduler."""
    return asyncio.run(run_checks_async(context, checks, progress=progress))


_STATUS_DISPLAY = {
    CheckStatus.OK: ("✅", "[ok]", "green"),
    CheckStatus.WARN: ("⚠️", "[!!]", "yellow"),
    CheckStatus.ERROR: ("❌", "[xx]", "red"),
}


def _result_label(result: CheckResult) -> Text:
    indicator, _, style = _STATUS_DISPLAY[result.status]
    label = Text()
    label.append(f"{indicator} ")
    label.append(result.name, style=f"bold {style}")
    if result.duration_ms is not None:
        label.append(
            f"  ({result.duration_ms}\N{NO-BREAK SPACE}ms)", style="dim"
        )
    if result.message:
        label.append("  ")
        label.append(result.message, style="dim")
    return label


def _print_children(
    console: Console, result: CheckResult, depth: int = 0
) -> None:
    for child in result.checks:
        status = child.status
        _, marker, style = _STATUS_DISPLAY[status]
        row = Text("  ")
        row.append(marker, f"bold {style}")
        row.append(f"  {'  ' * depth}{child.name}", style="bold")
        if child.message:
            row.append(f"  {child.message}", style="dim")
        console.print(row)
        _print_children(console, child, depth + 1)


def _leaf_statuses(result: CheckResult) -> list[CheckStatus]:
    if not result.checks:
        return [result.status]
    return [
        status for child in result.checks for status in _leaf_statuses(child)
    ]


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


def _print_report(console: Console, report: CheckResult) -> None:
    """Print a grouped checkhealth-style report."""
    console.print(_result_label(replace(report, message=None)))
    console.rule(style="bright_black")
    for index, section in enumerate(report.checks):
        console.print(_result_label(section))
        _print_children(console, section)
        if index < len(report.checks) - 1:
            console.print()
    console.rule(style="bright_black")

    counts = Counter(_leaf_statuses(report))
    summary = "  |  ".join((
        f"{counts[CheckStatus.OK]} ok",
        f"{counts[CheckStatus.WARN]} warn",
        _count_label(counts[CheckStatus.ERROR], "error"),
    ))
    console.print(summary, style=f"bold {_STATUS_DISPLAY[report.status][2]}")
    console.print(
        "Use `ursa self doctor --json` for the machine-readable report.",
        style="dim",
    )


def render_report(report: CheckResult) -> str:
    """Render a plain terminal snapshot, primarily for tests and logs."""
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=100,
    )
    _print_report(console, report)
    return output.getvalue().rstrip()


def _progress_text(
    completed: int,
    total: int,
    latest: CheckResult | None = None,
) -> Text:
    text = Text("Running URSA health checks", style="bold")
    text.append(f"  {completed}/{total} complete", style="cyan")
    if latest is not None:
        text.append("  ·  ", style="dim")
        text.append(latest.name)
        text.append(
            f": {latest.status.value}",
            style=_STATUS_DISPLAY[latest.status][2],
        )
    return text


def _run_human_checks(context: DoctorContext, console: Console) -> CheckResult:
    try:
        interactive = console.is_terminal
    except Exception:
        interactive = False
    if not interactive:
        return run_checks(context)

    try:
        status = console.status(
            _progress_text(0, len(BUILTIN_CHECKS)),
            spinner=_SPINNER_NAME,
            spinner_style="cyan",
            refresh_per_second=12.5,
        )
        status.start()
    except Exception:
        return run_checks(context)

    def update(completed: int, total: int, latest: CheckResult) -> None:
        with suppress(Exception):
            status.update(_progress_text(completed, total, latest))

    try:
        return run_checks(context, progress=update)
    finally:
        with suppress(Exception):
            status.stop()


def run_doctor(
    args=None,
    *,
    config: UrsaConfig | None = None,
    config_error: BaseException | None = None,
    config_paths: tuple[Path, ...] = (),
) -> int:
    """Run all doctor checks, print one format, and return an exit code."""
    context = DoctorContext(
        config=config,
        config_error=config_error,
        config_paths=config_paths,
    )
    try:
        context = _prime_keyring_credentials(context)
    except _CHECK_FAILURES as exc:
        context = replace(context, config=None, config_error=exc)
    json_output = bool(getattr(args, "json", False))
    console = None
    try:
        if json_output:
            report = run_checks(context)
        else:
            console = Console()
            report = _run_human_checks(context, console)
    except _CHECK_FAILURES as exc:
        report = CheckResult(
            "URSA Doctor",
            CheckStatus.ERROR,
            checks=(
                CheckResult(
                    "Doctor runtime", CheckStatus.ERROR, _exception_message(exc)
                ),
            ),
        )

    if json_output:
        print(  # noqa: T201
            json.dumps(report.as_dict(), indent=2, default=str)
        )
    else:
        try:
            _print_report(console or Console(), report)
        except Exception:
            print(json.dumps(report.as_dict(), indent=2))  # noqa: T201
    return 1 if report.status is CheckStatus.ERROR else 0
