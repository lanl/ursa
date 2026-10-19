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
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import InitVar, dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, SecretStr
from rich.console import Console
from rich.text import Text

from ursa.cli.config import UrsaConfig
from ursa.util.secrets import SecretReference, SecretTemplate

CHECK_DEADLINE_SECONDS = 60
_CHECK_FAILURES = (Exception, SystemExit, asyncio.CancelledError)


class CheckStatus(IntEnum):
    """Possible health-check outcomes, ordered from best to worst."""

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
    def marker(self) -> str:
        return ("[ok]", "[!!]", "[xx]")[self]

    @property
    def style(self) -> str:
        return ("green", "yellow", "red")[self]

    def result_label(self, result: CheckResult) -> Text:
        label = Text(f"{self.indicator} ")
        label.append(result.name, style=f"bold {self.style}")
        if result.duration_ms is not None:
            label.append(
                f"  ({result.duration_ms}\N{NO-BREAK SPACE}ms)", style="dim"
            )
        if result.message:
            label.append("  ")
            label.append(result.message, style="dim")
        return label


@dataclass(frozen=True)
class CheckResult:
    """One heading or subcheck in a doctor report."""

    name: str
    status: CheckStatus
    message: str | BaseException | None = None
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)
    duration_ms: int | None = None
    secrets: InitVar[Sequence[str]] = ()

    def __post_init__(self, secrets: Sequence[str]) -> None:
        if not isinstance(self.message, BaseException):
            return
        try:
            message = str(self.message).strip() or type(self.message).__name__
        except BaseException:
            message = type(self.message).__name__
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[redacted]")
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
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.label,
        }
        if self.message is not None:
            result["message"] = self.message
        if self.checks:
            result["checks"] = [check.as_dict() for check in self.checks]
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result

    def leaf_statuses(self) -> list[CheckStatus]:
        if not self.checks:
            return [self.status]
        return [
            status for child in self.checks for status in child.leaf_statuses()
        ]


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


DoctorCheck = Callable[[DoctorContext], CheckResult]
ProgressCallback = Callable[[int, int, CheckResult], None]


def _configuration_error(heading: str) -> CheckResult:
    return CheckResult(
        heading,
        CheckStatus.WARN,
        "Skipped: URSA configuration is unavailable",
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
                checks.append(CheckResult(str(path), CheckStatus.ERROR, exc))
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
        final_message = context.config_error or RuntimeError(
            "configuration is unavailable"
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
                message = CheckResult(
                    header_name, CheckStatus.ERROR, outcome
                ).message
                mcp_error_messages.setdefault(server_name, []).append(
                    f"{header_name}: {message}"
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
        return CheckResult("uv tool receipt", CheckStatus.ERROR, exc)
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
        return CheckResult("Python dependencies", CheckStatus.ERROR, exc)
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
        return CheckResult(name, CheckStatus.ERROR, exc)
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


def _ursa_executables_on_path() -> list[tuple[str, Path]]:
    found = []
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
            found.append((
                raw_directory
                if raw_directory
                else "<empty> (current directory)",
                path,
            ))
    return found


def _running_ursa_path() -> Path | None:
    invoked = Path(sys.argv[0])
    if invoked.stem.casefold() != "ursa":
        return None
    resolved = invoked if invoked.is_absolute() else shutil.which(str(invoked))
    return Path(resolved).absolute() if resolved else None


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
                for index, (_, executable) in enumerate(executables)
                if executable.samefile(running)
            ),
            None,
        )
    if current_index is None:
        checks = tuple(
            CheckResult(
                f"PATH {path_entry} · not current",
                CheckStatus.WARN,
                str(executable),
            )
            for path_entry, executable in executables
        )
        return CheckResult(
            "URSA on PATH",
            CheckStatus.ERROR,
            f"Running URSA is not on PATH: {running}",
            checks,
        )

    checks = []
    for index, (path_entry, executable) in enumerate(executables):
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
                f"PATH {path_entry} · {label}",
                status,
                str(executable),
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
        if credential_error := context.credential_errors.get(id(provider)):
            checks.append(
                CheckResult(name, CheckStatus.ERROR, credential_error)
            )
            continue
        secret = None
        try:
            secret = _secret_value(provider)
            models = list_provider_models(provider)
            checks.append(
                CheckResult(
                    name,
                    CheckStatus.OK,
                    "Connected; authentication accepted; "
                    f"provider advertised {len(models)} model(s)",
                )
            )
        except _CHECK_FAILURES as exc:
            checks.append(
                CheckResult(
                    name,
                    CheckStatus.ERROR,
                    exc,
                    secrets=(secret,) if secret else (),
                )
            )
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
    probes = []
    for name, probe in (
        ("Chat invocation", _probe_chat),
        ("Forced tool use", _probe_forced_tool_use),
        ("Structured output", _probe_structured_output),
    ):
        if credential_error := context.credential_errors.get(id(model)):
            probes.append(
                CheckResult(name, CheckStatus.ERROR, credential_error)
            )
            continue
        secret = None
        try:
            secret = _secret_value(model)
            probes.append(
                CheckResult(
                    name,
                    CheckStatus.OK,
                    probe(model.init_chat_model()),
                )
            )
        except _CHECK_FAILURES as exc:
            probes.append(
                CheckResult(
                    name,
                    CheckStatus.ERROR,
                    exc,
                    secrets=(secret,) if secret else (),
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
    name = _model_label(config)
    if credential_error := context.credential_errors.get(id(config)):
        result = CheckResult(name, CheckStatus.ERROR, credential_error)
    else:
        secret = None
        try:
            secret = _secret_value(config)
            result = CheckResult(
                name, CheckStatus.OK, _probe_embedding_model(config)
            )
        except _CHECK_FAILURES as exc:
            result = CheckResult(
                name,
                CheckStatus.ERROR,
                exc,
                secrets=(secret,) if secret else (),
            )
    return CheckResult.group("Embedding models", [result])


async def _probe_mcp_server(
    name: str,
    server: Any,
    credential_error: BaseException | None = None,
) -> CheckResult:
    from ursa.util.mcp import start_mcp_client

    if credential_error is not None:
        return CheckResult(name, CheckStatus.ERROR, credential_error)
    try:
        client = start_mcp_client({name: server})
        tools = await client.get_tools(server_name=name)
        return CheckResult(
            name,
            CheckStatus.OK,
            f"Connected; server advertised {len(tools)} tool(s)",
        )
    except _CHECK_FAILURES as exc:
        return CheckResult(name, CheckStatus.ERROR, exc)


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
            "MCP servers", CheckStatus.OK, "No MCP servers are configured"
        )
    checks = asyncio.run(
        _probe_mcp_servers(
            context.config.mcp_servers, context.mcp_credential_errors
        )
    )
    return CheckResult.group("MCP servers", checks)


BUILTIN_CHECKS: list[DoctorCheck] = [
    check_configuration,
    check_self_status,
    check_installation_integrity,
    check_ursa_on_path,
    check_inference_providers,
    check_llm_models,
    check_embedding_models,
    check_mcp_servers,
]


def run_checks(
    context: DoctorContext,
    checks: Sequence[DoctorCheck] | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> CheckResult:
    """Run heading functions concurrently, preserving their display order."""
    selected = BUILTIN_CHECKS if checks is None else checks

    def invoke(check: DoctorCheck) -> CheckResult:
        started = perf_counter()
        try:
            result = check(context)
            if not isinstance(result, CheckResult):
                raise TypeError(
                    "doctor heading functions must return CheckResult, got "
                    f"{type(result).__name__}"
                )
        except _CHECK_FAILURES as exc:
            name = getattr(check, "__name__", type(check).__name__)
            result = CheckResult(name, CheckStatus.ERROR, exc)
        return replace(
            result, duration_ms=round((perf_counter() - started) * 1000)
        )

    results = []
    with ThreadPoolExecutor() as executor:
        for completed, result in enumerate(
            executor.map(invoke, selected), start=1
        ):
            results.append(result)
            if progress is not None:
                with suppress(Exception):
                    progress(completed, len(selected), result)
    return CheckResult.group("URSA Doctor", results)


def _print_children(
    console: Console, result: CheckResult, depth: int = 0
) -> None:
    for child in result.checks:
        status = child.status
        row = Text("  ")
        row.append(status.marker, f"bold {status.style}")
        row.append(f"  {'  ' * depth}{child.name}", style="bold")
        if child.message:
            row.append(f"  {child.message}", style="dim")
        console.print(row)
        _print_children(console, child, depth + 1)


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


def _print_report(console: Console, report: CheckResult) -> None:
    """Print a grouped checkhealth-style report."""
    console.print(report.status.result_label(replace(report, message=None)))
    console.rule(style="bright_black")
    for index, section in enumerate(report.checks):
        console.print(section.status.result_label(section))
        _print_children(console, section)
        if index < len(report.checks) - 1:
            console.print()
    console.rule(style="bright_black")

    counts = Counter(report.leaf_statuses())
    summary = "  |  ".join((
        f"{counts[CheckStatus.OK]} ok",
        f"{counts[CheckStatus.WARN]} warn",
        _count_label(counts[CheckStatus.ERROR], "error"),
    ))
    console.print(summary, style=f"bold {report.status.style}")
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
            f": {latest.status.label}",
            style=latest.status.style,
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
            spinner="dots",
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
            checks=(CheckResult("Doctor runtime", CheckStatus.ERROR, exc),),
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
