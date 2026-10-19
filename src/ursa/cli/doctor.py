# ruff: noqa: TID251

"""Composable health checks for ``ursa self doctor``."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib.metadata import EntryPoint, entry_points
from inspect import isawaitable
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel
from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from ursa.cli.config import UrsaConfig

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
        checks.append(
            CheckResult(
                "Final resolved configuration",
                CheckStatus.ERROR,
                _exception_message(
                    context.config_error
                    or RuntimeError("configuration is unavailable")
                ),
            )
        )
    else:
        checks.append(
            CheckResult(
                "Final resolved configuration",
                CheckStatus.OK,
                "Merged and resolved successfully",
            )
        )
    return CheckResult.group("Configuration", checks)


def _secret_value(config: Any) -> str | None:
    reference = getattr(config, "api_key", None)
    if reference is None:
        return None
    return reference.get_secret_value()


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
    if not invoked.is_absolute():
        resolved = shutil.which(str(invoked))
        if resolved is None:
            return None
        invoked = Path(resolved)
    return invoked.absolute()


def _same_executable(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _current_path_index(
    executables: Sequence[_PathUrsa], running: Path
) -> int | None:
    normalized = os.path.normcase(str(running))
    exact = next(
        (
            index
            for index, entry in enumerate(executables)
            if os.path.normcase(str(entry.executable)) == normalized
        ),
        None,
    )
    if exact is not None:
        return exact
    return next(
        (
            index
            for index, entry in enumerate(executables)
            if _same_executable(entry.executable, running)
        ),
        None,
    )


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
        current_index = _current_path_index(executables, running)
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

    shadowing = current_index
    additional = len(executables) - 1
    message = None
    if additional:
        message = f"Found {len(executables)} URSA executables"
        if shadowing:
            noun = "entry" if shadowing == 1 else "entries"
            message += (
                f"; {shadowing} {noun} before and shadow the running URSA"
            )
        else:
            noun = "copy" if additional == 1 else "copies"
            message += f"; {additional} additional {noun}"
    return CheckResult.group("URSA on PATH", checks, message)


def check_inference_providers(context: DoctorContext) -> CheckResult:
    """Validate credentials and connectivity for every configured provider."""
    if context.config is None:
        return _configuration_error("Inference providers")

    from ursa.util.inference_providers import list_provider_models

    checks = []
    for name, provider in context.config.inference_providers.items():
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
            checks.append(_error_result(name, exc, (secret,) if secret else ()))
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


def _run_model_probe(config: Any, name: str, probe: ModelProbe) -> CheckResult:
    secret = None
    try:
        secret = _secret_value(config)
        model = config.init_chat_model()
        message = probe(model)
        return CheckResult(name, CheckStatus.OK, message)
    except _CHECK_FAILURES as exc:
        return _error_result(name, exc, (secret,) if secret else ())


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
        _run_model_probe(model, name, probe)
        for name, probe in (
            ("Chat invocation", _probe_chat),
            ("Forced tool use", _probe_forced_tool_use),
            ("Structured output", _probe_structured_output),
        )
    )
    model_result = CheckResult.group(_model_label(model), probes)
    return CheckResult.group("LLM models", [model_result])


def _probe_embedding_model(config: Any) -> CheckResult:
    secret = None
    try:
        secret = _secret_value(config)
        model = config.init_embedding()
        embedding = model.embed_query("URSA health check")
        if not isinstance(embedding, Sequence) or not embedding:
            raise ValueError("embedding invocation returned no vector")
        return CheckResult(
            _model_label(config),
            CheckStatus.OK,
            f"Embedding invocation returned {len(embedding)} dimensions",
        )
    except _CHECK_FAILURES as exc:
        return _error_result(
            _model_label(config), exc, (secret,) if secret else ()
        )


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
    return CheckResult.group(
        "Embedding models", [_probe_embedding_model(context.config.emb_model)]
    )


async def _probe_mcp_server(name: str, server: Any) -> CheckResult:
    from ursa.util.mcp import start_mcp_client

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


async def _probe_mcp_servers(servers: Mapping[str, Any]) -> list[CheckResult]:
    return list(
        await asyncio.gather(
            *(
                _probe_mcp_server(name, server)
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
    checks = asyncio.run(_probe_mcp_servers(context.config.mcp_servers))
    return CheckResult.group("MCP servers", checks)


def _additional_entry_points() -> list[EntryPoint]:
    return sorted(
        entry_points(group=HEALTHCHECK_ENTRYPOINT_GROUP),
        key=lambda item: item.name,
    )


def _normalize_result(result: CheckResult) -> CheckResult:
    """Defensively normalize a result before aggregation or serialization."""
    if not isinstance(result, CheckResult):
        raise TypeError(f"expected CheckResult, got {type(result).__name__}")
    name = str(result.name)
    message = None if result.message is None else str(result.message)
    checks = tuple(_normalize_result(check) for check in result.checks)
    return CheckResult(
        name=name,
        status=CheckStatus(result.status),
        message=message,
        checks=checks,
        duration_ms=result.duration_ms,
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
            result = replace(_normalize_result(value), name=entry_point.name)
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
    check_ursa_on_path: "URSA on PATH",
    check_inference_providers: "Inference providers",
    check_llm_models: "LLM models",
    check_embedding_models: "Embedding models",
    check_mcp_servers: "MCP servers",
    check_additional_healthchecks: "Additional health checks",
}


def _check_name(check: DoctorCheck) -> str:
    fallback = getattr(check, "__name__", type(check).__name__)
    try:
        return _BUILTIN_CHECK_NAMES.get(check, fallback)
    except TypeError:
        return fallback


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
        result = _normalize_result(result)
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


_INDICATORS = {
    CheckStatus.OK: "✅",
    CheckStatus.WARN: "⚠️",
    CheckStatus.ERROR: "❌",
}


_STATUS_STYLES = {
    CheckStatus.OK: "green",
    CheckStatus.WARN: "yellow",
    CheckStatus.ERROR: "red",
}

_STATUS_LABELS = {
    CheckStatus.OK: "[ok]",
    CheckStatus.WARN: "[!!]",
    CheckStatus.ERROR: "[xx]",
}


def _result_label(result: CheckResult) -> Text:
    style = _STATUS_STYLES[result.status]
    label = Text()
    label.append(f"{_INDICATORS[result.status]} ")
    label.append(result.name, style=f"bold {style}")
    if result.duration_ms is not None:
        label.append(
            f"  ({result.duration_ms}\N{NO-BREAK SPACE}ms)", style="dim"
        )
    if result.message:
        label.append("  ")
        label.append(result.message, style="dim")
    return label


def _add_result_rows(output: Text, result: CheckResult, depth: int = 0) -> None:
    for child in result.checks:
        status = child.status
        output.append("  ")
        output.append(
            _STATUS_LABELS[status],
            style=f"bold {_STATUS_STYLES[status]}",
        )
        output.append(f"  {'  ' * depth}{child.name}", style="bold")
        if child.message:
            output.append(f"  {child.message}", style="dim")
        output.append("\n")
        _add_result_rows(output, child, depth + 1)


def _section_renderable(result: CheckResult) -> Text:
    content = _result_label(result)
    if result.checks:
        content.append("\n")
        _add_result_rows(content, result)
        content.rstrip()
    return content


def _leaf_statuses(result: CheckResult) -> list[CheckStatus]:
    if not result.checks:
        return [result.status]
    return [
        status for child in result.checks for status in _leaf_statuses(child)
    ]


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun if count == 1 else noun + 's'}"


def report_renderable(report: CheckResult) -> Group:
    """Build grouped checkhealth-style human output."""
    sections: list[Any] = []
    for section in report.checks:
        if sections:
            sections.append(Text())
        sections.append(_section_renderable(section))
    counts = Counter(_leaf_statuses(report))
    summary = "  |  ".join((
        f"{counts[CheckStatus.OK]} ok",
        f"{counts[CheckStatus.WARN]} warn",
        _count_label(counts[CheckStatus.ERROR], "error"),
    ))
    summary_text = Text(summary, style=f"bold {_STATUS_STYLES[report.status]}")
    hint = Text(
        "Use `ursa self doctor --json` for the machine-readable report.",
        style="dim",
    )
    return Group(
        _result_label(replace(report, message=None)),
        Rule(style="bright_black"),
        *sections,
        Rule(style="bright_black"),
        summary_text,
        hint,
    )


def render_report(report: CheckResult) -> str:
    """Render a plain terminal snapshot, primarily for tests and logs."""
    output = io.StringIO()
    Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=100,
    ).print(report_renderable(report))
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
            f": {latest.status.value}", style=_STATUS_STYLES[latest.status]
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
    json_output = bool(getattr(args, "json", False))
    console = None
    try:
        if json_output:
            # Third-party checks are not allowed to corrupt the JSON document.
            with redirect_stdout(io.StringIO()):
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
            (console or Console()).print(report_renderable(report))
        except Exception:
            print(json.dumps(report.as_dict(), indent=2))  # noqa: T201
    return 1 if report.status is CheckStatus.ERROR else 0
