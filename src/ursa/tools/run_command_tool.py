import os
import platform
import subprocess
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ursa.agents.base import AgentContext
from ursa.prompt_library.safety_prompts import (
    get_safety_prompt,
)
from ursa.sandbox import (
    LinuxBwrapSandbox,
    MacSandbox,
    NoSandbox,
    SandboxError,
    _disable_sbx_suggestion,
    parse_extra_writable_paths,
)
from ursa.util.events import ToolEvents
from ursa.util.rendering import event_artifact
from ursa.util.structured_output import invoke_structured
from ursa.util.types import (
    AsciiValidationError,
    ascii_validation_message,
    validate_ascii,
)


class SafetyAssessment(BaseModel):
    is_safe: bool = Field(description="Whether the command is safe to execute.")
    reason: str = Field(description="Brief reason for the safety decision.")


@tool
def run_command(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Execute a shell command in the workspace and return its combined output.

    Runs the specified command using subprocess.run in the given workspace
    directory, captures stdout and stderr, enforces a maximum character budget,
    and formats both streams into a single string. KeyboardInterrupt during
    execution is caught and reported.

    The command timeout defaults to 600000 seconds. Users can set
    URSA_COMMAND_TIMEOUT_SECONDS to a positive integer to override it.

    Args:
        query: The shell command to execute.

    Returns:
        A formatted string with "STDOUT:" followed by the truncated stdout and
        "STDERR:" followed by the truncated stderr.
    """
    try:
        query = validate_ascii(query)
    except AsciiValidationError as exc:
        return ascii_validation_message("query", exc)
    try:
        timeout = int(os.getenv("URSA_COMMAND_TIMEOUT_SECONDS", "600000"))
        if timeout <= 0:
            raise ValueError
    except ValueError:
        return (
            "Invalid URSA_COMMAND_TIMEOUT_SECONDS: expected a positive integer "
            "number of seconds. Command was not executed."
        )
    workspace_dir = Path(runtime.context.workspace)
    if runtime.store is not None:
        search_results = runtime.store.search(
            ("workspace", "file_edit"), limit=1000
        )
        edited_files = [item.key for item in search_results]
    else:
        edited_files = []

    if runtime.store is not None:
        search_results = runtime.store.search(
            ("workspace", "safe_codes"), limit=1000
        )
        safe_codes = [item.key for item in search_results]
    else:
        safe_codes = []

    prompt_level = os.getenv("URSA_SAFETY_LEVEL", "default")
    llm = runtime.context.llm
    events = ToolEvents.from_runtime("run_command", runtime)
    if prompt_level.lower() in {"yolo", "none"}:
        safety_result = SafetyAssessment(
            is_safe=True, reason=f"User set safety level to {prompt_level}"
        )
    else:
        safety_result = invoke_structured(
            llm,
            SafetyAssessment,
            get_safety_prompt(
                query, safe_codes, edited_files, prompt_level=prompt_level
            ),
            context="run_command safety assessment",
            fallback=SafetyAssessment(
                is_safe=False,
                reason=(
                    "Could not parse command safety assessment from the model. "
                    "Command blocked."
                ),
            ),
            repair=1,
        )

        if not safety_result.is_safe:
            tool_response = f"[UNSAFE] That command `{query}` was deemed unsafe and cannot be run.\nFor reason: {safety_result.reason}"
            events.emit(
                "Command deemed unsafe",
                stage="safety_check",
                query=query,
                safe=False,
                reason=safety_result.reason,
            )
            return tool_response

    events.emit(
        "Command passed safety check",
        stage="safety_check",
        query=query,
        safe=True,
        reason=safety_result.reason,
    )

    try:
        with events.range(
            "execute",
            "Running command",
            done="Command finished",
            error="Command interrupted",
            query=query,
        ) as span:
            access_network = os.getenv("URSA_ACCESS_NETWORK", "1") == "1"
            enable_sbx = os.getenv("URSA_SBX", "1") == "1"
            extra_writable_paths = parse_extra_writable_paths(
                "URSA_EXTRA_WRITABLE_PATHS"
            )
            match (pf := platform.system()), prompt_level.lower(), enable_sbx:
                # NOTE:
                # * No sandbox if URSA_SAFETY_LEVEL=yolo or URSA_SBX=0
                # * To use sandbox but disable safety checks,
                #   set URSA_SAFETY_LEVEL=none, URSA_SBX=1 (default)
                case _, "yolo", _:
                    sbx = NoSandbox(cwd=workspace_dir, timeout=timeout)
                case _, _, 0:
                    sbx = NoSandbox(cwd=workspace_dir, timeout=timeout)
                case "Linux", _, 1:
                    sbx = LinuxBwrapSandbox(
                        cwd=workspace_dir,
                        access_network=access_network,
                        extra_writable_paths=extra_writable_paths,
                        timeout=timeout,
                    )
                case "Darwin", _, 1:
                    sbx = MacSandbox(
                        cwd=workspace_dir,
                        access_network=access_network,
                        extra_writable_paths=extra_writable_paths,
                        timeout=timeout,
                    )
                case "Windows", _, 1:
                    # TODO: Implement WindowsSandbox
                    raise SandboxError(
                        f"Unsupported platform {pf} for sandboxing. "
                        + _disable_sbx_suggestion
                    )
                case _, _, _:
                    raise SandboxError(
                        f"Unsupported platform {pf} for sandboxing."
                        + _disable_sbx_suggestion
                    )

            if not sbx.is_working():
                raise SandboxError(
                    f"Could not start sandbox: {sbx.backend}. "
                    + _disable_sbx_suggestion
                )

            result = sbx.execute(query)
            stdout, stderr = result.stdout, result.stderr
            stdout = stdout.encode("utf-8", errors="ignore").decode("utf-8")
            stderr = stderr.encode("utf-8", errors="ignore").decode("utf-8")
            # Fit BOTH streams under a single overall cap
            stdout_fit, stderr_fit = _fit_streams_to_budget(
                stdout or "",
                stderr or "",
                runtime.context.tool_character_limit,
            )
            artifacts = [
                event_artifact(
                    content,
                    "text/plain",
                    metadata={"title": stream},
                )
                for stream, content in (
                    ("stdout", stdout_fit),
                    ("stderr", stderr_fit),
                )
                if content
            ]
            span.update(
                returncode=getattr(result, "returncode", None),
                stdout_chars=len(stdout or ""),
                stderr_chars=len(stderr or ""),
                stdout_truncated=stdout_fit != (stdout or ""),
                stderr_truncated=stderr_fit != (stderr or ""),
                **({"artifacts": artifacts} if artifacts else {}),
            )
    except subprocess.TimeoutExpired:
        stdout_fit, stderr_fit = _fit_streams_to_budget(
            "",
            f"Command timed out after {timeout} seconds.",
            runtime.context.tool_character_limit,
        )
    except KeyboardInterrupt:
        stdout, stderr = "", "KeyboardInterrupt:"
        stdout_fit, stderr_fit = _fit_streams_to_budget(
            stdout,
            stderr,
            runtime.context.tool_character_limit,
        )
    except Exception as exc:  # noqa: BLE001
        stdout, stderr = "", f"{type(exc).__name__}: {exc}"
        stdout_fit, stderr_fit = _fit_streams_to_budget(
            stdout,
            stderr,
            runtime.context.tool_character_limit,
        )

    return f"STDOUT:\n{stdout_fit}\nSTDERR:\n{stderr_fit}"


def _fit_streams_to_budget(stdout: str, stderr: str, total_budget: int):
    """Allocate and truncate stdout and stderr to fit a total character budget.

    Args:
        stdout: The original stdout string.
        stderr: The original stderr string.
        total_budget: The combined character budget for stdout and stderr.

    Returns:
        A tuple of (possibly truncated stdout, possibly truncated stderr).
    """
    label_overhead = len("STDOUT:\n") + len("\nSTDERR:\n")
    budget = max(0, total_budget - label_overhead)

    if len(stdout) + len(stderr) <= budget:
        return stdout, stderr

    total_len = max(1, len(stdout) + len(stderr))
    stdout_budget = int(budget * (len(stdout) / total_len))
    stderr_budget = budget - stdout_budget

    stdout_snip, _ = _snip_text(stdout, stdout_budget)
    stderr_snip, _ = _snip_text(stderr, stderr_budget)

    return stdout_snip, stderr_snip


def _snip_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text to a maximum length and indicate if truncation occurred.

    Args:
        text: The original text to potentially truncate.
        max_chars: The maximum characters allowed in the output.

    Returns:
        A tuple of (possibly truncated text, boolean flag indicating
        if truncation occurred).
    """
    if text is None:
        return "", False
    if max_chars <= 0:
        return "", len(text) > 0
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + f"\n... [snipped {len(text) - max_chars} chars] ...\n"
        + text[-tail:],
        True,
    )
