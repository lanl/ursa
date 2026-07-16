import os
import subprocess
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ursa.agents.base import AgentContext
from ursa.prompt_library.safety_prompts import (
    get_safety_prompt,
)
from ursa.util.events import ToolEvents
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
            result = subprocess.run(
                query,
                text=True,
                shell=True,
                timeout=60000,
                capture_output=True,
                cwd=workspace_dir,
                check=False,
            )
            stdout, stderr = result.stdout, result.stderr
            # Fit BOTH streams under a single overall cap
            stdout_fit, stderr_fit = _fit_streams_to_budget(
                stdout or "",
                stderr or "",
                runtime.context.tool_character_limit,
            )
            span.update(
                returncode=getattr(result, "returncode", None),
                stdout_chars=len(stdout or ""),
                stderr_chars=len(stderr or ""),
                stdout_truncated=stdout_fit != (stdout or ""),
                stderr_truncated=stderr_fit != (stderr or ""),
            )
    except KeyboardInterrupt:
        stdout, stderr = "", "KeyboardInterrupt:"
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
