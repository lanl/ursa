import subprocess
from pathlib import Path
from typing import TypedDict

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.prompt_library.execution_prompts import (
    get_safety_prompt,
)
from ursa.util.types import AsciiStr

RUN_COMMAND_SAFETY_NAMESPACE = ("workspace", "command_safety")


class SafetyAssessment(TypedDict):
    is_safe: bool
    reason: str


def clean_env(workspace: Path):
    import os

    env = dict(os.environ)
    for k in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"]:
        env.pop(k, None)

    old = env.get("PATH", env.get("PATH", ""))
    parts = [p for p in old.split(os.pathsep) if "\\.venv\\" not in p.lower()]
    env["PATH"] = os.pathsep.join(parts)

    env["UV_PROJECT"] = str(workspace.resolve())
    env["UV_PROJECT_ENVIRONMENT"] = str(workspace.resolve() / ".venv")
    return env


def run_command_safety_key(tool_call_id: str | None, query: str) -> str:
    return tool_call_id or f"query:{query}"


@tool
def run_command(query: AsciiStr, runtime: ToolRuntime[AgentContext]) -> str:
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

    llm = runtime.context.llm
    safety_result = llm.with_structured_output(SafetyAssessment).invoke(
        get_safety_prompt(query, safe_codes, edited_files)
    )
    if runtime.store is not None:
        runtime.store.put(
            RUN_COMMAND_SAFETY_NAMESPACE,
            run_command_safety_key(runtime.tool_call_id, query),
            {
                "query": query,
                "is_safe": safety_result["is_safe"],
                "reason": safety_result["reason"],
            },
        )

    if not safety_result["is_safe"]:
        return (
            f"[UNSAFE] That command `{query}` was deemed unsafe and cannot be run.\n"
            f"For reason: {safety_result['reason']}"
        )

    try:
        result = subprocess.run(
            query,
            text=True,
            shell=True,
            timeout=60000,
            capture_output=True,
            env=clean_env(workspace_dir),
            cwd=workspace_dir,
            check=False,
        )
        stdout, stderr = result.stdout, result.stderr
    except KeyboardInterrupt:
        stdout, stderr = "", "KeyboardInterrupt:"

    # Fit BOTH streams under a single overall cap
    stdout_fit, stderr_fit = _fit_streams_to_budget(
        stdout or "", stderr or "", runtime.context.tool_character_limit
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
