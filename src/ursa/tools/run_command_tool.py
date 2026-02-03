import os
import subprocess
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from rich import get_console

from ursa.prompt_library.execution_prompts import (
    get_safety_prompt,
)

# Global variables for the module.

# Set a limit for message characters - the user could overload
# that in their env, or maybe we could pull this out of the LLM parameters
MAX_TOOL_MSG_CHARS = int(os.getenv("MAX_TOOL_MSG_CHARS", "30000"))

console = get_console()


@tool
def run_command(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Execute a shell command in the workspace and return its combined output.

    Runs the specified command using subprocess.run in the given workspace
    directory, captures stdout and stderr, enforces a maximum character budget,
    and formats both streams into a single string. KeyboardInterrupt during
    execution is caught and reported.

    Args:
        query: The shell command to execute.
        state: A dict with injected state; must include the 'workspace' path.

    Returns:
        A formatted string with "STDOUT:" followed by the truncated stdout and
        "STDERR:" followed by the truncated stderr.
    """
    workspace_dir = state["workspace"]
    llm = state["model"]

    safety_result = StrOutputParser().invoke(
        llm.invoke(
            get_safety_prompt(
                query, state["safe_codes"], state.get("code_files", [])
            )
        )
    )

    if "[NO]" in safety_result:
        tool_response = f"[UNSAFE] That command `{query}` was deemed unsafe and cannot be run.\nFor reason: {safety_result}"
        console.print(
            "[bold red][WARNING][/bold red] Command deemed unsafe:",
            query,
        )
        # Also surface the model's rationale for transparency.
        console.print("[bold red][WARNING][/bold red] REASON:", tool_response)
        return tool_response
    else:
        console.print(
            f"[green]Command passed safety check:[/green] {query}\nFor reason: {safety_result}"
        )

    print("RUNNING: ", query)
    try:
        result = subprocess.run(
            query,
            text=True,
            shell=True,
            timeout=60000,
            capture_output=True,
            cwd=workspace_dir,
        )
        stdout, stderr = result.stdout, result.stderr
    except KeyboardInterrupt:
        print("Keyboard Interrupt of command: ", query)
        stdout, stderr = "", "KeyboardInterrupt:"
    except UnicodeDecodeError:
        print(
            f"Invalid Command: {query} - only 'utf-8' decodable characters allowed."
        )
        stdout, stderr = (
            "",
            f"Invalid Command: {query} - only 'utf-8' decodable characters allowed.:",
        )

    # Fit BOTH streams under a single overall cap
    stdout_fit, stderr_fit = _fit_streams_to_budget(
        stdout or "", stderr or "", MAX_TOOL_MSG_CHARS
    )

    print("STDOUT: ", stdout_fit)
    print("STDERR: ", stderr_fit)

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
