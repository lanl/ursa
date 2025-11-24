"""Execution agent that builds a tool-enabled state graph to autonomously run tasks.

This module implements ExecutionAgent, a LangGraph-based agent that executes user
instructions by invoking LLM tool calls and coordinating a controlled workflow.

Key features:
- Workspace management with optional symlinking for external sources.
- Safety-checked shell execution via run_cmd with output size budgeting.
- Code authoring and edits through write_code and edit_code with rich previews.
- Web search capability through DuckDuckGoSearchResults.
- Summarization of the session and optional memory logging.
- Configurable graph with nodes for agent, safety_check, action, and summarize.

Implementation notes:
- LLM prompts are sourced from prompt_library.execution_prompts.
- Outputs from subprocess are trimmed under MAX_TOOL_MSG_CHARS to fit tool messages.
- The agent uses ToolNode and LangGraph StateGraph to loop until no tool calls remain.
- Safety gates block unsafe shell commands and surface the rationale to the user.

Environment:
- MAX_TOOL_MSG_CHARS caps combined stdout/stderr in tool responses.

Entry points:
- ExecutionAgent._invoke(...) runs the compiled graph.
- main() shows a minimal demo that writes and runs a script.
"""

# from langchain_core.runnables.graph import MermaidDrawMethod
# async stuff
import asyncio
import base64
import inspect
import json
import os
import re
import secrets
import subprocess
import threading
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
)

import randomname
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import BaseChatModel
from langchain_community.tools import (
    DuckDuckGoSearchResults,
)  # TavilySearchResults,
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import (
    BaseTool,
    InjectedToolCallId,
    StructuredTool,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

# Rich
from rich import get_console
from rich.console import Group
from rich.json import JSON
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from typing_extensions import TypedDict

from ..prompt_library.execution_prompts import (
    executor_prompt,
    get_safety_prompt,
    summarize_prompt,
)
from ..util.diff_renderer import DiffRenderer
from ..util.memory_logger import AgentMemory
from .base import BaseAgent

console = get_console()  # always returns the same instance

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# Global variables for the module.

# Set a limit for message characters - the user could overload
# that in their env, or maybe we could pull this out of the LLM parameters
MAX_TOOL_MSG_CHARS = int(os.getenv("MAX_TOOL_MSG_CHARS", "50000"))

# Set a search tool.
search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(
#                   max_results=10,
#                   search_depth="advanced",
#                   include_answer=True)


def merge_code_files(a: list[str] | None, b: list[str] | None) -> list[str]:
    a = a or []
    b = b or []
    out: list[str] = []
    for x in a + b:
        if x not in out:
            out.append(x)
    return out


# Classes for typing
class ExecutionState(TypedDict):
    """TypedDict representing the execution agent's mutable run state used by nodes.

    Fields:
    - messages: list of messages (System/Human/AI/Tool) with add_messages metadata.
    - current_progress: short status string describing agent progress.
    - code_files: list of filenames created or edited in the workspace.
    - workspace: path to the working directory where files and commands run.
    - symlinkdir: optional dict describing a symlink operation (source, dest,
      is_linked).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    current_progress: str
    code_files: Annotated[
        list[str], merge_code_files
    ]  # multiple code files coming in need a reducer
    workspace: str
    symlinkdir: dict


_BG_LOOP = None


def _ensure_bg_loop():
    """Start (or return) a persistent asyncio loop on a background thread."""
    global _BG_LOOP
    if _BG_LOOP and _BG_LOOP.is_running():
        return _BG_LOOP
    loop = asyncio.new_event_loop()

    def _runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    _BG_LOOP = loop
    return loop


# Tool instrumention helper functions
def _maybe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def to_jsonable(x: Any) -> Any:
    """Recursively convert objects to something JSON-serializable."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, BaseMessage):
        # keep it compact & safe
        return {"message_type": x.type, "content": x.content}
    # pydantic models
    if hasattr(x, "model_dump"):
        try:
            return to_jsonable(x.model_dump())
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return to_jsonable(x.dict())
        except Exception:
            pass
    # final fallback
    return str(x)


def normalize_tool_payload(out: Any):
    """
    Normalize tool outputs from MCP/LangChain so we can render nicely:
    - dict/list → return as-is
    - string that is JSON → parse to dict/list
    - ["{...}", null] style → unwrap and parse inner JSON
    - else → return original string
    """
    if isinstance(out, (dict, list)):
        # unwrap ["{...}", null] pattern
        if (
            isinstance(out, list)
            and len(out) == 2
            and isinstance(out[0], str)
            and (out[1] is None or isinstance(out[1], (str, int, float)))
        ):
            inner = _maybe_json_load(out[0])
            return inner if inner is not None else out
        return out

    if isinstance(out, str):
        parsed = _maybe_json_load(out)
        if parsed is not None:
            if (
                isinstance(parsed, list)
                and len(parsed) == 2
                and isinstance(parsed[0], str)
                and (
                    parsed[1] is None
                    or isinstance(parsed[1], (str, int, float))
                )
            ):
                inner = _maybe_json_load(parsed[0])
                return inner if inner is not None else parsed
            return parsed
        return out

    return out


def _redact_state(obj: Any) -> Any:
    """Recursively replace 'state' payloads with a compact summary."""
    try:
        # dict → copy & process
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "state":
                    if isinstance(v, dict):
                        summary = {}
                        ws = v.get("workspace")
                        if ws:
                            summary["workspace"] = ws
                        msgs = v.get("messages")
                        if isinstance(msgs, list):
                            summary["messages"] = f"<{len(msgs)} messages>"
                        cf = v.get("code_files")
                        if isinstance(cf, list):
                            summary["code_files"] = f"<{len(cf)} files>"
                        out[k] = summary or "<SNIPPED>"
                    else:
                        out[k] = "<SNIPPED>"
                else:
                    out[k] = _redact_state(v)
            return out
        # list/tuple/set → process each
        if isinstance(obj, (list, tuple, set)):
            t = type(obj)
            return t(_redact_state(x) for x in obj)
        # everything else unchanged
        return obj
    except Exception:
        return "<SNIPPED>"


def maybe_redact(obj: Any) -> Any:
    # you almost ALWAYS want this on . . .
    # default ON; set REDACT_TOOL_STATE=0 to disable
    if os.getenv("REDACT_TOOL_STATE", "1") == "0":
        return obj
    return _redact_state(obj)


def unwrap_adapter_result(out: Any):
    # Case: ["{...json...}", None] from langchain_mcp_adapters (content, artifact)
    if isinstance(out, list) and len(out) == 2 and (out[1] is None):
        first = out[0]
        if isinstance(first, str):
            try:
                return json.loads(first)
            except Exception:
                return first
        return first
    return out


MAX_LOG_STRING_CHARS = int(os.getenv("MAX_LOG_STRING_CHARS", "400"))


def _clip_string(s: str, limit: int = MAX_LOG_STRING_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[snipped {len(s) - limit} chars]…"


def _looks_base64(s: str) -> bool:
    # Fast-path heuristic: long, limited alphabet
    if len(s) < 128:
        return False
    for ch in s[:2048]:  # don't scan all
        if not (ch.isalnum() or ch in "+/=\n\r"):
            return False
    return True


def _approx_decoded_len_from_b64(s: str) -> int:
    # ≈ 3/4 of non-padding length
    t = s.rstrip("=\n\r")
    return (len(t) * 3) // 4


def _redact_large_payload(obj: Any) -> Any:
    """
    Recursively redact/clip large values in generic tool I/O payloads.
    - Known base64 fields -> replace with a compact summary
    - Any very long strings -> clip
    """
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if (
                    isinstance(k, str)
                    and k in _B64_FIELD_NAMES
                    and isinstance(v, str)
                ):
                    approx = _approx_decoded_len_from_b64(v)
                    out[k] = f"<base64 payload ~{approx} bytes; redacted>"
                else:
                    out[k] = _redact_large_payload(v)
            return out

        if isinstance(obj, (list, tuple, set)):
            t = type(obj)
            return t(_redact_large_payload(x) for x in obj)

        if isinstance(obj, str):
            if (len(obj) > MAX_LOG_STRING_CHARS) and _looks_base64(obj):
                approx = _approx_decoded_len_from_b64(obj)
                return f"<probable base64 ~{approx} bytes; redacted>"
            # generic long-string clipping
            return _clip_string(obj)

        # everything else
        return obj
    except Exception:
        return "<SNIPPED>"


def _render_tool_io(phase: str, name: str, payload, style: str) -> None:
    payload = maybe_redact(payload)
    norm = unwrap_adapter_result(payload)
    norm = normalize_tool_payload(norm)
    norm = _redact_large_payload(norm)
    header = f"[{style}]TOOL {phase}[/] [bold]{name}[/]"

    try:
        if isinstance(norm, (dict, list)):
            safe = to_jsonable(norm)
            body = JSON.from_data(safe)
        elif isinstance(norm, str):
            looks_jsonish = norm.strip().startswith(("{", "[", '"'))
            body = Syntax(
                norm, "json" if looks_jsonish else "text", word_wrap=True
            )
        else:
            body = Syntax(str(to_jsonable(norm)), "text", word_wrap=True)
    except Exception:
        # last-resort fallback to avoid crashing the demo
        body = Syntax(str(norm), "text", word_wrap=True)

    console.print(Panel(Group(header, body), border_style=style))


def instrument_tool(struct_tool):
    """Wrap a StructuredTool so each call prints args & result using Rich.
    We ALSO sanitize the *result* here before returning, in case the sanitizer
    wrapper didn’t hook this particular tool shape.
    """
    fn = getattr(struct_tool, "func", None)
    coro = getattr(struct_tool, "coroutine", None)
    name = getattr(struct_tool, "name", "<tool>")

    def _sanitize_out(out):
        # Normalize MCP adapter shapes and cache blobs; never return base64.
        out = unwrap_adapter_result(out)
        out = normalize_tool_payload(out)
        out = _extract_and_cache_blobs(out)
        return out

    # Sync function
    if fn and not inspect.iscoroutinefunction(fn):
        orig = fn

        def logged_fn(*args, **kwargs):
            _render_tool_io("→", name, kwargs or {"args": args}, "cyan")
            try:
                raw = orig(*args, **kwargs)
                safe = _sanitize_out(raw)
                _render_tool_io("←", name, safe, "green")
                return safe
            except Exception as e:
                _render_tool_io("!", name, {"error": str(e)}, "red")
                raise

        struct_tool.func = logged_fn

    # Async function
    if coro and inspect.iscoroutinefunction(coro):
        orig_coro = coro

        async def logged_coro(*args, **kwargs):
            _render_tool_io("→", name, kwargs or {"args": args}, "cyan")
            try:
                raw = await orig_coro(*args, **kwargs)
                safe = _sanitize_out(raw)
                _render_tool_io("←", name, safe, "green")
                return safe
            except Exception as e:
                _render_tool_io("!", name, {"error": str(e)}, "red")
                raise

        struct_tool.coroutine = logged_coro

    return struct_tool


def _render_tool_plan(tool_calls) -> None:
    """Pretty-print model's planned tool calls."""
    if not tool_calls:
        console.print(
            Panel.fit(
                "[bold yellow]No tool calls in this step[/]",
                border_style="yellow",
            )
        )
        return

    tbl = Table(
        box=None, show_edge=False, show_header=True, header_style="bold magenta"
    )
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Tool", style="bold cyan")
    tbl.add_column("Args", style="white")

    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name")
        args = tc.get("args") or {}
        tbl.add_row(str(i), f"[cyan]{name}[/]", JSON.from_data(args))

    console.print(
        Panel(
            Group("[bold magenta]Model wants to call[/]", tbl),
            border_style="magenta",
        )
    )


_B64_FIELD_NAMES = {
    "bytes_b64",
    "file_b64",
    "data_b64",
    "content_b64",
    "blob_b64",
}
_BLOB_CACHE: dict[str, bytes] = {}
_BLOB_CACHE_LOCK = threading.Lock()
_MAX_BLOB_CACHE_BYTES = int(
    os.getenv("MAX_BLOB_CACHE_BYTES", str(512 * 1024 * 1024))
)  # 512MB soft cap
_BLOB_CACHE_SIZE = 0


def _new_blob_id() -> str:
    return "blob_" + secrets.token_hex(8)


def _cache_put(b: bytes) -> str:
    global _BLOB_CACHE_SIZE
    with _BLOB_CACHE_LOCK:
        if _BLOB_CACHE_SIZE + len(b) > _MAX_BLOB_CACHE_BYTES:
            raise RuntimeError("Blob cache capacity exceeded")
        bid = _new_blob_id()
        _BLOB_CACHE[bid] = b
        _BLOB_CACHE_SIZE += len(b)
        return bid


def _cache_get(bid: str) -> bytes:
    return _BLOB_CACHE[bid]


def _cache_del(bid: str) -> None:
    global _BLOB_CACHE_SIZE
    with _BLOB_CACHE_LOCK:
        b = _BLOB_CACHE.pop(bid, None)
        if b is not None:
            _BLOB_CACHE_SIZE -= len(b)


def _extract_and_cache_blobs(obj):
    """
    Walk tool result; for any *known* base64 fields, store bytes and replace with a small ref:
    {'bytes_b64': 'AAAA...'} -> {'blob_ref': 'blob_ab12', 'bytes': N}
    Returns the transformed object.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _B64_FIELD_NAMES and isinstance(v, str):
                try:
                    raw = base64.b64decode(v)
                    bid = _cache_put(raw)
                    out["blob_ref"] = bid
                    out["bytes"] = len(raw)
                    # keep *no* base64 in the message
                except Exception:
                    out[k] = v  # if decode fails, leave as-is
            else:
                out[k] = _extract_and_cache_blobs(v)
        return out
    if isinstance(obj, list):
        return [_extract_and_cache_blobs(x) for x in obj]
    return obj


def _sanitize_tool_content(obj):
    out = unwrap_adapter_result(obj)
    out = normalize_tool_payload(out)
    out = _extract_and_cache_blobs(
        out
    )  # turns ..._b64 into {"blob_ref": "...", "bytes": N}
    return out


def prepare_tool(t, enable_logging: bool):
    # Normalize
    tool_obj = t if isinstance(t, BaseTool) else convert_to_tool(t)

    # Optionally add pretty logging
    if enable_logging:
        tool_obj = instrument_tool(tool_obj)
    return tool_obj


# Helper functions
def convert_to_tool(fn):
    # Already a LangChain tool? Just return it.
    if isinstance(fn, BaseTool):
        return fn
    # StructuredTool specifically (subclass of BaseTool); safe to pass through
    if isinstance(fn, StructuredTool):
        return fn
    # Plain callable -> wrap as StructuredTool
    if callable(fn):
        return StructuredTool.from_function(
            func=fn,
            name=getattr(fn, "__name__", "tool"),
            description=(fn.__doc__ or ""),
        )
    # Anything else is unsupported
    raise TypeError(
        f"Expected a callable or a LangChain tool, got: {type(fn).__name__}"
    )


def _strip_fences(snippet: str) -> str:
    """Remove markdown fences from a code snippet.

    This function strips leading triple backticks and any language
    identifiers from a markdown-formatted code snippet and returns
    only the contained code.

    Args:
        snippet: The markdown-formatted code snippet.

    Returns:
        The snippet content without leading markdown fences.
    """
    if "```" not in snippet:
        return snippet

    parts = snippet.split("```")
    if len(parts) < 3:
        return snippet

    body = parts[1]
    return "\n".join(body.split("\n")[1:]) if "\n" in body else body.strip()


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


def _last_ai(msgs: list[BaseMessage]) -> AIMessage | None:
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m
    return None


def should_continue(state: ExecutionState) -> Literal["summarize", "continue"]:
    """Return 'summarize' if no tool calls in the last message, else 'continue'.

    Args:
        state: The current execution state containing messages.

    Returns:
        A literal "summarize" if the last message has no tool calls,
        otherwise "continue".
    """
    msgs = state.get("messages", []) or []
    last_ai = _last_ai(msgs)
    decision = (
        "continue"
        if (last_ai and getattr(last_ai, "tool_calls", None))
        else "summarize"
    )
    # TRACE - may be helpful for debugging, uncomment if needed
    # tc = (last_ai.tool_calls if last_ai else None)
    # console.print(f"[red][TRACE][/red] should_continue -> {decision}; "
    #       f"last_ai={'yes' if last_ai else 'no'}; tool_calls={bool(tc)}; "
    #       f"ids={[t.get('id') for t in (tc or [])]}")
    return decision


def command_safe(state: ExecutionState) -> Literal["safe", "unsafe"]:
    """Return 'safe' if the last command was safe, otherwise 'unsafe'.

    Args:
        state: The current execution state containing messages and tool calls.
    Returns:
        A literal "safe" if no '[UNSAFE]' tags are in the last command,
        otherwise "unsafe".
    """
    msgs = state.get("messages", []) or []
    i = len(msgs) - 1
    unsafe = False
    while i >= 0 and isinstance(msgs[i], ToolMessage):
        c = msgs[i].content or ""
        if "[UNSAFE]" in c:
            unsafe = True
            break
        i -= 1
    decision = "unsafe" if unsafe else "safe"
    # TRACE - may be helpful for debugging, uncomment if needed
    # console.print(f"[red][TRACE][/red] command_safe -> {decision}")
    return decision


# Tools for ExecutionAgent
@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
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

    # Fit BOTH streams under a single overall cap
    stdout_fit, stderr_fit = _fit_streams_to_budget(
        stdout or "", stderr or "", MAX_TOOL_MSG_CHARS
    )

    print("STDOUT: ", stdout_fit)
    print("STDERR: ", stderr_fit)

    return f"STDOUT:\n{stdout_fit}\nSTDERR:\n{stderr_fit}"


@tool
def write_code(
    code: str,
    filename: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Write source code to a file and update the agent’s workspace state.

    Args:
        code: The source code content to be written to disk.
        filename: Name of the target file (including its extension).
        tool_call_id: Identifier for this tool invocation.
        state: Agent state dict holding workspace path and file list.

    Returns:
        Command: Contains an updated state (including code_files) and
        a ToolMessage acknowledging success or failure.
    """
    # Determine the full path to the target file
    workspace_dir = state["workspace"]
    console.print("[cyan]Writing file:[/]", filename)

    # Clean up markdown fences on submitted code.
    code = _strip_fences(code)

    # Show syntax-highlighted preview before writing to file
    try:
        lexer_name = Syntax.guess_lexer(filename, code)
    except Exception:
        lexer_name = "text"

    console.print(
        Panel(
            Syntax(code, lexer_name, line_numbers=True),
            title="File Preview",
            border_style="cyan",
        )
    )

    # Write cleaned code to disk
    code_file = os.path.join(workspace_dir, filename)
    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as exc:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Failed to write file:[/]",
            exc,
        )
        return f"Failed to write {filename}."

    console.print(
        f"[bold bright_white on green] :heavy_check_mark: [/] "
        f"[green]File written:[/] {code_file}"
    )

    # Append the file to the list in agent's state for later reference
    file_list = state.get("code_files", [])
    if filename not in file_list:
        file_list.append(filename)

    # Create a tool message to send back to acknowledge success.
    msg = ToolMessage(
        content=f"File {filename} written successfully.",
        tool_call_id=tool_call_id,
    )

    # Return updated code files list & the message
    return Command(
        update={
            "code_files": file_list,
            "messages": [msg],
        }
    )


@tool
def edit_code(
    old_code: str,
    new_code: str,
    filename: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Replace the **first** occurrence of *old_code* with *new_code* in *filename*.

    Args:
        old_code: Code fragment to search for.
        new_code: Replacement fragment.
        filename: Target file inside the workspace.

    Returns:
        Success / failure message.
    """
    workspace_dir = state["workspace"]
    console.print("[cyan]Editing file:[/cyan]", filename)

    code_file = os.path.join(workspace_dir, filename)
    try:
        with open(code_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]File not found:[/]",
            filename,
        )
        return f"Failed: {filename} not found."

    # Clean up markdown fences
    old_code_clean = _strip_fences(old_code)
    new_code_clean = _strip_fences(new_code)

    if old_code_clean not in content:
        console.print(
            "[yellow] ⚠️ 'old_code' not found in file'; no changes made.[/]"
        )
        return f"No changes made to {filename}: 'old_code' not found in file."

    updated = content.replace(old_code_clean, new_code_clean, 1)

    console.print(
        Panel(
            DiffRenderer(content, updated, filename),
            title="Diff Preview",
            border_style="cyan",
        )
    )

    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(updated)
    except Exception as exc:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Failed to write file:[/]",
            exc,
        )
        return f"Failed to edit {filename}."

    console.print(
        f"[bold bright_white on green] :heavy_check_mark: [/] "
        f"[green]File updated:[/] {code_file}"
    )
    file_list = state.get("code_files", [])
    if code_file not in file_list:
        file_list.append(filename)
    state["code_files"] = file_list

    return f"File {filename} updated successfully."


_CODE_LIKE_EXTS = {
    ".py",
    ".ipynb",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".sh",
    ".rb",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cxx",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
}


# Non-code text we’re OK writing via bytes because files can be large:
def _is_probably_text(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except Exception:
        return False


@tool
def write_bytes(
    data_b64: str | None = None,
    filename: str = "",  # (optional but nice to default, not required)
    tool_call_id: Annotated[
        str, InjectedToolCallId
    ] = "",  # (Injected at runtime)
    state: Annotated[dict, InjectedState] = {},  # (Injected at runtime)
    allow_text: bool = False,
    overwrite: bool = True,
    blob_ref: str | None = None,
) -> Command:
    """
    Save BYTES to a file inside the agent's workspace (binary-safe).

    Preferred input: `blob_ref` (a handle created by upstream tools where large base64
    payloads were cached out-of-band). Use `data_b64` only when a blob_ref is not available.

    Use this tool for:
      • binary data (PDFs, images, zips, parquet, etc.)
      • large text data files (CSV/TSV/JSONL/NDJSON). For these, set `allow_text=True`.

    Do NOT use this for source code or small human-edited text; use `write_code` instead.

    Parameters
    ----------
    blob_ref : str | None
        Reference to cached bytes (preferred). Produces no large messages.
    data_b64 : str | None
        Base64-encoded bytes (fallback when a blob_ref is unavailable).
    filename : str
        Path relative to the workspace (e.g., "downloads/file.pdf"). Directories
        are created as needed.
    allow_text : bool
        If True, allows writing texty payloads via bytes (e.g., large .csv/.tsv/.jsonl).
        If False and the content looks like source/text with a code-like extension,
        the tool will refuse and advise using `write_code`.
    overwrite : bool
        If False and the file already exists, the tool refuses to overwrite.

    Returns
    -------
    Command update with:
      • messages: human-readable confirmation (size, path)
      • code_files: updated list including `filename`

    Examples
    --------
    # Preferred, with blob_ref
    write_bytes(blob_ref="blob_ab12", filename="downloads/report.pdf")

    # Fallback, with base64 (small payloads only)
    write_bytes(data_b64="<base64>", filename="downloads/sample.csv", allow_text=True)
    """

    ws = state["workspace"]
    path = os.path.join(ws, filename)
    os.makedirs(os.path.dirname(path) or ws, exist_ok=True)

    # get raw bytes from either blob_ref (preferred) or base64
    raw = None
    if blob_ref:
        try:
            raw = _cache_get(blob_ref)
        except KeyError:
            # If the blob was already consumed (duplicate tool call),
            # treat it as idempotent if the target file already exists.
            if os.path.exists(path):
                files = state.get("code_files", [])
                if filename not in files:
                    files.append(filename)
                msg = ToolMessage(
                    content=f"File {filename} already exists; assuming prior write succeeded (blob {blob_ref} was already consumed).",
                    tool_call_id=tool_call_id,
                )
                return Command(update={"code_files": files, "messages": [msg]})
            msg = ToolMessage(
                content=f"Blob ref {blob_ref} is not available and file {filename} does not exist; cannot write.",
                tool_call_id=tool_call_id,
            )
            return Command(update={"messages": [msg]})
    elif data_b64:
        raw = base64.b64decode(data_b64)
    else:
        msg = ToolMessage(
            content="write_bytes requires data_b64 or blob_ref",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    _, ext = os.path.splitext(filename.lower())

    if _is_probably_text(raw) and not allow_text and (ext in _CODE_LIKE_EXTS):
        msg = ToolMessage(
            content=(
                f"Refusing to write probable source/text '{filename}' with write_bytes. "
                f"Use write_code for code or pass allow_text=True if this is large data."
            ),
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    if os.path.exists(path) and not overwrite:
        msg = ToolMessage(
            content=f"File exists, not overwriting: {filename}",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    with open(path, "wb") as f:
        f.write(raw)

    if blob_ref:
        _cache_del(blob_ref)

    files = state.get("code_files", [])
    if filename not in files:
        files.append(filename)

    msg = ToolMessage(
        content=(
            f"Wrote {len(raw)} bytes to {filename} in workspace.  "
            f"Next step: read the local path '{filename}' (if needed) instead of re-downloading."
        ),
        tool_call_id=tool_call_id,
    )
    return Command(update={"code_files": files, "messages": [msg]})


@tool
def append_bytes(
    data_b64: str | None = None,
    filename: str = "",
    expected_offset: int = 0,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = {},
    blob_ref: str | None = None,
) -> Command:
    """
    Append BYTES to an existing file inside the agent's workspace (streaming/chunked writes).

    Use when an upstream tool returns data in multiple chunks (e.g., byte ranges).
    Call `write_bytes(...)` once to create the file; then call `append_bytes(...)`
    for each subsequent chunk.

    Preferred input: `blob_ref`. Use `data_b64` only if a blob_ref is not available.

    Parameters
    ----------
    blob_ref : str | None
        Reference to cached bytes for this chunk (preferred).
    data_b64 : str | None
        Base64-encoded bytes for this chunk (fallback).
    filename : str
        Path relative to the workspace; must already exist if this is not the first chunk.
    expected_offset : int | None
        Integrity check. If provided, it must equal the current file size (bytes)
        before appending; otherwise the append is refused. Use this to catch
        out-of-order chunks.

    Returns
    -------
    Command update with:
      • messages: confirmation with appended byte count
      • code_files: updated list including `filename`

    Examples
    --------
    # Create file with first chunk
    write_bytes(blob_ref="blob_0001", filename="downloads/huge.csv", allow_text=True)

    # Append subsequent chunks with integrity checks
    append_bytes(blob_ref="blob_0002", filename="downloads/huge.csv", expected_offset=1048576)
    append_bytes(blob_ref="blob_0003", filename="downloads/huge.csv", expected_offset=2097152)
    """
    ws = state["workspace"]
    path = os.path.join(ws, filename)
    os.makedirs(os.path.dirname(path) or ws, exist_ok=True)

    current = os.path.getsize(path) if os.path.exists(path) else 0
    if expected_offset is not None and current != expected_offset:
        msg = ToolMessage(
            content=f"Offset mismatch: file={current}, expected={expected_offset}. Aborting append.",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    if blob_ref:
        raw = _cache_get(blob_ref)
        _cache_del(blob_ref)
    elif data_b64:
        raw = base64.b64decode(data_b64)
    else:
        msg = ToolMessage(
            content="append_bytes requires data_b64 or blob_ref",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    with open(path, "ab") as f:
        f.write(raw)

    files = state.get("code_files", [])
    if filename not in files:
        files.append(filename)

    msg = ToolMessage(
        content=f"Appended {len(raw)} bytes to {filename}.",
        tool_call_id=tool_call_id,
    )
    return Command(update={"code_files": files, "messages": [msg]})


def _collect_tool_links(msgs):
    """Map assistant tool-calls <-> tool messages so we never break pairs."""
    ai_idx_to_ids = {}
    id_to_tool_idxs = {}
    for i, m in enumerate(msgs):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            ids = [tc.get("id") for tc in m.tool_calls if tc.get("id")]
            if ids:
                ai_idx_to_ids[i] = ids
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                id_to_tool_idxs.setdefault(tcid, []).append(i)
    return ai_idx_to_ids, id_to_tool_idxs


def _shrink_tool_msg(m):
    """Keep the tool_call_id but replace content with a tiny stub."""
    return ToolMessage(
        content="<redacted tool output>",
        tool_call_id=getattr(m, "tool_call_id", None),
    )


def _size_of_msg(m) -> int:
    try:
        if isinstance(m, ToolMessage):
            c = m.content
            if isinstance(c, (dict, list)):
                return len(json.dumps(c))
            return len(str(c))
        if isinstance(m, AIMessage):
            base = len(str(m.content))
            if getattr(m, "tool_calls", None):
                base += len(json.dumps(m.tool_calls))
            return base
        return len(str(getattr(m, "content", "")))
    except Exception:
        return 10000


def _strip_orphan_tool_messages(msgs: list[AnyMessage]) -> list[AnyMessage]:
    """
    Keep ToolMessages only if there exists *some* earlier Assistant with a matching tool_call id.
    Never allow the conversation to start with a ToolMessage.
    Do not drop ToolMessages that are correctly paired, even if other messages appear later.
    """
    cleaned: list[AnyMessage] = []
    seen_tool_ids: set[str] = set()

    # Pass 1: record all assistant tool_call ids seen so far
    assistant_ids = set()
    for m in msgs:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tid = tc.get("id")
                if tid:
                    assistant_ids.add(tid)

    # Pass 2: filter
    for m in msgs:
        if isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid and tcid in assistant_ids:
                cleaned.append(m)
                seen_tool_ids.add(tcid)
            # else: drop true orphan
            continue

        cleaned.append(m)

    # Safety: cannot start with ToolMessage
    while cleaned and isinstance(cleaned[0], ToolMessage):
        cleaned.pop(0)

    return cleaned


def _has_pending_tool_calls(msgs: list) -> bool:
    """Return True if the last assistant-with-tools still lacks matching ToolMessage(s)."""
    last_ai_idx = None
    last_ids = []
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            ids = [tc.get("id") for tc in m.tool_calls if tc.get("id")]
            if ids:
                last_ai_idx = i
                last_ids = ids
                break
    if last_ai_idx is None:
        return False
    got = {
        getattr(m, "tool_call_id", None)
        for m in msgs[last_ai_idx + 1 :]
        if isinstance(m, ToolMessage)
    }
    # pending if any id not yet matched by a ToolMessage
    return any(tid not in got for tid in last_ids)


def _scrub_messages_for_llm(msgs: list[AnyMessage]) -> list[AnyMessage]:
    # If the last assistant-with-tools isn’t fully satisfied, do not alter the list.
    if _has_pending_tool_calls(msgs):
        return msgs

    cleaned: list[AnyMessage] = []
    for m in msgs:
        # Only mutate ToolMessage; pass others through unless they contain giant base64 in strings.
        if isinstance(m, ToolMessage):
            c = m.content
            # If dict/list, ensure any residual base64-ish fields are replaced with blob_ref summaries
            if isinstance(c, (dict, list)):
                # normalize + sanitize again defensively (cheap + safe)
                c2 = _extract_and_cache_blobs(
                    normalize_tool_payload(unwrap_adapter_result(c))
                )
                cleaned.append(
                    ToolMessage(
                        content=c2,
                        tool_call_id=getattr(m, "tool_call_id", None),
                    )
                )
                continue

            # If it's a giant base64-y string, nuke it to a placeholder
            if isinstance(c, str) and len(c) > 10_000 and _looks_base64(c):
                cleaned.append(
                    ToolMessage(
                        content="<redacted binary payload; see blob_ref>",
                        tool_call_id=getattr(m, "tool_call_id", None),
                    )
                )
                continue

        # Also trim absurdly long *string* content on Human/AI messages (rare but safe)
        if (
            hasattr(m, "content")
            and isinstance(m.content, str)
            and len(m.content) > 50_000
        ):
            trimmed = (
                m.content[:50_000]
                + f"\n…[snipped {len(m.content) - 50_000} chars]…"
            )
            nm = type(m)(
                content=trimmed,
                **{
                    k: v for k, v in m.__dict__.items() if k not in ("content",)
                },
            )
            cleaned.append(nm)
            continue

        cleaned.append(m)

    # First, drop true orphans (but keep valid tool windows intact)
    cleaned = _strip_orphan_tool_messages(cleaned)

    # Enforce a hard cap without breaking the last tool-call window
    MAX_TAIL = 60

    if len(cleaned) > MAX_TAIL:
        # Find the most recent assistant-with-tools and the full block of its replies
        last_ai_idx = None
        last_ids = set()
        for i in range(len(cleaned) - 1, -1, -1):
            mi = cleaned[i]
            if isinstance(mi, AIMessage) and getattr(mi, "tool_calls", None):
                ids = [tc.get("id") for tc in mi.tool_calls if tc.get("id")]
                if ids:
                    last_ai_idx = i
                    last_ids = set(ids)
                    break

        if last_ai_idx is not None:
            # Extend tail to include all following ToolMessages that match those ids
            j = last_ai_idx + 1
            while (
                j < len(cleaned)
                and isinstance(cleaned[j], ToolMessage)
                and getattr(cleaned[j], "tool_call_id", None) in last_ids
            ):
                j += 1

            head = cleaned[:last_ai_idx]
            tail = cleaned[last_ai_idx:j]

            # If the protected tail alone is bigger than the cap, DO NOT TRIM this turn.
            # Trimming here would drop required ToolMessages and re-trigger the 400.
            if len(tail) <= MAX_TAIL:
                # Keep as much head as fits, but never cut into the protected tail.
                keep_head_n = MAX_TAIL - len(tail)
                if keep_head_n <= 0:
                    cleaned = tail
                else:
                    # Preserve the very first message (often a System) + the newest head items
                    head_keep = []
                    if head:
                        head_keep = [head[0]]
                        needed = keep_head_n - 1  # we already kept head[0]
                        if needed > 0 and len(head) > 1:
                            head_keep += head[-needed:]
                    cleaned = head_keep + tail
            # else: leave 'cleaned' as-is (skip trimming this round)
        else:
            # No pending/last tool window → safe to keep first + newest messages
            head0 = [cleaned[0]] if cleaned else []
            tail = (
                cleaned[-(MAX_TAIL - len(head0)) :]
                if MAX_TAIL > len(head0)
                else []
            )
            cleaned = head0 + tail

    return cleaned


# this stuff deals w/ MCP tool name weirdness, things like . aren't allowed, this allows us
# to catch and allow them
_ALLOWED_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def _canonical_tool_name(raw: str) -> str:
    """Make a provider-safe tool name (OpenAI: ^[a-zA-Z0-9_-]+$)."""
    if not raw:
        return "tool"
    nm = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw)).strip("_")
    return nm or "tool"


def _dedupe_name(nm: str, existing: set[str]) -> str:
    """Ensure uniqueness after sanitization."""
    if nm not in existing:
        return nm
    base = nm
    i = 2
    while nm in existing:
        nm = f"{base}_{i}"
        i += 1
    return nm


# Main module class
class ExecutionAgent(BaseAgent):
    """Orchestrates model-driven code execution, tool calls, and state management.

    Orchestrates model-driven code execution, tool calls, and state management for
    iterative program synthesis and shell interaction.

    This agent wraps an LLM with a small execution graph that alternates
    between issuing model queries, invoking tools (run, write, edit, search),
    performing safety checks, and summarizing progress. It manages a
    workspace on disk, optional symlinks, and an optional memory backend to
    persist summaries.

    Args:
        llm (BaseChatModel): Model identifier or bound chat model
            instance. If a string is provided, the BaseAgent initializer will
            resolve it.
        agent_memory (Any | AgentMemory, optional): Memory backend used to
            store summarized agent interactions. If provided, summaries are
            saved here.
        log_state (bool): When True, the agent writes intermediate json state
            to disk for debugging and auditability.
        **kwargs: Passed through to the BaseAgent constructor (e.g., model
            configuration, checkpointer).

    Attributes:
        safe_codes (list[str]): List of trusted programming languages for the
            agent. Defaults to python and julia
        executor_prompt (str): Prompt used when invoking the executor LLM
            loop.
        summarize_prompt (str): Prompt used to request concise summaries for
            memory or final output.
        tools (list[Tool]): Tools available to the agent (run_cmd, write_code,
            edit_code, search_tool).
        tool_node (ToolNode): Graph node that dispatches tool calls.
        llm (BaseChatModel): LLM instance bound to the available tools.
        _action (StateGraph): Compiled execution graph that implements the
            main loop and branching logic.

    Methods:
        query_executor(state): Send messages to the executor LLM, ensure
            workspace exists, and handle symlink setup before returning the
            model response.
        summarize(state): Produce and optionally persist a summary of recent
            interactions to the memory backend.
        safety_check(state): Validate pending run_cmd calls via the safety
            prompt and append ToolMessages for unsafe commands.
        get_safety_prompt(query, safe_codes, created_files): Get the LLM prompt for safety_check
            that includes an editable list of available programming languages and gets the context
            of files that the agent has generated and can trust.
        _build_graph(): Construct and compile the StateGraph for the agent
            loop.
        _invoke(inputs, recursion_limit=...): Internal entry that invokes the
            compiled graph with a given recursion limit.
        action (property): Disabled; direct access is not supported. Use
            invoke or stream entry points instead.

    Raises:
        AttributeError: Accessing the .action attribute raises to encourage
            using .stream(...) or .invoke(...).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        agent_memory: Optional[Any | AgentMemory] = None,
        log_state: bool = False,
        extra_tools: Optional[list[Callable[..., Any]]] = None,
        tokens_before_summarize: int = 50000,
        messages_to_keep: int = 20,
        safe_codes: Optional[list[str]] = None,
        tool_log: Optional[bool] = None,
        **kwargs,
    ):
        """ExecutionAgent class initialization."""
        super().__init__(llm, **kwargs)

        self.agent_memory = agent_memory
        self.safe_codes = safe_codes or ["python", "julia"]
        self.get_safety_prompt = get_safety_prompt
        self.executor_prompt = executor_prompt
        self.summarize_prompt = summarize_prompt

        # opt-in tool logging: env or explicit arg
        env_val = os.getenv("URSA_AGENT_TOOL_LOG", "0").strip().lower()
        env_enabled = env_val in ("1", "true", "yes", "on")
        self.tool_log = bool(tool_log) if tool_log is not None else env_enabled

        base_tools = [
            run_cmd,
            write_code,
            edit_code,
            write_bytes,
            append_bytes,
            search_tool,
        ]
        self.extra_tools = extra_tools
        if self.extra_tools is not None:
            base_tools.extend(self.extra_tools)

        # Build + sanitize names + dedupe (so providers like OpenAI accept them)
        prepared = [
            prepare_tool(t, enable_logging=self.tool_log) for t in base_tools
        ]
        existing = set()
        for tool_obj in prepared:
            orig = tool_obj.name
            safe = _canonical_tool_name(orig)
            safe = _dedupe_name(safe, existing)
            # most LangChain tools allow name reassignment; if not, rebuild
            try:
                tool_obj.name = safe
            except Exception:
                tool_obj = StructuredTool.from_function(
                    func=getattr(tool_obj, "func", None)
                    or getattr(tool_obj, "coroutine", None),
                    name=safe,
                    description=getattr(tool_obj, "description", "") or "",
                )
            existing.add(tool_obj.name)

        self.tools = prepared

        # this is a catch for tool errors.
        def _tool_error_handler(e: Exception):
            # Let ToolNode wrap this into a ToolMessage *with the correct tool_call_id*.
            return f"[TOOL_ERROR] {type(e).__name__}: {e}"

        self._tool_error_handler = _tool_error_handler
        self._rebind_tools(self.tools)

        self.log_state = log_state

        self.context_summarizer = SummarizationMiddleware(
            model=self.llm,
            max_tokens_before_summary=tokens_before_summarize,
            messages_to_keep=messages_to_keep,
        )

    def _missing_tool_outputs(self, msgs: list[BaseMessage]) -> list[str]:
        """Return tool_call_ids from the last assistant-with-tools
        that are not followed by ToolMessages."""
        last_ai_idx = None
        last_ids = []
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                ids = [tc.get("id") for tc in m.tool_calls if tc.get("id")]
                if ids:
                    last_ai_idx = i
                    last_ids = ids
                    break
        if last_ai_idx is None:
            return []
        got = {
            getattr(m, "tool_call_id", None)
            for m in msgs[last_ai_idx + 1 :]
            if isinstance(m, ToolMessage)
        }
        return [tid for tid in last_ids if tid not in got]

    # Check message history length and summarize to shorten the token usage:
    def _summarize_context(self, state: ExecutionState) -> ExecutionState:
        msgs = state.get("messages", []) or []

        # 1) If any assistant-with-tools still lacks replies → do nothing.
        if self._missing_tool_outputs(msgs):
            return state

        # 2) Compute tokens BEFORE (for the whole sequence as it stands now).
        try:
            tokens_before = self.context_summarizer.token_counter(msgs)
        except Exception:
            tokens_before = None  # don't let logging failures break execution

        # 3) Find the most recent assistant-with-tools and protect its full tool window.
        last_ai_idx = None
        last_ids = set()
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                ids = [tc.get("id") for tc in m.tool_calls if tc.get("id")]
                if ids:
                    last_ai_idx = i
                    last_ids = set(ids)
                    break

        # Build (head, tail) split; NEVER summarize the tail.
        tail_start = 0 if last_ai_idx is None else last_ai_idx
        j = tail_start + 1
        while (
            j < len(msgs)
            and isinstance(msgs[j], ToolMessage)
            and getattr(msgs[j], "tool_call_id", None) in last_ids
        ):
            j += 1
        head = msgs[:tail_start] if tail_start > 0 else msgs[:0]
        tail = msgs[tail_start:] if tail_start < len(msgs) else []

        if not head:
            # Nothing to summarize—still emit a panel so operators know why
            try:
                tokens_after = self.context_summarizer.token_counter(msgs)
            except Exception:
                tokens_after = None
            console.print(
                Panel(
                    (
                        "Summarized Conversation History (skipped: protected tail only)\n"
                        f"Approximate tokens before: {tokens_before}\n"
                        f"Approximate tokens after:  {tokens_after}\n"
                    ),
                    title="[bold yellow1 on black]:clipboard: Plan",
                    border_style="yellow1",
                    style="bold yellow1 on black",
                )
            )
            return state

        # 4) Summarize only the head
        summarized = self.context_summarizer.before_model(
            {"messages": head}, None
        )

        if summarized and "messages" in summarized:
            state["messages"] = summarized["messages"] + tail

            # Match v1 behavior: count AFTER while skipping an initial SystemMessage if present
            try:
                msgs_after = state["messages"]
                if msgs_after and isinstance(msgs_after[0], SystemMessage):
                    msgs_after = msgs_after[1:]
                tokens_after = self.context_summarizer.token_counter(msgs_after)
            except Exception:
                tokens_after = None

            console.print(
                Panel(
                    (
                        "Summarized Conversation History:\n"
                        f"Approximate tokens before: {tokens_before}\n"
                        f"Approximate tokens after:  {tokens_after}\n"
                    ),
                    title="[bold yellow1 on black]:clipboard: Plan",
                    border_style="yellow1",
                    style="bold yellow1 on black",
                )
            )
        else:
            # Summarizer chose not to change anything; still report counts
            try:
                tokens_after = self.context_summarizer.token_counter(msgs)
            except Exception:
                tokens_after = None
            console.print(
                Panel(
                    (
                        "Summarized Conversation History (no change)\n"
                        f"Approximate tokens before: {tokens_before}\n"
                        f"Approximate tokens after:  {tokens_after}\n"
                    ),
                    title="[bold yellow1 on black]:clipboard: Plan",
                    border_style="yellow1",
                    style="bold yellow1 on black",
                )
            )

        return state

    # Define the function that calls the model
    def query_executor(self, state: ExecutionState) -> ExecutionState:
        """Prepare workspace, handle optional symlinks, and invoke the executor LLM.

        This method copies the incoming state, ensures a workspace directory exists
        (creating one with a random name when absent), optionally creates a symlink
        described by state["symlinkdir"], sets or injects the executor system prompt
        as the first message, and invokes the bound LLM. When logging is enabled,
        it persists the pre-invocation state to disk.

        Args:
            state: The current execution state. Expected keys include:
                - "messages": Ordered list of System/Human/AI/Tool messages.
                - "workspace": Optional path to the working directory.
                - "symlinkdir": Optional dict with "source" and "dest" keys.

        Returns:
            ExecutionState: Partial state update containing:
                - "messages": A list with the model's response as the latest entry.
                - "workspace": The resolved workspace path.
        """
        new_state = state.copy()

        # bail out immediately if any pending tool calls
        if self._missing_tool_outputs(new_state.get("messages", []) or []):
            return new_state

        # 1) Ensure a workspace directory exists, creating a named one if absent.
        if "workspace" not in new_state.keys():
            new_state["workspace"] = randomname.get_name()
            print(
                f"{RED}Creating the folder "
                f"{BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} "
                f"for this project.{RESET}"
            )
        os.makedirs(new_state["workspace"], exist_ok=True)

        # 1.5) Check message history length and summarize to shorten the token usage:
        new_state = self._summarize_context(new_state)

        # 2) Optionally create a symlink if symlinkdir is provided and not yet linked.
        sd = new_state.get("symlinkdir")
        if (
            isinstance(sd, dict)
            and sd  # ignore empty {}
            and "is_linked" not in sd
            and "source" in sd
            and "dest" in sd
        ):
            # symlinkdir structure: {"source": "/path/to/src", "dest": "link/name"}
            symlinkdir = sd

            src = Path(symlinkdir["source"]).expanduser().resolve()
            workspace_root = Path(new_state["workspace"]).expanduser().resolve()
            dst = (
                workspace_root / symlinkdir["dest"]
            )  # Link lives inside workspace.

            # If a file/link already exists at the destination, replace it.
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            # Ensure parent directories for the link exist.
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Create the symlink (tell pathlib if the target is a directory).
            dst.symlink_to(src, target_is_directory=src.is_dir())
            print(f"{RED}Symlinked {src} (source) --> {dst} (dest)")
            new_state["symlinkdir"]["is_linked"] = True

        # 3) Ensure executor prompt is present, but don't clobber an existing SystemMessage.
        msgs = new_state["messages"] or []
        if msgs and isinstance(msgs[0], SystemMessage):
            # Keep the existing system; prepend ours only if it's not already present.
            if (
                not msgs[0].content
                or msgs[0].content.strip() != self.executor_prompt.strip()
            ):
                new_state["messages"] = [
                    SystemMessage(content=self.executor_prompt)
                ] + msgs
        else:
            # No system at all → prepend ours
            new_state["messages"] = [
                SystemMessage(content=self.executor_prompt)
            ] + msgs

        # 4) Invoke the LLM with the prepared message sequence.
        try:
            # Never call the model while an assistant-with-tools is unresolved.
            missing = self._missing_tool_outputs(new_state["messages"])
            if missing:
                # TRACE - may be helpful for debugging, uncomment if needed
                # console.print(f"[red][TRACE][/red] query_executor: waiting for tool replies for ids={missing}")
                return new_state

            scrubbed = _scrub_messages_for_llm(new_state["messages"])
            # persist the scrubbed history so subsequent turns don't grow again
            response = self.llm.invoke(
                scrubbed, self.build_config(tags=["agent"])
            )
            # TRACE - may be helpful for debugging, uncomment if needed
            # tcs = getattr(response, "tool_calls", None) or []
            # console.print(f"[red][TRACE][/red] query_executor: model returned {len(tcs)} tool calls -> "
            #    f"{[ (t.get('name'), t.get('id')) for t in tcs ]}")

            if self.tool_log:
                _render_tool_plan(getattr(response, "tool_calls", None))

            # Append the assistant response to canonical state after logging.
            new_state["messages"].append(response)

        except Exception as e:
            print("Error: ", e, " ", new_state["messages"][-1].content)
            new_state["messages"].append(
                AIMessage(content=f"Response error {e}")
            )

        # 5) Optionally persist the pre-invocation state for audit/debugging.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)

        # Return the model's response and the workspace path as a partial state update.
        return new_state

    def summarize(self, state: ExecutionState) -> ExecutionState:
        """Produce a concise summary of the conversation and optionally persist memory.

        This method builds a summarization prompt, invokes the LLM to obtain a compact
        summary of recent interactions, optionally logs salient details to the agent
        memory backend, and writes debug state when logging is enabled.

        Args:
            state (ExecutionState): The execution state containing message history.

        Returns:
            ExecutionState: A partial update with a single string message containing
                the summary.
        """
        new_state = state.copy()

        # 0) Skip summarization if last assistant-with-tools isn't satisfied
        if self._missing_tool_outputs(new_state["messages"]):
            return new_state

        # 0.5) Check message history length and summarize to shorten the token usage:
        new_state = self._summarize_context(new_state)

        # 1) Construct the summarization message list (system prompt + prior messages).
        messages = (
            new_state["messages"]
            if isinstance(new_state["messages"][0], SystemMessage)
            else [SystemMessage(content=summarize_prompt)]
            + new_state["messages"]
        )

        # 2) Invoke the LLM to generate a summary; capture content even on failure.
        response_content = ""
        try:
            scrubbed_for_model = _scrub_messages_for_llm(messages)
            response = self.llm.invoke(
                scrubbed_for_model, self.build_config(tags=["summarize"])
            )

            response_content = response.content
            new_state["messages"].append(response)
        except Exception as e:
            print("Error: ", e, " ", messages[-1].content)
            new_state["messages"].append(
                AIMessage(content=f"Response error {e}")
            )

        # 3) Optionally persist salient details to the memory backend.
        if self.agent_memory:
            memories: list[str] = []
            # Use a safe projection for logging
            safe_view = _scrub_messages_for_llm(new_state["messages"])
            for msg in safe_view:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # Log tool names/arg keys but not big payloads
                    for tc in msg.tool_calls:
                        memories.append(f"Tool Name: {tc.get('name')}")
                        for k in tc.get("args") or {}:
                            memories.append(f"Arg: {k}")
                else:
                    memories.append(getattr(msg, "content", ""))
            memories.append(response_content)
            self.agent_memory.add_memories(memories)

        # 4) Optionally write state to disk for debugging/auditing.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)

        # 5) Return a partial state update with only the summary content.
        return new_state

    def safety_check(self, state: ExecutionState) -> ExecutionState:
        """Assess pending shell commands for safety and inject ToolMessages with results.

        This method inspects the most recent AI tool calls, evaluates any run_cmd
        queries against the safety prompt, and constructs ToolMessages that either
        flag unsafe commands with reasons or confirm safe execution. If any command
        is unsafe, the generated ToolMessages are appended to the state so the agent
        can react without executing the command.

        Args:
            state (ExecutionState): Current execution state.

        Returns:
            ExecutionState: Either the unchanged state (all safe) or a copy with one
                or more ToolMessages appended when unsafe commands are detected.
        """
        new_state = state.copy()

        last_msg = new_state["messages"][-1]
        # Only run safety checks when the model is proposing tool calls
        if not isinstance(last_msg, AIMessage) or not getattr(
            last_msg, "tool_calls", None
        ):
            return new_state

        # 1) Evaluate pending run_cmd calls (no history needed; use purpose-built system prompt)
        tool_responses: list[ToolMessage] = []
        any_unsafe = False

        for tc in last_msg.tool_calls:
            if tc.get("name") != "run_cmd":
                continue

            query = (tc.get("args") or {}).get("query", "")
            # Build the minimal safety prompt and scrub the view we send to the model
            safety_msgs = get_safety_prompt(
                query, self.safe_codes, new_state.get("code_files", [])
            )
            safety_result = self.llm.invoke(
                _scrub_messages_for_llm(safety_msgs),
                self.build_config(tags=["safety_check"]),
            )

            if "[NO]" in (safety_result.content or ""):
                any_unsafe = True
                tool_responses.append(
                    ToolMessage(
                        content=(
                            "[UNSAFE] That command `{q}` was deemed unsafe and cannot be run.\n"
                            "Reasoning: {r}"
                        ).format(q=query, r=safety_result.content),
                        tool_call_id=tc.get("id"),
                    )
                )
                console.print(
                    "[bold red]Command deemed unsafe:[/bold red] ", query
                )
            else:
                # Optional: you can log/print the pass, but do NOT emit a ToolMessage on success.
                console.print(
                    "[green]Command passed safety check:[/green] ", query
                )

        # 2) If any were unsafe, append tool responses now so the planner can react
        if any_unsafe and tool_responses:
            new_state["messages"].extend(tool_responses)

        return new_state

    def action_with_sanitize_sync(self, state: ExecutionState):
        """Sync adapter for the async tool node; runs the coroutine on
        the background loop and returns the final result."""
        loop = _ensure_bg_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self.action_with_sanitize_async(state), loop
        )
        return fut.result()

    async def action_with_sanitize_async(self, state: ExecutionState):
        # TRACE - may be helpful for debugging, uncomment if needed
        last_ai = _last_ai(state.get("messages", []) or [])
        planned = getattr(last_ai, "tool_calls", None) or []
        console.print(
            f"[red][TRACE][/red] action: entering with {len(planned)} planned calls -> "
            f"{[(t.get('name'), t.get('id')) for t in planned]}"
        )

        result = await self.tool_node.ainvoke(state)

        # TRACE - may be helpful for debugging, uncomment if needed
        tname = type(result).__name__
        preview = None
        if isinstance(result, dict):
            preview = list(result.keys())
        elif hasattr(result, "update"):
            preview = list((result.update or {}).keys())
        console.print(
            f"[red][TRACE][/red] action: ToolNode returned {tname} with keys={preview}"
        )

        def _sanitize_msg(m):
            if isinstance(m, ToolMessage):
                return ToolMessage(
                    content=_sanitize_tool_content(m.content),
                    tool_call_id=getattr(m, "tool_call_id", None),
                )
            if isinstance(m, BaseMessage):
                return m
            return None  # drop

        # 1) Command -> sanitize its update.messages
        if isinstance(result, Command):
            upd = dict(result.update or {})
            msgs = []
            for m in upd.get("messages") or []:
                sm = _sanitize_msg(m)
                if sm is not None:
                    msgs.append(sm)
            upd["messages"] = msgs
            return Command(update=upd)

        # 2) dict -> convert/clean messages in-place
        if isinstance(result, dict):
            if "messages" in result:
                msgs = []
                for m in list(result["messages"] or []):
                    sm = _sanitize_msg(m)
                    if sm is not None:
                        msgs.append(sm)
                out = dict(result)
                out["messages"] = msgs
                return Command(update=out)
            return Command(update=result)

        # 3) list -> merge to a single Command(update=...)
        if isinstance(result, list):
            merged = {"messages": [], "code_files": []}
            for item in result:
                if isinstance(item, Command):
                    upd = dict(item.update or {})
                    for f in upd.get("code_files") or []:
                        if f not in merged["code_files"]:
                            merged["code_files"].append(f)
                    for m in upd.get("messages") or []:
                        sm = _sanitize_msg(m)
                        if sm is not None:
                            merged["messages"].append(sm)
                elif isinstance(item, dict):
                    for f in item.get("code_files") or []:
                        if f not in merged["code_files"]:
                            merged["code_files"].append(f)
                    for m in item.get("messages") or []:
                        sm = _sanitize_msg(m)
                        if sm is not None:
                            merged["messages"].append(sm)
                else:
                    sm = _sanitize_msg(item)
                    if sm is not None:
                        merged["messages"].append(sm)
            return Command(update=merged)

        # 4) single-message results (the common edge case)
        if isinstance(result, ToolMessage):
            sm = _sanitize_msg(result)  # preserves tool_call_id
            return Command(update={"messages": [sm]})

        if isinstance(result, BaseMessage):
            return Command(update={"messages": [result]})

        # 5) scalar fallbacks (string/None/other)
        if isinstance(result, str):
            # Best-effort: attach as a tool message without an id (better than dropping it)
            return Command(
                update={
                    "messages": [ToolMessage(content=result, tool_call_id=None)]
                }
            )

        if result is None:
            return Command(update={})

        # 6) Unknown type: stringify to a safe ToolMessage stub
        return Command(
            update={
                "messages": [
                    ToolMessage(content=str(result), tool_call_id=None)
                ]
            }
        )

    def _build_graph(self, use_async_action: bool = True):
        """Construct and compile the agent's LangGraph state machine."""
        # Create a graph over the agent's execution state.
        graph = StateGraph(ExecutionState)

        # Register nodes:
        # - "agent": LLM planning/execution step
        # - "action": tool dispatch (run_cmd, write_code, etc.)
        # - "summarize": summary/finalization step
        # - "safety_check": gate for shell command safety
        self.add_node(graph, self.query_executor, "agent")

        if use_async_action:
            # native async node for ainvoke()
            self.add_node(graph, self.action_with_sanitize_async, "action")
        else:
            # sync adapter so invoke() stays fully sync-safe
            self.add_node(graph, self.action_with_sanitize_sync, "action")

        self.add_node(graph, self.summarize, "summarize")
        self.add_node(graph, self.safety_check, "safety_check")

        # Set entrypoint: execution starts with the "agent" node.
        graph.set_entry_point("agent")

        # From "agent", either continue (tools) or finish (summarize),
        # based on presence of tool calls in the last message.
        graph.add_conditional_edges(
            "agent",
            self._wrap_cond(should_continue, "should_continue", "execution"),
            {"continue": "safety_check", "summarize": "summarize"},
        )

        # From "safety_check", route to tools if safe, otherwise back to agent
        # to revise the plan without executing unsafe commands.
        graph.add_conditional_edges(
            "safety_check",
            self._wrap_cond(command_safe, "command_safe", "execution"),
            {"safe": "action", "unsafe": "agent"},
        )

        # After tools run, return control to the agent for the next step.
        graph.add_edge("action", "agent")

        # The graph completes at the "summarize" node.
        graph.set_finish_point("summarize")

        # Return the uncompiled graph; we'll compile variants in __init__ / add_tool.
        return graph

    async def add_mcp_tool(
        self, mcp_tools: Callable[..., Any] | list[Callable[..., Any]]
    ) -> None:
        client = MultiServerMCPClient(mcp_tools)
        tools = await client.get_tools()
        self.add_tool(tools)

    def _rebind_tools(self, tools):
        self.tools = list(tools)
        self.tool_node = ToolNode(
            self.tools, handle_tool_errors=self._tool_error_handler
        )
        self.llm = self.llm.bind_tools(self.tools)

        # recompile both graph variants if you need to (as you already do)
        graph_sync = self._build_graph(use_async_action=False)
        graph_async = self._build_graph(use_async_action=True)
        sync_cp = getattr(self, "checkpointer", None)
        self._action_sync = graph_sync.compile(checkpointer=sync_cp)
        from langgraph.checkpoint.memory import MemorySaver

        def _is_async_cp(cp):
            return hasattr(cp, "aget_tuple") or hasattr(cp, "alist")

        async_cp = sync_cp if _is_async_cp(sync_cp) else MemorySaver()
        self._action_async = graph_async.compile(checkpointer=async_cp)
        self._action = self._action_sync

    def add_tool(
        self, new_tools: BaseTool | StructuredTool | Any | Iterable[Any]
    ) -> None:
        if not isinstance(new_tools, (list, tuple, set)):
            candidates = [new_tools]
        else:
            candidates = list(new_tools)

        # prepare & sanitize each incoming tool
        existing = {t.name for t in self.tools}
        prepared: list[BaseTool] = []
        for t in candidates:
            tool_obj = prepare_tool(t, enable_logging=self.tool_log)

            # sanitize name for provider compatibility
            orig = tool_obj.name
            safe = _canonical_tool_name(orig)
            safe = _dedupe_name(safe, existing)

            try:
                tool_obj.name = safe
            except Exception:
                # some wrappers don't allow assignment: rebuild with the safe name
                tool_obj = StructuredTool.from_function(
                    func=getattr(tool_obj, "func", None)
                    or getattr(tool_obj, "coroutine", None),
                    name=safe,
                    description=getattr(tool_obj, "description", "") or "",
                )

            if tool_obj.name in existing:
                continue

            prepared.append(tool_obj)
            existing.add(tool_obj.name)

        if not prepared:
            return

        self._rebind_tools(self.tools + prepared)

    def list_tools(self) -> None:
        print(
            f"Available tool names are: {', '.join([x.name for x in self.tools])}."
        )

    def remove_tool(self, cut_tools: str | list[str]) -> None:
        names = [cut_tools] if isinstance(cut_tools, str) else list(cut_tools)
        self._rebind_tools([t for t in self.tools if t.name not in set(names)])

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 999_999, **_
    ):
        """Run the execution graph with a sync-first, async-fallback strategy.

        1) Try the SYNC-compiled graph first (preserves legacy `.invoke(...)` usage).
        2) On the canonical “Cannot invoke a coroutine function synchronously” error,
        fall back to the ASYNC-compiled graph, executed on a background event loop.
        If the configured checkpointer is sync-only (e.g., SqliteSaver), rebind the
        runnable with an async-safe saver (MemorySaver) **and** remove any per-call
        `checkpointer` from the config so the rebind is honored.
        """

        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        # 1) Try SYNC first (keeps old examples working)
        try:
            return self._action_sync.invoke(inputs, config)
        except TypeError as e:
            if "Cannot invoke a coroutine function synchronously" not in str(e):
                raise

        # 2) Async fallback — make sure the runnable has an async-safe checkpointer
        checkpointer = getattr(self, "checkpointer", None)
        is_async_cp = hasattr(checkpointer, "aget_tuple") or hasattr(
            checkpointer, "alist"
        )

        async_action = self._action_async
        if not is_async_cp:
            from langgraph.checkpoint.memory import MemorySaver

            async_action = self._action_async.with_config(
                checkpointer=MemorySaver()
            )

        # IMPORTANT: remove any per-call checkpointer so we don't override the rebind above
        cfg2 = dict(config) if isinstance(config, dict) else config
        if isinstance(cfg2, dict):
            cfg2.pop("checkpointer", None)

        loop = _ensure_bg_loop()
        fut = asyncio.run_coroutine_threadsafe(
            async_action.ainvoke(inputs, cfg2), loop
        )
        return fut.result()

    def _ainvoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 999_999, **_
    ):
        """Invoke the compiled graph with inputs under a specified recursion limit.

        This method builds a LangGraph config with the provided recursion limit
        and a "graph" tag, then delegates to the compiled graph's invoke method.
        """
        # Build invocation config with a generous recursion limit for long runs.
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        # Delegate execution to the compiled graph.
        return self._action_async.ainvoke(inputs, config)

    # This property is trying to stop people bypassing invoke
    @property
    def action(self):
        """Property used to affirm `action` attribute is unsupported."""
        raise AttributeError(
            "Use .stream(...) or .invoke(...); direct .action access is unsupported."
        )
