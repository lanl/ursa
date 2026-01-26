import time
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from rich import get_console
from rich.panel import Panel
from rich.syntax import Syntax

from ursa.agents.base import AgentContext
from ursa.util.diff_renderer import DiffRenderer
from ursa.util.parse import read_text_file
from ursa.util.types import AsciiStr

console = get_console()


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


@tool(description="Write source code to a file")
def write_code(
    code: str,
    filename: AsciiStr,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Write source code to a file

    Records successful file edits to the graph's store

    Args:
        code: The source code content to be written to disk.
        filename: Name of the target file (including its extension).

    """
    # Determine the full path to the target file
    workspace_dir = runtime.context.workspace
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
    code_file = workspace_dir.joinpath(filename)
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

    # Record the edit operation
    if (store := runtime.store) is not None:
        store.put(
            ("workspace", "file_edit"),
            filename,
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
            },
        )

    return f"File {filename} written successfully."


@tool
def edit_code(
    old_code: str,
    new_code: str,
    filename: AsciiStr,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Replace the **first** occurrence of *old_code* with *new_code* in *filename*.

    Args:
        old_code: Code fragment to search for.
        new_code: Replacement fragment.
        filename: Target file inside the workspace.

    Returns:
        Success / failure message.
    """
    workspace_dir = runtime.context.workspace
    console.print("[cyan]Editing file:[/cyan]", filename)

    code_file = Path(workspace_dir, filename)
    try:
        content = read_text_file(code_file)
    except FileNotFoundError:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]File not found:[/]",
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

    # Record the edit operation
    if (store := runtime.store) is not None:
        store.put(
            ("workspace", "file_edit"),
            filename,
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
            },
        )

    return f"File {filename} updated successfully."
