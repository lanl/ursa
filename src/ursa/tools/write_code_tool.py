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


def _validate_file_path(
    filename: str, workspace_dir: Path, repo_path: Path | None = None
) -> tuple[Path, str | None]:
    """Validate that a filename is within workspace and optionally within a repo.
    
    Args:
        filename: The requested filename to write to
        workspace_dir: The workspace directory (all files must be under this)
        repo_path: Optional repo directory (if provided, file must be under this)
    
    Returns:
        Tuple of (resolved_path, error_message). If error_message is not None,
        the resolution failed.
    """
    # Resolve the file path
    if Path(filename).is_absolute():
        file_path = Path(filename)
    else:
        file_path = workspace_dir / filename
    
    file_path = file_path.resolve()
    
    # Validate it's within the workspace
    try:
        file_path.relative_to(workspace_dir.resolve())
    except ValueError:
        return None, (
            f"File path '{filename}' resolves outside workspace directory. "
            "Files must be written within the workspace."
        )
    
    # If repo_path is specified, validate it's within the repo
    if repo_path is not None:
        # Resolve repo_path relative to workspace if it's not absolute
        repo_resolved = repo_path if repo_path.is_absolute() else workspace_dir / repo_path
        repo_resolved = repo_resolved.resolve()
        
        try:
            file_path.relative_to(repo_resolved)
        except ValueError:
            return None, (
                f"File path '{filename}' resolves outside repository directory. "
                "Files must be written within the repository."
            )
    
    return file_path, None


@tool(description="Write source code to a file")
def write_code(
    code: str,
    filename: AsciiStr,
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Write source code to a file

    Records successful file edits to the graph's store

    Args:
        code: The source code content to be written to disk.
        filename: Name of the target file (including its extension).
        repo_path: Optional repo path - if provided, file must be within this repo.

    """
    # Determine the full path to the target file
    workspace_dir = runtime.context.workspace
    console.print("[cyan]Writing file:[/]", filename)

    # Validate file path
    repo = None
    if repo_path:
        repo = Path(repo_path)
        if not repo.is_absolute():
            repo = workspace_dir / repo
        repo = repo.resolve()
    
    code_file, error = _validate_file_path(filename, workspace_dir, repo)
    if error:
        console.print(f"[bold bright_white on red] :heavy_multiplication_x: [/] [red]{error}[/]")
        return f"Failed to write {filename}: {error}"

    # Show syntax-highlighted preview before writing to file
    try:
        lexer_name = Syntax.guess_lexer(str(code_file), code)
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
    try:
        code_file.parent.mkdir(parents=True, exist_ok=True)
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
    repo_path: AsciiStr | None = None,
) -> str:
    """Replace the **first** occurrence of *old_code* with *new_code* in *filename*.

    Args:
        old_code: Code fragment to search for.
        new_code: Replacement fragment.
        filename: Target file inside the workspace.
        repo_path: Optional repo path - if provided, file must be within this repo.

    Returns:
        Success / failure message.
    """
    workspace_dir = runtime.context.workspace
    console.print("[cyan]Editing file:[/cyan]", filename)

    # Validate file path
    repo = None
    if repo_path:
        repo = Path(repo_path)
        if not repo.is_absolute():
            repo = workspace_dir / repo
        repo = repo.resolve()
    
    code_file, error = _validate_file_path(filename, workspace_dir, repo)
    if error:
        console.print(f"[bold bright_white on red] :heavy_multiplication_x: [/] [red]{error}[/]")
        return f"Failed to edit {filename}: {error}"

    try:
        content = read_text_file(code_file)
    except FileNotFoundError:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]File not found:[/]",
        )
        return f"Failed: {filename} not found."

    # Clean up markdown fences
    old_code_clean = old_code
    new_code_clean = new_code

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
