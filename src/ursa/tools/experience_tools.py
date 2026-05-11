import time
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from rich import get_console
from rich.panel import Panel

from ursa.agents.base import AgentContext
from ursa.util.diff_renderer import DiffRenderer
from ursa.util.parse import read_text_file, read_text_from_file
from ursa.util.types import AsciiStr

console = get_console()

EXPERIENCES_DIRNAME = "experiences"


def _experiences_dir(runtime: ToolRuntime[AgentContext]) -> Path:
    path = runtime.context.den / EXPERIENCES_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_experience_filename(filename: str) -> str:
    name = filename.strip()
    if not name:
        raise ValueError("Experience filename must not be empty.")

    path = Path(name)
    if path.is_absolute() or path.name != name or name in {".", ".."}:
        raise ValueError(
            "Experience filename must be a simple relative file name within the experiences directory."
        )

    if path.suffix.lower() != ".md":
        raise ValueError("Experience files must use the .md extension.")

    return name


def _experience_path(filename: str, runtime: ToolRuntime[AgentContext]) -> Path:
    safe_name = _validate_experience_filename(filename)
    return (_experiences_dir(runtime) / safe_name).resolve()


@tool
def write_experience(
    filename: AsciiStr,
    content: str,
    runtime: ToolRuntime[AgentContext],
    append: bool = True,
) -> str:
    """Write or append information to markdown files for later recall.

    Use it for lessons learned, task-specific notes, observations, summaries, and other
    context worth preserving in the future.

    Args:
        filename: Markdown filename to write inside the experiences directory.
        content: Text content to store.
        append: If True, append content to the file. If False, overwrite the file.

    Returns:
        Confirmation message describing the write operation.
    """
    experience_file = _experience_path(filename, runtime)
    experience_file.parent.mkdir(parents=True, exist_ok=True)

    existing_text = ""
    if experience_file.exists():
        existing_text = read_text_from_file(experience_file)

    text_to_write = content.rstrip() + "\n"
    if append and existing_text:
        separator = "\n\n" if not existing_text.endswith("\n\n") else ""
        updated_text = existing_text + separator + text_to_write
        action = "updated"
    elif append:
        updated_text = text_to_write
        action = "created"
    else:
        updated_text = text_to_write
        action = "overwritten"

    with open(experience_file, "w", encoding="utf-8") as f:
        f.write(updated_text)

    if (store := runtime.store) is not None:
        store.put(
            ("den", "experience_edit"),
            str(experience_file.relative_to(runtime.context.den)),
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
                "append": append,
            },
        )

    console.print("[cyan]Writing experience file:[/]", experience_file)
    return f"Experience file {experience_file.relative_to(runtime.context.den)} {action} successfully."


@tool
def edit_experience(
    old_content: str,
    new_content: str,
    filename: AsciiStr,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Replace the first occurrence of old_content with new_content in a markdown experience file.

    Args:
        old_content: Text fragment to search for.
        new_content: Replacement text fragment.
        filename: Markdown filename inside the experiences directory.

    Returns:
        Success or failure message describing the edit operation.
    """
    experience_file = _experience_path(filename, runtime)
    console.print("[cyan]Editing experience file:[/cyan]", filename)

    try:
        content = read_text_file(experience_file)
    except FileNotFoundError:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Experience file not found:[/]",
        )
        return f"Failed: {filename} not found."
    except ValueError as exc:
        return f"Failed to edit {filename}: {exc}"
    except OSError as exc:
        return f"Failed to edit {filename}: Could not read file: {exc}"

    if old_content not in content:
        console.print(
            "[yellow] ⚠️ 'old_content' not found in experience file; no changes made.[/]"
        )
        return f"No changes made to {filename}: 'old_content' not found in experience file."

    updated = content.replace(old_content, new_content, 1)

    console.print(
        Panel(
            DiffRenderer(content, updated, filename),
            title="Experience Diff Preview",
            border_style="cyan",
        )
    )

    try:
        with open(experience_file, "w", encoding="utf-8") as f:
            f.write(updated)
    except Exception as exc:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Failed to write experience file:[/]",
            exc,
        )
        return f"Failed to edit {filename}: {exc}"

    if (store := runtime.store) is not None:
        store.put(
            ("den", "experience_edit"),
            str(experience_file.relative_to(runtime.context.den)),
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
                "append": False,
            },
        )

    return f"Experience file {experience_file.relative_to(runtime.context.den)} updated successfully."


@tool
def read_experience(
    filename: AsciiStr,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Read a markdown experience file to recall previously stored context.

    This tool loads previously stored notes, experiences, and lessons learned from the
    experiences directory so they can be brought back into context for the current task.

    Args:
        filename: Markdown filename to read from the experiences directory.

    Returns:
        The contents of the requested experience file.
    """
    experience_file = _experience_path(filename, runtime)
    if not experience_file.exists() or not experience_file.is_file():
        return (
            "Experience file not found: "
            f"{experience_file.relative_to(runtime.context.den)}"
        )

    console.print("[cyan]Reading experience:[/]", experience_file)
    return read_text_from_file(experience_file)


@tool
def list_experiences(runtime: ToolRuntime[AgentContext]) -> str:
    """List available markdown experience files stored in the experiences directory.

    This helps identify which experience files can be read back into context later.

    Returns:
        A newline-separated list of available experience filenames, or a message if none exist.
    """
    experiences_dir = _experiences_dir(runtime)
    files = sorted(
        path.name
        for path in experiences_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )

    if not files:
        return "No experience files found in experiences/."

    console.print("[cyan]Listing experience files...")
    return "Available experience files:\n" + "\n".join(
        f"- {name}" for name in files
    )
