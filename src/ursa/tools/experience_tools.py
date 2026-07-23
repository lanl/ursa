import difflib
import time
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents
from ursa.util.parse import read_text_file, read_text_from_file
from ursa.util.rendering import event_artifact, file_artifact
from ursa.util.types import (
    AsciiValidationError,
    ascii_validation_message,
    validate_ascii,
)

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
        raise ValueError(
            f"Experience files, {name} , must use the .md extension."
        )

    return name


def _experience_path(filename: str, runtime: ToolRuntime[AgentContext]) -> Path:
    safe_name = _validate_experience_filename(filename)
    return (_experiences_dir(runtime) / safe_name).resolve()


def _relative_to_den(path: Path, runtime: ToolRuntime[AgentContext]) -> str:
    """Return a den-relative path for display/store keys.

    Tool paths are resolved before use. Resolve the den as well so Path.relative_to
    does not fail when one side is absolute/resolved and the other is not. If a
    future caller passes an unexpected path outside the den, fall back to the
    resolved absolute path rather than crashing a tool call.
    """
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(runtime.context.den.resolve()))
    except ValueError:
        return str(resolved_path)


@tool
def write_experience(
    filename: str,
    content: str,
    runtime: ToolRuntime[AgentContext],
    append: bool = True,
) -> str:
    """Write or append information to markdown files for later recall.

    Use it for lessons learned, task-specific notes, observations, summaries, and other
    context worth preserving in the future.

    Args:
        filename: Markdown filename to write inside the experiences directory. Must be a .md file!
        content: Text content to store.
        append: If True, append content to the file. If False, overwrite the file.

    Returns:
        Confirmation message describing the write operation.
    """
    try:
        filename = validate_ascii(filename)
    except AsciiValidationError as exc:
        return ascii_validation_message("filename", exc)
    experience_file = _experience_path(filename, runtime)
    events = ToolEvents.from_runtime("write_experience", runtime)
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

    try:
        with events.range(
            "write",
            "Writing experience",
            done="Experience written",
            error="Failed to write experience",
            path=str(experience_file),
        ) as span:
            with open(experience_file, "w", encoding="utf-8") as f:
                f.write(updated_text)
            span.update(
                artifact=file_artifact(
                    experience_file, title="Experience written"
                )
            )
    except OSError as exc:
        return f"Failed to write {filename}: {exc}"

    if (store := runtime.store) is not None:
        store.put(
            ("den", "experience_edit"),
            _relative_to_den(experience_file, runtime),
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
                "append": append,
            },
        )

    return f"Experience file {_relative_to_den(experience_file, runtime)} {action} successfully."


@tool
def edit_experience(
    old_content: str,
    new_content: str,
    filename: str,
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
    try:
        filename = validate_ascii(filename)
    except AsciiValidationError as exc:
        return ascii_validation_message("filename", exc)
    experience_file = _experience_path(filename, runtime)
    events = ToolEvents.from_runtime("edit_experience", runtime)
    events.emit(
        "Editing experience",
        stage="edit",
        phase="start",
        path=str(experience_file),
    )

    try:
        content = read_text_file(experience_file)
    except FileNotFoundError:
        events.emit(
            "Experience file not found",
            stage="edit",
            phase="error",
            path=str(experience_file),
        )
        return f"Failed: {filename} not found."
    except ValueError as exc:
        events.emit(
            "Failed to read experience",
            stage="edit",
            phase="error",
            path=str(experience_file),
            error=str(exc),
        )
        return f"Failed to edit {filename}: {exc}"
    except OSError as exc:
        events.emit(
            "Failed to read experience",
            stage="edit",
            phase="error",
            path=str(experience_file),
            error=str(exc),
        )
        return f"Failed to edit {filename}: Could not read file: {exc}"

    if old_content not in content:
        events.emit(
            "No changes made",
            stage="edit",
            phase="end",
            path=str(experience_file),
            reason="'old_content' not found in experience file.",
        )
        return f"No changes made to {filename}: 'old_content' not found in experience file."

    updated = content.replace(old_content, new_content, 1)

    diff = "".join(
        difflib.unified_diff(
            content.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=filename,
            tofile=filename,
        )
    )

    try:
        with open(experience_file, "w", encoding="utf-8") as f:
            f.write(updated)
    except OSError as exc:
        events.emit(
            "Failed to edit experience",
            stage="edit",
            phase="error",
            path=str(experience_file),
            error=str(exc),
        )
        return f"Failed to edit {filename}: {exc}"

    events.emit(
        "Experience updated",
        stage="edit",
        phase="end",
        path=str(experience_file),
        artifact=event_artifact(
            diff,
            "text/x-diff",
            metadata={"title": "Experience diff", "path": filename},
        ),
    )

    if (store := runtime.store) is not None:
        store.put(
            ("den", "experience_edit"),
            _relative_to_den(experience_file, runtime),
            {
                "modified": time.time(),
                "tool_call_id": runtime.tool_call_id,
                "thread_id": runtime.config.get("metadata", {}).get(
                    "thread_id", None
                ),
                "append": False,
            },
        )

    return f"Experience file {_relative_to_den(experience_file, runtime)} updated successfully."


@tool
def read_experience(
    filename: str,
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
    try:
        filename = validate_ascii(filename)
    except AsciiValidationError as exc:
        return ascii_validation_message("filename", exc)
    experience_file = _experience_path(filename, runtime)
    events = ToolEvents.from_runtime("read_experience", runtime)
    if not experience_file.exists() or not experience_file.is_file():
        events.emit(
            "Experience file not found",
            stage="read",
            phase="error",
            path=str(experience_file),
        )
        return (
            "Experience file not found: "
            f"{_relative_to_den(experience_file, runtime)}"
        )

    try:
        content = read_text_from_file(experience_file)
    except (OSError, ValueError) as exc:
        events.emit(
            "Failed to read experience",
            stage="read",
            phase="error",
            path=str(experience_file),
            error=str(exc),
        )
        return f"Failed to read {filename}: {exc}"
    events.emit(
        "Experience read",
        stage="read",
        phase="end",
        path=str(experience_file),
        artifact=file_artifact(experience_file, title="Experience read"),
    )
    return content


@tool
def list_experiences(runtime: ToolRuntime[AgentContext]) -> str:
    """List available markdown experience files stored in the experiences directory.

    This helps identify which experience files can be read back into context later.

    Returns:
        A newline-separated list of available experience filenames, or a message if none exist.
    """
    experiences_dir = _experiences_dir(runtime)
    events = ToolEvents.from_runtime("list_experiences", runtime)
    files = sorted(
        path.name
        for path in experiences_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )

    if not files:
        events.emit("No experience files found", stage="list", count=0)
        return "No experience files found in experiences/."

    events.emit(
        "Experience files listed",
        stage="list",
        count=len(files),
        artifact=event_artifact(
            "\n".join(files),
            "text/plain",
            metadata={"title": "Experiences"},
        ),
    )
    return "Available experience files:\n" + "\n".join(
        f"- {name}" for name in files
    )
