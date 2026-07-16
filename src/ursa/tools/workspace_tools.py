from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents


@tool
def list_workspace_files(
    runtime: ToolRuntime[AgentContext],
    pattern: Annotated[
        str,
        "Glob pattern relative to the workspace, for example '**/*.md' or '*'",
    ] = "**/*",
    max_results: Annotated[
        int,
        "Maximum number of paths to return. Use a smaller number for broad workspaces.",
    ] = 200,
) -> str:
    """List files in the current workspace, not the agent den.

    Use this before read_file when the user has pointed the agent at a workspace
    containing relevant documents. Returned paths are relative to the workspace
    and can be passed to read_file.
    """
    workspace = runtime.context.workspace.resolve()
    events = ToolEvents.from_runtime("list_workspace_files", runtime)
    events.emit(
        "Listing workspace files",
        stage="list_workspace",
        workspace=str(workspace),
        pattern=pattern,
        max_results=max_results,
    )

    try:
        paths = []
        for path in workspace.glob(pattern or "**/*"):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if workspace not in resolved.parents and resolved != workspace:
                continue
            if path.is_dir():
                continue
            rel = path.relative_to(workspace)
            if any(part.startswith(".") for part in rel.parts):
                continue
            paths.append(rel.as_posix())
        paths = sorted(dict.fromkeys(paths))[: max(1, max_results)]
    except Exception as exc:  # noqa: BLE001
        events.emit(
            "Workspace listing failed",
            stage="list_workspace_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return f"Error listing workspace files: {exc}"

    if not paths:
        return "No matching files found in the workspace."
    return "Workspace files:\n" + "\n".join(f"- {path}" for path in paths)
