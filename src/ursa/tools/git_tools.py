import subprocess
from collections.abc import Iterable
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.types import AsciiStr

# Git commands are typically instant; timeout indicates hanging (waiting for input or wrong directory)
GIT_TIMEOUT = 30  # seconds - git ops should be near-instant


def _format_result(stdout: str | None, stderr: str | None) -> str:
    return f"STDOUT:\n{stdout or ''}\nSTDERR:\n{stderr or ''}"


def _repo_path(
    repo_path: str | None, runtime: ToolRuntime[AgentContext]
) -> Path:
    base = Path(runtime.context.workspace).absolute()
    if not repo_path:
        candidate = base
    else:
        candidate = Path(repo_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.absolute()

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("repo_path must resolve inside the workspace") from exc

    return candidate


def _run_git(repo: Path, args: Iterable[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *list(args)],
            text=True,
            capture_output=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _format_result("", f"Error running git: {exc}")

    return _format_result(result.stdout, result.stderr)


def _check_ref_format(repo: Path, branch: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ref-format", "--branch", branch],
        text=True,
        capture_output=True,
        timeout=GIT_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        return result.stderr or result.stdout or "Invalid branch name for git"
    return None


@tool
def git_status(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Return git status for a repository inside the workspace.

    Args:
        repo_path: Path to repository relative to workspace. If None, uses workspace root.
                   Recommended to always specify a repo_path to avoid large untracked file lists.
    """
    repo = _repo_path(repo_path, runtime)

    # Warn if using workspace root without explicit repo_path
    workspace = Path(runtime.context.workspace).absolute()
    if repo_path is None and repo == workspace:
        return (
            "WARNING: git_status called on workspace root without specifying repo_path. "
            "This may show many untracked files. "
            "Please specify a specific repository path (e.g., repo_path='my-project'). "
            "Use list_directory to see available repositories in the workspace first."
        )

    result = _run_git(repo, ["status", "-sb"])

    # Limit output size for very large untracked file lists
    if len(result) > 10000:
        lines = result.split("\n")
        if len(lines) > 100:
            return (
                f"Git status output too large ({len(lines)} lines). "
                f"Showing first 50 and last 50 lines:\n"
                f"{''.join(lines[:50])}\n"
                f"... ({len(lines) - 100} lines omitted) ...\n"
                f"{''.join(lines[-50:])}\n"
                f"Consider using git_status on a specific subdirectory."
            )

    return result


@tool
def git_diff(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
    staged: bool = False,
    pathspecs: list[AsciiStr] | None = None,
) -> str:
    """Return git diff for a repository inside the workspace."""
    repo = _repo_path(repo_path, runtime)
    args = ["diff"]
    if staged:
        args.append("--staged")
    if pathspecs:
        args.append("--")
        args.extend(list(pathspecs))
    return _run_git(repo, args)


@tool
def git_log(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
    limit: int = 20,
) -> str:
    """Return recent git log entries for a repository."""
    repo = _repo_path(repo_path, runtime)
    limit = max(1, int(limit))
    return _run_git(repo, ["log", f"-n{limit}", "--oneline", "--decorate"])


@tool
def git_ls_files(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
    pathspecs: list[AsciiStr] | None = None,
) -> str:
    """List tracked files, optionally filtered by pathspecs."""
    repo = _repo_path(repo_path, runtime)
    args = ["ls-files"]
    if pathspecs:
        args.append("--")
        args.extend(list(pathspecs))
    return _run_git(repo, args)


@tool
def git_add(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
    pathspecs: list[AsciiStr] | None = None,
) -> str:
    """Stage files for commit using git add."""
    repo = _repo_path(repo_path, runtime)
    if not pathspecs:
        return _format_result("", "No pathspecs provided to git_add")
    return _run_git(repo, ["add", "--", *list(pathspecs)])


@tool
def git_commit(
    runtime: ToolRuntime[AgentContext],
    message: AsciiStr,
    repo_path: AsciiStr | None = None,
) -> str:
    """Create a git commit with the provided message."""
    repo = _repo_path(repo_path, runtime)
    if not message.strip():
        return _format_result("", "Commit message must not be empty")
    return _run_git(repo, ["commit", "--message", message])


@tool
def git_switch(
    runtime: ToolRuntime[AgentContext],
    branch: AsciiStr,
    repo_path: AsciiStr | None = None,
    create: bool = False,
) -> str:
    """Switch branches using git switch (optionally create)."""
    repo = _repo_path(repo_path, runtime)
    err = _check_ref_format(repo, branch)
    if err:
        return _format_result("", err)
    args = ["switch"]
    if create:
        args.append("-c")
    args.append(branch)
    return _run_git(repo, args)


@tool
def git_create_branch(
    runtime: ToolRuntime[AgentContext],
    branch: AsciiStr,
    repo_path: AsciiStr | None = None,
) -> str:
    """Create a branch without switching to it."""
    repo = _repo_path(repo_path, runtime)
    err = _check_ref_format(repo, branch)
    if err:
        return _format_result("", err)
    return _run_git(repo, ["branch", branch])


GIT_TOOLS = [
    git_status,
    git_diff,
    git_log,
    git_ls_files,
    git_add,
    git_commit,
    git_switch,
    git_create_branch,
]
