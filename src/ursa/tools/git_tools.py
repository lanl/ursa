import json
import subprocess
from pathlib import Path
from typing import Iterable

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.types import AsciiStr

# Differentiated timeouts by operation type
# Git commands are typically instant; timeout indicates hanging (waiting for input or wrong directory)
GIT_TIMEOUT = 30  # seconds - git ops should be near-instant
GO_FORMAT_TIMEOUT = 30  # seconds - gofmt is usually fast
GO_ANALYSIS_TIMEOUT = 60  # seconds - go vet, go mod tidy
GO_BUILD_TIMEOUT = 300  # seconds (5 min) - builds can take time for large projects
GO_TEST_TIMEOUT = 600  # seconds (10 min) - test suites can be slow
LINT_TIMEOUT = 180  # seconds (3 min) - linting is moderate speed


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
        raise ValueError(
            "repo_path must resolve inside the workspace"
        ) from exc

    return candidate


def _run_git(repo: Path, args: Iterable[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *list(args)],
            text=True,
            capture_output=True,
            timeout=TOOL_TIMEOUT,
        )
    except Exception as exc:
        return _format_result("", f"Error running git: {exc}")

    return _format_result(result.stdout, result.stderr)


def _check_ref_format(repo: Path, branch: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ref-format", "--branch", branch],
        text=True,
        capture_output=True,
        timeout=TOOL_TIMEOUT,
    )
    if result.returncode != 0:
        return (
            result.stderr
            or result.stdout
            or "Invalid branch name for git"
        )
    return None


@tool
def git_status(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Return git status for a repository inside the workspace."""
    repo = _repo_path(repo_path, runtime)
    return _run_git(repo, ["status", "-sb"])


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


@tool
def gofmt_files(
    runtime: ToolRuntime[AgentContext],
    paths: list[AsciiStr],
    repo_path: AsciiStr | None = None,
) -> str:
    """Format Go files in-place using gofmt."""
    if not paths:
        return _format_result("", "No paths provided to gofmt_files")
    if any(not str(p).endswith(".go") for p in paths):
        return _format_result("", "gofmt_files only accepts .go files")
    repo = _repo_path(repo_path, runtime)
    try:
        result = subprocess.run(
            ["gofmt", "-w", *list(paths)],
            text=True,
            capture_output=True,
            timeout=TOOL_TIMEOUT,
            cwd=repo,
        )
    except Exception as exc:
        return _format_result("", f"Error running gofmt: {exc}")
    return _format_result(result.stdout, result.stderr)


@tool
def go_build(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Build a Go module using go build ./..."""
    repo = _repo_path(repo_path, runtime)
    try:
        result = subprocess.run(
            ["go", "build", "./..."],
            text=True,
            capture_output=True,
            timeout=GO_BUILD_TIMEOUT,
            cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            "", f"go build timed out after {GO_BUILD_TIMEOUT}s (5 minutes). "
            "Large builds may need to be run in smaller chunks."
        )
    except Exception as exc:
        return _format_result("", f"Error running go build: {exc}")
    return _format_result(result.stdout, result.stderr)


@tool
def go_test(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
    verbose: bool = True,
) -> str:
    """Run Go tests using go test ./..."""
    repo = _repo_path(repo_path, runtime)
    args = ["go", "test"]
    if verbose:
        args.append("-v")
    args.append("./...")
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=GO_TEST_TIMEOUT,
            cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            "", f"go test timed out after {GO_TEST_TIMEOUT}s (10 minutes). "
            "Large test suites may need to run selectively."
        )
    except Exception as exc:
        return _format_result("", f"Error running go test: {exc}")
    return _format_result(result.stdout, result.stderr)


@tool
def go_vet(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Run Go vet for code pattern analysis using go vet ./..."""
    repo = _repo_path(repo_path, runtime)
    try:
        result = subprocess.run(
            ["go", "vet", "./..."],
            text=True,
            capture_output=True,
            timeout=GO_ANALYSIS_TIMEOUT,
            cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            "", f"go vet timed out after {GO_ANALYSIS_TIMEOUT}s."
        )
    except Exception as exc:
        return _format_result("", f"Error running go vet: {exc}")
    return _format_result(result.stdout, result.stderr)


@tool
def go_mod_tidy(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Clean up and validate Go module dependencies using go mod tidy."""
    repo = _repo_path(repo_path, runtime)
    try:
        result = subprocess.run(
            ["go", "mod", "tidy"],
            text=True,
            capture_output=True,
            timeout=GO_ANALYSIS_TIMEOUT,
            cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            "", f"go mod tidy timed out after {GO_ANALYSIS_TIMEOUT}s."
        )
    except Exception as exc:
        return _format_result("", f"Error running go mod tidy: {exc}")
    return _format_result(result.stdout, result.stderr)


@tool
def golangci_lint(
    runtime: ToolRuntime[AgentContext],
    repo_path: AsciiStr | None = None,
) -> str:
    """Run golangci-lint on the repository.
    
    Automatically detects and uses .golangci.yml if present,
    otherwise uses sensible defaults.
    """
    repo = _repo_path(repo_path, runtime)
    
    # Check if golangci-lint is installed
    try:
        subprocess.run(
            ["golangci-lint", "--version"],
            text=True,
            capture_output=True,
            timeout=GIT_TIMEOUT,
        )
    except FileNotFoundError:
        return _format_result(
            "", "Error: golangci-lint is not installed. "
            "Install it with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"
        )
    except Exception as exc:
        return _format_result("", f"Error checking golangci-lint: {exc}")
    
    # Run golangci-lint
    config_file = repo / ".golangci.yml"
    args = ["golangci-lint", "run"]
    if config_file.exists():
        args.extend(["--config", str(config_file)])
    
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=LINT_TIMEOUT,
            cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return _format_result(
            "", f"golangci-lint timed out after {LINT_TIMEOUT}s (3 minutes)."
        )
    except Exception as exc:
        return _format_result("", f"Error running golangci-lint: {exc}")
    
    return _format_result(result.stdout, result.stderr)
