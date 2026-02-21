import subprocess

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.tools.git_tools import _format_result, _repo_path
from ursa.util.types import AsciiStr

# Differentiated timeouts by operation type
GO_FORMAT_TIMEOUT = 30  # seconds - gofmt is usually fast
GO_ANALYSIS_TIMEOUT = 60  # seconds - go vet, go mod tidy
GO_BUILD_TIMEOUT = 300  # seconds (5 min) - builds can take time for large projects
GO_TEST_TIMEOUT = 600  # seconds (10 min) - test suites can be slow
LINT_TIMEOUT = 180  # seconds (3 min) - linting is moderate speed


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
            timeout=GO_FORMAT_TIMEOUT,
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
    from ursa.tools.git_tools import GIT_TIMEOUT

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


GO_TOOLS = [
    gofmt_files,
    go_build,
    go_test,
    go_vet,
    go_mod_tidy,
    golangci_lint,
]
