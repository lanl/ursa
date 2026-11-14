"""Tests for Go tooling functions in git_tools module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from ursa.tools.go_tools import (
        go_build,
        go_mod_tidy,
        go_test,
        go_vet,
        golangci_lint,
    )
except (ImportError, ModuleNotFoundError) as exc:
    pytest.skip(
        "Skipping legacy Go tooling tests: ursa.tools.go_tools is unavailable. "
        f"Import error: {exc}",
        allow_module_level=True,
    )

from ursa.agents.base import AgentContext


@pytest.fixture
def mock_runtime(go_repo):
    """Create a mock ToolRuntime with AgentContext."""
    runtime = MagicMock()
    context = MagicMock(spec=AgentContext)
    context.workspace = go_repo.parent
    runtime.context = context
    runtime.store = None
    return runtime


@pytest.fixture
def go_repo(tmpdir):
    """Create a minimal Go repo for testing."""
    repo_dir = Path(tmpdir) / "test_repo"
    repo_dir.mkdir(parents=True)

    # Create minimal go.mod
    mod_file = repo_dir / "go.mod"
    mod_file.write_text("module github.com/test/example\n\ngo 1.21\n")

    # Create a simple Go file
    main_file = repo_dir / "main.go"
    main_file.write_text(
        'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello")\n}\n'
    )

    # Create a test file
    test_file = repo_dir / "main_test.go"
    test_file.write_text(
        'package main\n\nimport "testing"\n\nfunc TestHello(t *testing.T) {\n    t.Log("Test")\n}\n'
    )

    return repo_dir


@pytest.fixture
def mock_run():
    with patch("ursa.tools.go_tools.subprocess.run") as run:
        yield run


@pytest.mark.parametrize(
    "tool,options,stdout,required,forbidden",
    [
        (go_build, {}, "", ["go", "build", "./..."], []),
        (go_test, {"verbose": True}, "ok\n", ["-v"], []),
        (go_test, {"verbose": False}, "", [], ["-v"]),
        (go_vet, {}, "", ["go", "vet"], []),
        (go_mod_tidy, {}, "", ["go", "mod", "tidy"], []),
    ],
    ids=["build", "test-verbose", "test-quiet", "vet", "mod-tidy"],
)
def test_go_tool_success(
    mock_runtime, go_repo, mock_run, tool, options, stdout, required, forbidden
):
    mock_run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")

    result = tool.func(mock_runtime, repo_path=str(go_repo.name), **options)

    assert "STDOUT:" in result
    if stdout:
        assert stdout in result
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert set(required) <= set(args)
    assert not set(forbidden) & set(args)


@pytest.mark.parametrize(
    "tool,command,seconds,extra_message",
    [(go_build, "build", 300, "5 minutes"), (go_test, "test", 600, None)],
    ids=["build", "test"],
)
def test_go_tool_timeout(
    mock_runtime, go_repo, mock_run, tool, command, seconds, extra_message
):
    mock_run.side_effect = subprocess.TimeoutExpired(
        ["go", command, "./..."], seconds
    )

    result = tool.func(mock_runtime, repo_path=str(go_repo.name))

    assert "timed out" in result.lower()
    assert str(seconds) in result
    if extra_message is not None:
        assert extra_message in result


def test_golangci_lint_not_installed(mock_runtime, go_repo, mock_run):
    mock_run.side_effect = FileNotFoundError()

    result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))

    assert "not installed" in result.lower()
    assert "go install" in result.lower()


@pytest.mark.parametrize("has_config", [True, False])
def test_golangci_lint_config(mock_runtime, go_repo, mock_run, has_config):
    config_file = go_repo / ".golangci.yml"
    if has_config:
        config_file.write_text("linters:\n  enable:\n    - gofmt\n")
    # First call checks the version; the second performs the linting.
    mock_run.side_effect = [
        MagicMock(
            returncode=0, stdout="golangci-lint version 1.0.0\n", stderr=""
        ),
        MagicMock(returncode=0, stdout="", stderr=""),
    ]

    result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))

    assert "STDOUT:" in result
    second_call_args = mock_run.call_args_list[1][0][0]
    if has_config:
        assert "--config" in second_call_args
        assert str(config_file) in second_call_args or ".golangci.yml" in str(
            second_call_args
        )
    else:
        assert "--config" not in second_call_args


@pytest.mark.parametrize(
    "tool,error,expected",
    [
        (go_build, "Some error", "Error running go build"),
        (go_test, "Some error", "Error running go test"),
        (golangci_lint, "Version check failed", "Error checking golangci-lint"),
    ],
    ids=["build", "test", "linter-version"],
)
def test_go_tool_error(mock_runtime, go_repo, mock_run, tool, error, expected):
    mock_run.side_effect = Exception(error)

    result = tool.func(mock_runtime, repo_path=str(go_repo.name))

    assert expected in result
