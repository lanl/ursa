"""Tests for Go tooling functions in git_tools module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ursa.agents.base import AgentContext
from ursa.tools.git_tools import (
    go_build,
    go_test,
    go_vet,
    go_mod_tidy,
    golangci_lint,
)


@pytest.fixture
def mock_runtime():
    """Create a mock ToolRuntime with AgentContext."""
    runtime = MagicMock()
    context = MagicMock(spec=AgentContext)
    context.workspace = Path("/tmp/workspace")
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


class TestGoTools:
    """Test suite for Go tooling functions."""
    
    def test_go_build_success(self, mock_runtime, go_repo):
        """Test successful go build execution."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = go_build.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "STDOUT:" in result
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "go" in args
            assert "build" in args
            assert "./..." in args
    
    def test_go_test_with_verbose(self, mock_runtime, go_repo):
        """Test go test with verbose flag."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="ok\n",
                stderr=""
            )
            
            result = go_test.func(mock_runtime, repo_path=str(go_repo.name), verbose=True)
            
            assert "STDOUT:" in result
            assert "ok\n" in result
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "-v" in args
    
    def test_go_test_without_verbose(self, mock_runtime, go_repo):
        """Test go test without verbose flag."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = go_test.func(mock_runtime, repo_path=str(go_repo.name), verbose=False)
            
            assert "STDOUT:" in result
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "-v" not in args
    
    def test_go_vet_success(self, mock_runtime, go_repo):
        """Test successful go vet execution."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = go_vet.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "STDOUT:" in result
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "go" in args
            assert "vet" in args
    
    def test_go_mod_tidy_success(self, mock_runtime, go_repo):
        """Test successful go mod tidy execution."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )
            
            result = go_mod_tidy.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "STDOUT:" in result
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "go" in args
            assert "mod" in args
            assert "tidy" in args
    
    def test_go_build_timeout(self, mock_runtime, go_repo):
        """Test go build timeout handling."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                ["go", "build", "./..."], 300
            )
            
            result = go_build.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "timed out" in result.lower()
            assert "300" in result
            assert "5 minutes" in result
    
    def test_go_test_timeout(self, mock_runtime, go_repo):
        """Test go test timeout handling."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                ["go", "test", "./..."], 600
            )
            
            result = go_test.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "timed out" in result.lower()
            assert "600" in result
    
    def test_golangci_lint_not_installed(self, mock_runtime, go_repo):
        """Test golangci_lint when linter is not installed."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "not installed" in result.lower()
            assert "go install" in result.lower()
    
    def test_golangci_lint_with_config(self, mock_runtime, go_repo):
        """Test golangci_lint detects and uses .golangci.yml."""
        # Create .golangci.yml in the repo
        config_file = go_repo / ".golangci.yml"
        config_file.write_text("linters:\n  enable:\n    - gofmt\n")
        
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            # First call for version check, second for actual linting
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="golangci-lint version 1.0.0\n", stderr=""),
                MagicMock(returncode=0, stdout="", stderr="")
            ]
            
            result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "STDOUT:" in result
            # Verify the second call used the config file
            second_call_args = mock_run.call_args_list[1][0][0]
            assert "--config" in second_call_args
            assert str(config_file) in second_call_args or ".golangci.yml" in str(second_call_args)
    
    def test_golangci_lint_without_config(self, mock_runtime, go_repo):
        """Test golangci_lint runs without config file."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            # First call for version check, second for actual linting
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="golangci-lint version 1.0.0\n", stderr=""),
                MagicMock(returncode=0, stdout="", stderr="")
            ]
            
            result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "STDOUT:" in result
            # Verify the second call did not use a config file
            second_call_args = mock_run.call_args_list[1][0][0]
            assert "--config" not in second_call_args


class TestGoToolsErrorHandling:
    """Test error handling in Go tools."""
    
    def test_go_build_general_error(self, mock_runtime, go_repo):
        """Test go build general exception handling."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Some error")
            
            result = go_build.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "Error running go build" in result
    
    def test_go_test_general_error(self, mock_runtime, go_repo):
        """Test go test general exception handling."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Some error")
            
            result = go_test.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "Error running go test" in result
    
    def test_golangci_lint_version_check_error(self, mock_runtime, go_repo):
        """Test golangci_lint version check failure."""
        mock_runtime.context.workspace = go_repo.parent
        
        with patch("ursa.tools.git_tools.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Version check failed")
            
            result = golangci_lint.func(mock_runtime, repo_path=str(go_repo.name))
            
            assert "Error checking golangci-lint" in result
