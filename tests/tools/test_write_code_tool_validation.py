"""Tests for write_code_tool path validation and file operations."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ursa.agents.base import AgentContext
from ursa.tools.write_code_tool import (
    _allow_unsafe_writes_enabled,
    _validate_file_path,
    edit_code,
    write_code,
    write_code_with_repo,
)


@pytest.fixture
def mock_runtime():
    """Create a mock ToolRuntime with AgentContext."""
    runtime = MagicMock()
    context = MagicMock(spec=AgentContext)
    runtime.context = context
    runtime.store = None
    runtime.tool_call_id = "test_tool_call"
    runtime.config = {"metadata": {"thread_id": "test_thread"}}
    return runtime


class TestPathValidation:
    """Test path validation in write_code and edit_code."""

    def test_valid_file_within_workspace(self, tmpdir):
        """Test that a file within workspace is accepted."""
        workspace = Path(tmpdir)
        filename = "test.py"

        result_path, error = _validate_file_path(filename, workspace)

        assert error is None
        assert result_path is not None
        assert (
            workspace in result_path.parents or result_path.parent == workspace
        )

    def test_valid_nested_file_within_workspace(self, tmpdir):
        """Test that nested files within workspace are accepted."""
        workspace = Path(tmpdir)
        filename = "src/main/test.py"

        result_path, error = _validate_file_path(filename, workspace)

        assert error is None
        assert result_path is not None

    def test_path_traversal_attempt_rejected(self, tmpdir):
        """Test that path traversal attempts are rejected."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        filename = "../../../etc/passwd"

        result_path, error = _validate_file_path(filename, workspace)

        assert error is not None
        assert "outside workspace" in error.lower()
        assert result_path is None

    def test_absolute_path_outside_workspace_rejected(self, tmpdir):
        """Test that absolute paths outside workspace are rejected."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        filename = "/etc/passwd"

        _result_path, error = _validate_file_path(filename, workspace)

        assert error is not None
        assert "outside workspace" in error.lower()

    def test_file_within_repo_path(self, tmpdir):
        """Test that files within specified repo are accepted."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        filename = "myrepo/test.py"

        result_path, error = _validate_file_path(filename, workspace, repo)

        assert error is None
        assert result_path is not None

    def test_file_outside_repo_path_rejected(self, tmpdir):
        """Test that files outside specified repo are rejected."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        other = workspace / "other"
        other.mkdir(parents=True)
        filename = "other/test.py"

        _result_path, error = _validate_file_path(filename, workspace, repo)

        assert error is not None
        assert "outside repository" in error.lower()

    def test_relative_repo_path(self, tmpdir):
        """Test that relative repo paths are resolved correctly."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        filename = "myrepo/test.py"

        # Test with relative repo path
        result_path, error = _validate_file_path(
            filename, workspace, Path("myrepo")
        )

        assert error is None
        assert result_path is not None

    def test_allow_unsafe_writes_allows_outside_workspace(self, tmpdir):
        """Test unsafe writes can bypass workspace validation when enabled."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        filename = "../../../etc/passwd"

        result_path, error = _validate_file_path(
            filename,
            workspace,
            allow_unsafe_writes=True,
        )

        assert error is None
        assert result_path is not None


class TestUnsafeWriteEnvToggle:
    """Test env var parsing for unsafe write mode."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_unsafe_writes(self, monkeypatch, value):
        monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", value)
        assert _allow_unsafe_writes_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsey_values_disable_unsafe_writes(self, monkeypatch, value):
        monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", value)
        assert _allow_unsafe_writes_enabled() is False

    def test_missing_env_defaults_to_safe_mode(self, monkeypatch):
        monkeypatch.delenv("URSA_ALLOW_UNSAFE_WRITES", raising=False)
        assert _allow_unsafe_writes_enabled() is False


class TestWriteCodePathValidation:
    """Test write_code function with path validation."""

    def test_write_code_within_workspace(self, mock_runtime, tmpdir):
        """Test write_code works for files within workspace."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        code = "print('hello')"
        filename = "test.py"

        result = write_code.func(code, filename, mock_runtime)

        assert "successfully" in result.lower()
        assert (workspace / filename).exists()

    def test_write_code_path_traversal_rejected(self, mock_runtime, tmpdir):
        """Test write_code rejects path traversal."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        code = "print('hello')"
        filename = "../../../etc/passwd"

        result = write_code.func(code, filename, mock_runtime)

        assert "failed" in result.lower()
        assert "outside workspace" in result.lower()

    def test_write_code_nested_directory_creation(self, mock_runtime, tmpdir):
        """Test write_code creates nested directories."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        code = "print('hello')"
        filename = "src/main/test.py"

        result = write_code.func(code, filename, mock_runtime)

        assert "successfully" in result.lower()
        assert (workspace / filename).exists()
        assert (workspace / "src" / "main").is_dir()

    def test_write_code_with_repo_boundary(self, mock_runtime, tmpdir):
        """Test write_code_with_repo respects repo boundary when specified."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        code = "print('hello')"
        filename = "myrepo/test.py"

        result = write_code_with_repo.func(
            code, filename, mock_runtime, repo_path=str(repo)
        )

        assert "successfully" in result.lower()

    def test_write_code_outside_repo_boundary_rejected(
        self, mock_runtime, tmpdir
    ):
        """Test write_code_with_repo rejects files outside repo boundary."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        other = workspace / "other"
        other.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        code = "print('hello')"
        filename = "other/test.py"

        result = write_code_with_repo.func(
            code, filename, mock_runtime, repo_path=str(repo)
        )

        assert "failed" in result.lower()
        assert "outside repository" in result.lower()

    def test_write_code_outside_workspace_allowed_when_env_enabled(
        self, mock_runtime, tmpdir, monkeypatch
    ):
        """Test write_code allows unsafe writes when env toggle is enabled."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", "1")
        target = Path(tmpdir) / "outside.py"

        result = write_code.func("print('hello')", str(target), mock_runtime)

        assert "successfully" in result.lower()
        assert target.exists()

    def test_write_code_repo_path_not_found_rejected(
        self, mock_runtime, tmpdir
    ):
        """Test write_code_with_repo fails clearly when repo path does not exist."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        missing_repo = workspace / "does-not-exist"
        result = write_code_with_repo.func(
            "print('hello')",
            "test.py",
            mock_runtime,
            repo_path=str(missing_repo),
        )

        assert "failed" in result.lower()
        assert "repository path not found" in result.lower()


class TestEditCodePathValidation:
    """Test edit_code function with path validation."""

    def test_edit_code_within_workspace(self, mock_runtime, tmpdir):
        """Test edit_code works for files within workspace."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        # Create initial file
        filename = "test.py"
        test_file = workspace / filename
        test_file.write_text("x = 1\n")

        old_code = "x = 1"
        new_code = "x = 2"

        result = edit_code.func(old_code, new_code, filename, mock_runtime)

        assert "successfully" in result.lower()
        assert test_file.read_text() == "x = 2\n"

    def test_edit_code_path_traversal_rejected(self, mock_runtime, tmpdir):
        """Test edit_code rejects path traversal."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        old_code = "x = 1"
        new_code = "x = 2"
        filename = "../../../etc/passwd"

        result = edit_code.func(old_code, new_code, filename, mock_runtime)

        assert "failed" in result.lower()
        assert "outside workspace" in result.lower()

    def test_edit_code_with_repo_boundary(self, mock_runtime, tmpdir):
        """Test edit_code respects repo boundary when specified."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        # Create initial file
        filename = "myrepo/test.py"
        test_file = workspace / filename
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("x = 1\n")

        old_code = "x = 1"
        new_code = "x = 2"

        result = edit_code.func(
            old_code, new_code, filename, mock_runtime, repo_path=str(repo)
        )

        assert "successfully" in result.lower()

    def test_edit_code_outside_repo_boundary_rejected(
        self, mock_runtime, tmpdir
    ):
        """Test edit_code rejects files outside repo boundary."""
        workspace = Path(tmpdir)
        repo = workspace / "myrepo"
        repo.mkdir(parents=True)
        other = workspace / "other"
        other.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        old_code = "x = 1"
        new_code = "x = 2"
        filename = "other/test.py"

        result = edit_code.func(
            old_code, new_code, filename, mock_runtime, repo_path=str(repo)
        )

        assert "failed" in result.lower()
        assert "outside repository" in result.lower()

    def test_edit_code_outside_workspace_allowed_when_env_enabled(
        self, mock_runtime, tmpdir, monkeypatch
    ):
        """Test edit_code allows unsafe edits when env toggle is enabled."""
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", "1")
        target = Path(tmpdir) / "outside.py"
        target.write_text("x = 1\n")

        result = edit_code.func("x = 1", "x = 2", str(target), mock_runtime)

        assert "successfully" in result.lower()
        assert target.read_text() == "x = 2\n"

    def test_edit_code_binary_file_returns_failure(self, mock_runtime, tmpdir):
        """Test edit_code handles binary files without raising."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        filename = "binary.bin"
        test_file = workspace / filename
        test_file.write_bytes(b"\xff\xfe\x00\x01")

        result = edit_code.func("x", "y", filename, mock_runtime)

        assert "failed" in result.lower()
        assert "binary" in result.lower()

    def test_edit_code_repo_path_not_found_rejected(self, mock_runtime, tmpdir):
        """Test edit_code fails clearly when repo path does not exist."""
        workspace = Path(tmpdir)
        mock_runtime.context.workspace = workspace

        filename = "test.py"
        (workspace / filename).write_text("x = 1\n")
        missing_repo = workspace / "does-not-exist"

        result = edit_code.func(
            "x = 1",
            "x = 2",
            filename,
            mock_runtime,
            repo_path=str(missing_repo),
        )

        assert "failed" in result.lower()
        assert "repository path not found" in result.lower()
