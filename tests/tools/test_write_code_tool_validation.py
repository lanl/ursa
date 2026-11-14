"""Tests for write_code_tool path validation and file operations."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.event_fixtures import (
    stub_event_dispatch as stub_event_dispatch,
)
from ursa.agents.base import AgentContext
from ursa.tools.write_code_tool import (
    _allow_unsafe_writes_enabled,
    _validate_file_path,
    edit_code,
    write_code,
    write_code_with_repo,
)


@pytest.fixture
def mock_runtime(tmp_path):
    """Create a mock ToolRuntime with AgentContext."""
    runtime = MagicMock()
    context = MagicMock(spec=AgentContext)
    context.workspace = tmp_path
    runtime.context = context
    runtime.store = None
    runtime.tool_call_id = "test_tool_call"
    runtime.config = {"metadata": {"thread_id": "test_thread"}}
    return runtime


class TestPathValidation:
    """Validate workspace and repository boundaries before touching a file."""

    @pytest.mark.parametrize("filename", ["test.py", "src/main/test.py"])
    def test_file_within_workspace(self, tmp_path, filename):
        result_path, error = _validate_file_path(filename, tmp_path)

        assert error is None
        assert result_path is not None
        assert tmp_path in result_path.parents or result_path.parent == tmp_path

    @pytest.mark.parametrize("filename", ["../../../etc/passwd", "/etc/passwd"])
    def test_path_outside_workspace_rejected(self, tmp_path, filename):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)

        result_path, error = _validate_file_path(filename, workspace)

        assert error is not None
        assert "outside workspace" in error.lower()
        assert result_path is None

    @pytest.mark.parametrize("relative", [False, True])
    def test_file_within_repo_path(self, tmp_path, relative):
        repo = tmp_path / "myrepo"
        repo.mkdir(parents=True)
        repo_path = Path("myrepo") if relative else repo

        result_path, error = _validate_file_path(
            "myrepo/test.py", tmp_path, repo_path
        )

        assert error is None
        assert result_path is not None

    def test_file_outside_repo_path_rejected(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir(parents=True)
        (tmp_path / "other").mkdir(parents=True)

        _result_path, error = _validate_file_path(
            "other/test.py", tmp_path, repo
        )

        assert error is not None
        assert "outside repository" in error.lower()

    def test_allow_unsafe_writes_allows_outside_workspace(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)

        result_path, error = _validate_file_path(
            "../../../etc/passwd", workspace, allow_unsafe_writes=True
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


def apply_code(tool, runtime, filename, *, repo_path=None):
    """Exercise the public write/edit tool functions with their native inputs."""
    options = {} if repo_path is None else {"repo_path": str(repo_path)}
    if tool is write_code:
        writer = write_code if repo_path is None else write_code_with_repo
        return writer.func("print('hello')", filename, runtime, **options)
    return tool.func("x = 1", "x = 2", filename, runtime, **options)


@pytest.mark.parametrize("tool", [write_code, edit_code], ids=["write", "edit"])
class TestCodePathValidation:
    def test_file_within_workspace(self, mock_runtime, tmp_path, tool):
        target = tmp_path / "test.py"
        if tool is edit_code:
            target.write_text("x = 1\n")

        result = apply_code(tool, mock_runtime, "test.py")

        assert "successfully" in result.lower()
        assert target.exists()
        if tool is edit_code:
            assert target.read_text() == "x = 2\n"

    def test_path_traversal_rejected(self, mock_runtime, tmp_path, tool):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace

        result = apply_code(tool, mock_runtime, "../../../etc/passwd")

        assert "failed" in result.lower()
        assert "outside workspace" in result.lower()

    def test_file_within_repo_boundary(self, mock_runtime, tmp_path, tool):
        repo = tmp_path / "myrepo"
        repo.mkdir(parents=True)
        if tool is edit_code:
            (repo / "test.py").write_text("x = 1\n")

        result = apply_code(
            tool, mock_runtime, "myrepo/test.py", repo_path=repo
        )

        assert "successfully" in result.lower()

    def test_file_outside_repo_boundary_rejected(
        self, mock_runtime, tmp_path, tool
    ):
        repo = tmp_path / "myrepo"
        repo.mkdir(parents=True)
        (tmp_path / "other").mkdir(parents=True)

        result = apply_code(tool, mock_runtime, "other/test.py", repo_path=repo)

        assert "failed" in result.lower()
        assert "outside repository" in result.lower()

    def test_file_outside_workspace_allowed_when_env_enabled(
        self, mock_runtime, tmp_path, monkeypatch, tool
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        mock_runtime.context.workspace = workspace
        monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", "1")
        target = tmp_path / "outside.py"
        if tool is edit_code:
            target.write_text("x = 1\n")

        result = apply_code(tool, mock_runtime, str(target))

        assert "successfully" in result.lower()
        assert target.exists()
        if tool is edit_code:
            assert target.read_text() == "x = 2\n"

    def test_repo_path_not_found_rejected(self, mock_runtime, tmp_path, tool):
        if tool is edit_code:
            (tmp_path / "test.py").write_text("x = 1\n")
        missing_repo = tmp_path / "does-not-exist"

        result = apply_code(
            tool, mock_runtime, "test.py", repo_path=missing_repo
        )

        assert "failed" in result.lower()
        assert "repository path not found" in result.lower()


def test_write_code_creates_nested_directories(mock_runtime, tmp_path):
    filename = "src/main/test.py"
    result = write_code.func("print('hello')", filename, mock_runtime)

    assert "successfully" in result.lower()
    assert (tmp_path / filename).exists()
    assert (tmp_path / "src" / "main").is_dir()


def test_edit_code_binary_file_returns_failure(mock_runtime, tmp_path):
    filename = "binary.bin"
    (tmp_path / filename).write_bytes(b"\xff\xfe\x00\x01")

    result = edit_code.func("x", "y", filename, mock_runtime)

    assert "failed" in result.lower()
    assert "binary" in result.lower()
