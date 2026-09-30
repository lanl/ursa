"""Tests for the ``python -m ursa.integrations.harbor`` CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

harbor = pytest.importorskip("harbor")

from ursa.integrations import harbor as harbor_integration  # noqa: E402
from ursa.integrations.harbor_validation import (  # noqa: E402
    discover_harbor_tasks,
)

runner = CliRunner()


def _task(tmp_path: Path) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Complete the task.\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "Dockerfile").write_text("FROM scratch\n")
    (task / "task.toml").write_text(
        """schema_version = "1.4"

[task]
name = "example/task"
version = "1.0.0"
description = "Example task"

[verifier]
environment_mode = "separate"

[verifier.environment]
network_mode = "no-network"

[environment]
network_mode = "public"
"""
    )
    return task


def test_harbor_validate_help_exposes_paths():
    result = runner.invoke(harbor_integration.app, ["validate", "--help"])

    assert result.exit_code == 0
    assert "PATHS" in result.stdout


def test_validate_harbor_paths_reports_success(tmp_path):
    task = _task(tmp_path)

    result = runner.invoke(harbor_integration.app, ["validate", str(tmp_path)])

    assert result.exit_code == 0
    assert f"OK   {task}" in result.stdout
    assert "Validated 1 Harbor task(s)." in result.stdout


def test_validate_harbor_paths_reports_failures(tmp_path):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").unlink()

    result = runner.invoke(harbor_integration.app, ["validate", str(tmp_path)])

    assert result.exit_code == 1
    assert f"FAIL {task}" in result.stderr


def test_harbor_runner_dispatches(monkeypatch):
    received = []
    monkeypatch.setattr("ursa.integrations.harbor_runner.main", received.append)

    result = runner.invoke(harbor_integration.app, ["runner", "payload"])

    assert result.exit_code == 0
    assert received == ["payload"]


def test_discover_harbor_tasks_ignores_hidden_directories(tmp_path):
    task = _task(tmp_path)
    hidden_task = _task(tmp_path / ".venv")

    assert hidden_task.exists()
    assert discover_harbor_tasks([tmp_path]) == [task]
