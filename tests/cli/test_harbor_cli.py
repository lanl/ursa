"""Tests for the ``ursa harbor`` CLI."""

from pathlib import Path

import pytest

from ursa.cli import build_parser, main
from ursa.cli.harbor import validate_harbor_paths
from ursa.integrations.harbor_validation import discover_harbor_tasks


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


def test_harbor_validate_parser(tmp_path):
    parsed = build_parser().parse_args(["harbor", "validate", str(tmp_path)])

    assert parsed.harbor.validate.paths == [tmp_path]


def test_validate_harbor_paths_reports_success(tmp_path, capsys):
    task = _task(tmp_path)

    validate_harbor_paths([tmp_path])

    output = capsys.readouterr().out
    assert f"OK   {task}" in output
    assert "Validated 1 Harbor task(s)." in output


def test_validate_harbor_paths_reports_failures(tmp_path, capsys):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").unlink()

    with pytest.raises(SystemExit, match="1"):
        validate_harbor_paths([tmp_path])

    assert f"FAIL {task}" in capsys.readouterr().err


def test_harbor_validate_main_dispatches(tmp_path, capsys):
    _task(tmp_path)

    main(["harbor", "validate", str(tmp_path)])

    assert "Validated 1 Harbor task(s)." in capsys.readouterr().out


def test_discover_harbor_tasks_ignores_hidden_directories(tmp_path):
    task = _task(tmp_path)
    hidden_task = _task(tmp_path / ".venv")

    assert hidden_task.exists()
    assert discover_harbor_tasks([tmp_path]) == [task]
