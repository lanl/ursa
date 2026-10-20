"""Tests for the ``python -m ursa.integrations.harbor`` CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

harbor = pytest.importorskip("harbor")

from harbor.models.task.config import NetworkMode  # noqa: E402
from harbor.models.task.task import Task  # noqa: E402

from ursa.integrations import harbor as harbor_integration  # noqa: E402
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
)
from ursa.integrations.harbor_validation import (  # noqa: E402
    _task_environments,
    discover_harbor_tasks,
    prebuild_harbor_task,
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


def test_harbor_network_phases_keep_environment_as_setup_baseline(tmp_path):
    task_dir = _task(tmp_path)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task_dir / "task.toml").write_text(
        """schema_version = "1.4"

[task]
name = "example/task"
version = "1.0.0"
description = "Example task"

[environment]
network_mode = "public"

[agent]
network_mode = "no-network"

[verifier]
environment_mode = "shared"
network_mode = "no-network"
"""
    )

    environments = _task_environments(Task(task_dir), purpose="validate")

    assert len(environments) == 1
    environment = environments[0]
    assert environment.network_policy.network_mode == NetworkMode.PUBLIC
    assert [
        policy.network_mode for policy in environment._phase_network_policies
    ] == [NetworkMode.NO_NETWORK, NetworkMode.NO_NETWORK]
    assert environment.capabilities.dynamic_network_policy


async def test_prebuild_harbor_task_builds_agent_and_verifier_sifs(
    tmp_path, monkeypatch
):
    task = _task(tmp_path)
    cache = tmp_path / "sif-cache"
    observed: list[tuple[Path, Path, bool]] = []

    async def build(
        self: DockerfileSingularityEnvironment,
        force_build: bool = False,
    ) -> list[Path]:
        observed.append((
            self.environment_dir,
            self._image_cache_dir,
            force_build,
        ))
        return [cache / f"{self.environment_dir.name}.sif"]

    monkeypatch.setattr(DockerfileSingularityEnvironment, "build_sifs", build)

    paths = await prebuild_harbor_task(
        task, force_build=True, image_cache_dir=cache
    )

    assert observed == [
        (task / "environment", cache, True),
        (task / "tests", cache, True),
    ]
    assert paths == [cache / "environment.sif", cache / "tests.sif"]


def test_harbor_prebuild_command_reports_built_sifs(tmp_path, monkeypatch):
    task = _task(tmp_path)
    cache = tmp_path / "sif-cache"
    observed = []

    async def build(
        task_dir: Path,
        *,
        force_build: bool,
        image_cache_dir: Path | None,
    ) -> list[Path]:
        observed.append((task_dir, force_build, image_cache_dir))
        return [cache / "agent.sif", cache / "verifier.sif"]

    monkeypatch.setattr(
        "ursa.integrations.harbor_validation.prebuild_harbor_task", build
    )

    result = runner.invoke(
        harbor_integration.app,
        [
            "prebuild",
            "--force",
            "--image-cache-dir",
            str(cache),
            str(task),
        ],
    )

    assert result.exit_code == 0
    assert observed == [(task, True, cache)]
    assert f"OK   {task}: 2 SIF file(s)" in result.stdout
    assert "Pre-built 2 SIF file(s) for 1 Harbor task(s)." in result.stdout


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
