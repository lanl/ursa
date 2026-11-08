"""Tests for the ``python -m ursa.integration.harbor`` CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

try:
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.task import Task

    from ursa.integration import harbor as harbor_integration
    from ursa.integration.harbor.singularity import (
        DockerfileSingularityEnvironment,
    )
    from ursa.integration.harbor.validation import (
        _task_environments,
        prebuild_harbor_task,
    )
except ImportError:
    pytest.skip("Harbor extra is not installed", allow_module_level=True)

runner = CliRunner()

_TASK_TOML = """schema_version = "1.4"

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


def _task(tmp_path: Path, task_toml: str = _TASK_TOML) -> Path:
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Complete the task.\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "Dockerfile").write_text("FROM scratch\n")
    (task / "task.toml").write_text(task_toml)
    return task


@pytest.mark.parametrize(
    ("remove_dockerfile", "exit_code", "output"),
    [
        (False, 0, "OK   {task}\nValidated 1 Harbor task(s)."),
        (True, 1, "FAIL {task}"),
    ],
)
def test_validate_harbor_paths(tmp_path, remove_dockerfile, exit_code, output):
    task = _task(tmp_path)
    if remove_dockerfile:
        (task / "environment" / "Dockerfile").unlink()

    result = runner.invoke(harbor_integration.app, ["validate", str(tmp_path)])

    assert result.exit_code == exit_code
    assert output.format(task=task) in (result.stdout + result.stderr)


def test_harbor_network_phases_keep_environment_as_setup_baseline(tmp_path):
    task_dir = _task(
        tmp_path,
        _TASK_TOML.replace(
            '[verifier]\nenvironment_mode = "separate"\n\n'
            '[verifier.environment]\nnetwork_mode = "no-network"',
            '[agent]\nnetwork_mode = "no-network"\n\n'
            '[verifier]\nenvironment_mode = "shared"\n'
            'network_mode = "no-network"',
        ),
    )
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")

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
        "ursa.integration.harbor.validation.prebuild_harbor_task", build
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
