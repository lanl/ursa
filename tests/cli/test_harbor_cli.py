"""Tests for the ``python -m ursa.integrations.harbor`` CLI."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from typer.testing import CliRunner

try:
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.task import Task

    from ursa.integrations import harbor as harbor_integration
    from ursa.integrations.harbor.singularity import (
        DockerfileSingularityEnvironment,
    )
    from ursa.integrations.harbor.validation import (
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


@pytest.mark.parametrize("force", [False, True])
async def test_prebuild_includes_image_only_agent_and_verifier_services(
    tmp_path, monkeypatch, force
):
    task = _task(
        tmp_path,
        _TASK_TOML.replace(
            "[verifier.environment]", '[verifier.environment]\nworkdir = "/"'
        )
        .replace("[environment]", '[environment]\nworkdir = "/"')
        .replace('network_mode = "no-network"', 'network_mode = "public"'),
    )
    cache = tmp_path / "cache"
    for name, services in (
        (
            "environment",
            {
                "main": {"build": ".", "image": "local-agent:latest"},
                "helper": {"image": "redis:7"},
                "replica": {"image": "docker://redis:7"},
            },
        ),
        (
            "tests",
            {
                "main": {"image": "alpine:3"},
                "replica": {"image": "docker://alpine:3"},
                "builder": {"build": "."},
            },
        ),
    ):
        (task / name / "docker-compose.yaml").write_text(
            yaml.safe_dump({"services": services})
        )
    pulls, builds = [], []

    async def pull(self, image, force_build):
        image = self._image_uri(image)
        pulls.append((image, force_build))
        return cache / f"{image.rsplit('/', 1)[-1].replace(':', '-')}.sif"

    async def build(self, force_build, **kwargs):
        builds.append((kwargs["context_dir"], force_build))
        return cache / f"{self.environment_dir.name}.sif"

    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_pull_image_sif", pull
    )
    monkeypatch.setattr(
        DockerfileSingularityEnvironment, "_build_dockerfile_sif", build
    )
    paths = await prebuild_harbor_task(
        task, force_build=force, image_cache_dir=cache
    )

    assert set(pulls) == {
        ("docker://redis:7", force),
        ("docker://alpine:3", force),
    }
    assert len(pulls) == 2
    assert set(builds) == {
        (task / "environment", force),
        (task / "tests", force),
    }
    assert set(paths) == {
        cache / "environment.sif",
        cache / "tests.sif",
        cache / "redis-7.sif",
        cache / "alpine-3.sif",
    }


@pytest.mark.parametrize("agents", [[], ["goose@v1.2.3", "codex"]])
def test_harbor_prebuild_command_reports_built_sifs(
    tmp_path, monkeypatch, agents
):
    task = _task(tmp_path)
    cache = tmp_path / "sif-cache"
    observed = []

    async def build(
        task_dir: Path,
        *,
        force_build: bool,
        image_cache_dir: Path | None,
        **kwargs,
    ) -> list[Path]:
        observed.append((task_dir, force_build, image_cache_dir, kwargs))
        return [cache / "agent.sif", cache / "verifier.sif"]

    monkeypatch.setattr(
        "ursa.integrations.harbor.validation.prebuild_harbor_task", build
    )

    result = runner.invoke(
        harbor_integration.app,
        [
            "prebuild",
            "--force",
            "--image-cache-dir",
            str(cache),
            "--agent-cache-dir",
            str(tmp_path / "agents"),
            "--agent-install-timeout-sec",
            "900",
            "--no-fakeroot",
            *(argument for agent in agents for argument in ("--agent", agent)),
            str(task),
        ],
    )

    assert result.exit_code == 0
    assert observed == [
        (
            task,
            True,
            cache,
            {
                "agents": agents or (),
                "agent_kwargs": {},
                "agent_cache_dir": tmp_path / "agents",
                "agent_install_timeout_sec": 900,
                "fakeroot": False,
            },
        )
    ]
    assert f"OK   {task}: 2 SIF file(s)" in result.stdout
    assert "Pre-built 2 SIF file(s) for 1 Harbor task(s)." in result.stdout


async def test_prebuild_builds_agent_layers_after_all_task_images(
    tmp_path, monkeypatch
):
    task = _task(tmp_path)
    events = []

    async def build(self, force_build=False):
        events.append(("image", self.environment_dir.name, force_build))
        return [tmp_path / f"{self.environment_dir.name}.sif"]

    async def layer(template, spec, **kwargs):
        assert template.environment_dir == task / "environment"
        assert template.network_policy.network_mode == NetworkMode.PUBLIC
        assert template._agent_cache_dir == tmp_path / "agents"
        assert template._fakeroot is False
        events.append(("layer", spec, kwargs))
        return tmp_path / "layer.img"

    monkeypatch.setattr(DockerfileSingularityEnvironment, "build_sifs", build)
    monkeypatch.setattr(
        "ursa.integrations.harbor.validation.build_agent_layer", layer
    )
    await prebuild_harbor_task(
        task,
        force_build=True,
        agents=["goose@v1.2.3", "codex", "goose@v1.2.3"],
        agent_cache_dir=tmp_path / "agents",
        fakeroot=False,
        agent_install_timeout_sec=600,
    )
    assert events == [
        ("image", "environment", True),
        ("image", "tests", True),
        ("layer", "goose@v1.2.3", {"force": True, "timeout_sec": 600}),
        ("layer", "codex", {"force": True, "timeout_sec": 600}),
    ]


@pytest.mark.parametrize(
    "agent", ["nop", "missing-agent", "goose@", "codex@1;bad"]
)
async def test_prebuild_rejects_invalid_installer_before_building(
    tmp_path, monkeypatch, agent
):
    build = AsyncMock()
    monkeypatch.setattr(DockerfileSingularityEnvironment, "build_sifs", build)
    with pytest.raises(ValueError):
        await prebuild_harbor_task(_task(tmp_path), agents=[agent])
    build.assert_not_awaited()


def test_prebuild_cli_forwards_ursa_install_options(tmp_path, monkeypatch):
    build = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "ursa.integrations.harbor.validation.prebuild_harbor_task", build
    )
    result = runner.invoke(
        harbor_integration.app,
        [
            "prebuild",
            str(_task(tmp_path)),
            "--agent",
            "ursa",
            "--agent-kwarg",
            "ursa_install_spec=ursa-ai==0.17.3",
            "--agent-kwarg",
            'extra_packages=["numpy==2.3.0"]',
        ],
    )
    assert result.exit_code == 0, result.output
    assert build.call_args.kwargs["agent_kwargs"] == {
        "ursa_install_spec": "ursa-ai==0.17.3",
        "extra_packages": ["numpy==2.3.0"],
    }


async def test_prebuild_rejects_runtime_options_before_building(
    tmp_path, monkeypatch
):
    build = AsyncMock()
    monkeypatch.setattr(DockerfileSingularityEnvironment, "build_sifs", build)
    with pytest.raises(ValueError, match="Unsupported cached install options"):
        await prebuild_harbor_task(
            _task(tmp_path), agents=["ursa"], agent_kwargs={"config": {}}
        )
    build.assert_not_awaited()
