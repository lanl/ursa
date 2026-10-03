import asyncio
import gc
import weakref
from pathlib import Path

import pytest
import yaml

pytest.importorskip("harbor")

from harbor.models.task.config import (  # noqa: E402
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from ursa.integrations import harbor_singularity  # noqa: E402
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
    docker_compose_to_singularity_compose,
)


def _environment(
    tmp_path: Path, *, compose: bool = False
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        (environment_dir / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    image: ubuntu:24.04\n"
            "  helper:\n"
            "    image: ubuntu:24.04\n"
        )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="unit-test",
        session_id=f"{tmp_path.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(workdir="/workspace"),
        mounts=[
            {
                "type": "bind",
                "source": str(workspace),
                "target": "/workspace",
                "read_only": False,
            }
        ],
        network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
        persistent_env={},
        singularity_fakeroot=False,
        singularity_overlay=False,
    )
    environment._runtime_path = str(tmp_path / "fake-singularity")
    return environment


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


async def test_direct_runtime_commands_use_isolated_host_workdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    staging = tmp_path / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    environment._sif_path = tmp_path / "image.sif"

    start = environment._instance_start_command()
    observed: list[str] = []

    async def capture(*command: str, **_kwargs: object) -> None:
        observed.extend(command)

    monkeypatch.setattr(environment, "_run", capture)
    environment._prepare_harbor_bind_sources()
    await environment._seed_harbor_bind_sources()

    main_workdir = Path(_option_value(start, "--workdir"))
    seed_workdir = Path(_option_value(observed, "--workdir"))
    assert main_workdir == root / "main"
    assert seed_workdir == root / "seed"
    assert main_workdir.is_dir()
    assert seed_workdir.is_dir()
    assert main_workdir != seed_workdir

    environment._cleanup_staging()
    assert not root.exists()


async def test_compose_services_get_distinct_host_workdirs(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "  helper:\n"
        "    image: redis:7\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    output = tmp_path / "singularity-compose.yml"
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (tmp_path / "workdirs" / "helper",),
    }
    for service_workdirs in workdirs.values():
        for workdir in service_workdirs:
            workdir.mkdir(parents=True)

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    await docker_compose_to_singularity_compose(
        compose,
        output,
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        service_workdirs=workdirs,
    )

    instances = yaml.safe_load(output.read_text())["instances"]
    assert len(instances) == 2
    assert not any(key.endswith("-r1") for key in instances)
    options = {
        key: instance["start"]["options"] for key, instance in instances.items()
    }
    configured = {
        option.removeprefix("workdir=")
        for instance_options in options.values()
        for option in instance_options
        if option.startswith("workdir=")
    }
    assert configured == {
        str(path) for paths in workdirs.values() for path in paths
    }
    assert all("containall" in value for value in options.values())


async def test_compose_conversion_rejects_missing_service_workdir(
    tmp_path: Path,
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "  helper:\n"
        "    image: redis:7\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    with pytest.raises(ValueError, match="workdir.*helper"):
        await docker_compose_to_singularity_compose(
            compose,
            tmp_path / "singularity-compose.yml",
            identity="unit",
            image_resolver=resolve,
            staging_dir=staging,
            fakeroot=False,
            service_workdirs={"main": (tmp_path / "main",)},
        )


async def test_compose_replicas_get_distinct_host_workdirs(tmp_path: Path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "    depends_on: [helper]\n"
        "  helper:\n"
        "    image: redis:7\n"
        "    deploy:\n"
        "      replicas: 2\n"
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (
            tmp_path / "workdirs" / "helper-1",
            tmp_path / "workdirs" / "helper-2",
        ),
    }

    async def resolve(_name: str, service: dict[str, object]) -> str:
        return str(service["image"])

    names = await docker_compose_to_singularity_compose(
        compose,
        tmp_path / "singularity-compose.yml",
        identity="unit",
        image_resolver=resolve,
        staging_dir=staging,
        fakeroot=False,
        service_workdirs=workdirs,
    )

    instances = yaml.safe_load(
        (tmp_path / "singularity-compose.yml").read_text()
    )["instances"]
    helper_workdirs = {
        option.removeprefix("workdir=")
        for key, instance in instances.items()
        if "helper" in key
        for option in instance["start"]["options"]
        if option.startswith("workdir=")
    }
    assert helper_workdirs == {str(path) for path in workdirs["helper"]}
    assert len(names["helper"]) == 2
    assert len(set(names["helper"])) == 2
    assert all("deploy" not in instance for instance in instances.values())
    main = next(
        instance for key, instance in instances.items() if "main" in key
    )
    assert len(main["depends_on"]) == 2


async def test_prepare_compose_project_wires_runtime_workdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._prepare_runtime_workdir()
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    observed: dict[str, object] = {}

    async def convert(*_args: object, **kwargs: object) -> dict[str, list[str]]:
        observed.update(kwargs)
        return {"main": ["main1"], "helper": ["helper1"]}

    monkeypatch.setattr(
        harbor_singularity,
        "docker_compose_to_singularity_compose",
        convert,
    )
    await environment._prepare_compose_project(force_build=False)

    workdirs = observed["service_workdirs"]
    assert isinstance(workdirs, dict)
    assert set(workdirs) == {"main", "helper"}
    assert all(
        path.is_dir()
        for service_workdirs in workdirs.values()
        for path in service_workdirs
    )
    assert environment._runtime_workdir_root is not None
    assert all(
        path.is_relative_to(environment._runtime_workdir_root)
        for service_workdirs in workdirs.values()
        for path in service_workdirs
    )
    environment._cleanup_compose_project()
    environment._cleanup_staging()


async def test_stop_removes_runtime_workdir_and_restart_uses_a_new_one(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    first = environment._runtime_workdir_root
    assert first is not None and first.is_dir()

    await environment.stop(delete=False)
    assert not first.exists()
    assert environment._runtime_workdir_root is None

    environment._prepare_runtime_workdir()
    second = environment._runtime_workdir_root
    assert second is not None and second.is_dir()
    assert second != first
    environment._cleanup_staging()
    assert not second.exists()


async def test_cleanup_removes_runtime_workdir_before_temporary_is_released(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    temporary = environment._runtime_workdir_temp
    root = environment._runtime_workdir_root
    assert temporary is not None
    assert root is not None and root.is_dir()

    environment._cleanup_staging()

    assert not root.exists()
    assert temporary.name == str(root)


async def test_abandoned_environment_does_not_remove_preserved_workdir(
    tmp_path: Path,
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    temporary = environment._runtime_workdir_temp
    root = environment._runtime_workdir_root
    assert temporary is not None
    assert root is not None and root.is_dir()
    temporary_reference = weakref.ref(temporary)
    del temporary
    del environment

    gc.collect()

    assert temporary_reference() is None
    assert root.is_dir()
    root.rmdir()


async def test_concurrent_start_cannot_clean_up_running_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    seed_started = asyncio.Event()
    release_seed = asyncio.Event()

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def no_op(*_args: object, **_kwargs: object) -> None:
        pass

    async def seed() -> None:
        seed_started.set()
        await release_seed.wait()

    monkeypatch.setattr(environment, "_build_main_sif", build)
    monkeypatch.setattr(environment, "_mark_sif_in_use", no_op)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", seed)
    monkeypatch.setattr(environment, "_run", no_op)
    monkeypatch.setattr(environment, "ensure_dirs", no_op)
    monkeypatch.setattr(
        environment, "_upload_environment_dir_after_start", no_op
    )

    first_start = asyncio.create_task(environment.start(force_build=False))
    await seed_started.wait()
    root = environment._runtime_workdir_root
    assert root is not None and root.is_dir()

    second_start = asyncio.create_task(environment.start(force_build=False))
    await asyncio.sleep(0)
    assert not second_start.done()

    release_seed.set()
    await first_start
    with pytest.raises(RuntimeError, match="already started"):
        await second_start

    assert environment._runtime_workdir_root == root
    assert root.is_dir()
    environment._instance_started = False
    environment._instance_start_attempted = False
    environment._cleanup_staging()


async def test_start_rejects_preserved_failed_runtime_state(tmp_path: Path):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    with pytest.raises(RuntimeError, match="requires stop"):
        await environment.start(force_build=False)

    assert environment._runtime_workdir_root == root
    assert root.is_dir()
    environment._instance_start_attempted = False
    environment._cleanup_staging()


async def test_cleanup_completion_survives_repeated_cancellation():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cleanup() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    waiter = asyncio.create_task(
        DockerfileSingularityEnvironment._await_cleanup(cleanup())
    )
    await cleanup_started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not waiter.done()

    release_cleanup.set()
    await waiter
    assert cleanup_finished.is_set()


async def test_compose_cleanup_accepts_confirmed_absent_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._compose_file = tmp_path / "singularity-compose.yml"
    environment._compose_instances = {
        "main": ["main1"],
        "helper": ["helper1", "helper2"],
    }
    environment._instance_started = True
    environment._instance_start_attempted = True

    async def fail_down(*_args: str) -> None:
        raise RuntimeError("compose unavailable")

    async def no_instances() -> set[str]:
        return set()

    async def unexpected_stop(*_args: str, **_kwargs: object) -> None:
        pytest.fail("absent instances must not be stopped")

    monkeypatch.setattr(environment, "_run_compose", fail_down)
    monkeypatch.setattr(environment, "_list_instances", no_instances)
    monkeypatch.setattr(environment, "_run", unexpected_stop)

    assert await environment._stop_compose(warn=True)
    assert await environment._stop_compose(warn=True)
    assert not environment._instance_started
    assert not environment._instance_start_attempted


async def test_compose_cleanup_preserves_privileged_runtime_on_down_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    environment._compose_file = tmp_path / "singularity-compose.yml"
    environment._compose_instances = {"main": ["main1"]}
    environment._compose_uses_sudo = True
    environment._instance_started = True
    environment._instance_start_attempted = True

    async def fail_down(*_args: str) -> None:
        raise RuntimeError("compose unavailable")

    async def unexpected_list() -> set[str]:
        pytest.fail("an unprivileged list cannot prove privileged absence")

    monkeypatch.setattr(environment, "_run_compose", fail_down)
    monkeypatch.setattr(environment, "_list_instances", unexpected_list)

    assert not await environment._stop_compose(warn=True)
    assert environment._instance_started
    assert environment._instance_start_attempted


async def test_failed_start_before_instance_attempt_removes_runtime_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    observed: list[Path] = []

    async def build(_force_build: bool) -> Path:
        return tmp_path / "image.sif"

    async def mark(_path: Path) -> None:
        pass

    async def fail_seed() -> None:
        assert environment._runtime_workdir_root is not None
        observed.append(environment._runtime_workdir_root)
        raise RuntimeError("seed failed")

    monkeypatch.setattr(environment, "_build_main_sif", build)
    monkeypatch.setattr(environment, "_mark_sif_in_use", mark)
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", fail_seed)

    with pytest.raises(RuntimeError, match="seed failed"):
        await environment.start(force_build=False)

    assert len(observed) == 1
    assert not observed[0].exists()
    assert environment._runtime_workdir_root is None


async def test_failed_stop_preserves_runtime_workdir_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    async def exists() -> bool:
        return True

    async def fail(*_command: str, **_kwargs: object) -> None:
        raise RuntimeError("stop failed")

    monkeypatch.setattr(environment, "_instance_exists", exists)
    monkeypatch.setattr(environment, "_run", fail)
    await environment.stop(delete=False)

    assert root.is_dir()
    assert environment._runtime_workdir_root == root

    async def succeed(*_command: str, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(environment, "_run", succeed)
    await environment.stop(delete=False)
    assert not root.exists()


async def test_stop_cleans_failed_start_when_instance_was_never_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    assert root is not None
    environment._instance_start_attempted = True

    async def missing() -> bool:
        return False

    async def unexpected(*_command: str, **_kwargs: object) -> None:
        pytest.fail("stop command should not run for an absent instance")

    monkeypatch.setattr(environment, "_instance_exists", missing)
    monkeypatch.setattr(environment, "_run", unexpected)
    await environment.stop(delete=False)

    assert not root.exists()
    assert not environment._instance_start_attempted


@pytest.mark.parametrize(
    "output",
    [
        b'{"instances": {}}',
        b'{"instances": "bad"}',
        b'{"instances": [1]}',
        b'{"instances": [{}]}',
        b'{"instances": [{"instance": 1}]}',
    ],
)
def test_instance_list_parser_rejects_malformed_schema(output: bytes):
    with pytest.raises(RuntimeError, match="parse Singularity instance list"):
        DockerfileSingularityEnvironment._parse_instance_list(output)


def test_instance_list_parser_returns_instance_names():
    output = b'{"instances": [{"instance": "first", "pid": 1}]}'
    assert DockerfileSingularityEnvironment._parse_instance_list(output) == {
        "first"
    }
