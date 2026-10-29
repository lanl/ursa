import asyncio
import gc
import os
import sys
import weakref
from pathlib import Path
from unittest.mock import AsyncMock

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
    tmp_path: Path,
    *,
    compose: bool = False,
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    phase_network_modes: tuple[NetworkMode, ...] = (),
    overlay: bool = False,
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
        network_policy=NetworkPolicy(network_mode=network_mode),
        phase_network_policies=[
            NetworkPolicy(network_mode=mode) for mode in phase_network_modes
        ],
        persistent_env={},
        singularity_fakeroot=False,
        singularity_overlay=overlay,
    )
    environment._runtime_path = str(tmp_path / "fake-singularity")
    return environment


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _prepare_started_environment(
    environment: DockerfileSingularityEnvironment, tmp_path: Path
) -> Path:
    environment._prepare_runtime_workdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    environment._scratch_dir = scratch
    staging = scratch / "staging"
    staging.mkdir()
    environment._staging_dir = staging
    overlay = scratch / "overlay.img"
    overlay.touch()
    environment._overlay_path = overlay
    environment._sif_path = tmp_path / "image.sif"
    environment._instance_started = True
    environment._instance_start_attempted = True
    return overlay


def test_host_ca_certificates_pem_uses_ssl(
    monkeypatch: pytest.MonkeyPatch,
):
    class Context:
        @staticmethod
        def get_ca_certs(*, binary_form: bool) -> list[bytes]:
            assert binary_form
            return [b"first", b"second"]

    monkeypatch.setattr(
        harbor_singularity.ssl,
        "create_default_context",
        lambda: Context(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type("VerifyPaths", (), {"cafile": None, "capath": None})(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda certificate: f"PEM:{certificate.decode()}\n",
    )

    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:first\nPEM:second\n"
    )


def test_host_ca_certificates_pem_loads_hashed_capath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    capath = tmp_path / "certs"
    capath.mkdir()
    certificate = capath / "1234abcd.0"
    certificate.write_bytes(b"certificate")
    (capath / "ignored.pem").write_bytes(b"ignored")

    class Context:
        loaded: list[bytes] = []

        def load_verify_locations(self, *, cafile: str) -> None:
            self.loaded.append(Path(cafile).read_bytes())

        def get_ca_certs(self, *, binary_form: bool) -> list[bytes]:
            assert binary_form
            return self.loaded

    monkeypatch.setattr(
        harbor_singularity.ssl,
        "create_default_context",
        lambda: Context(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type(
            "VerifyPaths",
            (),
            {"cafile": None, "capath": str(capath)},
        )(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda value: f"PEM:{value.decode()}\n",
    )

    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:certificate\n"
    )


def test_phase_network_change_requires_overlay(tmp_path: Path):
    with pytest.raises(ValueError, match="require a writable overlay"):
        _environment(
            tmp_path,
            phase_network_modes=(NetworkMode.NO_NETWORK,),
        )


def test_overlay_enables_dynamic_network_policy(tmp_path: Path):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )

    assert environment.capabilities.dynamic_network_policy


def test_compose_sidecars_reject_no_network_phase(tmp_path: Path):
    with pytest.raises(ValueError, match="sidecar networking"):
        _environment(
            tmp_path,
            compose=True,
            phase_network_modes=(NetworkMode.NO_NETWORK,),
            overlay=True,
        )


def test_compose_network_policy_is_rewritten(tmp_path: Path):
    environment = _environment(tmp_path)
    compose_file = tmp_path / "singularity-compose.yml"
    compose_file.write_text(
        "instances:\n"
        "  main:\n"
        "    network:\n"
        "      allocate_ip: true\n"
        "      enable: true\n"
    )
    environment._compose_file = compose_file

    environment._set_compose_network_policy(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    isolated = yaml.safe_load(compose_file.read_text())["instances"]["main"]
    assert isolated["network"] == {
        "allocate_ip": False,
        "enable": True,
        "type": "none",
    }

    environment._set_compose_network_policy(
        NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    )
    public = yaml.safe_load(compose_file.read_text())["instances"]["main"]
    assert public["network"] == {
        "allocate_ip": True,
        "enable": True,
    }


def test_host_ca_certificates_are_staged_and_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    monkeypatch.setattr(
        harbor_singularity,
        "_host_ca_certificates_pem",
        lambda: "test host CA bundle\n",
    )
    staged = environment._prepare_host_ca_certificates()
    environment._persistent_env["SSL_CERT_FILE"] = "/task/ca-bundle.pem"

    runtime_env = environment._runtime_environment(None)

    assert staged is not None
    assert staged.read_text() == "test host CA bundle\n"
    assert staged.stat().st_mode & 0o222 == 0
    assert runtime_env["SINGULARITYENV_SSL_CERT_FILE"] == (
        "/task/ca-bundle.pem"
    )
    assert runtime_env["SINGULARITYENV_REQUESTS_CA_BUNDLE"] == (
        "/staging/host-ca-certificates.crt"
    )
    assert runtime_env["SINGULARITYENV_CURL_CA_BUNDLE"] == (
        "/staging/host-ca-certificates.crt"
    )


async def test_network_policy_change_restarts_with_existing_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )
    overlay = _prepare_started_environment(environment, tmp_path)
    commands: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        assert not warn
        environment._instance_started = False
        environment._instance_start_attempted = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    await environment.set_network_policy(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )

    start = next(command for command in commands if "start" in command)
    assert start[start.index("--overlay") + 1] == str(overlay)
    assert start[start.index("--network") + 1] == "none"
    assert environment.network_policy.network_mode == NetworkMode.NO_NETWORK
    assert environment._instance_started


async def test_failed_network_policy_change_restores_previous_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(
        tmp_path,
        phase_network_modes=(NetworkMode.NO_NETWORK,),
        overlay=True,
    )
    _prepare_started_environment(environment, tmp_path)
    starts: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        environment._instance_started = False
        environment._instance_start_attempted = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        if "start" not in command:
            return
        starts.append(command)
        if "--network" in command:
            raise RuntimeError("isolated start failed")

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    with pytest.raises(RuntimeError, match="isolated start failed"):
        await environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        )

    assert len(starts) == 2
    assert "--network" in starts[0]
    assert "--network" not in starts[1]
    assert environment.network_policy.network_mode == NetworkMode.PUBLIC
    assert environment._instance_started


async def test_run_only_records_failed_build_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"

    with pytest.raises(RuntimeError) as raised:
        await environment._run(
            sys.executable,
            "-c",
            "import sys; print('builder stdout'); "
            "print('builder stderr', file=sys.stderr); raise SystemExit(7)",
            failure_log_path=build_log,
        )

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    message = str(raised.value)
    assert f"Build log: {build_log}" in message
    log = build_log.read_text()
    assert "exit code: 7" in log
    assert "builder stdout" in log
    assert "builder stderr" in log
    assert build_log.stat().st_mode & 0o777 == 0o600


async def test_run_discards_successful_build_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"

    await environment._run(
        sys.executable,
        "-c",
        "import sys; print('builder stdout'); "
        "print('builder stderr', file=sys.stderr)",
        failure_log_path=build_log,
    )

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert not build_log.exists()


async def test_build_surfaces_log_after_all_builders_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"

    async def cache_path(*_args: object, **_kwargs: object) -> Path:
        return output

    async def invalid(_path: Path) -> bool:
        return False

    async def fail_build(
        _builder: str,
        _output: Path,
        _temporary: Path,
        _force_build: bool,
        _dockerfile_path: Path,
        _context_dir: Path,
        _build_args: tuple[str, ...],
        _target: str | None,
        build_log_path: Path,
    ) -> None:
        build_log_path.write_text("builder failed\n")
        raise RuntimeError("failed")

    monkeypatch.setattr(environment, "_dockerfile_cache_path", cache_path)
    monkeypatch.setattr(environment, "_is_valid_sif", invalid)
    monkeypatch.setattr(environment, "_build_with", fail_build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )

    with pytest.raises(RuntimeError, match="No container builder succeeded"):
        await environment._build_dockerfile_sif(False)

    captured = capsys.readouterr()
    assert not captured.out
    assert "builder failed" in captured.err


async def test_build_discards_failed_attempt_log_after_fallback_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"
    attempts = 0

    async def cache_path(*_args: object, **_kwargs: object) -> Path:
        return output

    async def invalid(_path: Path) -> bool:
        return False

    async def build(
        _builder: str,
        output: Path,
        _temporary: Path,
        _force_build: bool,
        _dockerfile_path: Path,
        _context_dir: Path,
        _build_args: tuple[str, ...],
        _target: str | None,
        build_log_path: Path,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            build_log_path.write_text("first builder failed\n")
            raise RuntimeError("failed")
        output.write_bytes(b"sif")

    monkeypatch.setattr(environment, "_dockerfile_cache_path", cache_path)
    monkeypatch.setattr(environment, "_is_valid_sif", invalid)
    monkeypatch.setattr(environment, "_build_with", build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name != "podman" else None,
    )

    assert await environment._build_dockerfile_sif(False) == output

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert not output.with_suffix(".build.log").exists()


async def test_build_with_records_builder_and_apptainer_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"
    temporary = tmp_path / "cached-image.tmp.sif"
    build_log = tmp_path / "cached-image.build.log"
    observed: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def capture(*command: str, **kwargs: object) -> None:
        observed.append((command, kwargs))
        if command[:2] == (environment._runtime_path, "build"):
            Path(command[-2]).write_bytes(b"sif")

    monkeypatch.setattr(environment, "_run", capture)

    await environment._build_with(
        "docker",
        output,
        temporary,
        False,
        environment._dockerfile_path,
        environment.environment_dir,
        (),
        None,
        build_log,
    )

    builder_build = observed[0]
    assert builder_build[0][:2] == ("docker", "build")
    assert builder_build[1] == {"failure_log_path": build_log}
    apptainer_build = next(
        item
        for item in observed
        if item[0][:2] == (environment._runtime_path, "build")
    )
    assert apptainer_build[1] == {"failure_log_path": build_log}
    save = next(item for item in observed if item[0][:2] == ("docker", "save"))
    assert save[1] == {"failure_log_path": build_log}
    assert output.read_bytes() == b"sif"


@pytest.mark.parametrize("builder", ["buildah", "podman", "docker"])
@pytest.mark.parametrize("runtime", ["singularity", "apptainer"])
async def test_oci_cache_is_disabled_only_for_sif_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder: str,
    runtime: str,
):
    cache_variables = ("SINGULARITY_DISABLE_CACHE", "APPTAINER_DISABLE_CACHE")
    for variable in cache_variables:
        monkeypatch.delenv(variable, raising=False)
    environment = _environment(tmp_path)
    environment._runtime_path = runtime
    environment._image_cache_dir = tmp_path
    output = tmp_path / "cached-image.sif"
    observed: list[tuple[str, ...]] = []

    async def capture(*command: str, **kwargs: object) -> AsyncMock:
        observed.append(command)
        process_env = kwargs.get("env")
        if process_env is None:
            process_env = os.environ
        assert all(variable not in process_env for variable in cache_variables)
        if command[:2] == (runtime, "build"):
            Path(command[-2]).write_bytes(b"sif")
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (b"", b"")
        return process

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", capture
    )
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: builder if name == builder else None,
    )
    monkeypatch.setattr(
        environment, "_dockerfile_cache_path", AsyncMock(return_value=output)
    )

    assert await environment._build_dockerfile_sif(False) == output
    build_command = next(
        command for command in observed if command[:2] == (runtime, "build")
    )
    assert build_command[2:4] == (
        "--disable-cache",
        str(output.with_suffix(".tmp.sif")),
    )
    assert build_command[-1].startswith("docker-archive://")
    assert output.read_bytes() == b"sif"
    assert all(
        "--disable-cache" not in command
        for command in observed
        if command != build_command
    )

    observed.clear()
    assert await environment._build_dockerfile_sif(False) == output
    assert observed == [(runtime, "sif", "list", str(output))]

    environment._sif_path = output
    environment._staging_dir = tmp_path / "staging"
    environment._staging_dir.mkdir()
    environment._prepare_runtime_workdir()
    await environment._start_prepared_runtime(environment._network_policy)
    assert observed[-2][:3] == (runtime, "instance", "start")
    assert observed[-1][:2] == (runtime, "exec")
    assert all("--disable-cache" not in command for command in observed)
    environment._cleanup_staging()


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
    monkeypatch.setattr(
        harbor_singularity,
        "_host_ca_certificates_pem",
        lambda: "test host CA bundle\n",
    )
    staged_ca_certificates = environment._prepare_host_ca_certificates()
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
    assert staged_ca_certificates is not None
    assert staged_ca_certificates.read_text() == "test host CA bundle\n"
    assert staged_ca_certificates.stat().st_mode & 0o222 == 0
    assert observed["main_environment"] == {
        "CURL_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "REQUESTS_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "SSL_CERT_FILE": "/staging/host-ca-certificates.crt",
    }
    environment._cleanup_compose_project()
    environment._cleanup_staging()


async def test_build_sifs_builds_main_and_unique_compose_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment = _environment(tmp_path, compose=True)
    helper = environment.environment_dir / "helper"
    helper.mkdir()
    (helper / "Dockerfile").write_text("FROM busybox:latest\n")
    (environment.environment_dir / "docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n"
        "    build: .\n"
        "  helper:\n"
        "    build: helper\n"
        "  duplicate:\n"
        "    build: helper\n"
        "  external:\n"
        "    image: redis:7\n"
    )
    observed: list[tuple[bool, Path, Path]] = []

    async def build(
        force_build: bool,
        *,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        **_kwargs: object,
    ) -> Path:
        dockerfile = dockerfile_path or environment._dockerfile_path
        context = context_dir or environment.environment_dir
        observed.append((force_build, dockerfile, context))
        return tmp_path / f"{context.name}.sif"

    monkeypatch.setattr(environment, "_build_dockerfile_sif", build)

    paths = await environment.build_sifs(force_build=True)

    assert paths == [tmp_path / "environment.sif", tmp_path / "helper.sif"]
    assert observed == [
        (
            True,
            environment.environment_dir / "Dockerfile",
            environment.environment_dir,
        ),
        (True, helper / "Dockerfile", helper),
    ]


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
