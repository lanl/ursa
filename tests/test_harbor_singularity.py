import asyncio
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

pytest.importorskip("harbor")

from harbor.environments.base import ExecResult
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths

from ursa.integrations import harbor_singularity
from ursa.integrations.harbor_singularity import (
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
    workdir: str | None = "/workspace",
) -> DockerfileSingularityEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        (environment_dir / "docker-compose.yaml").write_text(
            yaml.safe_dump({
                "services": {
                    "main": {"image": "ubuntu:24.04"},
                    "helper": {"image": "ubuntu:24.04"},
                }
            })
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
        task_env_config=EnvironmentConfig(workdir=workdir),
        mounts=[
            dict(
                type="bind",
                source=str(workspace),
                target="/workspace",
                read_only=False,
            )
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


async def _await(value: object) -> object:
    return value


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


async def _convert_compose(
    tmp_path: Path,
    services: dict[str, object],
    workdirs: dict[str, tuple[Path, ...]],
) -> tuple[dict[str, list[str]], dict[str, object]]:
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(yaml.safe_dump({"services": services}))
    staging = tmp_path / "staging"
    staging.mkdir()

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
    return names, instances


def test_compose_image_requires_explicit_workdir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Compose main image.*workdir"):
        _environment(tmp_path, compose=True, workdir=None)


@pytest.mark.parametrize("hashed_capath", [False, True])
def test_host_ca_certificates_pem(
    tmp_path: Path, monkeypatch, hashed_capath: bool
):
    capath = tmp_path / "certs" if hashed_capath else None
    if capath:
        capath.mkdir()
        (capath / "1234abcd.0").write_bytes(b"certificate")
        (capath / "ignored.pem").write_bytes(b"ignored")
    loaded: list[bytes] = []

    class Context:
        def load_verify_locations(self, *, cafile: str) -> None:
            loaded.append(Path(cafile).read_bytes())

        def get_ca_certs(self, *, binary_form: bool) -> list[bytes]:
            assert binary_form
            return loaded if hashed_capath else [b"first", b"second"]

    monkeypatch.setattr(
        harbor_singularity.ssl, "create_default_context", Context
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "get_default_verify_paths",
        lambda: type(
            "Paths",
            (),
            {"cafile": None, "capath": str(capath) if capath else None},
        )(),
    )
    monkeypatch.setattr(
        harbor_singularity.ssl,
        "DER_cert_to_PEM_cert",
        lambda value: f"PEM:{value.decode()}\n",
    )
    assert harbor_singularity._host_ca_certificates_pem() == (
        "PEM:certificate\n" if hashed_capath else "PEM:first\nPEM:second\n"
    )


@pytest.mark.parametrize(
    ("compose", "overlay", "error"),
    [
        (False, False, "cannot change network policy after start"),
        (False, True, None),
        (True, True, "sidecar networking"),
    ],
)
def test_network_phase_requirements(
    tmp_path: Path, compose: bool, overlay: bool, error: str | None
):
    kwargs = {
        "compose": compose,
        "overlay": overlay,
        "phase_network_modes": (NetworkMode.NO_NETWORK,),
    }
    if error:
        with pytest.raises(ValueError, match=error):
            _environment(tmp_path, **kwargs)
    else:
        assert _environment(
            tmp_path, **kwargs
        ).capabilities.dynamic_network_policy


@pytest.mark.parametrize(
    ("port", "valid"),
    [
        ({"published": 8080, "target": 80}, True),
        ({"published": True, "target": 80}, False),
        ({"published": 8080, "target": False}, False),
        ({"published": [], "target": 80}, False),
        ({"published": 8080, "target": {}}, False),
        ({"published": -1, "target": 80}, False),
        ({"published": 8080, "target": "http"}, False),
    ],
)
def test_compose_parser_normalizes_null_dependencies_and_validates_ports(
    tmp_path: Path, port: dict[str, object], valid: bool
):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(
        yaml.safe_dump({
            "services": {
                "main": {
                    "image": "ubuntu:24.04",
                    "depends_on": None,
                    "ports": [port],
                }
            }
        })
    )

    if not valid:
        with pytest.raises(ValueError, match="TCP published ports"):
            harbor_singularity._load_docker_compose([compose])
        return

    config = harbor_singularity._load_docker_compose([compose])
    assert config["services"]["main"]["depends_on"] == []


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

    for mode, expected in (
        (
            NetworkMode.NO_NETWORK,
            {"allocate_ip": False, "enable": True, "type": "none"},
        ),
        (NetworkMode.PUBLIC, {"allocate_ip": True, "enable": True}),
    ):
        environment._set_compose_network_policy(
            NetworkPolicy(network_mode=mode)
        )
        assert (
            yaml.safe_load(compose_file.read_text())["instances"]["main"][
                "network"
            ]
            == expected
        )


@pytest.mark.parametrize("fail_isolated_start", [False, True])
async def test_network_policy_change_restarts_or_restores_previous_policy(
    tmp_path: Path, monkeypatch, fail_isolated_start: bool
):
    environment = _environment(
        tmp_path, phase_network_modes=(NetworkMode.NO_NETWORK,), overlay=True
    )
    overlay = _prepare_started_environment(environment, tmp_path)
    starts: list[tuple[str, ...]] = []

    async def stop(*, warn: bool) -> bool:
        environment._instance_started = (
            environment._instance_start_attempted
        ) = False
        return True

    async def run(*command: str, **_kwargs: object) -> None:
        if "start" in command:
            starts.append(command)
        if fail_isolated_start and "--network" in command:
            raise RuntimeError("isolated start failed")

    monkeypatch.setattr(environment, "_stop_instance", stop)
    monkeypatch.setattr(environment, "_run", run)

    if fail_isolated_start:
        with pytest.raises(RuntimeError, match="isolated start failed"):
            await environment.set_network_policy(
                NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
            )
        assert (
            len(starts) == 2
            and "--network" in starts[0]
            and "--network" not in starts[1]
        )
        assert environment.network_policy.network_mode == NetworkMode.PUBLIC
    else:
        await environment.set_network_policy(
            NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        )
        start = starts[0]
        assert (
            _option_value(list(start), "--overlay"),
            _option_value(list(start), "--network"),
        ) == (str(overlay), "none")
        assert environment.network_policy.network_mode == NetworkMode.NO_NETWORK
    assert environment._instance_started


@pytest.mark.parametrize("fails", [False, True])
async def test_run_build_log_is_retained_only_on_failure(
    tmp_path: Path, capsys, fails: bool
):
    environment = _environment(tmp_path)
    build_log = tmp_path / "cached-image.build.log"
    program = "import sys; print('builder stdout'); print('builder stderr', file=sys.stderr)"
    if fails:
        program += "; raise SystemExit(7)"

    if fails:
        with pytest.raises(RuntimeError) as raised:
            await environment._run(
                sys.executable, "-c", program, failure_log_path=build_log
            )
        message = str(raised.value)
        assert f"Build log: {build_log}" in message
        log = build_log.read_text()
        assert all(
            value in log
            for value in ("exit code: 7", "builder stdout", "builder stderr")
        )
        assert build_log.stat().st_mode & 0o777 == 0o600
    else:
        await environment._run(
            sys.executable, "-c", program, failure_log_path=build_log
        )
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err
    assert build_log.exists() is fails


@pytest.mark.parametrize("fallback_succeeds", [False, True])
async def test_build_attempt_logs_are_retained_only_after_all_builders_fail(
    tmp_path: Path, monkeypatch, capsys, fallback_succeeds: bool
):
    environment = _environment(tmp_path)
    output = tmp_path / "cached-image.sif"

    attempts = 0

    async def build(*args: object) -> None:
        nonlocal attempts
        attempts += 1
        build_log_path = args[-1]
        assert isinstance(build_log_path, Path)
        if not fallback_succeeds or attempts == 1:
            build_log_path.write_text("builder failed\n")
            raise RuntimeError("failed")
        output.write_bytes(b"sif")

    monkeypatch.setattr(
        environment,
        "_dockerfile_cache_path",
        lambda *_args, **_kwargs: _await(output),
    )
    monkeypatch.setattr(
        environment, "_is_valid_sif", lambda _path: _await(False)
    )
    monkeypatch.setattr(environment, "_build_with", build)
    monkeypatch.setattr(
        harbor_singularity.shutil,
        "which",
        lambda name: (
            f"/usr/bin/{name}"
            if fallback_succeeds and name != "podman"
            else "/usr/bin/docker"
            if name == "docker"
            else None
        ),
    )

    if fallback_succeeds:
        assert await environment._build_dockerfile_sif(False) == output
    else:
        with pytest.raises(
            RuntimeError, match="No container builder succeeded"
        ):
            await environment._build_dockerfile_sif(False)
    captured = capsys.readouterr()
    assert not captured.out
    assert ("builder failed" in captured.err) is not fallback_succeeds
    assert output.with_suffix(".build.log").exists() is not fallback_succeeds


async def test_marking_a_live_sif_twice_keeps_its_usage_marker(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    image = tmp_path / "image.sif"
    image.touch()

    await environment._mark_sif_in_use(image)
    await environment._mark_sif_in_use(image)

    marker = environment._sif_in_use_path(image)
    assert marker.read_text().splitlines() == [environment._sif_use_token]


@pytest.mark.parametrize("scenario", ["simple", "missing", "replicas"])
async def test_compose_conversion_assigns_each_service_a_workdir(
    tmp_path: Path, scenario: str
):
    services: dict[str, object] = {
        "main": {"image": "ubuntu:24.04"},
        "helper": {"image": "redis:7"},
    }
    workdirs = {
        "main": (tmp_path / "workdirs" / "main",),
        "helper": (tmp_path / "workdirs" / "helper",),
    }
    if scenario == "missing":
        with pytest.raises(ValueError, match="workdir.*helper"):
            await _convert_compose(
                tmp_path, services, {"main": workdirs["main"]}
            )
        return
    if scenario == "replicas":
        services["main"] = {"image": "ubuntu:24.04", "depends_on": ["helper"]}
        services["helper"] = {"image": "redis:7", "deploy": {"replicas": 2}}
        workdirs["helper"] = (
            tmp_path / "workdirs" / "helper-1",
            tmp_path / "workdirs" / "helper-2",
        )
    for paths in workdirs.values():
        for path in paths:
            path.mkdir(parents=True)
    names, instances = await _convert_compose(tmp_path, services, workdirs)
    options = [item["start"]["options"] for item in instances.values()]
    configured = {
        option.removeprefix("workdir=")
        for values in options
        for option in values
        if option.startswith("workdir=")
    }
    assert configured == {
        str(path) for paths in workdirs.values() for path in paths
    }
    assert all("containall" in values for values in options)
    if scenario == "simple":
        assert len(instances) == 2
        assert not any(key.endswith("-r1") for key in instances)
        return
    assert len(names["helper"]) == len(set(names["helper"])) == 2
    assert all("deploy" not in instance for instance in instances.values())
    assert (
        len(
            next(v for key, v in instances.items() if "main" in key)[
                "depends_on"
            ]
        )
        == 2
    )


async def test_prepare_compose_project_wires_runtime_workdirs(
    tmp_path: Path, monkeypatch
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
        harbor_singularity, "docker_compose_to_singularity_compose", convert
    )
    await environment._prepare_compose_project(force_build=False)

    workdirs = observed["service_workdirs"]
    assert isinstance(workdirs, dict) and set(workdirs) == {"main", "helper"}
    paths = [path for values in workdirs.values() for path in values]
    assert all(path.is_dir() for path in paths)
    assert environment._runtime_workdir_root and all(
        path.is_relative_to(environment._runtime_workdir_root) for path in paths
    )
    assert (
        staged_ca_certificates
        and staged_ca_certificates.read_text() == "test host CA bundle\n"
        and not staged_ca_certificates.stat().st_mode & 0o222
    )
    assert observed["main_environment"] == dict.fromkeys(
        ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"),
        "/staging/host-ca-certificates.crt",
    )
    environment._persistent_env["SSL_CERT_FILE"] = "/task/ca-bundle.pem"
    runtime_environment = environment._runtime_environment(None)
    assert {
        key.removeprefix("SINGULARITYENV_"): value
        for key, value in runtime_environment.items()
        if key.startswith("SINGULARITYENV_")
    } == {
        "SSL_CERT_FILE": "/task/ca-bundle.pem",
        "REQUESTS_CA_BUNDLE": "/staging/host-ca-certificates.crt",
        "CURL_CA_BUNDLE": "/staging/host-ca-certificates.crt",
    }
    environment._cleanup_compose_project()
    environment._cleanup_staging()


async def test_start_failure_cleans_runtime_workdir(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    roots = []

    async def fail_seed():
        roots.append(environment._runtime_workdir_root)
        raise RuntimeError("seed failed")

    monkeypatch.setattr(
        environment, "_build_main_sif", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(environment, "_seed_harbor_bind_sources", fail_seed)
    with pytest.raises(RuntimeError, match="seed failed"):
        await environment.start(force_build=False)

    assert len(roots) == 1 and roots[0] is not None and not roots[0].exists()
    assert environment._runtime_workdir_root is None


async def test_exec_stages_multiline_environment_values(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    environment._fakeroot = True
    environment._persistent_env["PERSISTENT"] = "persistent\nsecret"
    observed: dict[str, object] = {}

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"ok", b""

    async def create_process(*command: str, **kwargs: object) -> Process:
        observed["command"] = command
        observed["env"] = kwargs["env"]
        staged = next(environment._staging.glob("*exec-env"))
        observed["contents"] = staged.read_text()
        observed["mode"] = staged.stat().st_mode & 0o777
        return Process()

    monkeypatch.setattr(
        harbor_singularity.asyncio, "create_subprocess_exec", create_process
    )
    result = await environment._exec_instance(
        "printf ready",
        cwd=None,
        env={"PROMPT": "private line\nsecond line", "SAFE": "value"},
        timeout_sec=None,
        user="taskuser",
    )

    assert result.stdout == "ok"
    command = " ".join(observed["command"])
    assert "private line" not in command
    assert ". /staging/" in command
    assert command.index(". /staging/") < command.index("su taskuser")
    assert observed["contents"] == (
        "export PERSISTENT='persistent\nsecret'\n"
        "export PROMPT='private line\nsecond line'\n"
    )
    assert observed["mode"] == 0o600
    runtime_env = observed["env"]
    assert isinstance(runtime_env, dict)
    assert runtime_env["SINGULARITYENV_SAFE"] == "value"
    assert "SINGULARITYENV_PROMPT" not in runtime_env
    assert "SINGULARITYENV_PERSISTENT" not in runtime_env
    assert not list(environment._staging.glob("*exec-env"))


async def test_directory_transfers_preserve_symlinks(tmp_path, monkeypatch):
    environment = _environment(tmp_path)
    _prepare_started_environment(environment, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("data")
    (source / "link").symlink_to("target")
    external = tmp_path / "external"
    external.write_text("fixture")
    (source / "external-link").symlink_to("../external")
    commands: list[str] = []

    async def upload_execute(command: str, **_kwargs: object) -> ExecResult:
        commands.append(command)
        if command.startswith("cp -rP"):
            staged = next(environment._staging.glob("*source"))
            assert (staged / "link").is_symlink()
            assert not (staged / "external-link").is_symlink()
            assert (staged / "external-link").read_text() == "fixture"
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(environment, "exec", upload_execute)
    await environment.upload_dir(source, "/workspace")
    assert any(command.startswith("cp -rP") for command in commands)

    async def download_execute(command: str) -> ExecResult:
        staged = environment._staging / Path(shlex.split(command)[-1]).name
        staged.mkdir()
        os.symlink("/container-only/path", staged / "link")
        return ExecResult(stdout="", stderr="", return_code=0)

    destination = tmp_path / "download"
    await environment._download(
        "/workspace", destination, download_execute, directory=True
    )
    assert (destination / "link").is_symlink()
    assert os.readlink(destination / "link") == "/container-only/path"


@pytest.mark.parametrize(
    "scenario", ["instance-retry", "compose-fallback", "compose-privileged"]
)
async def test_stop_failure_cleanup_contracts(tmp_path, monkeypatch, scenario):
    compose = scenario.startswith("compose")
    environment = _environment(tmp_path, compose=compose)
    environment._prepare_runtime_workdir()
    root = environment._runtime_workdir_root
    environment._instance_started = environment._instance_start_attempted = True

    if compose:
        environment._compose_file = tmp_path / "singularity-compose.yml"
        environment._compose_instances = {"main": ["main"]}
        environment._compose_uses_sudo = scenario == "compose-privileged"
        monkeypatch.setattr(
            environment,
            "_run_compose",
            AsyncMock(side_effect=RuntimeError("compose unavailable")),
        )
        if scenario == "compose-fallback":
            monkeypatch.setattr(
                environment, "_list_instances", AsyncMock(return_value=set())
            )
    else:
        monkeypatch.setattr(
            environment, "_instance_exists", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            environment,
            "_run",
            AsyncMock(side_effect=RuntimeError("stop failed")),
        )

    await environment.stop(delete=False)
    preserved = scenario != "compose-fallback"
    assert (environment._runtime_workdir_root is not None) is preserved

    if scenario == "instance-retry":
        assert (
            root is not None and root.exists() and environment._instance_started
        )
        monkeypatch.setattr(environment, "_run", AsyncMock(return_value=None))
        await environment.stop(delete=False)
        assert not root.exists() and environment._runtime_workdir_root is None


async def test_start_lock_rejects_a_concurrent_start(tmp_path):
    environment = _environment(tmp_path)
    async with environment._lifecycle_lock:
        pending = asyncio.create_task(environment.start(force_build=False))
        await asyncio.sleep(0)
        assert not pending.done()
        environment._instance_started = True
    with pytest.raises(RuntimeError, match="already started"):
        await pending


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (b'{"instances": {}}', None),
        (b'{"instances": "bad"}', None),
        (b'{"instances": [1]}', None),
        (b'{"instances": [{}]}', None),
        (b'{"instances": [{"instance": 1}]}', None),
        (b'{"instances": [{"instance": "first", "pid": 1}]}', {"first"}),
    ],
)
def test_instance_list_parser(output: bytes, expected: set[str] | None):
    if expected is None:
        with pytest.raises(
            RuntimeError, match="parse Singularity instance list"
        ):
            DockerfileSingularityEnvironment._parse_instance_list(output)
    else:
        assert (
            DockerfileSingularityEnvironment._parse_instance_list(output)
            == expected
        )
