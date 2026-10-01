import asyncio
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from filelock import FileLock  # noqa: E402
from harbor.models.task.config import (  # noqa: E402
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from ursa.integrations.harbor import UrsaHarborAgent  # noqa: E402
from ursa.integrations.harbor_singularity import (  # noqa: E402
    DockerfileSingularityEnvironment,
)

SINGULARITY = shutil.which("apptainer") or shutil.which("singularity")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        SINGULARITY is None,
        reason="Apptainer or Singularity is required",
    ),
]


def _shared_session_dir(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
    name: str,
) -> Path:
    base = tmp_path_factory.getbasetemp()
    root = (base if worker_id == "master" else base.parent) / name
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session")
def ubuntu_sif(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Path:
    """Build the test image directly from Ubuntu with Singularity."""
    root = _shared_session_dir(tmp_path_factory, worker_id, "harbor-ubuntu-sif")
    image = root / "ubuntu-24.04.sif"
    with FileLock(f"{image}.lock"):
        if image.is_file():
            return image
        temporary = image.with_suffix(".tmp.sif")
        temporary.unlink(missing_ok=True)
        result = subprocess.run(
            [
                SINGULARITY,
                "build",
                str(temporary),
                "docker://ubuntu:24.04",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            pytest.fail(
                "Could not build the Ubuntu Singularity fixture:\n"
                f"{result.stderr or result.stdout}"
            )
        temporary.replace(image)
    return image


@pytest.fixture(scope="session")
def ursa_wheel(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Path:
    """Build the package under test for installation inside Singularity."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("Building the URSA wheel requires uv")
    output_dir = _shared_session_dir(
        tmp_path_factory, worker_id, "harbor-ursa-wheel"
    )
    repository = Path(__file__).resolve().parents[2]
    with FileLock(f"{output_dir}.lock"):
        wheels = list(output_dir.glob("ursa_ai-*.whl"))
        if not wheels:
            build_dir = output_dir / "build"
            build_dir.mkdir(exist_ok=True)
            result = subprocess.run(
                [
                    uv,
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(build_dir),
                    str(repository),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                pytest.fail(f"Could not build the URSA wheel:\n{result.stderr}")
            built_wheels = list(build_dir.glob("ursa_ai-*.whl"))
            if len(built_wheels) != 1:
                pytest.fail(
                    f"Expected one built URSA wheel, found: {built_wheels}"
                )
            built_wheels[0].replace(output_dir / built_wheels[0].name)
            wheels = list(output_dir.glob("ursa_ai-*.whl"))
        if len(wheels) != 1:
            pytest.fail(f"Expected one URSA wheel, found: {wheels}")
    return wheels[0]


async def _start_environment(
    root: Path,
    image: Path,
    *,
    fakeroot: bool = False,
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    overlay: bool = False,
    compose: bool = False,
    image_cache_dir: Path | None = None,
) -> DockerfileSingularityEnvironment:
    environment_dir = root / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        sidecar = environment_dir / "sidecar"
        sidecar.mkdir()
        (sidecar / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        (environment_dir / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    build: .\n"
            "  helper:\n"
            "    build: ./sidecar\n"
        )
    workspace = root / "workspace"
    workspace.mkdir(mode=0o777)
    trial_paths = TrialPaths(root / "trial")
    trial_paths.mkdir()

    environment = DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="local-test",
        session_id=f"{root.name}__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            workdir="/workspace",
            storage_mb=32,
        ),
        mounts=[
            {
                "type": "bind",
                "source": str(workspace),
                "target": "/workspace",
                "read_only": False,
            }
        ],
        network_policy=NetworkPolicy(network_mode=network_mode),
        persistent_env={"PERSISTED_VALUE": "from-environment"},
        singularity_fakeroot=fakeroot,
        singularity_overlay=overlay,
        singularity_image_cache_dir=image_cache_dir or root / "sif-cache",
        singularity_startup_timeout_sec=30,
    )
    if compose:
        services = environment._load_compose_config()["services"].values()
        build_inputs = [
            environment._compose_build_inputs(service["build"])
            for service in services
        ]
    else:
        build_inputs = [
            (environment._dockerfile_path, environment_dir, (), None)
        ]
    for dockerfile, context, build_args, target in build_inputs:
        cache_path = await environment._dockerfile_cache_path(
            dockerfile,
            context,
            build_args,
            target,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            shutil.copy2(image, cache_path)
    try:
        await environment.start(force_build=False)
    except BaseException:
        await _cleanup_environment(environment)
        raise
    return environment


def _force_stop(
    environment: DockerfileSingularityEnvironment,
    known_instances: set[str] | None = None,
) -> None:
    instance_names = set(known_instances or ()) | {
        instance
        for instances in environment._compose_instances.values()
        for instance in instances
    }
    instance_names.add(environment._instance_name)
    for instance_name in instance_names:
        try:
            subprocess.run(
                [
                    environment._instance_runtime(),
                    "instance",
                    "stop",
                    "--force",
                    instance_name,
                ],
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            continue


def _require_fakeroot(image: Path) -> None:
    result = subprocess.run(
        [SINGULARITY, "exec", "--fakeroot", str(image), "true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Singularity fakeroot is unavailable: {result.stderr.strip()}"
        )


async def _cleanup_environment(
    environment: DockerfileSingularityEnvironment,
    known_instances: set[str] | None = None,
) -> None:
    try:
        await environment.stop(delete=False)
    finally:
        _force_stop(environment, known_instances)
        environment._cleanup_compose_project()
        environment._cleanup_staging()


@pytest.fixture
async def environment(tmp_path: Path, ubuntu_sif: Path):
    running = await _start_environment(tmp_path, ubuntu_sif)
    try:
        yield running
    finally:
        await _cleanup_environment(running)


async def test_real_instance_lifecycle_and_file_transfer(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
):
    result = await environment.exec(
        'printf \'%s:%s\' "$PERSISTED_VALUE" "$PER_CALL_VALUE"',
        env={"PER_CALL_VALUE": "per-call"},
    )

    assert result.return_code == 0
    assert result.stdout == "from-environment:per-call"

    written = await environment.exec("printf container-data > result.txt")
    assert written.return_code == 0
    assert (tmp_path / "workspace" / "result.txt").read_text() == (
        "container-data"
    )

    source = tmp_path / "source with spaces.txt"
    source.write_text("round trip")
    await environment.upload_file(source, "/workspace/uploaded with spaces.txt")
    downloaded = tmp_path / "downloaded with spaces.txt"
    await environment.download_file(
        "/workspace/uploaded with spaces.txt", downloaded
    )
    assert downloaded.read_text() == "round trip"

    scratch = environment._scratch_dir
    await environment.stop(delete=False)
    assert scratch is not None and not scratch.exists()
    stopped = subprocess.run(
        [
            environment._instance_runtime(),
            "exec",
            f"instance://{environment._instance_name}",
            "true",
        ],
        capture_output=True,
        timeout=10,
    )
    assert stopped.returncode != 0
    with pytest.raises(RuntimeError, match="not started"):
        await environment.exec("true")


async def test_cached_sif_is_touched_and_marked_until_its_last_user_stops(
    tmp_path: Path,
    ubuntu_sif: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cache = tmp_path / "shared-sif-cache"
    old_image = tmp_path / "old-ubuntu.sif"
    shutil.copy2(ubuntu_sif, old_image)
    os.utime(old_image, (1, 1))
    first = await _start_environment(
        tmp_path / "first",
        old_image,
        image_cache_dir=cache,
    )
    second = None
    try:
        sif = first._sif_path
        assert sif is not None
        marker = first._sif_in_use_path(sif)
        assert sif.stat().st_mtime_ns > old_image.stat().st_mtime_ns
        assert marker.is_file()
        with monkeypatch.context() as patch:
            patch.setattr(
                "ursa.integrations.harbor_singularity.archspec.cpu.host",
                lambda: SimpleNamespace(name="different-microarchitecture"),
            )
            other_microarchitecture = await first._dockerfile_cache_path()
        assert other_microarchitecture != sif

        second = await _start_environment(
            tmp_path / "second",
            old_image,
            image_cache_dir=cache,
        )
        assert second._sif_path == sif

        await first.stop(delete=False)
        assert marker.is_file()

        await second.stop(delete=False)
        assert not marker.exists()
    finally:
        if second is not None:
            await _cleanup_environment(second)
        await _cleanup_environment(first)


async def test_timeout_kills_the_command_but_keeps_instance_usable(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
):
    result = await environment.exec(
        "sleep 30 & child=$!; printf '%s' \"$child\" > timeout.pid; "
        'wait "$child"',
        timeout_sec=1,
    )

    assert result.return_code == 124
    assert "timed out" in result.stderr
    child_pid = (tmp_path / "workspace" / "timeout.pid").read_text()
    reaped = await environment.exec(f"kill -0 {child_pid}")
    assert reaped.return_code != 0
    follow_up = await environment.exec("printf still-running")
    assert follow_up.return_code == 0
    assert follow_up.stdout == "still-running"


class _LoopbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"local-only"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        pass


async def test_no_network_policy_is_enforced_by_the_runtime(
    tmp_path: Path,
    ubuntu_sif: Path,
):
    _require_fakeroot(ubuntu_sif)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    public = isolated = None
    try:
        public = await _start_environment(tmp_path / "public", ubuntu_sif)
        isolated = await _start_environment(
            tmp_path / "isolated",
            ubuntu_sif,
            fakeroot=True,
            network_mode=NetworkMode.NO_NETWORK,
        )

        request = (
            f'bash -c "exec 3<>/dev/tcp/127.0.0.1/{port}; '
            "printf 'GET / HTTP/1.0\\r\\n\\r\\n' >&3; cat <&3\""
        )
        reachable = await public.exec(request, timeout_sec=5)
        blocked = await isolated.exec(request, timeout_sec=5)

        assert reachable.return_code == 0
        assert "local-only" in reachable.stdout
        assert blocked.return_code != 0
    finally:
        if isolated is not None:
            await _cleanup_environment(isolated)
        if public is not None:
            await _cleanup_environment(public)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def test_fakeroot_overlay_provides_a_writable_container_root(
    tmp_path: Path,
    ubuntu_sif: Path,
):
    _require_fakeroot(ubuntu_sif)
    environment = await _start_environment(
        tmp_path,
        ubuntu_sif,
        fakeroot=True,
        overlay=True,
    )
    try:
        result = await environment.exec(
            "mkdir /created-at-runtime && printf overlay > "
            "/created-at-runtime/value"
        )
        read_back = await environment.exec("cat /created-at-runtime/value")

        assert result.return_code == 0
        assert read_back.stdout == "overlay"
    finally:
        await _cleanup_environment(environment)


async def test_compose_sidecar_lifecycle_and_file_transfer(
    tmp_path: Path,
    ubuntu_sif: Path,
):
    _require_fakeroot(ubuntu_sif)
    environment = await _start_environment(
        tmp_path,
        ubuntu_sif,
        fakeroot=True,
        compose=True,
    )
    instances: set[str] = set()
    try:
        cached_sifs = list((tmp_path / "sif-cache").glob("*.sif"))
        assert len(cached_sifs) == 2
        assert all(
            environment._sif_in_use_path(path).is_file() for path in cached_sifs
        )

        main = await environment.service_exec(
            "printf main-service",
            service="main",
        )
        sidecar = await environment.service_exec(
            "printf sidecar-service",
            service="helper",
            cwd="/",
        )
        await environment.service_exec(
            "printf sidecar-file > /tmp/result",
            service="helper",
            cwd="/",
        )
        main_is_isolated = await environment.service_exec(
            "test ! -e /tmp/result",
            service="main",
            cwd="/",
        )
        downloaded = tmp_path / "sidecar-result"
        await environment.service_download_file(
            "/tmp/result",
            downloaded,
            service="helper",
        )

        assert main.stdout == "main-service"
        assert sidecar.stdout == "sidecar-service"
        assert main_is_isolated.return_code == 0
        assert downloaded.read_text() == "sidecar-file"

        instances = {
            instance
            for names in environment._compose_instances.values()
            for instance in names
        }
        await environment.stop(delete=False)
        assert all(
            not environment._sif_in_use_path(path).exists()
            for path in cached_sifs
        )
        for instance in instances:
            probe = subprocess.run(
                [
                    environment._instance_runtime(),
                    "exec",
                    f"instance://{instance}",
                    "true",
                ],
                capture_output=True,
                timeout=10,
            )
            assert probe.returncode != 0
    finally:
        await _cleanup_environment(environment, instances)


@pytest.mark.parametrize("fakeroot", [True, False])
@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="requires a genuinely unprivileged host user",
)
async def test_agent_install_works_in_an_unprivileged_instance(
    tmp_path: Path,
    ubuntu_sif: Path,
    ursa_wheel: Path,
    monkeypatch: pytest.MonkeyPatch,
    fakeroot: bool,
):
    monkeypatch.setenv("OPENAI_API_KEY", "host-only-secret")
    if fakeroot:
        _require_fakeroot(ubuntu_sif)
    environment = await _start_environment(
        tmp_path,
        ubuntu_sif,
        fakeroot=fakeroot,
    )
    config = tmp_path / "ursa.yaml"
    config.write_text("llm_model:\n  model: gpt-4.1-nano\n")
    agent = UrsaHarborAgent(
        logs_dir=tmp_path / "agent-logs",
        model_name="openai/gpt-4.1-nano",
        config_file=config,
        ursa_install_spec=ursa_wheel,
    )

    try:
        immutable_root = await environment.exec("touch /outside-writable-binds")
        assert immutable_root.return_code != 0
        identity = await environment.exec("id -u")
        assert identity.stdout.strip() == (
            "0" if fakeroot else str(os.geteuid())
        )

        await agent.install(environment)

        installed = await environment.exec(
            "test -x /installed-agent/bin/ursa-harbor-runner && "
            "test -x /installed-agent/tools/ursa-ai/bin/python"
        )
        cli_result = await environment.exec(
            "/installed-agent/bin/ursa --help", timeout_sec=30
        )
        runner_result = await environment.exec(
            "/installed-agent/bin/ursa-harbor-runner --help", timeout_sec=30
        )
        config_result = await environment.exec(
            "cat /installed-agent/tmp/ursa-config.json"
        )

        assert installed.return_code == 0
        assert cli_result.return_code == 0
        assert "usage: ursa" in cli_result.stdout
        assert runner_result.return_code == 0
        assert "Run the container-side Harbor agent process" in (
            runner_result.stdout
        )
        runtime_config = json.loads(config_result.stdout)
        assert runtime_config["llm_model"]["model"] == "gpt-4.1-nano"
        assert "host-only-secret" not in config_result.stdout
        assert agent._workspace == "/workspace"
    finally:
        await _cleanup_environment(environment)


async def test_cancelling_exec_reaps_the_container_process(
    environment: DockerfileSingularityEnvironment,
    tmp_path: Path,
):
    pid_file = tmp_path / "workspace" / "cancel.pid"
    command = asyncio.create_task(
        environment.exec(
            "sleep 30 & child=$!; printf '%s' \"$child\" > cancel.pid; "
            'wait "$child"'
        )
    )
    for _ in range(500):
        if pid_file.is_file():
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("container command did not publish its child PID")
    command.cancel()

    with pytest.raises(asyncio.CancelledError):
        await command
    child_pid = pid_file.read_text()
    reaped = await environment.exec(f"kill -0 {child_pid}")
    assert reaped.return_code != 0
    follow_up = await environment.exec("printf recovered")
    assert follow_up.return_code == 0
    assert follow_up.stdout == "recovered"
