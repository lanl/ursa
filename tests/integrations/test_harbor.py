import asyncio
import json
import os
import shutil
import subprocess
import threading
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from filelock import FileLock
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths

from ursa.integrations.harbor import UrsaHarborAgent
from ursa.integrations.harbor.singularity import (
    DockerfileSingularityEnvironment,
)

SINGULARITY = shutil.which("apptainer") or shutil.which("singularity")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        SINGULARITY is None, reason="Apptainer or Singularity is required"
    ),
]


def _shared(factory, worker, name):
    root = (
        factory.getbasetemp()
        if worker == "master"
        else factory.getbasetemp().parent
    ) / name
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session")
def ubuntu_sif(tmp_path_factory, worker_id):
    image = (
        _shared(tmp_path_factory, worker_id, "harbor-ubuntu-sif")
        / "ubuntu-24.04.sif"
    )
    with FileLock(f"{image}.lock"):
        if not image.is_file():
            temporary = image.with_suffix(".tmp.sif")
            result = subprocess.run(
                [SINGULARITY, "build", str(temporary), "docker://ubuntu:24.04"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode:
                pytest.fail(
                    f"Could not build Ubuntu SIF:\n{result.stderr or result.stdout}"
                )
            temporary.replace(image)
    return image


@pytest.fixture(scope="session")
def ursa_wheel(tmp_path_factory, worker_id):
    uv = shutil.which("uv") or pytest.fail(
        "Building the URSA wheel requires uv"
    )
    root = _shared(tmp_path_factory, worker_id, "harbor-ursa-wheel")
    with FileLock(f"{root}.lock"):
        wheels = list(root.glob("ursa_ai-*.whl"))
        if not wheels:
            build = root / "build"
            build.mkdir(exist_ok=True)
            result = subprocess.run(
                [
                    uv,
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(build),
                    str(Path(__file__).resolve().parents[2]),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                pytest.fail(f"Could not build URSA wheel:\n{result.stderr}")
            wheels = list(build.glob("ursa_ai-*.whl"))
            if len(wheels) != 1:
                pytest.fail(f"Expected one URSA wheel, found: {wheels}")
            wheels[0].replace(root / wheels[0].name)
            wheels = list(root.glob("ursa_ai-*.whl"))
    if len(wheels) != 1:
        pytest.fail(f"Expected one URSA wheel, found: {wheels}")
    return wheels[0]


async def _start(
    root,
    image,
    *,
    fakeroot=False,
    network=NetworkMode.PUBLIC,
    phases=(),
    overlay=False,
    compose=False,
    sidecar=True,
    cache=None,
):
    env_dir, workspace = root / "environment", root / "workspace"
    env_dir.mkdir(parents=True)
    workspace.mkdir(mode=0o777)
    (env_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\n"
    )
    if compose:
        text = "services:\n  main:\n    build: .\n"
        if sidecar:
            (env_dir / "sidecar").mkdir()
            (env_dir / "sidecar/Dockerfile").write_text("FROM ubuntu:24.04\n")
            text += "  helper:\n    build: ./sidecar\n"
        (env_dir / "docker-compose.yaml").write_text(text)
    env = DockerfileSingularityEnvironment(
        environment_dir=env_dir,
        environment_name="local-test",
        session_id=f"{root.name}__env",
        trial_paths=TrialPaths(root / "trial"),
        task_env_config=EnvironmentConfig(workdir="/workspace", storage_mb=64),
        mounts=[
            {
                "type": "bind",
                "source": str(workspace),
                "target": "/workspace",
                "read_only": False,
            }
        ],
        network_policy=NetworkPolicy(network_mode=network),
        phase_network_policies=[
            NetworkPolicy(network_mode=mode) for mode in phases
        ],
        persistent_env={"PERSISTED_VALUE": "from-environment"},
        singularity_fakeroot=fakeroot,
        singularity_overlay=overlay,
        singularity_image_cache_dir=cache or root / "sif-cache",
        singularity_startup_timeout_sec=30,
    )
    env.trial_paths.mkdir()
    builds = (
        env._load_compose_config()["services"].values()
        if compose
        else ({"build": "."},)
    )
    for service in builds:
        args = (
            env._compose_build_inputs(service["build"])
            if compose
            else (env._dockerfile_path, env_dir, (), None)
        )
        path = await env._dockerfile_cache_path(*args)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copy2(image, path)
    try:
        await env.start(force_build=False)
    except BaseException:
        await _cleanup(env)
        raise
    return env


def _force_stop(env, names=()):
    names = (
        set(names)
        | {env._instance_name}
        | {n for values in env._compose_instances.values() for n in values}
    )
    for name in names:
        try:
            subprocess.run(
                [env._instance_runtime(), "instance", "stop", "--force", name],
                capture_output=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            pass


async def _cleanup(env, names=()):
    try:
        await env.stop(delete=False)
    finally:
        _force_stop(env, names)
        env._cleanup_compose_project()
        env._cleanup_staging()


@asynccontextmanager
async def _environment(root, image, **kwargs):
    env = await _start(root, image, **kwargs)
    try:
        yield env
    finally:
        await _cleanup(env)


@pytest.fixture
async def environment(tmp_path, ubuntu_sif):
    async with _environment(tmp_path, ubuntu_sif) as env:
        yield env


def _require_fakeroot(image):
    result = subprocess.run(
        [SINGULARITY, "exec", "--fakeroot", str(image), "true"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        pytest.skip(
            f"Singularity fakeroot is unavailable: {result.stderr.strip()}"
        )


class _Loopback(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"local-only"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        pass


@contextmanager
def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Loopback)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(port):
    return f"bash -c \"exec 3<>/dev/tcp/127.0.0.1/{port}; printf 'GET / HTTP/1.0\\r\\n\\r\\n' >&3; cat <&3\""


async def _assert_reaped(env, pid, text):
    assert (await env.exec(f"kill -0 {pid}")).return_code != 0
    result = await env.exec(f"printf {text}")
    assert (result.return_code, result.stdout) == (0, text)


async def _assert_persistent_after_isolation(env):
    assert (
        await env.exec(
            "mkdir /installed && printf persisted > /installed/value"
        )
    ).return_code == 0
    await env.set_network_policy(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    assert (await env.exec("cat /installed/value")).stdout == "persisted"


def _assert_instances_stopped(env, names):
    for name in names:
        assert subprocess.run(
            [env._instance_runtime(), "exec", f"instance://{name}", "true"],
            capture_output=True,
            timeout=10,
        ).returncode


async def test_lifecycle_transfer_capacity_and_stop(environment, tmp_path):
    assert (
        await environment.exec(
            'printf \'%s:%s\' "$PERSISTED_VALUE" "$PER_CALL_VALUE"',
            env={"PER_CALL_VALUE": "per-call"},
        )
    ).stdout == "from-environment:per-call"
    assert (
        await environment.exec("printf container-data > result.txt")
    ).return_code == 0
    assert (tmp_path / "workspace/result.txt").read_text() == "container-data"
    source, destination = (
        tmp_path / "source with spaces.txt",
        tmp_path / "downloaded with spaces.txt",
    )
    source.write_text("round trip")
    await environment.upload_file(source, "/workspace/uploaded with spaces.txt")
    await environment.download_file(
        "/workspace/uploaded with spaces.txt", destination
    )
    assert destination.read_text() == "round trip"
    assert (
        await environment.exec(
            "dd if=/dev/zero of=/tmp/ursa-workdir-capacity bs=1M count=65 status=none"
        )
    ).return_code == 0
    runtime, scratch = (
        environment._runtime_workdir_root,
        environment._scratch_dir,
    )
    assert (
        runtime
        and (runtime / "main/tmp/ursa-workdir-capacity").stat().st_size
        == 65 * 1024 * 1024
    )
    await environment.stop(delete=False)
    assert scratch and not scratch.exists() and not runtime.exists()
    _assert_instances_stopped(environment, [environment._instance_name])
    with pytest.raises(RuntimeError, match="not started"):
        await environment.exec("true")


async def test_timeout_reaps_process_and_preserves_instance(
    environment, tmp_path
):
    result = await environment.exec(
        'sleep 30 & child=$!; printf \'%s\' "$child" > timeout.pid; wait "$child"',
        timeout_sec=1,
    )
    assert result.return_code == 124 and "timed out" in result.stderr
    await _assert_reaped(
        environment,
        (tmp_path / "workspace/timeout.pid").read_text(),
        "still-running",
    )


async def test_network_policy_blocks_loopback(tmp_path, ubuntu_sif):
    _require_fakeroot(ubuntu_sif)
    with _server() as port:
        async with _environment(tmp_path / "public", ubuntu_sif) as public:
            async with _environment(
                tmp_path / "isolated",
                ubuntu_sif,
                fakeroot=True,
                network=NetworkMode.NO_NETWORK,
            ) as isolated:
                reachable = await public.exec(_request(port), timeout_sec=5)
                blocked = await isolated.exec(_request(port), timeout_sec=5)
                assert reachable.return_code == 0
                assert "local-only" in reachable.stdout
                assert blocked.return_code != 0


async def test_host_ca_mount_is_read_only(tmp_path, ubuntu_sif, monkeypatch):
    monkeypatch.setattr(
        "ursa.integrations.harbor.singularity._host_ca_certificates_pem",
        lambda: "test host CA bundle\n",
    )
    async with _environment(tmp_path, ubuntu_sif) as env:
        result = await env.exec(
            'printf \'%s\\n\' "$SSL_CERT_FILE" "$REQUESTS_CA_BUNDLE" "$CURL_CA_BUNDLE"; cat "$SSL_CERT_FILE"'
        )
        assert (
            result.stdout
            == "/staging/host-ca-certificates.crt\n" * 3
            + "test host CA bundle\n"
        )
        assert (
            await env.exec('printf tampered >> "$SSL_CERT_FILE"')
        ).return_code != 0


async def test_compose_overlay_survives_network_transition(
    tmp_path, ubuntu_sif
):
    _require_fakeroot(ubuntu_sif)
    async with _environment(
        tmp_path,
        ubuntu_sif,
        fakeroot=True,
        phases=(NetworkMode.NO_NETWORK,),
        overlay=True,
        compose=True,
        sidecar=False,
    ) as env:
        await _assert_persistent_after_isolation(env)


async def test_network_transition_restores_connectivity(tmp_path, ubuntu_sif):
    _require_fakeroot(ubuntu_sif)
    with _server() as port:
        async with _environment(
            tmp_path,
            ubuntu_sif,
            fakeroot=True,
            phases=(NetworkMode.NO_NETWORK,),
            overlay=True,
        ) as env:
            assert (
                await env.exec(_request(port), timeout_sec=5)
            ).return_code == 0
            await _assert_persistent_after_isolation(env)
            assert (
                await env.exec(_request(port), timeout_sec=5)
            ).return_code != 0
            await env.set_network_policy(
                NetworkPolicy(network_mode=NetworkMode.PUBLIC)
            )
            assert (
                "local-only"
                in (await env.exec(_request(port), timeout_sec=5)).stdout
            )


async def test_compose_sidecar_isolation_transfer_and_cleanup(
    tmp_path, ubuntu_sif
):
    _require_fakeroot(ubuntu_sif)
    env = await _start(tmp_path, ubuntu_sif, fakeroot=True, compose=True)
    names = set()
    try:
        sifs = list((tmp_path / "sif-cache").glob("*.sif"))
        assert len(sifs) == 2
        assert (
            await env.service_exec("printf main-service", service="main")
        ).stdout == "main-service"
        assert (
            await env.service_exec(
                "printf sidecar-service", service="helper", cwd="/"
            )
        ).stdout == "sidecar-service"
        await env.service_exec(
            "printf sidecar-file > /tmp/result", service="helper", cwd="/"
        )
        assert (
            await env.service_exec(
                "test ! -e /tmp/result", service="main", cwd="/"
            )
        ).return_code == 0
        output = tmp_path / "sidecar-result"
        await env.service_download_file("/tmp/result", output, service="helper")
        assert output.read_text() == "sidecar-file"
        names = {
            name
            for values in env._compose_instances.values()
            for name in values
        }
        await env.stop(delete=False)
        assert all(not env._sif_in_use_path(path).exists() for path in sifs)
        _assert_instances_stopped(env, names)
    finally:
        await _cleanup(env, names)


@pytest.mark.parametrize("fakeroot", [None, True, False])
@pytest.mark.skipif(
    os.geteuid() == 0, reason="requires a genuinely unprivileged host user"
)
async def test_agent_install_unprivileged_instance(
    tmp_path, ubuntu_sif, ursa_wheel, monkeypatch, fakeroot
):
    monkeypatch.setenv("OPENAI_API_KEY", "host-only-secret")
    enabled = (
        Path(SINGULARITY).name == "apptainer" if fakeroot is None else fakeroot
    )
    if enabled:
        _require_fakeroot(ubuntu_sif)
    async with _environment(tmp_path, ubuntu_sif, fakeroot=fakeroot) as env:
        config = tmp_path / "ursa.yaml"
        config.write_text("llm_model:\n  model: gpt-4.1-nano\n")
        agent = UrsaHarborAgent(
            logs_dir=tmp_path / "agent-logs",
            model_name="openai/gpt-4.1-nano",
            config=config,
            ursa_install_spec=f"ursa-ai @ {ursa_wheel.as_uri()}",
        )
        assert (
            await env.exec("touch /outside-writable-binds")
        ).return_code != 0
        assert (await env.exec("id -u")).stdout.strip() == (
            "0" if enabled else str(os.geteuid())
        )
        await agent.install(env)
        assert (
            await env.exec(
                "test -x /installed-agent/bin/ursa-harbor-runner && test -x /installed-agent/tools/ursa-ai/bin/python"
            )
        ).return_code == 0
        assert (
            "usage: ursa"
            in (
                await env.exec(
                    "/installed-agent/bin/ursa --help", timeout_sec=30
                )
            ).stdout
        )
        assert (
            "Run the container-side Harbor agent process"
            in (
                await env.exec(
                    "/installed-agent/bin/ursa-harbor-runner --help",
                    timeout_sec=30,
                )
            ).stdout
        )
        output = (
            await env.exec("cat /installed-agent/tmp/ursa-config.json")
        ).stdout
        assert (
            json.loads(output)["llm_model"]["model"] == "gpt-4.1-nano"
            and "host-only-secret" not in output
            and agent._workspace == "/workspace"
        )


async def test_cancelling_exec_reaps_process(environment, tmp_path):
    path = tmp_path / "workspace/cancel.pid"
    command = asyncio.create_task(
        environment.exec(
            'sleep 30 & child=$!; printf \'%s\' "$child" > cancel.pid; wait "$child"'
        )
    )
    for _ in range(500):
        if path.is_file():
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("container command did not publish its child PID")
    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command
    await _assert_reaped(environment, path.read_text(), "recovered")


def test_retention_plugin_through_harbor_job(tmp_path, ubuntu_sif, monkeypatch):
    from harbor.cli.main import app
    from typer.testing import CliRunner

    task = tmp_path / "task"
    for directory in ("environment", "solution", "tests"):
        (task / directory).mkdir(parents=True)
    (task / "instruction.md").write_text("Create the requested artifacts.\n")
    (task / "task.toml").write_text(
        '[environment]\nnetwork_mode = "public"\n'
        "[agent]\ntimeout_sec = 30\n"
        "[verifier]\ntimeout_sec = 30\n"
    )
    (task / "environment/Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "solution/solve.sh").write_text(
        "#!/bin/bash\nset -e\n"
        "echo payload > /logs/artifacts/data.txt\n"
        "echo discard > /logs/artifacts/scratch.discard\n"
    )
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\nset -e\n"
        "test -f /logs/artifacts/scratch.discard\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    policy = tmp_path / "retention.yaml"
    policy.write_text(
        'rules:\n  - match: ["**/*.discard"]\n    action: delete\n'
        "default_action: archive\narchive: {}\n"
    )
    monkeypatch.setattr(
        DockerfileSingularityEnvironment,
        "_build_main_sif",
        AsyncMock(return_value=ubuntu_sif),
    )
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--path",
            str(task),
            "--agent",
            "oracle",
            "--env",
            "ursa.integrations.harbor.singularity:DockerfileSingularityEnvironment",
            "--environment-kwarg",
            "singularity_fakeroot=false",
            "--environment-kwarg",
            "singularity_overlay=false",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "retention",
            "--plugin",
            "ursa.integrations.harbor.retention:RetentionPlugin",
            "--plugin-kwarg",
            f"policy={policy}",
            "--quiet",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    root = tmp_path / "jobs/retention"
    status = json.loads((root / "retention-status.json").read_text())
    assert status["status"] == "complete", status
    trial = next(root.glob("*/result.json"))
    detail = json.loads(trial.read_text())
    assert detail["exception_info"] is None, detail
    assert detail["verifier_result"]["rewards"] == {"reward": 1.0}
    assert not list(trial.parent.rglob("scratch.discard"))
    assert not list(trial.parent.rglob("data.txt"))
    assert (root / "retained-artifacts.tar.xz").is_file()
    assert (trial.parent / "verifier/reward.txt").read_text() == "1\n"
