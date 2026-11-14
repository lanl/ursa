"""Exercise the same agent protocol across threads, pools, and launch jobs."""

import asyncio
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest

from ursa.cli.config import UrsaConfig
from ursa.runtime import (
    AgentInvocationError,
    AgentManager,
    AgentProcessExitedError,
    AgentRuntime,
    AgentSpec,
    AgentStatus,
    RuntimeConfig,
)
from ursa.runtime.worker import authenticate

FACTORY = "tests.runtime.fixtures:stateful_factory"


class OpaqueAgent:
    def __init__(self, config, context):
        self.calls = 0

    async def ainvoke(self, inputs, config=None):
        self.calls += 1
        return self.calls


@pytest.fixture(params=["thread", "pool"])
async def manager(request):
    async with AgentManager(
        runtime=RuntimeConfig(backend=request.param, rebalance_interval=60),
        heartbeat_interval=0.05,
        heartbeat_timeout=2,
        startup_timeout=10,
    ) as manager:
        yield manager


async def register(manager, count):
    for i in range(count):
        await manager.register(AgentSpec(str(i), FACTORY, {"label": str(i)}))
    return await asyncio.gather(
        *(manager.invoke(str(i), {}) for i in range(count))
    )


async def test_shared_worker_threads_preserve_state_messages_callbacks(
    manager, capfd
):
    results = await register(manager, 4)
    assert [r["label"] for r in results] == [str(i) for i in range(4)]
    assert len({r["pid"] for r in results}) == 1
    assert (results[0]["pid"] == os.getpid()) == (
        manager.runtime.backend == "thread"
    )
    await manager.send_message("1", "user input", sender="user")
    callback = type("Callback", (), {"on_custom_event": AsyncMock()})()
    await manager.invoke(
        "0", {"send": "1", "payload": "peer input"}, callbacks=[callback]
    )
    assert await manager.call("1", "received") == [
        {"sender": "user", "payload": "user input"},
        {"sender": "0", "payload": "peer input"},
    ]
    callback.on_custom_event.assert_awaited_once()
    await manager.restart_agent("0")
    assert (await manager.invoke("0", {}))["calls"] == 3
    assert len(manager.worker_snapshots()) == 1
    await manager.register(
        AgentSpec("output", "tests.runtime.fixtures:output_factory")
    )
    assert await manager.invoke("output", {}) == "ok"
    assert "agent stdout marker" not in capfd.readouterr().out


async def test_stop_is_prompt_during_work_and_does_not_stop_other_agents(
    manager,
):
    await register(manager, 2)
    pending = asyncio.create_task(
        manager.invoke("0", {"announce": "host", "sleep": 30})
    )
    await manager.receive_message(timeout=2)
    await asyncio.wait_for(manager.kill_agent("0", force=True), 1)
    with pytest.raises(AgentProcessExitedError):
        await pending
    assert (await manager.invoke("1", {}))["calls"] == 2
    assert (await manager.invoke("0", {}))["calls"] == 2


async def test_factory_failure_is_local_to_one_agent(manager):
    await register(manager, 1)
    await manager.register(AgentSpec("bad", "does_not_exist:factory"))
    with pytest.raises(AgentInvocationError, match="ModuleNotFoundError"):
        await manager.start_agent("bad")
    assert (await manager.invoke("0", {}))["calls"] == 2


async def test_threads_reject_process_global_overrides(manager):
    await manager.register(
        AgentSpec("bad", FACTORY, environment={"ONLY": "this"})
    )
    with pytest.raises(ValueError, match="process backend"):
        await manager.start_agent("bad")


async def test_unserializable_request_does_not_take_down_shared_worker(manager):
    await register(manager, 2)
    with pytest.raises((AttributeError, TypeError)):
        await manager.invoke("0", lambda: None)
    assert (await manager.invoke("0", {}))["calls"] == 2
    assert (await manager.invoke("1", {}))["calls"] == 2


@pytest.mark.parametrize("backend", ["process", "thread", "pool"])
async def test_receivers_do_not_starve_command_and_shutdown_threads(backend):
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(executor)
    try:
        async with AgentManager(
            runtime=RuntimeConfig(backend=backend)
        ) as manager:
            try:
                results = await asyncio.wait_for(register(manager, 6), 10)
                assert len(results) == 6
            finally:
                for entry in manager._agents.values():
                    if entry.process is not None:
                        entry.process.terminate()
    finally:
        loop.set_default_executor(previous or ThreadPoolExecutor())
        executor.shutdown(wait=True)


async def test_elastic_rebalance_migrates_idle_state_without_replaying_busy_work():
    config = RuntimeConfig(
        backend="pool",
        max_workers=3,
        agents_per_worker=2,
        rebalance_interval=60,
        idle_timeout=0,
    )
    async with AgentManager(runtime=config, startup_timeout=10) as manager:
        await register(manager, 6)
        pending = asyncio.create_task(
            manager.invoke("0", {"announce": "host", "sleep": 5})
        )
        await manager.receive_message(timeout=2)
        assert not pending.done()
        original = (await manager.snapshot("0")).pid
        await manager._pool.rebalance()
        assert (await manager.snapshot("0")).pid == original
        assert (await pending)["calls"] == 2
        assert sorted(len(w.agents) for w in manager.worker_snapshots()) == [
            2,
            2,
            2,
        ]
        for i in range(1, 6):
            assert (await manager.invoke(str(i), {}))["calls"] == 2
        for i in range(1, 6):
            await manager.unregister(str(i), force=True)
        await manager._pool.rebalance()
        assert len(manager.worker_snapshots()) == 1
        assert (await manager.invoke("0", {}))["calls"] == 3
        for i in range(1, 6):
            await manager.register(AgentSpec(str(i), FACTORY))
        await asyncio.gather(*(manager.invoke(str(i), {}) for i in range(1, 6)))
        await manager._pool.rebalance()
        assert len(manager.worker_snapshots()) == 3
        assert sum(p.returncode is None for p in manager._pool.processes) <= 3


async def test_periodic_elastic_growth_and_worker_loss():
    config = RuntimeConfig(
        backend="pool",
        max_workers=2,
        agents_per_worker=1,
        rebalance_interval=0.1,
    )
    async with AgentManager(runtime=config, startup_timeout=10) as manager:
        await register(manager, 4)
        async with asyncio.timeout(10):
            while len(manager.worker_snapshots()) != 2:
                await asyncio.sleep(0.02)
        process = manager._pool.processes[0]
        process.kill()
        await process.wait()
        async with asyncio.timeout(5):
            while not any(
                entry.status is AgentStatus.FAILED
                for entry in manager._agents.values()
            ):
                await asyncio.sleep(0.02)
        assert all(
            r["calls"] == 2
            for r in await asyncio.gather(
                *(manager.invoke(str(i), {}) for i in range(4))
            )
        )


async def test_rebalance_does_not_discard_unexported_agent_state():
    config = RuntimeConfig(
        backend="pool",
        max_workers=2,
        agents_per_worker=1,
        rebalance_interval=60,
    )
    async with AgentManager(runtime=config, startup_timeout=10) as manager:
        for index in range(3):
            await manager.register(
                AgentSpec(str(index), "tests.runtime.test_pool:OpaqueAgent")
            )
            assert await manager.invoke(str(index), {}) == 1
        original = [
            (await manager.snapshot(str(index))).pid for index in range(3)
        ]
        await manager._pool.rebalance()
        assert [
            (await manager.snapshot(str(index))).pid for index in range(3)
        ] == original
        assert await manager.invoke("0", {}) == 2


async def test_configured_thread_runtime_supports_real_hierarchy_tools(
    tmp_path, monkeypatch
):
    from tests.runtime.fixtures import spawn_managed

    monkeypatch.setattr(
        "ursa.runtime.agent_runtime.validate_model_provider", lambda *_: None
    )
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            runtime={"backend": "thread"},
        )
    )
    try:
        assert await runtime.run_agent("chat", "hello") == "ok"
        root = runtime.agents["chat"]
        await spawn_managed(runtime, "root/reviewer", "review this")
        assert (
            await runtime.run_active_agent("root/reviewer", "user follow-up")
            == "ok"
        )
        assert (
            await root.manager.call(
                root.runtime_id,
                "send_message",
                "root/reviewer",
                "peer follow-up",
            )
            == "Message queued for root/reviewer."
        )
        assert "Stopped agent root/reviewer" in await root.manager.call(
            root.runtime_id, "stop_subagent", "root/reviewer"
        )
        assert await runtime.run_active_agent("root", "still here") == "ok"
    finally:
        await runtime.aclose()


async def test_synchronous_prompt_formatting_keeps_control_loop_responsive(
    process_runner,
):
    runner = process_runner(result="ok")
    format_query = runner.agent.format_query
    release = threading.Event()

    def slow_format(*args, **kwargs):
        if not release.wait(0.5):
            raise RuntimeError("Prompt formatting blocked the control loop")
        return format_query(*args, **kwargs)

    runner.agent.format_query = slow_format
    timer = asyncio.get_running_loop().call_later(0.02, release.set)
    try:
        assert await asyncio.wait_for(runner.ainvoke("hello"), 1) == "ok"
    finally:
        release.set()
        timer.cancel()


@pytest.mark.parametrize("launcher_kind", ["python", "mpi", "tls"])
async def test_external_launcher_has_fixed_workers_and_variable_agents(
    tmp_path,
    launcher_kind,
):
    # A real subprocess launcher using the exact MPI/srun argv substitution.
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import subprocess, sys\n"
        "children = [subprocess.Popen([sys.executable, '-m', 'ursa.runtime.worker', '--rendezvous', sys.argv[2]]) for _ in range(int(sys.argv[1]))]\n"
        "for child in children: child.wait()\n"
    )
    config = RuntimeConfig(
        backend="pool",
        max_workers=2,
        launch_command=["{python}", str(launcher), "{workers}", "{rendezvous}"],
        rendezvous_dir=tmp_path,
    )
    if launcher_kind == "mpi":
        if not shutil.which("mpiexec"):
            pytest.skip("MPI launcher is not installed")
        config.launch_command = [
            "mpiexec",
            "-n",
            "{workers}",
            "{python}",
            "-m",
            "ursa.runtime.worker",
            "--rendezvous",
            "{rendezvous}",
        ]
    elif launcher_kind == "tls":
        if not shutil.which("openssl"):
            pytest.skip("openssl is needed to create a test certificate")
        certificate, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-config",
                os.devnull,
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1",
                "-keyout",
                str(key),
                "-out",
                str(certificate),
            ],
            check=True,
            capture_output=True,
        )
        config = RuntimeConfig.model_validate(
            config.model_dump()
            | {"tls_certificate": certificate, "tls_key": key}
        )
    async with AgentManager(runtime=config, startup_timeout=10) as manager:
        results = await register(manager, 8)
        assert len({r["pid"] for r in results}) == 2
        ids = {w.worker_id for w in manager.worker_snapshots()}
        for i in range(4):
            await manager.unregister(str(i), force=True)
        await manager._pool.rebalance()
        assert {w.worker_id for w in manager.worker_snapshots()} == ids
        path = manager._pool.rendezvous
        assert path.stat().st_mode & 0o777 == 0o600
        settings = json.loads(path.read_text())
        tls = (
            ssl.create_default_context(cafile=settings["certificate"])
            if settings.get("certificate")
            else None
        )
        reader, writer = await asyncio.open_connection(
            settings["host"], settings["port"], ssl=tls
        )
        with pytest.raises((asyncio.IncompleteReadError, PermissionError)):
            await authenticate(reader, writer, b"wrong key", server=False)
        writer.close()
        await writer.wait_closed()
        assert (await manager.invoke("7", {}))["calls"] == 2
    assert not path.exists()


async def test_launch_failure_does_not_hang(tmp_path):
    config = RuntimeConfig(
        backend="pool",
        launch_command=[sys.executable, "-c", "raise SystemExit(7)"],
        rendezvous_dir=tmp_path,
    )
    async with AgentManager(runtime=config, startup_timeout=1) as manager:
        await manager.register(AgentSpec("a", FACTORY))
        with pytest.raises(
            AgentProcessExitedError, match="launch command exited"
        ):
            await asyncio.wait_for(manager.start_agent("a"), 3)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "values",
    [
        {"backend": "invalid"},
        {"min_workers": 0},
        {"max_workers": True},
        {"min_workers": 3, "max_workers": 2},
        {"agents_per_worker": 0},
        {"rebalance_interval": 0},
        {"rebalance_interval": float("nan")},
        {"launch_command": ["srun"]},
        {"backend": "pool", "launch_command": []},
    ],
)
def test_invalid_runtime_config(values):
    with pytest.raises(ValueError):
        RuntimeConfig(**values)


def test_runtime_config_roundtrip_sparse_merge():
    base = UrsaConfig(
        runtime={
            "backend": "pool",
            "max_workers": 8,
            "launch_command": "srun -n {workers} {python} -m ursa.runtime.worker --rendezvous {rendezvous}",
        }
    )
    merged = base.model_merge({"runtime": {"agents_per_worker": 4}})
    assert merged.runtime.backend == "pool"
    assert merged.runtime.max_workers == 8
    assert merged.runtime.agents_per_worker == 4
    assert (
        UrsaConfig.model_validate_json(merged.model_dump_json()).runtime
        == merged.runtime
    )
