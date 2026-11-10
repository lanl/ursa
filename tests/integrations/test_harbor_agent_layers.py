import asyncio
import sys
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.goose import Goose as UpstreamGoose

from tests.test_harbor_singularity import _environment
from ursa.integrations.harbor import agent_layers
from ursa.integrations.harbor.installed_agents import Goose

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux install layers"
)


@pytest.fixture
def template(tmp_path):
    environment = _environment(tmp_path)
    environment._agent_cache_dir = tmp_path / "agents"
    environment._sif_path = tmp_path / "base.sif"
    environment._sif_path.write_bytes(b"base image")
    return environment


@pytest.fixture
def builder(monkeypatch):
    environments, events = [], []

    async def start(self, force_build):
        assert not force_build
        assert not self._mounts
        assert not self._harbor_writable_binds()
        assert not self._enable_environment_dir_upload
        assert not self._uses_compose
        assert self.task_env_config.storage_mb >= 4096
        self._overlay_path = self.trial_paths.trial_dir / "overlay.img"
        self._overlay_path.write_bytes(b"fresh overlay")
        environments.append(self)
        events.append("start")

    async def setup(self, environment):
        assert self.model_name is None
        assert not self._extra_env
        events.append("install")
        await asyncio.sleep(0.01)
        environment._overlay_path.write_bytes(b"installed agent")

    async def unmount(self, warn):
        assert warn is False
        events.append("unmount")
        return True

    async def stop(self, delete):
        assert delete is True
        events.append("cleanup")

    monkeypatch.setattr(agent_layers, "_layer_key", lambda *args: "test-key")
    monkeypatch.setattr(agent_layers._InstallEnvironment, "start", start)
    monkeypatch.setattr(
        agent_layers._InstallEnvironment, "_stop_prepared_runtime", unmount
    )
    monkeypatch.setattr(agent_layers._InstallEnvironment, "stop", stop)
    monkeypatch.setattr(BaseInstalledAgent, "setup", setup)
    return environments, events


async def test_concurrent_prebuilds_publish_once_and_reuse_complete_layer(
    template, builder
):
    layers = await asyncio.gather(
        *(agent_layers.build_agent_layer(template, "goose") for _ in range(3))
    )
    environments, events = builder
    assert len(set(layers)) == 1
    assert layers[0].read_bytes() == b"installed agent"
    assert layers[0].stat().st_mode & 0o777 == 0o400
    assert events == ["start", "install", "unmount", "cleanup"]
    assert not environments[0].trial_paths.trial_dir.exists()
    assert template.task_env_config.storage_mb != 4096
    pinned = await agent_layers.build_agent_layer(
        template, "goose", require_cached=True
    )
    assert pinned.samefile(layers[0])
    replacement = layers[0].with_suffix(".new")
    replacement.write_bytes(b"new installation")
    replacement.replace(layers[0])
    assert pinned.read_bytes() == b"installed agent"
    template._agent_layer_path = pinned
    template._cleanup_staging()
    assert not pinned.exists()
    assert layers[0].read_bytes() == b"new installation"
    assert events.count("install") == 1


@pytest.mark.parametrize(
    "failure", ["start", "install", "timeout", "cancel", "unmount", "copy"]
)
async def test_failed_refresh_retains_previous_layer_and_cleans_up(
    template, builder, monkeypatch, failure
):
    previous = await agent_layers.build_agent_layer(template, "goose")
    previous_bytes = previous.read_bytes()

    async def fail(*args, **kwargs):
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "cancel":
            raise asyncio.CancelledError
        raise RuntimeError("expected failure")

    target = {
        "start": "start",
        "unmount": "_stop_prepared_runtime",
        "copy": "_run",
    }.get(failure)
    if target:
        monkeypatch.setattr(agent_layers._InstallEnvironment, target, fail)
    else:
        monkeypatch.setattr(BaseInstalledAgent, "setup", fail)
    error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(error):
        await agent_layers.build_agent_layer(
            template, "goose", force=True, timeout_sec=0.01
        )
    assert previous.read_bytes() == previous_bytes
    assert len(list(template._agent_cache_dir.glob("*.img"))) == 1
    assert not list(template._agent_cache_dir.glob(".*.img"))
    assert builder[1][-1] == "cleanup"
    assert all(not env.trial_paths.trial_dir.exists() for env in builder[0])


async def test_missing_layer_requires_explicit_prebuild(template, builder):
    with pytest.raises(FileNotFoundError, match="prebuild --agent goose"):
        await agent_layers.build_agent_layer(
            template, "goose", require_cached=True
        )
    assert not builder[1]


def test_layer_identity_tracks_image_agent_version_and_fakeroot(
    template, monkeypatch
):
    monkeypatch.setattr(
        agent_layers.subprocess,
        "check_output",
        lambda *a, **k: "apptainer 1.5.5",
    )
    base = template._sif_path
    key = agent_layers._layer_key(template, base, UpstreamGoose, None)
    assert key != agent_layers._layer_key(
        template, base, UpstreamGoose, "v1.2.3"
    )
    template._fakeroot = True
    assert key != agent_layers._layer_key(template, base, UpstreamGoose, None)
    template._fakeroot = False
    base.write_bytes(b"changed image contents")
    assert key != agent_layers._layer_key(template, base, UpstreamGoose, None)


async def test_agent_adapter_skips_install_only_for_matching_selected_layer(
    template, monkeypatch, tmp_path
):
    install = AsyncMock()
    monkeypatch.setattr(UpstreamGoose, "install", install)
    agent = Goose(logs_dir=tmp_path)
    await agent.install(template)
    install.assert_awaited_once_with(template)
    install.reset_mock()
    template._agent_spec = "goose"
    template._agent_layer_path = tmp_path / "layer.img"
    await agent.install(template)
    agent._version = "detected-version"
    await agent.install(template)
    install.assert_not_awaited()
    template._agent_spec = "goose@v1.2.3"
    with pytest.raises(ValueError, match="does not match"):
        await agent.install(template)
    template._agent_spec = "goose"
    template._agent_layer_path = None
    with pytest.raises(RuntimeError, match="not loaded"):
        await agent.install(template)
