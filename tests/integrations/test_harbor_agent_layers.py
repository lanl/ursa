import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.goose import Goose as UpstreamGoose

from tests.test_harbor_singularity import _environment
from ursa.integrations.harbor import UrsaHarborAgent, agent_layers
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
    assert not pinned.samefile(layers[0])
    assert pinned.read_bytes() == b"installed agent"
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


async def test_runtime_mount_metadata_cannot_modify_cached_layer(
    template, builder
):
    layer = await agent_layers.build_agent_layer(template, "goose")
    snapshot = await agent_layers.build_agent_layer(
        template, "goose", require_cached=True
    )
    snapshot.chmod(0o600)
    snapshot.write_bytes(b"runtime filesystem metadata")
    assert layer.read_bytes() == b"installed agent"


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
    assert agent_layers._layer_key(
        template, base, UrsaHarborAgent, None, "one"
    ) != agent_layers._layer_key(template, base, UrsaHarborAgent, None, "two")
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


@pytest.mark.parametrize(
    "spec", ["ursa", "ursa.integrations.harbor:UrsaHarborAgent"]
)
@pytest.mark.parametrize("custom", [False, True])
async def test_ursa_layer_caches_packages_but_prepares_each_trial_config(
    template, builder, monkeypatch, tmp_path, spec, custom
):
    monkeypatch.setattr(
        UrsaHarborAgent,
        "_default_install_spec",
        staticmethod(lambda: "ursa-ai==0.17.3"),
    )
    template._agent_kwargs = (
        {"ursa_install_spec": "ursa-ai==0.17.3"} if custom else {}
    )

    async def packages(self, environment):
        assert self.model_name is None
        assert self.config_source is None
        assert str(self.ursa_install_spec) == "ursa-ai==0.17.3"
        environment._overlay_path.write_bytes(b"ursa installed")

    install = AsyncMock()
    monkeypatch.setattr(UrsaHarborAgent, "_install_packages", packages)
    monkeypatch.setattr(
        UrsaHarborAgent,
        "_materialize_native_config",
        lambda *a, **k: pytest.fail("prebuild must not prepare model config"),
    )
    layer = await agent_layers.build_agent_layer(template, spec)
    assert layer.read_bytes() == b"ursa installed"
    assert "install" not in builder[1]  # No BaseInstalledAgent.setup.
    template._agent_spec = spec
    template._agent_layer_path = await agent_layers.build_agent_layer(
        template, spec, require_cached=True
    )
    for group in ("first", "second"):
        agent = UrsaHarborAgent(
            logs_dir=tmp_path / group, **template._agent_kwargs
        )
        monkeypatch.setattr(agent, "_install_packages", install)
        monkeypatch.setattr(
            agent,
            "_materialize_native_config",
            lambda **k: ({"group": group}, {"KEY": group}),
        )
        monkeypatch.setattr(
            agent,
            "exec_as_agent",
            AsyncMock(return_value=SimpleNamespace(stdout="/workspace")),
        )
        upload = AsyncMock()
        monkeypatch.setattr(agent, "_upload_native_config", upload)
        await agent.install(template)
        install.assert_not_awaited()
        upload.assert_awaited_once_with(template, {"group": group})
        assert agent._secret_env == {"KEY": group}
        assert agent._workspace == "/workspace"
        assert (
            agent._remote_config_file == "/installed-agent/tmp/ursa-config.json"
        )
    agent.extra_packages = ("numpy",)
    with pytest.raises(ValueError, match="does not match"):
        await agent.install(template)
    agent.extra_packages = ()
    template._agent_layer_path = None
    with pytest.raises(RuntimeError, match="not loaded"):
        await agent.install(template)


def test_ursa_cache_key_tracks_source_and_install_options(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    project = source / "pyproject.toml"
    project.write_text('[project]\nname = "ursa-ai"\nversion = "1.0"\n')
    options = {"ursa_install_spec": f"ursa-ai @ {source.as_uri()}"}
    agent = UrsaHarborAgent(logs_dir=tmp_path, **options)
    key = agent_layers._ursa_install_key(agent)
    (source / ".env").write_text("SECRET=not-an-install-input")
    assert agent_layers._ursa_install_key(agent) == key
    # Runtime settings do not select a different installation.
    agent.agent_import_path = "ursa.agents:SimulatorAgent"
    agent._config = {"group": "different"}
    assert agent_layers._ursa_install_key(agent) == key
    for extra in (
        {"ursa_extras": "image"},
        {"extra_packages": "numpy==2.3.0"},
        {"ursa_install_spec": "ursa-ai==0.17.3"},
    ):
        changed = UrsaHarborAgent(logs_dir=tmp_path, **(options | extra))
        assert agent_layers._ursa_install_key(changed) != key
    project.write_text(project.read_text().replace("1.0", "1.1"))
    assert agent_layers._ursa_install_key(agent) != key


def test_ursa_version_uses_pep508_install_spec():
    with pytest.raises(ValueError, match="ursa_install_spec"):
        agent_layers.agent_installer("ursa@0.17.3")


def test_default_checkout_identity_accepts_symlinks_and_tracks_targets(
    tmp_path, monkeypatch
):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname="ursa-ai"\n')
    docs = tmp_path / "docs"
    docs.mkdir()
    linked = docs / "guide.md"
    linked.write_text("original")
    (source / "docs").symlink_to(docs, target_is_directory=True)
    monkeypatch.setattr(
        UrsaHarborAgent,
        "_default_install_spec",
        staticmethod(lambda: f"ursa-ai @ {source.as_uri()}"),
    )
    agent = UrsaHarborAgent(logs_dir=tmp_path)
    assert agent._build_default_install_wheel
    key = agent_layers._ursa_install_key(agent)
    linked.write_text("changed")
    assert agent_layers._ursa_install_key(agent) != key


@pytest.mark.parametrize("name", ["goose", "codex", "claude_code"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_installed_agent_uses_model_forward_and_restores_connection(
    template, tmp_path, monkeypatch, name, cancel
):
    from ursa.integrations.harbor import installed_agents

    cls = {
        "goose": installed_agents.Goose,
        "codex": installed_agents.Codex,
        "claude_code": installed_agents.ClaudeCode,
    }[name]
    upstream_cls = {
        "goose": installed_agents._Goose,
        "codex": installed_agents._Codex,
        "claude_code": installed_agents._ClaudeCode,
    }[name]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = "anthropic" if name == "claude_code" else "openai"
    agent = cls(
        logs_dir=tmp_path / "logs",
        model_name=f"{provider}/test-model",
        base_url="http://localhost:8000/v1",
    )
    original = agent.model_connection.execution_base_url
    forwarded = "http://localhost:12345/v1"
    mapping = AsyncMock(return_value=forwarded)
    monkeypatch.setattr(template, "_forward_model_url", mapping)

    async def run(self, instruction, environment, context):
        assert self.model_connection.execution_base_url == forwarded
        if cancel:
            raise asyncio.CancelledError

    monkeypatch.setattr(upstream_cls, "run", run)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await agent.run("test", template, None)
    else:
        await agent.run("test", template, None)
    mapping.assert_awaited_once_with(original)
    assert agent._base_url == "http://localhost:8000/v1"
    assert agent.model_connection.execution_base_url == original


@pytest.mark.parametrize("source", ["provider", "llm_model", "emb_model"])
async def test_ursa_remaps_native_provider_urls_and_refreshes_each_proxy_phase(
    template, tmp_path, monkeypatch, source
):
    import json

    from harbor.models.agent.context import AgentContext

    agent = UrsaHarborAgent(logs_dir=tmp_path / "logs")
    agent._remote_config_file = "/installed-agent/tmp/ursa-config.json"
    agent._workspace = "/workspace"
    agent._runtime_group = "default"
    agent._runtime_use_web = True
    agent._model_env = {}
    agent._secret_env = {}
    upload = AsyncMock()
    monkeypatch.setattr(agent, "_upload_config_text", upload)
    monkeypatch.setattr(agent, "exec_as_root", AsyncMock())
    monkeypatch.setattr(agent, "exec_as_agent", AsyncMock())
    monkeypatch.setattr(agent, "_best_effort_root_exec", AsyncMock())
    urls = [
        "http://localhost:12345/v1",
        "http://localhost:12346/v1",
        "http://localhost:8000/v1",
    ]
    monkeypatch.setattr(
        template, "_forward_model_url", AsyncMock(side_effect=urls)
    )
    monkeypatch.setattr(
        agent,
        "_materialize_native_config",
        lambda **kw: (
            (
                {
                    "inference_providers": {
                        "openai": {"base_url": "http://localhost:8000/v1"}
                    }
                }
                if source == "provider"
                else {source: {"base_url": "http://localhost:8000/v1"}}
            )
            | {"mcp_servers": {"local": {"url": "http://localhost:9000/mcp"}}},
            {},
        ),
    )
    for phase, expected in zip([object(), object(), None], urls):
        template._proxy_stack = phase
        await agent.run("test", template, AgentContext())
        config = json.loads(upload.await_args.kwargs["content"])
        model = (
            config["inference_providers"]["openai"]
            if source == "provider"
            else config[source]
        )
        assert model["base_url"] == expected
        assert (
            config["mcp_servers"]["local"]["url"] == "http://localhost:9000/mcp"
        )
    assert upload.await_count == 3
