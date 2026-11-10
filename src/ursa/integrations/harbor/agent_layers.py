"""Build immutable agent installation overlays for Apptainer/Singularity."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path

from filelock import AsyncFileLock
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import TrialPaths

from ursa.integrations.harbor import UrsaHarborAgent
from ursa.integrations.harbor.singularity import (
    DockerfileSingularityEnvironment,
)


def agent_installer(
    spec: str, kwargs: dict | None = None
) -> tuple[type[BaseInstalledAgent], str | None]:
    name, separator, version = spec.partition("@")
    if not name or (separator and not re.fullmatch(r"[\w.+-]+", version)):
        raise ValueError(
            f"Expected an agent NAME or NAME@VERSION, got {spec!r}"
        )
    agent_class = AgentFactory.get_agent_class_from_config(
        AgentConfig(
            name="ursa.integrations.harbor:UrsaHarborAgent"
            if name == "ursa"
            else name
        )
    )
    if not issubclass(agent_class, BaseInstalledAgent):
        raise ValueError(
            f"Agent {name!r} has no installed-agent setup to cache"
        )
    allowed = set()
    if issubclass(agent_class, UrsaHarborAgent):
        if version:
            raise ValueError(
                "Use --agent ursa --agent-kwarg ursa_install_spec=ursa-ai==VERSION "
                "to select an URSA package version"
            )
        allowed = {"ursa_install_spec", "ursa_extras", "extra_packages"}
    if unsupported := (kwargs or {}).keys() - allowed:
        raise ValueError(
            f"Unsupported cached install options for {name}: "
            f"{', '.join(sorted(unsupported))}"
        )
    return agent_class, version or None


def _ursa_install_key(agent: UrsaHarborAgent) -> str:
    """Identify package inputs, excluding model configuration and credentials."""
    source_hash = None
    revision = None
    if (source := agent._install_source) is not None:
        root = source if source.is_dir() else source.parent
        files = [source]
        if source.is_dir() and (source / ".git").exists():
            paths = subprocess.check_output([
                "git",
                "-C",
                str(source),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ])
            files = [
                source / os.fsdecode(path)
                for path in paths.split(b"\0")
                if path
            ]
            if agent._build_default_install_wheel:
                revision = subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip()
        digest = hashlib.sha256()
        visited = set()

        def hash_path(path):
            relative = path.relative_to(root)
            if agent._is_sensitive_source_path(relative):
                return
            if not path.exists() and not path.is_symlink():
                return  # Deleted tracked file.
            digest.update(str(relative).encode() + b"\0")
            digest.update(str(path.lstat().st_mode & 0o777).encode() + b"\0")
            if path.is_symlink():
                # Default wheel builds accept repository symlinks, including
                # documentation links. Track their targets without staging.
                digest.update(os.fsencode(path.readlink()) + b"\0")
            if path.is_dir():
                resolved = path.resolve()
                if resolved not in visited:
                    visited.add(resolved)
                    for child in sorted(path.iterdir()):
                        hash_path(child)
            elif path.is_file():
                with path.open("rb") as stream:
                    digest.update(
                        hashlib.file_digest(stream, "sha256").digest()
                    )

        for path in sorted(files):
            hash_path(path)
        source_hash = digest.hexdigest()
    identity = {
        "spec": str(agent.ursa_install_spec),
        "extras": sorted(agent.ursa_extras),
        "packages": agent.extra_packages,
        "source": source_hash,
        "revision": revision,
        "build_wheel": agent._build_default_install_wheel,
        "python": agent.URSA_PYTHON_VERSION,
        "uv": agent._UV_VERSION,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()


async def cached_ursa_install(agent, environment) -> bool:
    if not isinstance(environment, DockerfileSingularityEnvironment):
        return False
    if environment._agent_spec is None:
        return False
    selected, _ = agent_installer(
        environment._agent_spec, environment._agent_kwargs
    )
    if not isinstance(agent, selected) or (
        await asyncio.to_thread(_ursa_install_key, agent)
        != environment._agent_install_key
    ):
        raise ValueError(
            "Selected install layer does not match this URSA installation; "
            "match singularity_agent_kwargs to the agent's ursa_install_spec, "
            "ursa_extras, and extra_packages, then prebuild it"
        )
    if environment._agent_layer_path is None:
        raise RuntimeError("The selected agent install layer is not loaded")
    return True


class _InstallEnvironment(DockerfileSingularityEnvironment):
    """Write the complete installation into an overlay without trial mounts."""

    def __init__(self, base_image: Path, **kwargs):
        self._base_image = base_image
        super().__init__(
            **kwargs, auto_mounts=[], enable_environment_dir_upload=False
        )

    @property
    def _uses_compose(self) -> bool:
        return False

    def _validate_definition(self) -> None:
        # The caller has already built/validated the task's actual main SIF.
        if not self._base_image.is_file():
            raise FileNotFoundError(self._base_image)

    async def _build_main_sif(self, force_build: bool) -> Path:
        return self._base_image

    def _harbor_writable_binds(self):
        # Includes /installed-agent and HOME in the layer. Runtime trials seed
        # their private binds from the base SIF plus this read-only overlay.
        return ()

    async def _seed_harbor_bind_sources(self) -> None:
        # Ordinary trials get a writable workdir bind even when the image does
        # not contain that path. Create it inside the install layer instead.
        await self._run(
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            *self._isolated_runtime_args(
                self._runtime_workdir("seed"), include_overlay=True, pwd="/"
            ),
            str(self._base_image),
            "sh",
            "-c",
            f"mkdir -p {shlex.quote(self._workdir)}",
        )


def _layer_key(
    environment, base_image, agent_class, requested_version, install_key=None
) -> str:
    runtime = environment._instance_runtime()
    runtime_version = subprocess.check_output(
        [runtime, "--version"], text=True, timeout=10
    ).strip()
    with base_image.open("rb") as image:
        base_hash = hashlib.file_digest(image, "sha256").hexdigest()
    identity = {
        "format": 1,
        "base": base_hash,
        "agent": f"{agent_class.__module__}:{agent_class.__qualname__}",
        "version": requested_version,
        "harbor": package_version("harbor"),
        "runtime": runtime_version,
        "fakeroot": environment._fakeroot,
        "uid": os.getuid(),
        "home": str(Path.home()),
        "workdir": environment._workdir,
    }
    if install_key is not None:
        identity["install"] = install_key
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()


async def build_agent_layer(
    template: DockerfileSingularityEnvironment,
    spec: str,
    *,
    force: bool = False,
    timeout_sec: float = 1800,
    require_cached: bool = False,
) -> Path:
    """Return a completed install overlay, publishing only successful installs."""
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError(
            "agent_install_timeout_sec must be positive and finite"
        )
    agent_class, requested_version = agent_installer(
        spec, template._agent_kwargs
    )
    base_image = template._sif_path or await template._build_main_sif(False)
    with tempfile.TemporaryDirectory(
        prefix="ursa-harbor-install-"
    ) as directory:
        trial_paths = TrialPaths(Path(directory))
        trial_paths.mkdir()
        installer = agent_class(
            logs_dir=trial_paths.agent_dir,
            version=requested_version,
            **template._agent_kwargs,
        )
        template._agent_install_key = (
            await asyncio.to_thread(_ursa_install_key, installer)
            if isinstance(installer, UrsaHarborAgent)
            else None
        )
        key = await asyncio.to_thread(
            _layer_key,
            template,
            base_image,
            agent_class,
            requested_version,
            template._agent_install_key,
        )
        root = template._agent_cache_dir
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        layer = root / f"{key}.img"
        async with AsyncFileLock(root / f"{key}.lock"):
            if layer.is_file() and not force:
                if require_cached:
                    # Keep a private snapshot through restarts and refreshes.
                    # fuse2fs updates ext3 mount metadata even with -o ro, so a
                    # hardlink would still let mounts modify the shared cache.
                    pinned = root / f".{key}-{template._sif_use_token}.img"
                    try:
                        await template._run(
                            "cp",
                            "--sparse=always",
                            "--reflink=auto",
                            str(layer),
                            str(pinned),
                        )
                    except BaseException:
                        pinned.unlink(missing_ok=True)
                        raise
                    return pinned
                return layer
            if require_cached:
                raise FileNotFoundError(
                    f"No cached install layer for {spec!r}. Run prebuild --agent "
                    f"{spec} with the same image, cache, and fakeroot settings first."
                )
            environment = _InstallEnvironment(
                base_image,
                environment_dir=template.environment_dir,
                environment_name=template.environment_name,
                session_id=Path(directory).name,
                trial_paths=trial_paths,
                task_env_config=template.task_env_config.model_copy(
                    update={
                        "workdir": template._workdir,
                        # Native installers stage multiple copies of large
                        # binaries. This sparse build layer is independent of
                        # the trial's writable overlay/storage setting.
                        "storage_mb": max(
                            4096, template.task_env_config.storage_mb or 0
                        ),
                    }
                ),
                network_policy=template.network_policy,
                singularity_image_cache_dir=template._image_cache_dir,
                singularity_fakeroot=template._fakeroot,
            )
            environment._runtime_path = template._instance_runtime()
            try:
                await environment.start(force_build=False)
                try:
                    async with asyncio.timeout(timeout_sec):
                        if isinstance(installer, UrsaHarborAgent):
                            await installer._install_packages(environment)
                        else:
                            await installer.setup(environment)
                except TimeoutError as exc:
                    raise TimeoutError(
                        f"{spec} installation exceeded {timeout_sec:g}s; "
                        "increase --agent-install-timeout-sec"
                    ) from exc
                # Unmount before publishing: an active ext3 filesystem cannot
                # safely be shared as a read-only lower overlay.
                await environment._stop_prepared_runtime(warn=False)
                assert environment._overlay_path is not None
                # Copy into the destination filesystem, then atomically replace
                # the entry. Failed refreshes leave the previous layer intact.
                temporary = root / f".{key}-{Path(directory).name}.img"
                try:
                    await environment._run(
                        "cp",
                        "--sparse=always",
                        "--reflink=auto",
                        str(environment._overlay_path),
                        str(temporary),
                    )
                    temporary.chmod(0o400)
                    temporary.replace(layer)
                finally:
                    temporary.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not build {spec} install layer: {exc}"
                ) from exc
            finally:
                await environment._await_cleanup(environment.stop(delete=True))
    return layer
