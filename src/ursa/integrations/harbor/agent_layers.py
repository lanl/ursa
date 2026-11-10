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

from ursa.integrations.harbor.singularity import (
    DockerfileSingularityEnvironment,
)


def agent_installer(spec: str) -> tuple[type[BaseInstalledAgent], str | None]:
    name, separator, version = spec.partition("@")
    if not name or (separator and not re.fullmatch(r"[\w.+-]+", version)):
        raise ValueError(
            f"Expected an agent NAME or NAME@VERSION, got {spec!r}"
        )
    agent_class = AgentFactory.get_agent_class_from_config(
        AgentConfig(name=name)
    )
    if not issubclass(agent_class, BaseInstalledAgent):
        raise ValueError(
            f"Agent {name!r} has no installed-agent setup to cache"
        )
    return agent_class, version or None


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


def _layer_key(environment, base_image, agent_class, requested_version) -> str:
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
    agent_class, requested_version = agent_installer(spec)
    base_image = template._sif_path or await template._build_main_sif(False)
    key = await asyncio.to_thread(
        _layer_key, template, base_image, agent_class, requested_version
    )
    root = template._agent_cache_dir
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    layer = root / f"{key}.img"
    async with AsyncFileLock(root / f"{key}.lock"):
        if layer.is_file() and not force:
            if require_cached:
                # Pin this inode through network-policy restarts, even if a
                # concurrent prebuild atomically refreshes the cache entry.
                pinned = root / f".{key}-{template._sif_use_token}.img"
                os.link(layer, pinned)
                return pinned
            return layer
        if require_cached:
            raise FileNotFoundError(
                f"No cached install layer for {spec!r}. Run prebuild --agent "
                f"{spec} with the same image, cache, and fakeroot settings first."
            )
        with tempfile.TemporaryDirectory(
            prefix="ursa-harbor-install-"
        ) as directory:
            trial_paths = TrialPaths(Path(directory))
            trial_paths.mkdir()
            installer = agent_class(
                logs_dir=trial_paths.agent_dir, version=requested_version
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
