"""A Harbor Singularity environment that builds task Dockerfiles on demand."""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import shutil
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import override

from harbor.environments.singularity.singularity import SingularityEnvironment

if sys.platform != "win32":
    import fcntl


class DockerfileSingularityEnvironment(SingularityEnvironment):
    """Build ``environment/Dockerfile`` and cache the resulting SIF image."""

    @override
    def _validate_definition(self) -> None:
        if not self._dockerfile_path.is_file():
            raise FileNotFoundError(
                f"Singularity environment requires {self._dockerfile_path}"
            )

    def _dockerfile_cache_path(self) -> Path:
        digest = hashlib.sha256()
        digest.update(platform.machine().encode())
        for path in sorted(self.environment_dir.rglob("*")):
            relative = path.relative_to(self.environment_dir).as_posix()
            digest.update(relative.encode())
            digest.update(str(path.lstat().st_mode).encode())
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(path.read_bytes())
        return (
            self._image_cache_dir / f"dockerfile-{digest.hexdigest()[:24]}.sif"
        )

    async def _is_valid_sif(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            await self._run("singularity", "sif", "list", str(path))
        except RuntimeError:
            return False
        return True

    async def _build_with(
        self,
        builder: str,
        output: Path,
        temporary: Path,
        force_build: bool,
    ) -> None:
        builder_name = Path(builder).name
        tag = f"ursa-harbor-{output.stem}-{builder_name}"
        pull_args = ["--pull"] if force_build else []
        if builder_name == "buildah":
            remove_command = (builder, "rmi", "--force", tag)
        else:
            remove_command = (builder, "image", "rm", "--force", tag)
        temporary.unlink(missing_ok=True)
        try:
            await self._run(
                builder,
                "build",
                *pull_args,
                "--tag",
                tag,
                "--file",
                str(self._dockerfile_path),
                str(self.environment_dir),
            )
            with tempfile.TemporaryDirectory(
                prefix="ursa-harbor-oci-"
            ) as temp_dir:
                archive = Path(temp_dir) / "image.tar"
                if builder_name == "buildah":
                    await self._run(
                        builder, "push", tag, f"docker-archive:{archive}"
                    )
                else:
                    await self._run(builder, "save", "-o", str(archive), tag)
                await self._run(
                    "singularity",
                    "build",
                    str(temporary),
                    f"docker-archive://{archive}",
                )
                temporary.replace(output)
        finally:
            try:
                await self._run(*remove_command)
            except RuntimeError as exc:
                self.logger.warning(
                    "Failed to remove build image %s: %s", tag, exc
                )
            finally:
                temporary.unlink(missing_ok=True)

    async def _run(self, *command: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not start command ({' '.join(command)}): {exc}"
            ) from exc
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise
        if process.returncode:
            detail = stderr.decode(errors="replace") or stdout.decode(
                errors="replace"
            )
            raise RuntimeError(
                f"Command failed ({' '.join(command)}): {detail}"
            )

    async def _build_dockerfile_sif(self, force_build: bool) -> Path:
        output = self._dockerfile_cache_path()
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_build and await self._is_valid_sif(output):
            return output
        lock_file = output.with_suffix(".lock").open("w")
        try:
            while True:
                try:
                    fcntl.flock(
                        lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.1)
            if not force_build and await self._is_valid_sif(output):
                return output
            builders = [
                builder
                for name in ("buildah", "podman", "docker")
                if (builder := shutil.which(name)) is not None
            ]
            if not builders:
                raise RuntimeError(
                    "Building a Dockerfile for Singularity requires buildah, podman, or docker"
                )
            temporary = output.with_suffix(".tmp.sif")
            failures = []
            for candidate in builders:
                try:
                    await self._build_with(
                        candidate, output, temporary, force_build
                    )
                    return output
                except RuntimeError as exc:
                    failures.append(str(exc))
            raise RuntimeError(
                "No container builder succeeded: " + "; ".join(failures)
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    @override
    async def start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("Singularity is unavailable on Windows")
        self._validate_definition()
        self._sif_path = await self._build_dockerfile_sif(
            force_build or self._force_pull
        )
        await self._start_server()
        await self._upload_environment_dir_after_start()


__all__ = ["DockerfileSingularityEnvironment"]
