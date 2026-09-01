"""A Harbor Singularity environment that builds task Dockerfiles on demand."""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import secrets
import shlex
import shutil
import signal
import sys
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import override

from harbor.environments.base import ExecResult
from harbor.environments.singularity import singularity as harbor_singularity
from harbor.environments.singularity.singularity import SingularityEnvironment
from pathspec import GitIgnoreSpec

if sys.platform != "win32":
    import fcntl


class DockerfileSingularityEnvironment(SingularityEnvironment):
    """Build ``environment/Dockerfile`` and cache the resulting SIF image."""

    _runtime_path_lock = asyncio.Lock()
    _harbor_owned_timeout_sec = 7 * 24 * 60 * 60

    def _ensure_bootstrap_mounts(self) -> None:
        """Overlay Harbor's bootstrap with compatibility for older Python."""
        targets = {mount.get("target") for mount in self._mounts}
        resources = (
            (
                Path(harbor_singularity.__file__).parent / "bootstrap.sh",
                "/staging/bootstrap-upstream.sh",
            ),
            (
                Path(__file__).with_name("harbor_singularity_bootstrap.sh"),
                "/staging/bootstrap.sh",
            ),
        )
        for source, target in resources:
            if target not in targets:
                self._mounts.append({
                    "type": "bind",
                    "source": str(source),
                    "target": target,
                })

    @staticmethod
    def _runtime() -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise RuntimeError("Apptainer or Singularity is required")
        return runtime

    @override
    def _validate_definition(self) -> None:
        if not self._dockerfile_path.is_file():
            raise FileNotFoundError(
                f"Singularity environment requires {self._dockerfile_path}"
            )

    @override
    def _resolve_workdir(self) -> str:
        """Resolve task overrides and Dockerfile WORKDIR instructions."""
        if self.task_env_config.workdir is not None:
            return self.task_env_config.workdir
        workdir = PurePosixPath("/")
        for line in self._dockerfile_path.read_text().splitlines():
            instruction = line.strip()
            if instruction.upper().startswith("FROM "):
                workdir = PurePosixPath("/")
            elif instruction.upper().startswith("WORKDIR "):
                value = line.split(None, 1)[1].strip()
                if "$" in value:
                    raise ValueError(
                        "Singularity cannot resolve variables in Dockerfile "
                        f"WORKDIR: {value}"
                    )
                path = PurePosixPath(value)
                workdir = path if path.is_absolute() else workdir / path
        return str(workdir)

    def _dockerfile_cache_path_sync(self) -> Path:
        digest = hashlib.sha256()
        digest.update(platform.machine().encode())
        ignore_file = self.environment_dir / ".dockerignore"
        ignore = (
            GitIgnoreSpec.from_lines(ignore_file.read_text().splitlines())
            if ignore_file.is_file()
            else None
        )
        for path in sorted(self.environment_dir.rglob("*")):
            relative = path.relative_to(self.environment_dir).as_posix()
            if (
                ignore is not None
                and relative not in {"Dockerfile", ".dockerignore"}
                and ignore.match_file(relative)
            ):
                continue
            digest.update(relative.encode())
            digest.update(str(path.lstat().st_mode).encode())
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(path.read_bytes())
        return (
            self._image_cache_dir / f"dockerfile-{digest.hexdigest()[:24]}.sif"
        )

    async def _dockerfile_cache_path(self) -> Path:
        return await asyncio.to_thread(self._dockerfile_cache_path_sync)

    async def _is_valid_sif(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            await self._run(self._runtime(), "sif", "list", str(path))
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
                    self._runtime(),
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

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # The HTTP server otherwise lets subprocesses inherit the Apptainer
        # launch terminal. Non-interactive installers may try to read it and
        # be suspended by job control instead of returning an error.
        command = f"bash -c {shlex.quote(command)} </dev/null"
        # The upstream transport otherwise imposes a hidden 600-second limit
        # when Harbor owns the surrounding agent-phase timeout.
        if timeout_sec is not None:
            return await super().exec(command, cwd, env, timeout_sec, user)
        pid_file = f"/tmp/harbor-exec-{secrets.token_hex(8)}.pid"
        quoted_pid_file = shlex.quote(pid_file)
        command = (
            f"echo $$ > {quoted_pid_file}; "
            f"trap 'rm -f {quoted_pid_file}' EXIT; "
            f"{command}"
        )
        try:
            return await super().exec(
                command,
                cwd,
                env,
                self._harbor_owned_timeout_sec,
                user,
            )
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                super().exec(
                    self._terminate_process_tree_command(pid_file),
                    timeout_sec=10,
                )
            )
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(cleanup)
            raise

    @staticmethod
    def _terminate_process_tree_command(pid_file: str) -> str:
        quoted_pid_file = shlex.quote(pid_file)
        return (
            "i=0; "
            f'while [ ! -s {quoted_pid_file} ] && [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            f"if [ -s {quoted_pid_file} ]; then "
            f"pid=$(cat {quoted_pid_file}); "
            "case $pid in *[!0-9]*|'') exit 0;; esac; "
            "descendants() { for child in "
            '$(cat "/proc/$1/task/$1/children" 2>/dev/null); '
            'do descendants "$child"; echo "$child"; done; }; '
            'children=$(descendants "$pid"); '
            'kill -TERM $children "$pid" 2>/dev/null || true; '
            'i=0; while kill -0 "$pid" 2>/dev/null '
            '&& [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            'kill -KILL $children "$pid" 2>/dev/null || true; '
            f"rm -f {quoted_pid_file}; fi"
        )

    async def _build_dockerfile_sif(self, force_build: bool) -> Path:
        output = await self._dockerfile_cache_path()
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
        if getattr(self, "_singularity_no_mount", None) is None:
            self._singularity_no_mount = "home,tmp,bind-paths"
        no_mounts = {
            item.strip() for item in self._singularity_no_mount.split(",")
        }
        self._ensure_bootstrap_mounts()
        resolver = Path("/etc/resolv.conf")
        mounts = getattr(self, "_mounts", [])
        if (
            "bind-paths" in no_mounts
            and resolver.is_file()
            and not any(
                mount.get("target") == str(resolver) for mount in mounts
            )
        ):
            mounts.append({
                "type": "bind",
                "source": str(resolver),
                "target": str(resolver),
            })
            self._mounts = mounts
        self._validate_definition()
        self._sif_path = await self._build_dockerfile_sif(
            force_build or self._force_pull
        )
        runtime = self._runtime()
        if Path(runtime).name == "singularity":
            await self._start_server()
        else:
            # Harbor currently spells the executable as ``singularity`` in
            # its server startup. Supply a scoped compatibility shim until
            # Harbor exposes a runtime-command hook.
            async with self._runtime_path_lock:
                with tempfile.TemporaryDirectory(
                    prefix="ursa-harbor-apptainer-"
                ) as shim_dir:
                    Path(shim_dir, "singularity").symlink_to(runtime)
                    old_path = os.environ.get("PATH", "")
                    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{old_path}"
                    try:
                        await self._start_server()
                    finally:
                        os.environ["PATH"] = old_path
        await self._upload_environment_dir_after_start()


__all__ = ["DockerfileSingularityEnvironment"]
