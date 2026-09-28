"""A Harbor Singularity environment that builds task Dockerfiles on demand."""

from __future__ import annotations

import asyncio
import hashlib
import math
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

from filelock import AsyncFileLock
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import NetworkMode
from pathspec import GitIgnoreSpec


class DockerfileSingularityEnvironment(SingularityEnvironment):
    """Build a Dockerfile and run it as a Singularity 3.6 instance."""

    _supported_no_mounts = frozenset({"home", "tmp"})

    def __init__(
        self,
        *args,
        singularity_startup_timeout_sec: float = 300,
        **kwargs,
    ) -> None:
        if (
            not math.isfinite(singularity_startup_timeout_sec)
            or singularity_startup_timeout_sec <= 0
        ):
            raise ValueError(
                "singularity_startup_timeout_sec must be positive and finite"
            )
        if not kwargs.get("singularity_image_cache_dir"):
            kwargs["singularity_image_cache_dir"] = (
                self._default_image_cache_dir()
            )
        no_mount = kwargs.get("singularity_no_mount")
        if no_mount:
            requested = {part.strip() for part in no_mount.split(",")}
            unsupported = requested - self._supported_no_mounts
            if unsupported:
                values = ", ".join(sorted(unsupported))
                raise ValueError(
                    "Singularity 3.6 cannot disable these mount types: "
                    f"{values}. Only 'home' and 'tmp' are supported."
                )

        super().__init__(*args, **kwargs)
        for policy in self._phase_network_policies:
            if policy != self._network_policy:
                raise ValueError(
                    "Singularity 3.6 cannot change network policy after start"
                )
        self._startup_timeout_sec = singularity_startup_timeout_sec
        identity = hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
        self._instance_name = f"ursa{identity}{secrets.token_hex(4)}"
        self._instance_started = False

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # Singularity 3.6 can create an isolated network namespace, but cannot
        # securely enforce allowlists or switch a running instance's network.
        return EnvironmentCapabilities(mounted=True, disable_internet=True)

    @staticmethod
    def _default_image_cache_dir() -> Path:
        configured = os.environ.get("URSA_HARBOR_SIF_CACHE")
        if configured:
            return Path(configured).expanduser()
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(cache_home).expanduser()
            if cache_home
            else Path.home() / ".cache"
        )
        return root / "ursa" / "harbor" / "sif"

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
        tag = f"ursa-harbor-{output.stem}-{builder_name}-{secrets.token_hex(8)}"
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

    @staticmethod
    async def _terminate_host_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def _run(self, *command: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
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
            await self._terminate_host_process(process)
            raise
        if process.returncode:
            detail = stderr.decode(errors="replace") or stdout.decode(
                errors="replace"
            )
            raise RuntimeError(
                f"Command failed ({' '.join(command)}): {detail}"
            )

    async def _build_dockerfile_sif(self, force_build: bool) -> Path:
        output = await self._dockerfile_cache_path()
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        if not force_build and await self._is_valid_sif(output):
            return output
        async with AsyncFileLock(output.with_suffix(".lock")):
            if not force_build and await self._is_valid_sif(output):
                return output
            builders = [
                builder
                for name in ("buildah", "podman", "docker")
                if (builder := shutil.which(name)) is not None
            ]
            if not builders:
                raise RuntimeError(
                    "Building a Dockerfile for Singularity requires buildah, "
                    "podman, or docker"
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

    @property
    def _instance_ref(self) -> str:
        return f"instance://{self._instance_name}"

    def _instance_exec_prefix(self, cwd: str | None = None) -> list[str]:
        return [
            self._runtime(),
            "exec",
            "--cleanenv",
            "--pwd",
            cwd or self._workdir,
            self._instance_ref,
        ]

    def _instance_start_command(self) -> list[str]:
        if self._staging_dir is None or self._sif_path is None:
            raise RuntimeError("Singularity instance is not prepared")
        command = [
            self._runtime(),
            "instance",
            "start",
            "--fakeroot",
            "--containall",
            "--no-home",
            "--writable-tmpfs",
        ]
        if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
            command.extend(["--net", "--network", "none"])
        command.extend(["-B", f"{self._staging_dir}:/staging"])
        for mount in self._mounts:
            if mount.get("type") != "bind":
                continue
            if mount.get("target") == "/staging":
                raise ValueError(
                    "/staging is reserved by the Harbor environment"
                )
            command.extend([
                "-B",
                f"{mount['source']}:{mount['target']}",
            ])
        command.extend([str(self._sif_path), self._instance_name])
        return command

    async def _stop_instance(self, *, warn: bool) -> None:
        try:
            await self._run(
                self._runtime(),
                "instance",
                "stop",
                "--force",
                self._instance_name,
            )
        except RuntimeError as exc:
            if warn:
                self.logger.warning(
                    "Failed to stop Singularity instance %s: %s",
                    self._instance_name,
                    exc,
                )
        finally:
            self._instance_started = False

    @override
    async def start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("Singularity is unavailable on Windows")
        self._validate_definition()
        self._sif_path = await self._build_dockerfile_sif(
            force_build or self._force_pull
        )
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-singularity-")
        )
        self._staging_dir.chmod(0o755)
        try:
            async with asyncio.timeout(self._startup_timeout_sec):
                await self._run(*self._instance_start_command())
                self._instance_started = True
                await self._run(*self._instance_exec_prefix(), "true")
            await self._upload_environment_dir_after_start()
        except TimeoutError as exc:
            await self._stop_instance(warn=False)
            self._cleanup_staging()
            raise TimeoutError(
                "Singularity instance did not become ready within "
                f"{self._startup_timeout_sec:g} seconds"
            ) from exc
        except BaseException:
            # An interrupted instance-start command can still leave an instance
            # behind, so cleanup is attempted before start reports success.
            await self._stop_instance(warn=False)
            self._cleanup_staging()
            raise

    def _cleanup_staging(self) -> None:
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None

    @override
    async def stop(self, delete: bool) -> None:
        try:
            if self._instance_started:
                await self._stop_instance(warn=True)
        finally:
            self._cleanup_staging()
        if delete:
            self.logger.debug(
                "Singularity image preserved at %s for reuse", self._sif_path
            )

    def _exec_shell_command(
        self,
        command: str,
        cwd: str | None,
        user: str | int | None,
        pid_file: str,
    ) -> str:
        command = f"cd {shlex.quote(cwd or self._workdir)} && {command}"
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            if isinstance(resolved_user, int):
                user_arg = f"$(getent passwd {resolved_user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(str(resolved_user))
            command = f"su {user_arg} -s /bin/bash -c {shlex.quote(command)}"
        quoted_pid_file = shlex.quote(pid_file)
        return (
            f"echo $$ > {quoted_pid_file}; "
            f"trap 'rm -f {quoted_pid_file}' EXIT; "
            f"{command}"
        )

    def _runtime_environment(
        self, env: dict[str, str] | None
    ) -> dict[str, str]:
        runtime_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        prefix = (
            "APPTAINERENV_"
            if Path(self._runtime()).name == "apptainer"
            else "SINGULARITYENV_"
        )
        for key, value in (self._merge_env(env) or {}).items():
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError(f"Invalid environment variable: {key!r}")
            runtime_env[f"{prefix}{key}"] = value
        return runtime_env

    async def _cleanup_exec_process(
        self,
        process: asyncio.subprocess.Process,
        pid_file: str,
    ) -> None:
        cleanup = asyncio.create_task(
            self._run(
                *self._instance_exec_prefix(),
                "bash",
                "-c",
                self._terminate_process_tree_command(pid_file),
            )
        )
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=10)
        await self._terminate_host_process(process)

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._instance_started:
            raise RuntimeError("Singularity environment not started")
        if self._memory_limit_exceeded:
            raise RuntimeError(self._memory_limit_exceeded)

        pid_file = f"/tmp/harbor-exec-{secrets.token_hex(8)}.pid"
        shell_command = self._exec_shell_command(command, cwd, user, pid_file)
        runtime_command = [
            *self._instance_exec_prefix(cwd),
            "bash",
            "-c",
            shell_command,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *runtime_command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env=self._runtime_environment(env),
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not execute command in Singularity: {exc}"
            ) from exc

        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            if timeout_sec is None:
                stdout, stderr = await communication
            else:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication), timeout=timeout_sec
                )
        except TimeoutError:
            timed_out = True
            await self._cleanup_exec_process(process, pid_file)
            stdout, stderr = await communication
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self._cleanup_exec_process(process, pid_file)
            )
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleanup)
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(communication)
            raise

        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        return_code = (
            process.returncode if process.returncode is not None else 1
        )
        if timed_out:
            return_code = 124
            timeout_message = f"Command timed out after {timeout_sec} seconds"
            stderr_text = "\n".join(
                part for part in (stderr_text.rstrip(), timeout_message) if part
            )
        if return_code != 0:
            error_output = stderr_text or stdout_text or "<no output>"
            self.logger.debug(
                "Command exited with rc=%s: %s", return_code, error_output
            )
        return ExecResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=return_code,
        )

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


__all__ = ["DockerfileSingularityEnvironment"]
