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
        super().__init__(*args, **kwargs)
        self._startup_timeout_sec = singularity_startup_timeout_sec

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

    @staticmethod
    def _process_start_time(pid: int) -> str | None:
        """Return Linux's stable identity component for a live process."""
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            _, separator, fields = stat.rpartition(") ")
            return fields.split()[19] if separator else None
        except (IndexError, OSError):
            return None

    @classmethod
    def _descendant_processes(cls, pid: int) -> dict[int, str]:
        """Return current descendants keyed by PID and process start time."""
        descendants: dict[int, str] = {}
        pending = [pid]
        while pending:
            parent = pending.pop()
            for children_file in Path(f"/proc/{parent}/task").glob(
                "*/children"
            ):
                with suppress(OSError):
                    for value in children_file.read_text().split():
                        child = int(value)
                        if child not in descendants:
                            if (
                                start_time := cls._process_start_time(child)
                            ) is None:
                                continue
                            descendants[child] = start_time
                            pending.append(child)
        return descendants

    async def _terminate_processes(self, processes: dict[int, str]) -> None:
        """Terminate only the captured container helper processes."""
        for process_signal in (signal.SIGTERM, signal.SIGKILL):
            processes = {
                pid: start_time
                for pid, start_time in processes.items()
                if self._process_start_time(pid) == start_time
            }
            for pid in processes:
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, process_signal)
            for _ in range(20):
                processes = {
                    pid: start_time
                    for pid, start_time in processes.items()
                    if self._process_start_time(pid) == start_time
                }
                if not processes:
                    return
                await asyncio.sleep(0.1)
        if processes:
            self.logger.warning(
                "Could not terminate Singularity helper processes: %s",
                sorted(processes),
            )

    async def _cleanup_server_attempt(
        self, server_pid: int, server_start_time: str | None
    ) -> None:
        """Clean one failed startup without orphaning runtime helpers."""
        descendants: dict[int, str] = {}
        try:
            if (
                server_start_time is not None
                and self._process_start_time(server_pid) == server_start_time
            ):
                descendants.update(self._descendant_processes(server_pid))
            await self._terminate_processes(descendants)
        finally:
            try:
                if (
                    server_start_time is not None
                    and self._process_start_time(server_pid)
                    == server_start_time
                ):
                    descendants.update(self._descendant_processes(server_pid))
                await self._terminate_processes(descendants)
            finally:
                process = self._server_process
                current_start_time = self._process_start_time(server_pid)
                if (
                    process is not None
                    and process.returncode is None
                    and process.pid == server_pid
                    and (
                        server_start_time is None
                        or current_start_time == server_start_time
                    )
                ):
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        if (
                            process.pid == server_pid
                            and process.returncode is None
                            and (
                                server_start_time is None
                                or self._process_start_time(server_pid)
                                == server_start_time
                            )
                        ):
                            process.kill()
                            await process.wait()
                if (
                    server_start_time is not None
                    and self._process_start_time(server_pid)
                    == server_start_time
                ):
                    descendants.update(self._descendant_processes(server_pid))
                await self._terminate_processes(descendants)
                if self._stream_task is not None:
                    self._stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._stream_task
                    self._stream_task = None
                if self._http_client is not None:
                    await self._http_client.aclose()
                    self._http_client = None

    @override
    async def _start_server(self) -> None:
        """Start Harbor's server with a configurable readiness deadline."""
        await self._cleanup_server_resources()

        self._staging_dir = Path(
            tempfile.mkdtemp(prefix="singularity_staging_")
        )
        self._staging_dir.chmod(0o755)
        upstream_dir = Path(harbor_singularity.__file__).parent
        staging_server = self._staging_dir / "_hbexec.py"
        shutil.copy(upstream_dir / "server.py", staging_server)
        bootstrap_script = self._staging_dir / "bootstrap.sh"
        shutil.copy(upstream_dir / "bootstrap.sh", bootstrap_script)
        bootstrap_script.chmod(0o755)

        last_error: Exception | None = None
        for port_attempt in range(3):
            attempt_error: Exception | None = None
            reserved_socket, port = self._reserve_port()
            self._server_port = port
            env_files_dir = self.environment_dir / "files"
            bind_mounts = ["-B", f"{self._staging_dir}:/staging"]
            for mount in self._mounts:
                if mount.get("type") == "bind":
                    bind_mounts.extend([
                        "-B",
                        f"{mount['source']}:{mount['target']}",
                    ])
            if env_files_dir.exists():
                bind_mounts.extend([
                    "-B",
                    f"{env_files_dir}:/staging/env_files",
                ])

            no_mount_args: list[str] = []
            singularity_no_mount = self._singularity_no_mount
            if singularity_no_mount is None:
                singularity_no_mount = "home,tmp,bind-paths"
            if singularity_no_mount:
                for part in singularity_no_mount.split(","):
                    if part := part.strip():
                        no_mount_args.extend(["--no-mount", part])

            bootstrap_cmd = [
                "bash",
                "-c",
                'exec /staging/bootstrap.sh "$@"',
                "bash",
                self._workdir,
                "/staging/_hbexec.py",
                "--port",
                str(port),
                "--workdir",
                self._workdir,
            ]
            command = [
                "singularity",
                "exec",
                *no_mount_args,
                "--pwd",
                self._workdir,
                "--writable-tmpfs",
                "--fakeroot",
                "--containall",
                "--pid",
                *bind_mounts,
                str(self._sif_path),
                *bootstrap_cmd,
            ]
            reserved_socket.close()
            self._server_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            server_pid = self._server_process.pid
            server_start_time = self._process_start_time(server_pid)
            self._stream_task = asyncio.create_task(
                self._stream_server_output()
            )
            self._http_client = harbor_singularity.httpx.AsyncClient(
                timeout=30.0
            )
            deadline = (
                asyncio.get_running_loop().time() + self._startup_timeout_sec
            )
            server_ready = False
            cancelled = False
            try:
                while asyncio.get_running_loop().time() < deadline:
                    try:
                        response = await self._http_client.get(
                            f"http://localhost:{port}/health"
                        )
                        if response.status_code == 200:
                            if self._server_process.returncode is not None:
                                attempt_error = RuntimeError(
                                    f"Port collision on {port}: health check "
                                    "succeeded but our server process died."
                                )
                                break
                            server_ready = True
                            break
                    except harbor_singularity.httpx.RequestError:
                        pass
                    if self._server_process.returncode is not None:
                        attempt_error = RuntimeError(
                            f"Server process died on port {port}. Check "
                            "trial.log for server output."
                        )
                        break
                    await asyncio.sleep(1)
                if server_ready:
                    if self._memory_limit_bytes is not None:
                        self._memory_watchdog_task = asyncio.create_task(
                            self._memory_watchdog()
                        )
                    return
                if attempt_error is None:
                    attempt_error = TimeoutError(
                        "Singularity server did not become ready within "
                        f"{self._startup_timeout_sec:g} seconds "
                        f"(attempt {port_attempt + 1}/3)"
                    )
                last_error = attempt_error
            except asyncio.CancelledError:
                cancelled = True
                raise
            finally:
                if not server_ready:
                    cleanup = asyncio.create_task(
                        self._cleanup_server_attempt(
                            server_pid, server_start_time
                        )
                    )
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise
                    finally:
                        if cancelled and self._staging_dir is not None:
                            shutil.rmtree(self._staging_dir, ignore_errors=True)
                            self._staging_dir = None

        await self._cleanup_server_resources()
        raise last_error or RuntimeError(
            "Failed to start Singularity FastAPI server after 3 attempts"
        )

    @override
    async def stop(self, delete: bool) -> None:
        """Stop the container and reap helpers before they can be orphaned."""
        server_pid = (
            self._server_process.pid
            if self._server_process is not None
            else None
        )
        server_start_time = (
            self._process_start_time(server_pid)
            if server_pid is not None
            else None
        )
        descendants: dict[int, str] = {}
        try:
            if server_pid is not None and server_start_time is not None:
                descendants.update(self._descendant_processes(server_pid))
            await self._terminate_processes(descendants)
        finally:
            try:
                if (
                    server_pid is not None
                    and self._process_start_time(server_pid)
                    == server_start_time
                ):
                    descendants.update(self._descendant_processes(server_pid))
                await self._terminate_processes(descendants)
            finally:
                try:
                    await super().stop(delete)
                finally:
                    await self._terminate_processes(descendants)

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
