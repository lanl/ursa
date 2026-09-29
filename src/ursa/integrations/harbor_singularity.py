"""A Dockerfile and singularity-compose environment for Harbor tasks."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, override

import yaml
from filelock import AsyncFileLock
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import NetworkMode
from harbor.utils.env import resolve_env_vars
from pathspec import GitIgnoreSpec


class DockerfileSingularityEnvironment(SingularityEnvironment):
    """Run Dockerfile and supported Compose tasks with Singularity 3.6."""

    _supported_no_mounts = frozenset({"home", "tmp"})
    _compose_filename = "docker-compose.yaml"
    _compose_service_fields = frozenset({
        "build",
        "command",
        "depends_on",
        "deploy",
        "env_file",
        "environment",
        "expose",
        "image",
        "ports",
        "volumes",
    })

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
        self._compose_identity = f"{identity[:8]}{secrets.token_hex(4)}"
        self._instance_started = False
        self._compose_project_dir: Path | None = None
        self._compose_file: Path | None = None
        self._compose_instances: dict[str, list[str]] = {}
        self._running_instances: set[str] = set()

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        # Singularity 3.6 can create an isolated network namespace, but cannot
        # securely enforce allowlists or switch a running instance's network.
        return EnvironmentCapabilities(
            mounted=True,
            disable_internet=True,
            docker_compose=True,
        )

    @property
    @override
    def _uses_compose(self) -> bool:
        environment_dir = getattr(self, "environment_dir", None)
        task_compose = bool(
            environment_dir
            and (environment_dir / self._compose_filename).is_file()
        )
        return task_compose or bool(
            getattr(self, "extra_docker_compose_paths", [])
        )

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

    def _instance_runtime(self) -> str:
        if self._uses_compose:
            runtime = shutil.which("singularity")
            if runtime is None:
                raise RuntimeError(
                    "singularity-compose requires the 'singularity' command"
                )
            return runtime
        return self._runtime()

    @classmethod
    @override
    def preflight(cls) -> None:
        """Verify the runtime and a Dockerfile builder before queueing work."""
        try:
            runtime = cls._runtime()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        help_checks = (
            (
                ("instance", "start", "--help"),
                (
                    "--fakeroot",
                    "--containall",
                    "--hostname",
                    "--no-home",
                    "--bind",
                    "--writable-tmpfs",
                    "--net",
                    "--network",
                    "--network-args",
                ),
            ),
            (("exec", "--help"), ("--cleanenv", "--pwd")),
            (("instance", "stop", "--help"), ("--force",)),
        )
        for arguments, required_options in help_checks:
            command = [runtime, *arguments]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SystemExit(
                    f"Could not inspect the Singularity runtime: {exc}"
                ) from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise SystemExit(
                    f"Singularity runtime check failed ({' '.join(command)}): "
                    f"{detail or f'exit code {result.returncode}'}"
                )
            help_text = f"{result.stdout}\n{result.stderr}"
            missing = [
                option
                for option in required_options
                if re.search(
                    rf"(?<![\w-]){re.escape(option)}(?![\w-])",
                    help_text,
                )
                is None
            ]
            if missing:
                raise SystemExit(
                    f"{Path(runtime).name} does not support required options: "
                    + ", ".join(missing)
                )

        builders = [
            (name, path)
            for name in ("buildah", "podman", "docker")
            if (path := shutil.which(name)) is not None
        ]
        if not builders:
            raise SystemExit(
                "Building Harbor Dockerfiles requires buildah, podman, or docker"
            )
        failures: list[str] = []
        usable_builder = False
        for name, path in builders:
            try:
                result = subprocess.run(
                    [path, "info"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"{name}: {exc}")
                continue
            if result.returncode == 0:
                usable_builder = True
                break
            detail = (result.stderr or result.stdout).strip()
            failures.append(
                f"{name}: {detail or f'exit code {result.returncode}'}"
            )
        if not usable_builder:
            raise SystemExit(
                "No usable Dockerfile builder found. " + "; ".join(failures)
            )

        compose = shutil.which("singularity-compose")
        if compose is None:
            raise SystemExit(
                "Docker Compose tasks require singularity-compose 0.1 or later"
            )
        try:
            result = subprocess.run(
                [compose, "version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(
                f"Could not inspect singularity-compose: {exc}"
            ) from exc
        version_text = f"{result.stdout}\n{result.stderr}".strip()
        version = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", version_text)
        if result.returncode != 0 or version is None:
            raise SystemExit(
                "singularity-compose version check failed: "
                + (version_text or f"exit code {result.returncode}")
            )
        parsed_version = tuple(int(part or 0) for part in version.groups())
        if parsed_version < (0, 1, 0):
            raise SystemExit(
                "Docker Compose tasks require singularity-compose 0.1 or later"
            )

    @override
    def _validate_definition(self) -> None:
        config = None
        if self._uses_compose:
            if shutil.which("singularity") is None:
                raise RuntimeError(
                    "singularity-compose requires the 'singularity' command; "
                    "Apptainer-only installations can run single-container "
                    "tasks but not docker-compose.yaml tasks"
                )
            config = self._load_compose_config()
            sidecars = set(config["services"]) - {"main"}
            if (
                sidecars
                and self._network_policy.network_mode == NetworkMode.NO_NETWORK
            ):
                raise ValueError(
                    "singularity-compose cannot provide sidecar networking "
                    "while Harbor network_mode is 'no-network'"
                )
        main = config["services"]["main"] if config is not None else {}
        if (
            not ({"build", "image"} & set(main))
            and not self._dockerfile_path.is_file()
        ):
            raise FileNotFoundError(
                f"Singularity environment requires {self._dockerfile_path}"
            )

    def _compose_paths(self) -> list[Path]:
        paths = []
        task_compose = self.environment_dir / self._compose_filename
        if task_compose.is_file():
            paths.append(task_compose)
        paths.extend(self.extra_docker_compose_paths)
        return paths

    @staticmethod
    def _merge_compose(base: dict[str, Any], overlay: dict[str, Any]) -> None:
        """Apply singularity-compose's recursive overlay semantics."""
        for key, value in overlay.items():
            current = base.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                DockerfileSingularityEnvironment._merge_compose(current, value)
            else:
                base[key] = value

    @staticmethod
    def _absolute_compose_path(value: str, base_dir: Path) -> str:
        if "$" in value:
            raise ValueError(
                "singularity-compose cannot interpolate variables in paths: "
                f"{value!r}"
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())

    def _normalize_compose_paths(
        self, config: dict[str, Any], base_dir: Path
    ) -> None:
        services = config.get("services")
        if not isinstance(services, dict):
            return
        for name, service in services.items():
            if not isinstance(service, dict):
                continue
            build = service.get("build")
            if isinstance(build, str):
                service["build"] = self._absolute_compose_path(build, base_dir)
            elif isinstance(build, dict):
                context = build.get("context", ".")
                if isinstance(context, str):
                    build["context"] = self._absolute_compose_path(
                        context, base_dir
                    )

            env_files = service.get("env_file")
            if env_files is not None:
                if not isinstance(env_files, list):
                    env_files = [env_files]
                normalized_env_files = []
                for entry in env_files:
                    required = True
                    if isinstance(entry, dict):
                        unsupported = set(entry) - {"path", "required"}
                        if unsupported or not isinstance(
                            entry.get("path"), str
                        ):
                            raise ValueError(
                                "singularity-compose cannot represent env_file "
                                f"options on service {name!r}: {entry!r}"
                            )
                        required = entry.get("required", True)
                        if not isinstance(required, bool):
                            raise ValueError(
                                f"env_file.required on service {name!r} must be boolean"
                            )
                        entry = entry["path"]
                    if not isinstance(entry, str):
                        normalized_env_files.append(entry)
                        continue
                    path = self._absolute_compose_path(entry, base_dir)
                    if required or Path(path).is_file():
                        normalized_env_files.append(path)
                service["env_file"] = normalized_env_files

            depends_on = service.get("depends_on")
            if isinstance(depends_on, dict):
                dependencies = []
                for dependency, condition in depends_on.items():
                    if condition is None:
                        condition = {}
                    if not isinstance(condition, dict):
                        raise ValueError(
                            f"Invalid depends_on entry on service {name!r}: "
                            f"{dependency!r}"
                        )
                    unsupported = set(condition) - {
                        "condition",
                        "required",
                        "restart",
                    }
                    supported = (
                        not unsupported
                        and condition.get("condition", "service_started")
                        == "service_started"
                        and condition.get("required", True) is True
                        and condition.get("restart", False) is False
                    )
                    if not supported:
                        raise ValueError(
                            "singularity-compose supports only service_started "
                            f"depends_on conditions on service {name!r}"
                        )
                    dependencies.append(dependency)
                service["depends_on"] = dependencies

            volumes = service.get("volumes")
            if isinstance(volumes, list):
                normalized_volumes = []
                for volume in volumes:
                    if isinstance(volume, dict):
                        unsupported = set(volume) - {
                            "read_only",
                            "source",
                            "target",
                            "type",
                        }
                        if (
                            unsupported
                            or volume.get("type") != "bind"
                            or volume.get("read_only", False) is not False
                            or not isinstance(volume.get("source"), str)
                            or not isinstance(volume.get("target"), str)
                        ):
                            raise ValueError(
                                "singularity-compose supports only writable bind "
                                f"mounts on service {name!r}: {volume!r}"
                            )
                        volume = f"{volume['source']}:{volume['target']}"
                    if not isinstance(volume, str) or volume.count(":") != 1:
                        normalized_volumes.append(volume)
                        continue
                    source, target = volume.split(":", 1)
                    if source.startswith(("/", ".", "~")):
                        source = self._absolute_compose_path(source, base_dir)
                    normalized_volumes.append(f"{source}:{target}")
                service["volumes"] = normalized_volumes

            ports = service.get("ports")
            if isinstance(ports, list):
                normalized_ports = []
                for port in ports:
                    if isinstance(port, dict):
                        unsupported = set(port) - {
                            "protocol",
                            "published",
                            "target",
                        }
                        if (
                            unsupported
                            or port.get("protocol", "tcp") != "tcp"
                            or port.get("published") is None
                            or port.get("target") is None
                        ):
                            raise ValueError(
                                "singularity-compose supports only TCP published "
                                f"ports on service {name!r}: {port!r}"
                            )
                        port = f"{port['published']}:{port['target']}"
                    normalized_ports.append(str(port))
                service["ports"] = normalized_ports

    def _load_compose_config(self) -> dict[str, Any]:
        merged: dict[str, Any] = {"services": {}}
        for path in self._compose_paths():
            try:
                loaded = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Invalid Docker Compose file {path}: {exc}"
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Docker Compose file must be a mapping: {path}"
                )
            unsupported = set(loaded) - {"name", "services", "version"}
            if unsupported:
                fields = ", ".join(sorted(unsupported))
                raise ValueError(
                    "singularity-compose does not support these top-level "
                    f"Docker Compose fields: {fields}"
                )
            self._normalize_compose_paths(loaded, path.parent)
            self._merge_compose(merged, loaded)

        services = merged.get("services")
        if not isinstance(services, dict):
            raise ValueError("Docker Compose 'services' must be a mapping")
        services.setdefault("main", {})
        for name, service in services.items():
            self._validate_compose_service(name, service, set(services))
        self._validate_compose_dependency_graph(services)
        return merged

    @staticmethod
    def _validate_compose_dependency_graph(
        services: dict[str, dict[str, Any]],
    ) -> None:
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(
                    f"Docker Compose depends_on contains a cycle at {name!r}"
                )
            visiting.add(name)
            for dependency in services[name].get("depends_on", []):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in services:
            visit(name)

    def _validate_compose_service(
        self,
        name: Any,
        service: Any,
        service_names: set[str],
    ) -> None:
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
        ):
            raise ValueError(f"Invalid Docker Compose service name: {name!r}")
        if not isinstance(service, dict):
            raise ValueError(
                f"Docker Compose service {name!r} must be a mapping"
            )
        unsupported = set(service) - self._compose_service_fields
        if unsupported:
            fields = ", ".join(sorted(unsupported))
            raise ValueError(
                "singularity-compose does not support these fields on service "
                f"{name!r}: {fields}"
            )
        if name != "main" and not ({"build", "image"} & set(service)):
            raise ValueError(
                f"Docker Compose sidecar {name!r} requires 'build' or 'image'"
            )
        if "build" in service:
            self._validate_compose_build(name, service["build"])
        image = service.get("image")
        if image is not None and (
            not isinstance(image, str) or not image or "$" in image
        ):
            raise ValueError(
                f"Docker Compose image on service {name!r} must be a "
                "non-interpolated string"
            )

        depends_on = service.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(dependency, str) for dependency in depends_on
        ):
            raise ValueError(
                "singularity-compose supports only the list form of "
                f"depends_on on service {name!r}; health conditions are unsupported"
            )
        missing = set(depends_on) - service_names
        if missing:
            raise ValueError(
                f"Docker Compose service {name!r} depends on unknown services: "
                + ", ".join(sorted(missing))
            )

        self._validate_compose_volumes(name, service.get("volumes", []))
        self._validate_compose_ports(name, service.get("ports", []))
        self._validate_compose_environment(name, service.get("environment"))
        expose = service.get("expose", [])
        if not isinstance(expose, list) or not all(
            isinstance(port, (str, int)) and not isinstance(port, bool)
            for port in expose
        ):
            raise ValueError(
                f"Docker Compose expose on service {name!r} must be a list of ports"
            )

        env_files = service.get("env_file", [])
        if isinstance(env_files, str):
            env_files = [env_files]
        if not isinstance(env_files, list) or not all(
            isinstance(path, str) for path in env_files
        ):
            raise ValueError(
                f"Docker Compose env_file on service {name!r} must be a path or list"
            )
        missing_env_files = [
            path for path in env_files if not Path(path).is_file()
        ]
        if missing_env_files:
            raise FileNotFoundError(
                f"Docker Compose env_file not found for service {name!r}: "
                + ", ".join(missing_env_files)
            )

        command = service.get("command")
        if command is not None and not (
            isinstance(command, str)
            or (
                isinstance(command, list)
                and all(isinstance(argument, str) for argument in command)
            )
        ):
            raise ValueError(
                f"Docker Compose command on service {name!r} must be a string or list"
            )

        deploy = service.get("deploy", {})
        if not isinstance(deploy, dict) or set(deploy) - {"replicas"}:
            raise ValueError(
                "singularity-compose supports only deploy.replicas on service "
                f"{name!r}"
            )
        replicas = deploy.get("replicas", 1)
        if (
            not isinstance(replicas, int)
            or isinstance(replicas, bool)
            or replicas < 1
        ):
            raise ValueError(
                f"Docker Compose deploy.replicas on service {name!r} "
                "must be a positive integer"
            )

    @staticmethod
    def _validate_compose_build(name: str, build: Any) -> None:
        if isinstance(build, str):
            context = Path(build)
            dockerfile = context / "Dockerfile"
        elif isinstance(build, dict):
            unsupported = set(build) - {
                "args",
                "context",
                "dockerfile",
                "target",
            }
            if unsupported:
                fields = ", ".join(sorted(unsupported))
                raise ValueError(
                    "The Dockerfile builder does not support these build fields "
                    f"on service {name!r}: {fields}"
                )
            context_value = build.get("context", ".")
            dockerfile_value = build.get("dockerfile", "Dockerfile")
            if not isinstance(context_value, str) or not isinstance(
                dockerfile_value, str
            ):
                raise ValueError(
                    f"Docker Compose build paths on service {name!r} must be strings"
                )
            context = Path(context_value)
            dockerfile = context / dockerfile_value
            args = build.get("args", {})
            if not isinstance(args, (dict, list)):
                raise ValueError(
                    f"Docker Compose build.args on service {name!r} "
                    "must be a mapping or list"
                )
            target = build.get("target")
            if target is not None and not isinstance(target, str):
                raise ValueError(
                    f"Docker Compose build.target on service {name!r} must be a string"
                )
        else:
            raise ValueError(
                f"Docker Compose build on service {name!r} must be a path or mapping"
            )
        if not context.is_dir():
            raise FileNotFoundError(
                f"Docker Compose build context not found for service {name!r}: {context}"
            )
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"Dockerfile not found for service {name!r}: {dockerfile}"
            )

    @staticmethod
    def _validate_compose_volumes(name: str, volumes: Any) -> None:
        if not isinstance(volumes, list):
            raise ValueError(
                f"Docker Compose volumes on service {name!r} must be a list"
            )
        for volume in volumes:
            if not isinstance(volume, str) or volume.count(":") != 1:
                raise ValueError(
                    "singularity-compose supports only 'host:container' bind "
                    f"mounts on service {name!r}: {volume!r}"
                )
            source, target = volume.split(":", 1)
            if (
                not Path(source).is_absolute()
                or not Path(source).exists()
                or not PurePosixPath(target).is_absolute()
            ):
                raise ValueError(
                    "singularity-compose requires an existing host path and an "
                    f"absolute container path on service {name!r}: {volume!r}"
                )

    @staticmethod
    def _validate_compose_ports(name: str, ports: Any) -> None:
        if not isinstance(ports, list):
            raise ValueError(
                f"Docker Compose ports on service {name!r} must be a list"
            )
        pattern = re.compile(r"^\d+:\d+$")
        if any(
            not isinstance(port, str) or not pattern.fullmatch(port)
            for port in ports
        ):
            raise ValueError(
                "singularity-compose supports only TCP 'host:container' port "
                f"mappings on service {name!r}"
            )

    @staticmethod
    def _validate_compose_environment(name: str, environment: Any) -> None:
        if environment is None:
            return
        if isinstance(environment, dict):
            if not all(
                isinstance(key, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                for key in environment
            ):
                raise ValueError(
                    f"Docker Compose environment keys on service {name!r} are invalid"
                )
            return
        if isinstance(environment, list) and all(
            isinstance(entry, str)
            and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:=.*)?", entry, re.DOTALL
            )
            for entry in environment
        ):
            return
        raise ValueError(
            f"Docker Compose environment on service {name!r} must be a mapping or list"
        )

    @override
    def _resolve_workdir(self) -> str:
        """Resolve task overrides and Dockerfile WORKDIR instructions."""
        if self.task_env_config.workdir is not None:
            return self.task_env_config.workdir
        dockerfile_path = self._dockerfile_path
        compose_build = self._main_compose_build_inputs()
        if compose_build is not None:
            dockerfile_path = compose_build[0]
        elif not dockerfile_path.is_file():
            return "/"
        workdir = PurePosixPath("/")
        for line in dockerfile_path.read_text().splitlines():
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

    def _dockerfile_cache_path_sync(
        self,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> Path:
        dockerfile_path = (dockerfile_path or self._dockerfile_path).resolve()
        context_dir = (context_dir or self.environment_dir).resolve()
        digest = hashlib.sha256()
        digest.update(platform.machine().encode())
        digest.update("\0".join(build_args).encode())
        digest.update((target or "").encode())
        ignore_file = context_dir / ".dockerignore"
        ignore = (
            GitIgnoreSpec.from_lines(ignore_file.read_text().splitlines())
            if ignore_file.is_file()
            else None
        )
        dockerfile_in_context: str | None = None
        with suppress(ValueError):
            dockerfile_in_context = dockerfile_path.relative_to(
                context_dir
            ).as_posix()
        for path in sorted(context_dir.rglob("*")):
            relative = path.relative_to(context_dir).as_posix()
            if (
                ignore is not None
                and relative not in {dockerfile_in_context, ".dockerignore"}
                and ignore.match_file(relative)
            ):
                continue
            digest.update(relative.encode())
            digest.update(str(path.lstat().st_mode).encode())
            if path.is_symlink():
                digest.update(os.readlink(path).encode())
            elif path.is_file():
                digest.update(path.read_bytes())
        if dockerfile_in_context is None:
            digest.update(str(dockerfile_path).encode())
            digest.update(dockerfile_path.read_bytes())
        return (
            self._image_cache_dir / f"dockerfile-{digest.hexdigest()[:24]}.sif"
        )

    async def _dockerfile_cache_path(
        self,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> Path:
        return await asyncio.to_thread(
            self._dockerfile_cache_path_sync,
            dockerfile_path,
            context_dir,
            build_args,
            target,
        )

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
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> None:
        dockerfile_path = dockerfile_path or self._dockerfile_path
        context_dir = context_dir or self.environment_dir
        builder_name = Path(builder).name
        tag = f"ursa-harbor-{output.stem}-{builder_name}-{secrets.token_hex(8)}"
        pull_args = ["--pull"] if force_build else []
        if builder_name == "buildah":
            remove_command = (builder, "rmi", "--force", tag)
        else:
            remove_command = (builder, "image", "rm", "--force", tag)
        temporary.unlink(missing_ok=True)
        builder_options = [
            argument
            for build_arg in build_args
            for argument in ("--build-arg", build_arg)
        ]
        if target is not None:
            builder_options.extend(["--target", target])
        try:
            await self._run(
                builder,
                "build",
                *pull_args,
                *builder_options,
                "--tag",
                tag,
                "--file",
                str(dockerfile_path),
                str(context_dir),
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

    async def _run(
        self,
        *command: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                cwd=cwd,
                env=env,
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

    async def _build_dockerfile_sif(
        self,
        force_build: bool,
        *,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> Path:
        output = await self._dockerfile_cache_path(
            dockerfile_path,
            context_dir,
            build_args,
            target,
        )
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
                        candidate,
                        output,
                        temporary,
                        force_build,
                        dockerfile_path,
                        context_dir,
                        build_args,
                        target,
                    )
                    return output
                except RuntimeError as exc:
                    failures.append(str(exc))
            raise RuntimeError(
                "No container builder succeeded: " + "; ".join(failures)
            )

    def _compose_key(self, service: str) -> str:
        session = getattr(
            self,
            "_compose_identity",
            hashlib.sha256(self.session_id.encode()).hexdigest()[:16],
        )
        service_hash = hashlib.sha256(service.encode()).hexdigest()[:6]
        safe_service = re.sub(r"[^A-Za-z0-9_-]", "_", service)[:24]
        return f"u{session}_{safe_service}_{service_hash}"

    @staticmethod
    def _compose_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    @classmethod
    def _resolve_compose_value(cls, key: str, value: Any) -> str:
        if value is None:
            try:
                return os.environ[key]
            except KeyError as exc:
                raise ValueError(
                    f"Environment variable {key!r} is not set on the host"
                ) from exc
        scalar = cls._compose_scalar(value)
        literal_marker = "\0URSA_COMPOSE_DOLLAR\0"
        template = scalar.replace("$$", literal_marker)
        resolved = resolve_env_vars({key: template})[key]
        if "$" in template and resolved == template:
            raise ValueError(
                "Compose interpolation is supported only when the whole value "
                f"is a ${{VARIABLE}} template: {scalar!r}"
            )
        return resolved.replace(literal_marker, "$")

    def _compose_build_inputs(
        self, build: str | dict[str, Any]
    ) -> tuple[Path, Path, tuple[str, ...], str | None]:
        if isinstance(build, str):
            context = Path(build)
            return context / "Dockerfile", context, (), None
        context = Path(build.get("context", "."))
        dockerfile = context / build.get("dockerfile", "Dockerfile")
        raw_args = build.get("args", {})
        resolved_args: dict[str, str] = {}
        if isinstance(raw_args, dict):
            resolved_args = {
                str(key): self._resolve_compose_value(str(key), value)
                for key, value in raw_args.items()
            }
        else:
            for entry in raw_args:
                key, separator, value = entry.partition("=")
                resolved_args[key] = self._resolve_compose_value(
                    key, value if separator else None
                )
        args = tuple(
            f"{key}={value}" for key, value in sorted(resolved_args.items())
        )
        return dockerfile, context, args, build.get("target")

    def _main_compose_build_inputs(
        self,
    ) -> tuple[Path, Path, tuple[str, ...], str | None] | None:
        if not self._uses_compose:
            return None
        build = self._load_compose_config()["services"]["main"].get("build")
        if build is None:
            return None
        return self._compose_build_inputs(build)

    async def _build_main_sif(self, force_build: bool) -> Path | None:
        build = self._main_compose_build_inputs()
        if build is None and self._uses_compose:
            main = self._load_compose_config()["services"]["main"]
            if "image" in main:
                return None
        if build is None:
            return await self._build_dockerfile_sif(force_build)
        dockerfile, context, build_args, target = build
        return await self._build_dockerfile_sif(
            force_build,
            dockerfile_path=dockerfile,
            context_dir=context,
            build_args=build_args,
            target=target,
        )

    @classmethod
    def _read_compose_env_file(cls, path: Path) -> dict[str, str]:
        environment: dict[str, str] = {}
        for number, raw_line in enumerate(path.read_text().splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, value = line.partition("=")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(
                    f"Invalid environment entry at {path}:{number}: {raw_line!r}"
                )
            if not separator:
                environment[key] = cls._resolve_compose_value(key, None)
                continue
            if len(value) >= 2 and value[0] == value[-1] == "'":
                environment[key] = value[1:-1]
                continue
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            environment[key] = cls._resolve_compose_value(key, value)
        return environment

    def _compose_environment(
        self, service_name: str, service: dict[str, Any]
    ) -> dict[str, str]:
        environment: dict[str, str] = {}
        env_files = service.get("env_file", [])
        if isinstance(env_files, str):
            env_files = [env_files]
        for path in env_files:
            environment.update(self._read_compose_env_file(Path(path)))

        configured = service.get("environment", {})
        if isinstance(configured, dict):
            environment.update({
                key: self._resolve_compose_value(key, value)
                for key, value in configured.items()
            })
        else:
            for entry in configured:
                key, separator, value = entry.partition("=")
                environment[key] = self._resolve_compose_value(
                    key, value if separator else None
                )
        if service_name == "main":
            environment.update(self._startup_env())
        return environment

    def _write_compose_environment(
        self,
        service_name: str,
        environment: dict[str, str],
    ) -> Path | None:
        if not environment:
            return None
        if self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        invalid = [
            key
            for key in environment
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
        ]
        if invalid:
            raise ValueError(
                "Invalid Compose environment variable names: "
                + ", ".join(sorted(invalid))
            )
        path = (
            self._compose_project_dir
            / f"{self._compose_key(service_name)}.env.sh"
        )
        content = "".join(
            f"export {key}={shlex.quote(value)}\n"
            for key, value in sorted(environment.items())
        )
        path.write_text(content)
        path.chmod(0o600)
        return path

    async def _compose_image(
        self,
        service_name: str,
        service: dict[str, Any],
        force_build: bool,
    ) -> str:
        if service_name == "main":
            if "build" in service or "image" not in service:
                if self._sif_path is None:
                    raise RuntimeError("Main Singularity image is not prepared")
                return str(self._sif_path)
        if "build" in service:
            dockerfile, context, build_args, target = (
                self._compose_build_inputs(service["build"])
            )
            return str(
                await self._build_dockerfile_sif(
                    force_build,
                    dockerfile_path=dockerfile,
                    context_dir=context,
                    build_args=build_args,
                    target=target,
                )
            )
        image = service["image"]
        if not isinstance(image, str) or not image:
            raise ValueError(
                f"Docker Compose image on service {service_name!r} "
                "must be a non-empty string"
            )
        if "://" not in image:
            image = f"docker://{image}"
        return image

    def _compose_volumes(
        self,
        service_name: str,
        service: dict[str, Any],
        environment_file: Path | None,
    ) -> list[str]:
        volumes = list(service.get("volumes", []))
        targets = {volume.split(":", 1)[1] for volume in volumes}
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        if "/staging" in targets:
            raise ValueError("/staging is reserved by the Harbor environment")
        volumes.append(f"{self._staging_dir}:/staging")
        targets.add("/staging")
        if service_name == "main":
            for mount in self._mounts:
                if mount.get("type") != "bind":
                    continue
                target = str(mount["target"])
                if target in targets or target == "/staging":
                    raise ValueError(
                        f"Duplicate or reserved main service mount target: {target}"
                    )
                targets.add(target)
                volumes.append(f"{mount['source']}:{target}")
        if environment_file is not None:
            target = "/.singularity.d/env/91-ursa-compose.sh"
            if target in targets:
                raise ValueError(
                    f"Compose environment mount conflicts on service {service_name!r}"
                )
            volumes.append(f"{environment_file}:{target}")
        return volumes

    async def _prepare_compose_project(self, force_build: bool) -> None:
        config = self._load_compose_config()
        self._compose_project_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-compose-")
        )
        self._compose_project_dir.chmod(0o700)
        instances: dict[str, dict[str, Any]] = {}
        service_keys = {
            name: self._compose_key(name) for name in config["services"]
        }
        self._compose_instances = {}

        for name, service in config["services"].items():
            key = service_keys[name]
            environment_file = self._write_compose_environment(
                name, self._compose_environment(name, service)
            )
            instance: dict[str, Any] = {
                "image": await self._compose_image(name, service, force_build),
                "network": {
                    "allocate_ip": self._network_policy.network_mode
                    != NetworkMode.NO_NETWORK,
                    "enable": True,
                },
                "start": {"options": ["fakeroot", "containall", "no-home"]},
            }
            if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
                instance["network"]["type"] = "none"
            volumes = self._compose_volumes(name, service, environment_file)
            if volumes:
                instance["volumes"] = volumes
            depends_on = service.get("depends_on", [])
            if depends_on:
                instance["depends_on"] = [
                    service_keys[dependency] for dependency in depends_on
                ]
            ports = service.get("ports", [])
            if ports:
                instance["ports"] = ports
            command = service.get("command")
            if command:
                instance["start"]["args"] = (
                    shlex.join(command)
                    if isinstance(command, list)
                    else command
                )
            replicas = service.get("deploy", {}).get("replicas", 1)
            if replicas != 1:
                instance["deploy"] = {"replicas": replicas}
            instances[key] = instance
            self._compose_instances[name] = [
                f"{key}{replica}" for replica in range(1, replicas + 1)
            ]

        self._instance_name = self._compose_instances["main"][0]
        self._compose_file = (
            self._compose_project_dir / "singularity-compose.yml"
        )
        self._compose_file.write_text(
            yaml.safe_dump(
                {"version": "2.0", "instances": instances},
                sort_keys=False,
            )
        )

    async def _run_compose(self, *arguments: str) -> None:
        if self._compose_file is None or self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        executable = shutil.which("singularity-compose")
        if executable is None:
            raise RuntimeError(
                "Docker Compose tasks require singularity-compose 0.1 or later"
            )
        await self._run(
            executable,
            "--file",
            str(self._compose_file),
            "--project-name",
            self._compose_key("project"),
            *arguments,
            cwd=self._compose_project_dir,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
            },
        )

    def _add_compose_service_aliases(self) -> None:
        if self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        hosts_path = self._compose_project_dir / "etc.hosts"
        if not hosts_path.is_file():
            raise RuntimeError("singularity-compose did not create etc.hosts")
        lines = hosts_path.read_text().splitlines()
        aliases = []
        for service, instance_names in self._compose_instances.items():
            instance_name = instance_names[0]
            address = next(
                (
                    line.split()[0]
                    for line in lines
                    if len(line.split()) >= 2
                    and line.split()[1] == instance_name
                ),
                None,
            )
            if address is None:
                raise RuntimeError(
                    "singularity-compose did not assign an address to "
                    f"service {service!r}"
                )
            aliases.append(f"{address}\t{service}\n")
        with hosts_path.open("a") as hosts:
            hosts.writelines(aliases)

    def _cleanup_compose_project(self) -> None:
        project_dir = getattr(self, "_compose_project_dir", None)
        if project_dir is not None:
            shutil.rmtree(project_dir, ignore_errors=True)
        self._compose_project_dir = None
        self._compose_file = None
        self._compose_instances = {}
        self._running_instances = set()

    @property
    def _instance_ref(self) -> str:
        return f"instance://{self._instance_name}"

    def _instance_exec_prefix(
        self,
        cwd: str | None = None,
        *,
        instance_name: str | None = None,
        default_cwd: str | None = None,
    ) -> list[str]:
        return [
            self._instance_runtime(),
            "exec",
            "--cleanenv",
            "--pwd",
            cwd or default_cwd or self._workdir,
            f"instance://{instance_name or self._instance_name}",
        ]

    def _instance_start_command(self) -> list[str]:
        if self._staging_dir is None or self._sif_path is None:
            raise RuntimeError("Singularity instance is not prepared")
        command = [
            self._instance_runtime(),
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
                self._instance_runtime(),
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

    async def _stop_compose(self, *, warn: bool) -> None:
        try:
            await self._run_compose("down", "--timeout", "0")
        except RuntimeError as exc:
            if warn:
                self.logger.warning(
                    "Failed to stop singularity-compose project: %s", exc
                )
        finally:
            self._instance_started = False
            self._running_instances = set()

    @override
    async def start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("Singularity is unavailable on Windows")
        self._validate_definition()
        self._sif_path = await self._build_main_sif(
            force_build or self._force_pull
        )
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-singularity-")
        )
        self._staging_dir.chmod(0o755)
        try:
            async with asyncio.timeout(self._startup_timeout_sec):
                if self._uses_compose:
                    await self._prepare_compose_project(
                        force_build or self._force_pull
                    )
                    compose_lock = self._image_cache_dir / "compose-start.lock"
                    async with AsyncFileLock(compose_lock):
                        await self._run_compose("up")
                    self._add_compose_service_aliases()
                    self._running_instances = {
                        instance
                        for instances in self._compose_instances.values()
                        for instance in instances
                    }
                else:
                    await self._run(*self._instance_start_command())
                self._instance_started = True
                await self._run(*self._instance_exec_prefix(), "true")
            await self._upload_environment_dir_after_start()
        except TimeoutError as exc:
            if self._uses_compose:
                if self._compose_file is not None:
                    await self._stop_compose(warn=False)
                self._cleanup_compose_project()
            else:
                await self._stop_instance(warn=False)
            self._cleanup_staging()
            raise TimeoutError(
                "Singularity instance did not become ready within "
                f"{self._startup_timeout_sec:g} seconds"
            ) from exc
        except BaseException:
            # An interrupted instance-start command can still leave an instance
            # behind, so cleanup is attempted before start reports success.
            if self._uses_compose:
                if self._compose_file is not None:
                    await self._stop_compose(warn=False)
                self._cleanup_compose_project()
            else:
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
            if self._uses_compose:
                if self._compose_file is not None and self._running_instances:
                    await self._stop_compose(warn=True)
            elif self._instance_started:
                await self._stop_instance(warn=True)
        finally:
            self._cleanup_compose_project()
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
        *,
        default_cwd: str | None = None,
        resolve_default_user: bool = True,
    ) -> str:
        command = f"cd {shlex.quote(cwd or default_cwd or self._workdir)} && {command}"
        resolved_user = (
            self._resolve_user(user) if resolve_default_user else user
        )
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
        self,
        env: dict[str, str] | None,
        *,
        merge_persistent: bool = True,
    ) -> dict[str, str]:
        runtime_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        prefix = (
            "APPTAINERENV_"
            if Path(self._instance_runtime()).name == "apptainer"
            else "SINGULARITYENV_"
        )
        merged = self._merge_env(env) if merge_persistent else env
        for key, value in (merged or {}).items():
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError(f"Invalid environment variable: {key!r}")
            runtime_env[f"{prefix}{key}"] = value
        return runtime_env

    async def _cleanup_exec_process(
        self,
        process: asyncio.subprocess.Process,
        pid_file: str,
        *,
        instance_name: str | None = None,
        default_cwd: str | None = None,
    ) -> None:
        cleanup = asyncio.create_task(
            self._run(
                *self._instance_exec_prefix(
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                ),
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

        return await self._exec_instance(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _exec_instance(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
        instance_name: str | None = None,
        default_cwd: str | None = None,
        merge_persistent: bool = True,
    ) -> ExecResult:
        pid_file = f"/tmp/harbor-exec-{secrets.token_hex(8)}.pid"
        shell_command = self._exec_shell_command(
            command,
            cwd,
            user,
            pid_file,
            default_cwd=default_cwd,
            resolve_default_user=merge_persistent,
        )
        runtime_command = [
            *self._instance_exec_prefix(
                cwd,
                instance_name=instance_name,
                default_cwd=default_cwd,
            ),
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
                env=self._runtime_environment(
                    env, merge_persistent=merge_persistent
                ),
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
            if instance_name is None:
                await self._cleanup_exec_process(process, pid_file)
            else:
                await self._cleanup_exec_process(
                    process,
                    pid_file,
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                )
            stdout, stderr = await communication
        except asyncio.CancelledError:
            cleanup_call = (
                self._cleanup_exec_process(process, pid_file)
                if instance_name is None
                else self._cleanup_exec_process(
                    process,
                    pid_file,
                    instance_name=instance_name,
                    default_cwd=default_cwd,
                )
            )
            cleanup = asyncio.create_task(cleanup_call)
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

    def _compose_instance(self, service: str) -> str:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        instance = instances[0]
        if instance not in self._running_instances:
            raise RuntimeError(
                f"Docker Compose service {service!r} is not running"
            )
        return instance

    @override
    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.is_main_service(service):
            return await self.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
        instance = self._compose_instance(service)
        return await self._exec_instance(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
            instance_name=instance,
            default_cwd="/",
            merge_persistent=False,
        )

    @override
    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await self.download_file(source_path, target_path)
            return
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        staged_name = f"service-download-{secrets.token_hex(8)}"
        result = await self.service_exec(
            f"cp {shlex.quote(source_path)} /staging/{staged_name}",
            service=service,
        )
        if result.return_code != 0:
            error = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download sidecar file: {error}")
        staged = self._staging_dir / staged_name
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(staged, target)
        finally:
            staged.unlink(missing_ok=True)

    @override
    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        if self.is_main_service(service):
            await self.download_dir(source_dir, target_dir)
            return
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        staged_name = f"service-download-{secrets.token_hex(8)}"
        result = await self.service_exec(
            f"cp -r {shlex.quote(source_dir)} /staging/{staged_name}",
            service=service,
        )
        if result.return_code != 0:
            error = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to download sidecar directory: {error}")
        staged = self._staging_dir / staged_name
        target = Path(target_dir)
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staged, target, dirs_exist_ok=True)
        finally:
            shutil.rmtree(staged, ignore_errors=True)

    @override
    async def stop_service(self, service: str) -> None:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        for instance in instances:
            if instance not in self._running_instances:
                continue
            await self._run(
                self._instance_runtime(),
                "instance",
                "stop",
                "--force",
                instance,
            )
            self._running_instances.discard(instance)
        if service == "main":
            self._instance_started = False

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
