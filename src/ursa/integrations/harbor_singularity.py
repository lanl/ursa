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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, override

import yaml
from filelock import AsyncFileLock
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.env import resolve_env_vars
from pathspec import GitIgnoreSpec

_COMPOSE_SERVICE_FIELDS = frozenset({
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
_COMPOSE_BUILD_FIELDS = frozenset({"args", "context", "dockerfile", "target"})


def _merge_compose(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_compose(current, value)
        else:
            base[key] = value


def _compose_path(value: str, base_dir: Path) -> str:
    if "$" in value:
        raise ValueError(
            "singularity-compose cannot interpolate variables in paths: "
            f"{value!r}"
        )
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base_dir / path).resolve())


def _normalize_compose_service(
    name: str, service: dict[str, Any], base_dir: Path
) -> None:
    unsupported = set(service) - _COMPOSE_SERVICE_FIELDS
    if unsupported:
        raise ValueError(
            "singularity-compose does not support these fields on service "
            f"{name!r}: {', '.join(sorted(unsupported))}"
        )

    build = service.get("build")
    if isinstance(build, str):
        service["build"] = _compose_path(build, base_dir)
    elif isinstance(build, dict):
        unsupported = set(build) - _COMPOSE_BUILD_FIELDS
        if unsupported:
            raise ValueError(
                "singularity-compose does not support these build fields on "
                f"service {name!r}: {', '.join(sorted(unsupported))}"
            )
        context = build.get("context", ".")
        if not isinstance(context, str):
            raise ValueError(
                f"Docker Compose build context on service {name!r} must be a string"
            )
        build["context"] = _compose_path(context, base_dir)
        if not isinstance(build.get("dockerfile", "Dockerfile"), str):
            raise ValueError(
                f"Docker Compose dockerfile on service {name!r} must be a string"
            )
        if not isinstance(build.get("target", ""), str):
            raise ValueError(
                f"Docker Compose build target on service {name!r} must be a string"
            )
        if not isinstance(build.get("args", {}), (dict, list)):
            raise ValueError(
                f"Docker Compose build args on service {name!r} must be a mapping or list"
            )
        build_args = build.get("args", {})
        if (
            isinstance(build_args, dict)
            and not all(isinstance(key, str) for key in build_args)
        ) or (
            isinstance(build_args, list)
            and not all(isinstance(entry, str) for entry in build_args)
        ):
            raise ValueError(
                f"Docker Compose build args on service {name!r} must use string names"
            )
    elif build is not None:
        raise ValueError(
            f"Docker Compose build on service {name!r} must be a path or mapping"
        )

    command = service.get("command", "")
    if not isinstance(command, (str, list)) or (
        isinstance(command, list)
        and not all(isinstance(argument, str) for argument in command)
    ):
        raise ValueError(
            f"Docker Compose command on service {name!r} must contain only strings"
        )
    environment = service.get("environment", {})
    if (
        not isinstance(environment, (dict, list))
        or (
            isinstance(environment, dict)
            and not all(isinstance(key, str) for key in environment)
        )
        or (
            isinstance(environment, list)
            and not all(isinstance(entry, str) for entry in environment)
        )
    ):
        raise ValueError(
            f"Docker Compose environment on service {name!r} must contain only strings"
        )

    deploy = service.get("deploy")
    if deploy is not None:
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
                f"Docker Compose replicas on service {name!r} must be a positive integer"
            )

    env_files = service.get("env_file")
    if env_files is not None:
        entries = env_files if isinstance(env_files, list) else [env_files]
        normalized = []
        for entry in entries:
            required = True
            if isinstance(entry, dict):
                if (
                    set(entry) - {"path", "required"}
                    or not isinstance(entry.get("path"), str)
                    or not isinstance(entry.get("required", True), bool)
                ):
                    raise ValueError(
                        "singularity-compose cannot represent env_file options "
                        f"on service {name!r}: {entry!r}"
                    )
                required = entry.get("required", True)
                entry = entry["path"]
            if not isinstance(entry, str):
                raise ValueError(
                    f"Docker Compose env_file on service {name!r} must contain paths"
                )
            path = _compose_path(entry, base_dir)
            if required or Path(path).is_file():
                normalized.append(path)
        service["env_file"] = normalized

    depends_on = service.get("depends_on")
    if isinstance(depends_on, dict):
        for dependency, options in depends_on.items():
            if not isinstance(dependency, str):
                raise ValueError(
                    f"Docker Compose depends_on on service {name!r} must use string names"
                )
            options = options or {}
            if not isinstance(options, dict) or (
                set(options) - {"condition", "required", "restart"}
                or options.get("condition", "service_started")
                != "service_started"
                or options.get("required", True) is not True
                or options.get("restart", False) is not False
            ):
                raise ValueError(
                    "singularity-compose supports only service_started "
                    f"depends_on conditions on service {name!r}"
                )
        service["depends_on"] = list(depends_on)
    elif depends_on is not None and (
        not isinstance(depends_on, list)
        or not all(isinstance(dependency, str) for dependency in depends_on)
    ):
        raise ValueError(
            f"Docker Compose depends_on on service {name!r} must be a list of names"
        )

    volumes = service.get("volumes")
    if volumes is not None and not isinstance(volumes, list):
        raise ValueError(
            f"Docker Compose volumes on service {name!r} must be a list"
        )
    if isinstance(volumes, list):
        for index, volume in enumerate(volumes):
            if not isinstance(volume, dict):
                continue
            if (
                set(volume) - {"read_only", "source", "target", "type"}
                or volume.get("type") != "bind"
                or volume.get("read_only", False) is not False
                or not isinstance(volume.get("source"), str)
                or not isinstance(volume.get("target"), str)
            ):
                raise ValueError(
                    "singularity-compose supports only writable bind mounts "
                    f"on service {name!r}: {volume!r}"
                )
            volumes[index] = f"{volume['source']}:{volume['target']}"
        for index, volume in enumerate(volumes):
            if not isinstance(volume, str) or volume.count(":") != 1:
                raise ValueError(
                    "singularity-compose supports only writable bind mounts "
                    f"on service {name!r}: {volume!r}"
                )
            source, target = volume.split(":", 1)
            if not source.startswith(("/", ".", "~")) or not target.startswith(
                "/"
            ):
                raise ValueError(
                    "singularity-compose supports only writable bind mounts "
                    f"on service {name!r}: {volume!r}"
                )
            source = _compose_path(source, base_dir)
            volumes[index] = f"{source}:{target}"

    ports = service.get("ports")
    if ports is not None and not isinstance(ports, list):
        raise ValueError(
            f"Docker Compose ports on service {name!r} must be a list"
        )
    if isinstance(ports, list):
        for index, port in enumerate(ports):
            if not isinstance(port, dict):
                port = str(port)
                if re.fullmatch(r"[0-9]+:[0-9]+", port) is None:
                    raise ValueError(
                        "singularity-compose supports only TCP published ports "
                        f"on service {name!r}: {port!r}"
                    )
                ports[index] = port
                continue
            if (
                set(port) - {"protocol", "published", "target"}
                or port.get("protocol", "tcp") != "tcp"
                or port.get("published") is None
                or port.get("target") is None
            ):
                raise ValueError(
                    "singularity-compose supports only TCP published ports "
                    f"on service {name!r}: {port!r}"
                )
            ports[index] = f"{port['published']}:{port['target']}"


def _load_docker_compose(paths: list[Path]) -> dict[str, Any]:
    config: dict[str, Any] = {"services": {}}
    for path in paths:
        try:
            overlay = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid Docker Compose file {path}: {exc}"
            ) from exc
        if not isinstance(overlay, dict):
            raise ValueError(f"Docker Compose file must be a mapping: {path}")
        unsupported = set(overlay) - {"name", "services", "version"}
        if unsupported:
            raise ValueError(
                "singularity-compose does not support these top-level Docker "
                f"Compose fields: {', '.join(sorted(unsupported))}"
            )
        services = overlay.get("services", {})
        if not isinstance(services, dict):
            raise ValueError("Docker Compose 'services' must be a mapping")
        for name, service in services.items():
            if not isinstance(name, str) or not isinstance(service, dict):
                raise ValueError(
                    "Docker Compose services must be named mappings"
                )
            _normalize_compose_service(name, service, path.parent)
        _merge_compose(config, overlay)
    config["services"].setdefault("main", {})
    services = config["services"]
    for name, service in services.items():
        unknown = set(service.get("depends_on", [])) - set(services)
        if unknown:
            raise ValueError(
                f"Docker Compose service {name!r} depends on unknown services: "
                + ", ".join(sorted(unknown))
            )

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            cycle = visiting[visiting.index(name) :] + [name]
            raise ValueError(
                "Docker Compose dependency cycle: " + " -> ".join(cycle)
            )
        if name in visited:
            return
        visiting.append(name)
        for dependency in services[name].get("depends_on", []):
            visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in services:
        visit(name)
    return config


def _compose_key(identity: str, service: str) -> str:
    service_hash = hashlib.sha256(service.encode()).hexdigest()[:6]
    safe_service = re.sub(r"[^a-z0-9-]", "-", service.lower())[:24].rstrip("-")
    return f"u{identity}-{safe_service}-{service_hash}"


def _resolve_compose_value(key: str, value: Any) -> str:
    if value is None:
        try:
            return os.environ[key]
        except KeyError as exc:
            raise ValueError(
                f"Environment variable {key!r} is not set on the host"
            ) from exc
    scalar = str(value).lower() if isinstance(value, bool) else str(value)
    literal_marker = "\0URSA_COMPOSE_DOLLAR\0"
    template = scalar.replace("$$", literal_marker)
    resolved = resolve_env_vars({key: template})[key]
    if "$" in template and resolved == template:
        raise ValueError(
            "Compose interpolation is supported only when the whole value "
            f"is a ${{VARIABLE}} template: {scalar!r}"
        )
    return resolved.replace(literal_marker, "$")


def _read_compose_env_file(path: Path) -> dict[str, str]:
    environment = {}
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
            environment[key] = _resolve_compose_value(key, None)
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            environment[key] = value[1:-1]
        else:
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            environment[key] = _resolve_compose_value(key, value)
    return environment


def _compose_environment(
    service_name: str,
    service: dict[str, Any],
    main_environment: Mapping[str, str],
) -> dict[str, str]:
    environment = {}
    for path in service.get("env_file", []):
        environment.update(_read_compose_env_file(Path(path)))
    configured = service.get("environment", {})
    if isinstance(configured, dict):
        environment.update({
            key: _resolve_compose_value(key, value)
            for key, value in configured.items()
        })
    else:
        for entry in configured:
            key, separator, value = entry.partition("=")
            environment[key] = _resolve_compose_value(
                key, value if separator else None
            )
    if service_name == "main":
        environment.update(main_environment)
    return environment


def _write_compose_environment(
    project_dir: Path,
    identity: str,
    service_name: str,
    environment: dict[str, str],
) -> Path | None:
    if not environment:
        return None
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
    path = project_dir / f"{_compose_key(identity, service_name)}.env.sh"
    path.write_text(
        "".join(
            f"export {key}={shlex.quote(value)}\n"
            for key, value in sorted(environment.items())
        )
    )
    path.chmod(0o600)
    return path


def _compose_volumes(
    service_name: str,
    service: dict[str, Any],
    *,
    staging_dir: Path,
    environment_file: Path | None,
    main_mounts: Sequence[Mapping[str, Any]],
) -> list[str]:
    volumes = list(service.get("volumes", []))
    targets = {volume.split(":", 1)[1] for volume in volumes}
    if "/staging" in targets:
        raise ValueError("/staging is reserved by the Harbor environment")
    volumes.append(f"{staging_dir}:/staging")
    targets.add("/staging")
    if service_name == "main":
        for mount in main_mounts:
            if mount.get("type") != "bind":
                raise ValueError(
                    "Singularity supports only Harbor bind mounts, got "
                    f"{mount.get('type')!r}"
                )
            if mount.get("read_only"):
                raise ValueError(
                    "singularity-compose cannot preserve read-only Harbor mounts"
                )
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


async def docker_compose_to_singularity_compose(
    docker_compose_paths: Path | Sequence[Path],
    singularity_compose_path: Path,
    *,
    identity: str,
    image_resolver: Callable[[str, dict[str, Any]], Awaitable[str]],
    staging_dir: Path,
    main_environment: Mapping[str, str] | None = None,
    main_mounts: Sequence[Mapping[str, Any]] = (),
    network_mode: NetworkMode = NetworkMode.PUBLIC,
    fakeroot: bool = True,
) -> dict[str, list[str]]:
    """Convert one or more Docker Compose files into a singularity-compose file."""
    paths = (
        [docker_compose_paths]
        if isinstance(docker_compose_paths, Path)
        else list(docker_compose_paths)
    )
    config = _load_docker_compose(paths)
    service_keys = {
        name: _compose_key(identity, name) for name in config["services"]
    }
    instances = {}
    instance_names = {}
    project_dir = singularity_compose_path.parent
    project_dir.mkdir(parents=True, exist_ok=True)
    for name, service in config["services"].items():
        key = service_keys[name]
        environment_file = _write_compose_environment(
            project_dir,
            identity,
            name,
            _compose_environment(name, service, main_environment or {}),
        )
        start_options = ["containall", "no-home"]
        if fakeroot:
            start_options.insert(0, "fakeroot")
        instance: dict[str, Any] = {
            "image": await image_resolver(name, service),
            "network": {
                "allocate_ip": network_mode != NetworkMode.NO_NETWORK,
                "enable": True,
            },
            "start": {"options": start_options},
        }
        if network_mode == NetworkMode.NO_NETWORK:
            instance["network"]["type"] = "none"
        volumes = _compose_volumes(
            name,
            service,
            staging_dir=staging_dir,
            environment_file=environment_file,
            main_mounts=main_mounts,
        )
        if volumes:
            instance["volumes"] = volumes
        if depends_on := service.get("depends_on", []):
            instance["depends_on"] = [
                service_keys[dependency] for dependency in depends_on
            ]
        if ports := service.get("ports", []):
            instance["ports"] = ports
        if command := service.get("command"):
            instance["start"]["args"] = (
                shlex.join(command) if isinstance(command, list) else command
            )
        replicas = service.get("deploy", {}).get("replicas", 1)
        if replicas != 1:
            instance["deploy"] = {"replicas": replicas}
        instances[key] = instance
        instance_names[name] = [
            f"{key}{replica}" for replica in range(1, replicas + 1)
        ]
    singularity_compose_path.write_text(
        yaml.safe_dump(
            {"version": "2.0", "instances": instances}, sort_keys=False
        )
    )
    return instance_names


class DockerfileSingularityEnvironment(BaseEnvironment):
    """Run Dockerfile and supported Compose tasks with Singularity or Apptainer."""

    _compose_filename = "docker-compose.yaml"

    def __init__(
        self,
        *args,
        singularity_image_cache_dir: Path | str | None = None,
        singularity_force_pull: bool = False,
        singularity_fakeroot: bool = True,
        singularity_no_mount: str | None = None,
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
        if singularity_no_mount is not None:
            raise ValueError(
                "singularity_no_mount is not configurable; this environment "
                "always isolates mounts with --containall and --no-home"
            )

        self._image_cache_dir = (
            Path(singularity_image_cache_dir)
            if singularity_image_cache_dir
            else self._default_image_cache_dir()
        )
        self._force_pull = singularity_force_pull
        self._fakeroot = singularity_fakeroot
        self._runtime_path: str | None = None
        super().__init__(*args, **kwargs)
        for policy in self._phase_network_policies:
            if policy != self._network_policy:
                raise ValueError(
                    "Singularity 3.6 cannot change network policy after start"
                )
        self._startup_timeout_sec = singularity_startup_timeout_sec
        self._staging_dir: Path | None = None
        self._sif_path: Path | None = None
        self._overlay_path: Path | None = None
        self._workdir = self._resolve_workdir()
        identity = hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
        self._instance_name = f"ursa{identity}{secrets.token_hex(4)}"
        self._compose_identity = f"{identity[:8]}{secrets.token_hex(4)}"
        self._instance_started = False
        self._warned_user_switch_without_fakeroot = False
        self._compose_project_dir: Path | None = None
        self._compose_file: Path | None = None
        self._compose_instances: dict[str, list[str]] = {}

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

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
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _staging(self) -> Path:
        if self._staging_dir is None:
            raise RuntimeError(
                "Staging directory not initialized — call start() first"
            )
        return self._staging_dir

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
    def _compose_start_lock_path() -> Path:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(cache_home).expanduser()
            if cache_home
            else Path.home() / ".cache"
        )
        return root / "ursa" / "harbor" / "singularity-compose.lock"

    @staticmethod
    def _runtime() -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise RuntimeError("Apptainer or Singularity is required")
        return runtime

    def _instance_runtime(self) -> str:
        runtime = getattr(self, "_runtime_path", None)
        if runtime is None:
            runtime = self._runtime()
            self._runtime_path = runtime
        return runtime

    @classmethod
    @override
    def preflight(cls) -> None:
        """Verify that a Singularity-compatible runtime is installed."""
        try:
            cls._runtime()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    @override
    def _validate_definition(self) -> None:
        unsupported_mounts = [
            mount for mount in self._mounts if mount.get("type") != "bind"
        ]
        if unsupported_mounts:
            raise ValueError("Singularity supports only Harbor bind mounts")
        if self._uses_compose and any(
            mount.get("read_only") for mount in self._mounts
        ):
            raise ValueError(
                "singularity-compose cannot preserve read-only Harbor mounts"
            )
        config = None
        if self._uses_compose:
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

    def _load_compose_config(self) -> dict[str, Any]:
        return _load_docker_compose(self._compose_paths())

    @override
    def _resolve_workdir(self) -> str:
        """Resolve task overrides and Dockerfile WORKDIR instructions."""
        if self.task_env_config.workdir is not None:
            return self.task_env_config.workdir
        dockerfile_path = self._dockerfile_path
        main = (
            self._load_compose_config()["services"]["main"]
            if self._uses_compose
            else {}
        )
        compose_build = main.get("build")
        if compose_build is not None:
            dockerfile_path = self._compose_build_paths(compose_build)[0]
        elif self._uses_compose and "image" in main:
            return "/"
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
        dockerfile_path: Path,
        context_dir: Path,
        build_args: tuple[str, ...],
        target: str | None,
    ) -> Path:
        dockerfile_path = dockerfile_path.resolve()
        context_dir = context_dir.resolve()
        digest = hashlib.sha256()
        digest.update(platform.machine().encode())
        digest.update("\0".join(build_args).encode())
        digest.update((target or "").encode())
        ignore_path = context_dir / ".dockerignore"
        ignore = (
            GitIgnoreSpec.from_lines(ignore_path.read_text().splitlines())
            if ignore_path.is_file()
            else None
        )
        dockerfile_in_context = None
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
            dockerfile_path or self._dockerfile_path,
            context_dir or self.environment_dir,
            build_args,
            target,
        )

    async def _is_valid_sif(self, path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            await self._run(self._instance_runtime(), "sif", "list", str(path))
        except RuntimeError:
            return False
        return True

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

    async def _build_with(
        self,
        builder: str,
        output: Path,
        temporary: Path,
        force_build: bool,
        dockerfile_path: Path,
        context_dir: Path,
        build_args: tuple[str, ...],
        target: str | None,
    ) -> None:
        builder_name = Path(builder).name
        tag = f"ursa-harbor-{output.stem}-{builder_name}-{secrets.token_hex(8)}"
        remove_command = (
            (builder, "rmi", "--force", tag)
            if builder_name == "buildah"
            else (builder, "image", "rm", "--force", tag)
        )
        builder_options = [
            option
            for build_arg in build_args
            for option in ("--build-arg", build_arg)
        ]
        if target is not None:
            builder_options.extend(["--target", target])
        temporary.unlink(missing_ok=True)
        try:
            await self._run(
                builder,
                "build",
                *(["--pull"] if force_build else []),
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
                    self._instance_runtime(),
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

    async def _build_dockerfile_sif(
        self,
        force_build: bool,
        *,
        dockerfile_path: Path | None = None,
        context_dir: Path | None = None,
        build_args: tuple[str, ...] = (),
        target: str | None = None,
    ) -> Path:
        dockerfile_path = dockerfile_path or self._dockerfile_path
        context_dir = context_dir or self.environment_dir
        output = await self._dockerfile_cache_path(
            dockerfile_path, context_dir, build_args, target
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
            for builder in builders:
                try:
                    await self._build_with(
                        builder,
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
        identity = getattr(
            self,
            "_compose_identity",
            hashlib.sha256(self.session_id.encode()).hexdigest()[:16],
        )
        return _compose_key(identity, service)

    @staticmethod
    def _compose_build_paths(
        build: str | dict[str, Any],
    ) -> tuple[Path, Path]:
        if isinstance(build, str):
            context = Path(build)
            return context / "Dockerfile", context
        context = Path(build.get("context", "."))
        dockerfile = context / build.get("dockerfile", "Dockerfile")
        return dockerfile, context

    def _compose_build_inputs(
        self, build: str | dict[str, Any]
    ) -> tuple[Path, Path, tuple[str, ...], str | None]:
        dockerfile, context = self._compose_build_paths(build)
        if isinstance(build, str):
            return dockerfile, context, (), None
        raw_args = build.get("args", {})
        resolved_args: dict[str, str] = {}
        if isinstance(raw_args, dict):
            resolved_args = {
                str(key): _resolve_compose_value(str(key), value)
                for key, value in raw_args.items()
            }
        else:
            for entry in raw_args:
                key, separator, value = entry.partition("=")
                resolved_args[key] = _resolve_compose_value(
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

    async def _prepare_compose_project(self, force_build: bool) -> None:
        self._compose_project_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-compose-")
        )
        self._compose_project_dir.chmod(0o700)
        runtime_bin = self._compose_project_dir / "bin"
        runtime_bin.mkdir(mode=0o700)
        (runtime_bin / "singularity").symlink_to(self._instance_runtime())
        self._compose_file = (
            self._compose_project_dir / "singularity-compose.yml"
        )
        self._compose_instances = await docker_compose_to_singularity_compose(
            self._compose_paths(),
            self._compose_file,
            identity=self._compose_identity,
            image_resolver=partial(
                self._compose_image, force_build=force_build
            ),
            staging_dir=self._staging,
            main_environment=self._startup_env(),
            main_mounts=self._mounts,
            network_mode=self._network_policy.network_mode,
            fakeroot=self._fakeroot,
        )
        self._instance_name = self._compose_instances["main"][0]

    async def _run_compose(self, *arguments: str) -> None:
        if self._compose_file is None or self._compose_project_dir is None:
            raise RuntimeError("Singularity Compose project is not prepared")
        executable = shutil.which("singularity-compose")
        if executable is None:
            raise RuntimeError(
                "Docker Compose tasks require singularity-compose; "
                "install `ursa-ai[harbor]`"
            )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        runtime_bin = self._compose_project_dir / "bin"
        environment["PATH"] = os.pathsep.join(
            part for part in (str(runtime_bin), environment.get("PATH")) if part
        )
        await self._run(
            executable,
            "--file",
            str(self._compose_file),
            "--project-name",
            self._compose_key("project"),
            *arguments,
            cwd=self._compose_project_dir,
            env=environment,
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
            "--containall",
            "--no-home",
        ]
        if self._fakeroot:
            command.insert(3, "--fakeroot")
            command.append("--writable-tmpfs")
        else:
            if self._overlay_path is None:
                raise RuntimeError(
                    "Singularity writable overlay is not prepared"
                )
            command.extend(["--overlay", str(self._overlay_path)])
            for source, target in self._rootless_harbor_binds():
                command.extend(["-B", f"{source}:{target}"])
        if self._network_policy.network_mode == NetworkMode.NO_NETWORK:
            command.extend(["--net", "--network", "none"])
        command.extend(["-B", f"{self._staging_dir}:/staging"])
        for mount in self._mounts:
            if mount.get("type") != "bind":
                raise ValueError("Singularity supports only Harbor bind mounts")
            if mount.get("target") == "/staging":
                raise ValueError(
                    "/staging is reserved by the Harbor environment"
                )
            bind = f"{mount['source']}:{mount['target']}"
            if mount.get("read_only"):
                bind += ":ro"
            command.extend(["-B", bind])
        command.extend([str(self._sif_path), self._instance_name])
        return command

    def _rootless_harbor_binds(self) -> tuple[tuple[Path, PurePosixPath], ...]:
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        root = self._staging_dir / "harbor-writable"
        return (
            (root / "logs", EnvironmentPaths.logs_dir),
            (root / "solution", EnvironmentPaths.solution_dir),
            (root / "tests", EnvironmentPaths.tests_dir),
            (root / "skills", EnvironmentPaths.default_skills_dir),
        )

    def _prepare_disk_overlay(self) -> None:
        if self._staging_dir is None:
            raise RuntimeError("Singularity staging directory is not prepared")
        mkfs = shutil.which("mkfs.ext3") or shutil.which("mke2fs")
        if mkfs is None:
            raise RuntimeError(
                "Rootless Singularity requires mkfs.ext3 or mke2fs to "
                "create a writable disk overlay"
            )

        storage_mb = self.task_env_config.storage_mb or 1024
        layout = self._staging_dir / "overlay-layout"
        for directory in (layout / "upper", layout / "work"):
            directory.mkdir(parents=True)
            directory.chmod(0o777)
        writable_paths = (
            EnvironmentPaths.solution_dir,
            EnvironmentPaths.tests_dir,
            EnvironmentPaths.default_skills_dir,
            PurePosixPath(self._workdir),
        )
        for target in writable_paths:
            if not target.is_absolute():
                raise ValueError(
                    f"Singularity writable path must be absolute: {target}"
                )
            relative = target.relative_to("/")
            directory = layout / "upper" / relative
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o777)
        for source, _target in self._rootless_harbor_binds():
            source.mkdir(parents=True, exist_ok=True)
            source.chmod(0o777)
        overlay = self._staging_dir / "overlay.img"
        overlay.touch()
        os.truncate(overlay, storage_mb * 1024 * 1024)
        try:
            subprocess.run(
                [mkfs, "-q", "-d", str(layout), str(overlay)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            overlay.unlink(missing_ok=True)
            detail = getattr(exc, "stderr", None) or str(exc)
            raise RuntimeError(
                f"Could not create Singularity writable overlay: {detail}"
            ) from exc
        finally:
            shutil.rmtree(layout, ignore_errors=True)
        self._overlay_path = overlay

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
            failures = []
            for instance in {
                instance
                for instances in self._compose_instances.values()
                for instance in instances
            }:
                try:
                    await self._run(
                        self._instance_runtime(),
                        "instance",
                        "stop",
                        "--force",
                        instance,
                    )
                except RuntimeError as stop_exc:
                    failures.append(str(stop_exc))
            if failures and warn:
                self.logger.warning(
                    "Failed to stop singularity-compose project (%s) and "
                    "instances (%s)",
                    exc,
                    "; ".join(failures),
                )
            if failures and not warn:
                raise
        finally:
            self._instance_started = False

    async def _cleanup_failed_start(self) -> None:
        """Best-effort runtime cleanup without masking the startup error."""
        try:
            if self._uses_compose:
                if self._compose_file is not None:
                    await self._stop_compose(warn=True)
            else:
                await self._stop_instance(warn=True)
        except Exception as exc:
            self.logger.warning(
                "Failed to clean up Singularity startup: %s", exc
            )
        finally:
            self._cleanup_compose_project()
            self._cleanup_staging()

    @override
    async def start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("Singularity is unavailable on Windows")
        self._sif_path = await self._build_main_sif(
            force_build or self._force_pull
        )
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix="ursa-harbor-singularity-")
        )
        self._staging_dir.chmod(0o755)
        try:
            if not self._uses_compose and not self._fakeroot:
                self._prepare_disk_overlay()
            async with asyncio.timeout(self._startup_timeout_sec):
                if self._uses_compose:
                    await self._prepare_compose_project(
                        force_build or self._force_pull
                    )
                    lock_path = self._compose_start_lock_path()
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    async with AsyncFileLock(lock_path):
                        await self._run_compose("up")
                    self._add_compose_service_aliases()
                else:
                    await self._run(*self._instance_start_command())
                self._instance_started = True
                await self._run(*self._instance_exec_prefix(), "true")
            await self._upload_environment_dir_after_start()
        except TimeoutError as exc:
            await self._cleanup_failed_start()
            raise TimeoutError(
                "Singularity instance did not become ready within "
                f"{self._startup_timeout_sec:g} seconds"
            ) from exc
        except BaseException:
            # An interrupted instance-start command can still leave an instance
            # behind, so cleanup is attempted before start reports success.
            await self._cleanup_failed_start()
            raise

    def _cleanup_staging(self) -> None:
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
            self._staging_dir = None
        self._overlay_path = None

    @override
    async def stop(self, delete: bool) -> None:
        try:
            if self._uses_compose:
                if self._compose_file is not None:
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

    @override
    async def attach(self) -> None:
        """Replace this process with an interactive shell in the main instance."""
        if not self._instance_started:
            raise RuntimeError("Singularity instance is not running")
        command = [
            self._instance_runtime(),
            "shell",
            "--cleanenv",
            "--pwd",
            self._workdir,
            f"instance://{self._instance_name}",
        ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SINGULARITYENV_", "APPTAINERENV_"))
        }
        os.execvpe(command[0], command, environment)

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
        requested_user = (
            self._resolve_user(user) if resolve_default_user else user
        )
        resolved_user = requested_user
        if not self._fakeroot:
            # A rootless Singularity instance can only execute as the invoking
            # host user. Attempting to honor Harbor's requested image user with
            # ``su`` prompts for a container password and prevents every
            # command from starting.
            resolved_user = None
            if requested_user is not None and not getattr(
                self, "_warned_user_switch_without_fakeroot", False
            ):
                self.logger.warning(
                    "Singularity fakeroot is disabled; cannot switch to "
                    "requested container user %r, so commands will run as "
                    "the invoking host user",
                    requested_user,
                )
                self._warned_user_switch_without_fakeroot = True
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

    def _transfer_staging_path(self, name: str) -> Path:
        return self._staging / (
            f"harbor-transfer-{secrets.token_hex(8)}-{Path(name).name}"
        )

    @staticmethod
    def _remove_staging_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _check_transfer(result: ExecResult, operation: str) -> None:
        if result.return_code != 0:
            error = result.stderr or result.stdout or "<no output>"
            raise RuntimeError(f"Failed to {operation}: {error}")

    @override
    async def upload_file(
        self, source_path: Path | str, target_path: str
    ) -> None:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        staged = self._transfer_staging_path(source.name)
        try:
            self._remove_staging_path(staged)
            shutil.copy2(source, staged)
            result = await self.exec(
                f"cp {shlex.quote('/staging/' + staged.name)} "
                f"{shlex.quote(target_path)}"
            )
            self._check_transfer(result, "upload file")
        finally:
            self._remove_staging_path(staged)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        staged = self._transfer_staging_path(source.name)
        try:
            self._remove_staging_path(staged)
            shutil.copytree(source, staged)
            result = await self.exec(f"mkdir -p {shlex.quote(target_dir)}")
            self._check_transfer(result, "prepare upload directory")
            result = await self.exec(
                f"cp -r {shlex.quote('/staging/' + staged.name + '/.')} "
                f"{shlex.quote(target_dir + '/')}"
            )
            self._check_transfer(result, "upload directory")
        finally:
            self._remove_staging_path(staged)

    @override
    async def download_file(
        self, source_path: str, target_path: Path | str
    ) -> None:
        await self._download_file(source_path, target_path, self.exec)

    async def _download_file(
        self,
        source_path: str,
        target_path: Path | str,
        execute: Callable[[str], Awaitable[ExecResult]],
    ) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = self._transfer_staging_path(Path(source_path).name)
        try:
            self._remove_staging_path(staged)
            result = await execute(
                f"cp {shlex.quote(source_path)} "
                f"{shlex.quote('/staging/' + staged.name)}"
            )
            self._check_transfer(result, "download file")
            if not staged.is_file():
                raise RuntimeError(f"File not found in staging: {staged}")
            shutil.copy2(staged, target)
        finally:
            self._remove_staging_path(staged)

    @override
    async def download_dir(
        self, source_dir: str, target_dir: Path | str
    ) -> None:
        await self._download_dir(source_dir, target_dir, self.exec)

    async def _download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        execute: Callable[[str], Awaitable[ExecResult]],
    ) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        staged = self._transfer_staging_path(Path(source_dir).name)
        try:
            self._remove_staging_path(staged)
            result = await execute(
                f"cp -r {shlex.quote(source_dir)} "
                f"{shlex.quote('/staging/' + staged.name)}"
            )
            self._check_transfer(result, "download directory")
            if not staged.is_dir():
                raise RuntimeError(f"Directory not found in staging: {staged}")
            for item in staged.iterdir():
                destination = target / item.name
                if item.is_dir():
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
        finally:
            self._remove_staging_path(staged)

    def _compose_instance(self, service: str) -> str:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        return instances[0]

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
        await self._download_file(
            source_path,
            target_path,
            partial(self.service_exec, service=service),
        )

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
        await self._download_dir(
            source_dir,
            target_dir,
            partial(self.service_exec, service=service),
        )

    @override
    async def stop_service(self, service: str) -> None:
        instances = self._compose_instances.get(service)
        if not instances:
            raise ValueError(f"Unknown Docker Compose service: {service!r}")
        await self._run_compose("stop", "--timeout", "0", *instances)
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


__all__ = [
    "DockerfileSingularityEnvironment",
    "docker_compose_to_singularity_compose",
]
