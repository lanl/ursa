"""Run URSA agents as installed agents in the Harbor benchmark framework.

Harbor starts the adapter on the host.  The adapter then installs URSA and runs
the selected URSA agent *inside* the task container, so filesystem and shell
tools operate on the benchmark workspace rather than on the submit host.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import hashlib
import importlib.metadata
import json
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote, urlparse

import typer
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pydantic import SecretStr

try:
    from harbor.agents.agent_auth_resolution import (
        PROVIDERS,
        ModelConnectionSpec,
    )
    from harbor.agents.capabilities import AgentCapabilities
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import NetworkMode
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The Harbor integration requires `uv add 'ursa-ai[harbor]'`."
    ) from exc

from ursa.agents import BaseAgent as UrsaBaseAgent
from ursa.cli.config import (
    ENV_SUB_REGEX,
    UrsaConfig,
    load_config_file,
)
from ursa.security import DEFAULT_GROUP_NAME, validate_group_name
from ursa.util.secrets import externalize_secret_references

TaskPaths = Annotated[
    list[Path],
    typer.Argument(
        help="Task files, task directories, or roots to scan recursively"
    ),
]
_USAGE_KEYS = ("input_tokens", "cached_tokens", "output_tokens")


def _jsonl_token_usage(
    path: Path,
) -> tuple[int | None, int | None, int | None]:
    totals: dict[str, int | None] = dict.fromkeys(_USAGE_KEYS)
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or (
                    record.get("type"),
                    record.get("event"),
                ) != ("chat", "end"):
                    continue
                usage = record.get("usage")
                if not isinstance(usage, dict):
                    continue
                for key in _USAGE_KEYS:
                    value = usage.get(key)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                    ):
                        totals[key] = (totals[key] or 0) + value
    except OSError:
        return None, None, None

    return (
        totals["input_tokens"],
        totals["cached_tokens"],
        totals["output_tokens"],
    )


class UrsaHarborAgent(BaseInstalledAgent):
    """Generic Harbor binding for an importable URSA ``BaseAgent`` subclass.

    Args:
        agent_import_path: ``module:Class`` path for the URSA agent. Defaults
            to :class:`ursa.agents.ExecutionAgent`.
        config: URSA's native YAML/JSON file or inline JSON-compatible mapping.
            This is Harbor's standard installed-agent configuration input.
        ursa_install_spec: PEP 508 requirement for URSA, including named Git or
            file URL references to a project, wheel, or sdist. Defaults to this
            URSA checkout or installed version.
        ursa_extras: Additional URSA package extras to install, as a sequence or
            comma-separated string. The ``harbor`` extra is always installed.
        extra_packages: Additional Python packages to install. Pass one requirement
            string or a sequence of requirement strings.
    """

    MODEL_CONNECTION = ModelConnectionSpec(
        base_url_destination_names={"default": None},
    )
    capabilities = AgentCapabilities(
        handoff=True, native_config=True, base_url=True
    )
    URSA_PYTHON_VERSION = "3.13"
    URSA_RUNNER = "/installed-agent/bin/ursa-harbor-runner"
    _INSTALL_ROOT = "/installed-agent"
    _UV_VERSION = "0.12.21"
    ENV_AUTH_PROVIDERS = frozenset({
        "amazon-bedrock",
        "sagemaker",
        "vertex_ai",
    })

    def __init__(
        self,
        *args: Any,
        agent_import_path: str = "ursa.agents:ExecutionAgent",
        config: str | Path | dict[str, Any] | None = None,
        ursa_install_spec: str | None = None,
        ursa_extras: str | Sequence[str] | None = None,
        extra_packages: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, config=config, **kwargs)
        self.agent_import_path = agent_import_path
        install_spec = (
            self._default_install_spec()
            if ursa_install_spec is None
            else ursa_install_spec
        )
        self.ursa_install_spec = self._parse_install_spec(install_spec)
        self._install_source = self._local_install_source(
            self.ursa_install_spec
        )
        self._build_default_install_wheel = (
            ursa_install_spec is None
            and self._install_source is not None
            and self._install_source.is_dir()
        )
        requested_extras = self._parse_list(
            ursa_extras, name="ursa_extras", split_commas=True
        )
        self.ursa_extras = self._merge_extras(requested_extras, ("harbor",))
        self.extra_packages = self._parse_list(
            extra_packages, name="extra_packages"
        )
        self._secret_env: dict[str, str] = {}
        self._model_env: dict[str, str] = {}
        self._runtime_group = DEFAULT_GROUP_NAME
        self._workspace = "/"
        self._runner_pid_file = (
            f"{self._INSTALL_ROOT}/tmp/ursa-harbor-runner-"
            f"{secrets.token_hex(16)}.pid"
        )

    @staticmethod
    def _parse_list(
        value: str | Sequence[str] | None,
        *,
        name: str,
        split_commas: bool = False,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            if split_commas:
                values = value.split(",")
            elif value.lstrip().startswith("["):
                message = (
                    f"{name} must be a requirement or JSON array of strings"
                )
                try:
                    values = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(message) from exc
                if not isinstance(values, list):
                    raise ValueError(message)
            else:
                values = [value]
        else:
            values = value
        if not all(isinstance(item, str) for item in values):
            raise ValueError(f"{name} must contain only strings")
        return tuple(item.strip() for item in values if item.strip())

    @staticmethod
    def _is_package_archive(path: Path) -> bool:
        return path.name.endswith((".whl", ".zip", ".tar.gz"))

    @staticmethod
    def _default_install_spec() -> str:
        checkout = Path(__file__).resolve().parents[4]
        if (checkout / "pyproject.toml").is_file():
            return f"ursa-ai @ {checkout.as_uri()}"
        distribution = importlib.metadata.distribution("ursa-ai")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            provenance = json.loads(direct_url)
            url = provenance.get("url")
            if isinstance(url, str):
                vcs = provenance.get("vcs_info", {})
                if isinstance(vcs, dict) and isinstance(vcs.get("vcs"), str):
                    revision = vcs.get("commit_id") or vcs.get(
                        "requested_revision"
                    )
                    suffix = f"@{revision}" if revision else ""
                    url = f"{vcs['vcs']}+{url}{suffix}"
                return f"ursa-ai @ {url}"
        return f"ursa-ai=={distribution.version}"

    @staticmethod
    def _parse_install_spec(value: str) -> Requirement:
        try:
            requirement = Requirement(value)
        except (InvalidRequirement, TypeError) as exc:
            raise ValueError(
                "ursa_install_spec must be a PEP 508 requirement, such as "
                "'ursa-ai==0.17.3' or 'ursa-ai @ file:///path/to/checkout'"
            ) from exc
        if canonicalize_name(requirement.name) != "ursa-ai":
            raise ValueError("ursa_install_spec must name the ursa-ai package")
        return requirement

    @staticmethod
    def _local_install_source(requirement: Requirement) -> Path | None:
        parsed = urlparse(requirement.url or "")
        if parsed.scheme != "file":
            return None
        path = Path(urllib.request.url2pathname(parsed.path))
        if parsed.netloc not in ("", "localhost") or not path.is_absolute():
            raise ValueError(
                "ursa_install_spec must use an absolute local file URL"
            )
        path = path.resolve()
        if not path.exists():
            raise ValueError(f"ursa_install_spec path does not exist: {path}")
        if path.is_file() and UrsaHarborAgent._is_package_archive(path):
            return path
        if not path.is_dir() or not (path / "pyproject.toml").is_file():
            raise ValueError(
                "ursa_install_spec is not a Python project or wheel/sdist "
                f"archive: {path}"
            )
        return path

    @staticmethod
    def _merge_extras(*extra_groups: Sequence[str]) -> tuple[str, ...]:
        extras: dict[str, None] = {}
        for group in extra_groups:
            for extra in group:
                normalized = canonicalize_name(extra.strip())
                if normalized:
                    extras[normalized] = None
        return tuple(extras)

    @staticmethod
    def _github_archive_target(target: str) -> str:
        """Use a GitHub archive so task images do not need a Git client."""
        match = re.fullmatch(
            r"git\+https://github\.com/(?P<repository>[^?#]+?)"
            r"(?:\.git)?@(?P<revision>[^#;\s]+)",
            target,
        )
        if match is None:
            return target
        repository = match.group("repository").removesuffix(".git")
        revision = quote(match.group("revision"), safe="")
        return f"https://github.com/{repository}/archive/{revision}.tar.gz"

    def _mcp_config(self) -> dict[str, dict[str, Any]]:
        """Convert Harbor's MCP list to URSA's named mapping."""
        return {
            server.name: server.model_dump(exclude={"name"}, exclude_none=True)
            for server in self.mcp_servers
        }

    def _harbor_config(
        self,
        config: UrsaConfig,
        *,
        use_web: bool | None = None,
    ) -> dict[str, Any]:
        """Convert Harbor-owned model and MCP settings to an URSA layer."""
        layer: dict[str, Any] = {}
        if use_web is not None:
            layer["use_web"] = use_web
        if self.model_name:
            provider, separator, model = self.model_name.partition("/")
            if not separator or not provider or not model:
                raise ValueError(
                    "Harbor model must use inference_provider/model syntax"
                )
            connection = self.model_connection
            provider_config: dict[str, Any] = {}
            configured_provider = config.inference_providers.get(provider)
            if configured_provider is None:
                raise ValueError(
                    f"Harbor inference provider '{provider}' must be "
                    "defined in the merged URSA config"
                )
            model_provider = (configured_provider.model_extra or {}).get(
                "model_provider"
            )
            if model_provider:
                prefix = f"{model_provider}:"
                if not model.startswith(prefix):
                    model = f"{prefix}{model}"
            elif provider != "openai" and ":" not in model:
                raise ValueError(
                    f"Harbor model for inference provider '{provider}' must "
                    "include model_provider:model or the URSA inference "
                    "provider must define model_provider"
                )
            layer["llm_model"] = {
                "model": model,
                "inference_provider": provider,
            }
            env_authenticated = connection.provider in self.ENV_AUTH_PROVIDERS
            if connection.api_key is not None and not env_authenticated:
                provider_config["api_key"] = SecretStr(connection.api_key)
            if connection.user_supplied_base_url is not None:
                provider_config["base_url"] = connection.user_supplied_base_url
            provider_access = PROVIDERS.get(connection.provider or "")
            projected_names = (
                set()
                if env_authenticated or provider_access is None
                else {
                    *provider_access.api_key_candidate_names,
                    *provider_access.base_url_candidate_names,
                }
            )
            self._model_env = {
                name: value
                for name, value in connection.env.items()
                if name not in projected_names
            }
            if provider_config:
                layer["inference_providers"] = {provider: provider_config}
        if mcp_servers := self._mcp_config():
            layer["mcp_servers"] = mcp_servers
        return layer

    def _config_layers(self) -> list[dict[str, Any]]:
        source = self.config_source
        if isinstance(source, Path):
            source = source.expanduser().resolve()
        layers = (
            []
            if source is None
            else [
                source
                if isinstance(source, dict)
                else load_config_file(source, interpolate=False)
            ]
        )
        for layer in layers:
            self._reject_environment_interpolation(layer)
        return layers

    @staticmethod
    def _qualify_tagged_models(config: dict[str, Any]) -> None:
        """Keep normalized model tags valid for URSA's provider parser."""
        for field_name in ("llm_model", "emb_model"):
            model_config = config.get(field_name)
            if not isinstance(model_config, dict):
                continue
            model = model_config.get("model")
            model_provider = model_config.get("model_provider")
            if (
                isinstance(model, str)
                and ":" in model
                and isinstance(model_provider, str)
            ):
                model_config["model"] = f"{model_provider}:{model}"

    @staticmethod
    def _prune_implicit_unreferenced_providers(
        config: dict[str, Any], explicit_providers: set[str]
    ) -> None:
        """Remove defaults whose secrets are unrelated to this run."""
        referenced_providers = {
            str(provider)
            for field_name in ("llm_model", "emb_model")
            if isinstance(model := config.get(field_name), dict)
            if (provider := model.get("inference_provider")) is not None
        }
        provider_configs = config.get("inference_providers")
        if not isinstance(provider_configs, dict):
            return
        for name in tuple(provider_configs):
            if (
                name not in explicit_providers
                and name not in referenced_providers
            ):
                provider_configs.pop(name)

    def _materialize_native_config(
        self,
        *,
        use_web: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        config_layers = self._config_layers()
        base_config = UrsaConfig().model_merge(*config_layers)
        harbor_layer = self._harbor_config(base_config, use_web=use_web)
        config = UrsaConfig().model_merge(*config_layers, harbor_layer)

        config_data = config.model_dump(mode="python", exclude_unset=True)
        explicit_providers = {
            str(name)
            for layer in config_layers
            if isinstance(layer.get("inference_providers"), dict)
            for name in layer["inference_providers"]
        }
        self._prune_implicit_unreferenced_providers(
            config_data, explicit_providers
        )
        self._qualify_tagged_models(config_data)
        projected, secret_env = externalize_secret_references(config_data)
        native_config = UrsaConfig.model_validate(projected).model_dump(
            mode="json", exclude_none=True, exclude_unset=True
        )
        self._prune_implicit_unreferenced_providers(
            native_config, explicit_providers
        )
        self._qualify_tagged_models(native_config)
        self._runtime_group = validate_group_name(native_config.get("group"))
        return native_config, secret_env

    @staticmethod
    def _network_use_web(environment: BaseEnvironment) -> bool | None:
        policy = getattr(environment, "network_policy", None)
        if policy is None:
            return None
        return policy.network_mode != NetworkMode.NO_NETWORK

    async def _upload_native_config(
        self,
        environment: BaseEnvironment,
        native_config: dict[str, Any],
    ) -> None:
        await self._upload_config_text(
            environment,
            content=json.dumps(native_config),
            remote_path=self._remote_config_file,
            filename="ursa-config.json",
        )
        # Docker upload implementations can preserve the root-owned 0600 mode
        # of the host temporary file. The JSON contains secret references,
        # never secret values, and must be readable by a non-root agent.
        await self.exec_as_root(
            environment,
            command=f"chmod 644 {shlex.quote(self._remote_config_file)}",
            timeout_sec=30,
        )

    @classmethod
    def _reject_environment_interpolation(
        cls, value: Any, path: tuple[str, ...] = ()
    ) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                cls._reject_environment_interpolation(child, (*path, str(name)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                cls._reject_environment_interpolation(
                    child, (*path, str(index))
                )
        elif isinstance(value, str) and ENV_SUB_REGEX.search(value):
            location = ".".join(path) or "config"
            raise ValueError(
                f"Environment interpolation at {location} is not allowed in "
                "Harbor configs; use an explicit {env: VARIABLE} secret reference"
            )

    @staticmethod
    def _stage_source(source: Path, destination: Path) -> None:
        """Stage package files, respecting Git ignores when available."""
        if (source / ".git").exists():
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                check=True,
                capture_output=True,
            )
            destination.mkdir()
            for raw_path in result.stdout.split(b"\0"):
                if not raw_path:
                    continue
                relative = Path(raw_path.decode())
                if UrsaHarborAgent._is_sensitive_source_path(relative):
                    continue
                source_file = source / relative
                destination_file = destination / relative
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                if source_file.is_symlink():
                    raise ValueError(
                        "ursa_install_spec source must not contain symlinks: "
                        f"{relative}"
                    )
                elif source_file.is_file():
                    shutil.copy2(source_file, destination_file)
            return

        for source_file in source.rglob("*"):
            relative = source_file.relative_to(source)
            if (
                not UrsaHarborAgent._is_sensitive_source_path(relative)
                and source_file.is_symlink()
            ):
                raise ValueError(
                    "ursa_install_spec source must not contain symlinks: "
                    f"{relative}"
                )

        def ignored(directory: str, names: list[str]) -> list[str]:
            relative = Path(directory).relative_to(source)
            return [
                name
                for name in names
                if UrsaHarborAgent._is_sensitive_source_path(relative / name)
            ]

        shutil.copytree(source, destination, ignore=ignored)

    @staticmethod
    def _build_wheel(source: Path, destination: Path) -> Path:
        """Build one wheel for the implicit local URSA installation."""
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "Building the default local URSA installation requires uv; "
                "pass a PEP 508 wheel reference or version as ursa_install_spec"
            )
        destination.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(destination),
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            detail = result.stderr or result.stdout
            raise RuntimeError(
                f"Could not build the default URSA wheel: {detail}"
            )
        wheels = list(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                "Expected one wheel from the default URSA installation, found "
                f"{len(wheels)}"
            )
        return wheels[0]

    @staticmethod
    def _is_sensitive_source_path(path: Path) -> bool:
        excluded_names = {
            ".git",
            ".venv",
            ".env",
            ".netrc",
            ".pypirc",
            ".aws",
            "gcloud",
            "credentials",
            "credentials.json",
            "credentials.yaml",
            "credentials.yml",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "jobs",
        }
        patterns = (".env.*", "*.key", "*.pem")
        return any(
            part in excluded_names
            or any(fnmatch.fnmatch(part, pattern) for pattern in patterns)
            for part in path.parts
        )

    @staticmethod
    def _terminate_runner_command(pid_file: str) -> str:
        quoted_pid_file = shlex.quote(pid_file)
        return (
            f"if [ -s {quoted_pid_file} ]; then "
            f"pid=$(cat {quoted_pid_file}); "
            "case $pid in ''|*[!0-9]*) exit 0 ;; esac; "
            'command=$(tr "\\000" " " < "/proc/$pid/cmdline" 2>/dev/null) '
            "|| exit 0; "
            'case "$command" in *"ursa.integration.harbor runner"*) ;; '
            "*) exit 0 ;; esac; "
            "descendants() { for child in "
            '$(cat "/proc/$1/task/$1/children" 2>/dev/null); '
            'do descendants "$child"; echo "$child"; done; }; '
            'children=$(descendants "$pid"); '
            'kill -TERM $children "$pid" 2>/dev/null || true; '
            'i=0; while kill -0 "$pid" 2>/dev/null '
            '&& [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            'kill -KILL $children "$pid" 2>/dev/null || true; '
            f"rm -f -- {quoted_pid_file}; fi"
        )

    async def _best_effort_root_exec(
        self, environment: BaseEnvironment, command: str
    ) -> None:
        try:
            await asyncio.shield(
                self.exec_as_root(
                    environment,
                    command=command,
                    timeout_sec=10,
                )
            )
        except Exception:
            pass

    @staticmethod
    def name() -> str:
        return "ursa"

    @staticmethod
    def _handoff_agent_name(trial_dir: Path) -> str:
        """Return a valid local persistent-agent name for a Harbor trial."""
        trial_name = re.sub(r"[^A-Za-z0-9._-]+", "-", trial_dir.name).strip(
            "._-"
        )
        return f"harbor-{trial_name or 'trial'}"

    @staticmethod
    def _handoff_group(trial_dir: Path) -> str:
        """Read the evaluation group recorded in Harbor's trial result."""
        result_path = trial_dir / "result.json"
        if not result_path.is_file():
            raise ValueError(
                f"Harbor trial result {result_path} does not record the URSA "
                "evaluation group; refusing to resume"
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read Harbor trial result {result_path}: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError(
                f"Harbor trial result {result_path} is not an object"
            )

        agent_results = [result.get("agent_result")]
        step_results = result.get("step_results")
        if isinstance(step_results, list):
            agent_results.extend(
                step.get("agent_result")
                for step in step_results
                if isinstance(step, dict)
            )

        groups: set[str] = set()
        for agent_result in agent_results:
            if not isinstance(agent_result, dict):
                continue
            metadata = agent_result.get("metadata")
            if not isinstance(metadata, dict) or "group" not in metadata:
                continue
            group = metadata["group"]
            if not isinstance(group, str):
                raise ValueError(
                    f"Harbor trial result {result_path} has an invalid URSA group"
                )
            groups.add(validate_group_name(group))

        if len(groups) > 1:
            raise ValueError(
                f"Harbor trial result {result_path} contains multiple URSA groups"
            )
        if not groups:
            raise ValueError(
                f"Harbor trial result {result_path} does not record the URSA "
                "evaluation group; refusing to resume"
            )
        return next(iter(groups))

    @classmethod
    def handoff(cls, trial_dir: Path, cwd: Path) -> list[str]:
        """Import a trial checkpoint and resume it in the local URSA TUI."""
        del cwd  # URSA uses the directory from which Harbor launches the CLI.
        if shutil.which("ursa") is None:
            raise ValueError(
                "ursa CLI not found on PATH; install URSA first: "
                "uv tool install 'ursa-ai[harbor]'"
            )

        checkpoint = trial_dir / "agent/db/checkpointer.db"
        if not checkpoint.is_file():
            raise ValueError(
                f"URSA checkpoint not found at {checkpoint}; "
                "handoff requires a completed trial with a native session"
            )

        from ursa.cli.agent_management import import_agent

        agent_name = cls._handoff_agent_name(trial_dir)
        group = cls._handoff_group(trial_dir)
        try:
            import_agent(checkpoint, group_name=group, agent_name=agent_name)
        except (FileExistsError, FileNotFoundError) as exc:
            raise ValueError(str(exc)) from exc
        return [
            "ursa",
            "--name",
            agent_name,
            *(["--group", group] if group != DEFAULT_GROUP_NAME else []),
        ]

    def version(self) -> str | None:
        try:
            from ursa import __version__

            return __version__
        except (ImportError, AttributeError):
            return None

    @classmethod
    def _stage_uv_binary(cls, machine: str, destination: Path) -> None:
        """Download and verify a portable uv binary for the architecture."""
        match machine.strip().lower():
            case "amd64" | "x86_64":
                architecture = "x86_64"
            case "arm64" | "aarch64":
                architecture = "aarch64"
            case unsupported:
                raise RuntimeError(
                    f"unsupported architecture for uv: {unsupported}"
                )
        archive_name = f"uv-{architecture}-unknown-linux-musl.tar.gz"
        url = (
            "https://releases.astral.sh/github/uv/releases/download/"
            f"{cls._UV_VERSION}/{archive_name}"
        )
        checksum_request = urllib.request.Request(
            f"{url}.sha256", headers={"User-Agent": "ursa-harbor"}
        )
        with urllib.request.urlopen(checksum_request, timeout=120) as response:
            checksum_fields = response.read(1024).decode("ascii").split()
        if (
            len(checksum_fields) != 2
            or checksum_fields[1] != archive_name
            or not re.fullmatch(r"[0-9a-f]{64}", checksum_fields[0])
        ):
            raise RuntimeError("uv release checksum file is invalid")
        expected_sha256 = checksum_fields[0]
        with tempfile.TemporaryDirectory(prefix="ursa-harbor-uv-") as temp_dir:
            archive = Path(temp_dir) / archive_name
            digest = hashlib.sha256()
            archive_request = urllib.request.Request(
                url, headers={"User-Agent": "ursa-harbor"}
            )
            with urllib.request.urlopen(
                archive_request, timeout=120
            ) as response:
                with archive.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError(
                    "downloaded uv archive failed SHA-256 validation"
                )
            with tarfile.open(archive, "r:gz") as bundle:
                member = next(
                    (
                        item
                        for item in bundle.getmembers()
                        if item.isfile()
                        and PurePosixPath(item.name).name == "uv"
                    ),
                    None,
                )
                if member is None:
                    raise RuntimeError(
                        "downloaded uv archive does not contain uv"
                    )
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        "could not read uv from downloaded archive"
                    )
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        destination.chmod(0o755)

    async def install(self, environment: BaseEnvironment) -> None:
        # Reject host-side configuration errors before doing any work in the
        # benchmark container.
        self._runtime_use_web = self._network_use_web(environment)
        native_config, self._secret_env = self._materialize_native_config(
            use_web=self._runtime_use_web
        )
        working_directory = await self.exec_as_agent(
            environment, command="pwd", timeout_sec=30
        )
        self._workspace = (working_directory.stdout or "").strip()
        if not self._workspace.startswith("/") or "\n" in self._workspace:
            raise RuntimeError(
                f"Invalid task working directory: {self._workspace!r}"
            )
        install_root = self._INSTALL_ROOT
        install_env = {
            "HOME": f"{install_root}/home",
            "TMPDIR": f"{install_root}/tmp",
            "UV_CACHE_DIR": f"{install_root}/cache",
            "UV_PYTHON_INSTALL_BIN": "0",
            "UV_PYTHON_INSTALL_DIR": f"{install_root}/python",
            "UV_TOOL_BIN_DIR": f"{install_root}/bin",
            "UV_TOOL_DIR": f"{install_root}/tools",
        }
        architecture = await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {install_root}/bin {install_root}/cache "
                f"{install_root}/home {install_root}/python "
                f"{install_root}/tmp {install_root}/tools && "
                f"chmod 1777 {install_root}/tmp && uname -m"
            ),
            env=install_env,
            timeout_sec=30,
        )
        machine = (architecture.stdout or "").strip()
        with tempfile.TemporaryDirectory(
            prefix="ursa-harbor-uv-upload-"
        ) as temp_dir:
            staged_uv = Path(temp_dir) / "uv"
            await asyncio.to_thread(self._stage_uv_binary, machine, staged_uv)
            await environment.upload_file(
                staged_uv, f"{install_root}/tmp/uv-upload"
            )
        await self.exec_as_root(
            environment,
            command=(
                f"mv {install_root}/tmp/uv-upload {install_root}/bin/uv && "
                f"chmod 755 {install_root}/bin/uv && "
                f"{install_root}/bin/uv python install "
                f"{self.URSA_PYTHON_VERSION}"
            ),
            env=install_env,
            timeout_sec=600,
        )
        install_target = Requirement(str(self.ursa_install_spec))
        install_target.extras = set(
            self._merge_extras(tuple(install_target.extras), self.ursa_extras)
        )
        if (source := self._install_source) is not None:
            if self._build_default_install_wheel:
                with tempfile.TemporaryDirectory(
                    prefix="ursa-harbor-wheel-"
                ) as temp_dir:
                    wheel = await asyncio.to_thread(
                        self._build_wheel, source, Path(temp_dir)
                    )
                    remote_source = f"{install_root}/tmp/{wheel.name}"
                    await environment.upload_file(wheel, remote_source)
            elif source.is_file():
                remote_source = f"{install_root}/tmp/{source.name}"
                await environment.upload_file(source, remote_source)
            else:
                remote_source = f"{install_root}/tmp/ursa-source"
                with tempfile.TemporaryDirectory(
                    prefix="ursa-harbor-source-"
                ) as temp_dir:
                    staged_source = Path(temp_dir) / "ursa"
                    self._stage_source(source, staged_source)
                    await environment.upload_dir(staged_source, remote_source)
            install_target.url = (
                urlparse(install_target.url)
                ._replace(netloc="", path=quote(remote_source))
                .geturl()
            )
        elif install_target.url:
            install_target.url = self._github_archive_target(install_target.url)
        extra_packages = " ".join(
            f"--with {shlex.quote(package)}" for package in self.extra_packages
        )
        await self.exec_as_root(
            environment,
            command=(
                f"{install_root}/bin/uv tool install --force "
                "--refresh-package ursa-ai --python "
                f"{self.URSA_PYTHON_VERSION} "
                f"{extra_packages} {shlex.quote(str(install_target))} && "
                f"test -x {install_root}/bin/ursa && "
                f"test -x {install_root}/tools/ursa-ai/bin/python && "
                "printf '%s\\n' '#!/bin/sh' "
                "'exec /installed-agent/tools/ursa-ai/bin/python "
                '-m ursa.integration.harbor runner "$@"\' '
                f"> {install_root}/bin/ursa-harbor-runner && "
                f"chmod 755 {install_root}/bin/ursa-harbor-runner"
            ),
            env=install_env,
            timeout_sec=900,
        )
        self._remote_config_file = f"{install_root}/tmp/ursa-config.json"
        await self._upload_native_config(environment, native_config)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        use_web = self._network_use_web(environment)
        if use_web is not None and use_web != getattr(
            self, "_runtime_use_web", None
        ):
            native_config, self._secret_env = self._materialize_native_config(
                use_web=use_web
            )
            await self._upload_native_config(environment, native_config)
            self._runtime_use_web = use_web
        payload = {
            "agent_import_path": self.agent_import_path,
            "config_file": self._remote_config_file,
            "instruction": instruction,
            "workspace": self._workspace,
            "log_dir": str(self.environment_logs_dir),
            "use_web": use_web,
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
        runner_pid_file = self._runner_pid_file

        # Add context
        context.metadata = {
            **(context.metadata or {}),
            "agent": self.agent_import_path,
            "group": self._runtime_group,
            "ursa_install_spec": str(self.ursa_install_spec),
            "host": platform.node(),
            "host_platform": sysconfig.get_platform(),
        }

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    f"echo $$ > {runner_pid_file}; "
                    f"exec {self.URSA_RUNNER} " + shlex.quote(encoded)
                ),
                env={**self._model_env, **self._secret_env},
                cwd=self._workspace,
                timeout_sec=None,
            )
        except asyncio.CancelledError:
            await self._best_effort_root_exec(
                environment,
                self._terminate_runner_command(runner_pid_file),
            )
            raise
        finally:
            await self._best_effort_root_exec(
                environment,
                f"rm -f -- {shlex.quote(runner_pid_file)}",
            )
            (
                context.n_input_tokens,
                context.n_cache_tokens,
                context.n_output_tokens,
            ) = _jsonl_token_usage(self.logs_dir / "ursa.jsonl")


def make_harbor_agent(
    agent_class: type[UrsaBaseAgent],
    config: str | Path | dict[str, Any] | None = None,
    /,
) -> type[UrsaHarborAgent]:
    """Create a Harbor agent class bound to any URSA ``BaseAgent`` subclass."""
    if not issubclass(agent_class, UrsaBaseAgent):
        raise TypeError(
            "agent_class must be a subclass of ursa.agents.BaseAgent"
        )
    import_path = f"{agent_class.__module__}:{agent_class.__qualname__}"

    class BoundUrsaHarborAgent(UrsaHarborAgent):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(
                *args,
                agent_import_path=import_path,
                config=config,
                **kwargs,
            )

    BoundUrsaHarborAgent.__name__ = f"{agent_class.__name__}HarborAgent"
    BoundUrsaHarborAgent.__qualname__ = BoundUrsaHarborAgent.__name__
    return BoundUrsaHarborAgent


app = typer.Typer(
    no_args_is_help=True,
    help="Run and validate URSA's Harbor integration.",
)


artifacts_app = typer.Typer(
    help="Preview, apply, resume, or restore completed Harbor job artifacts."
)
app.add_typer(artifacts_app, name="artifacts")


@artifacts_app.command()
def plan(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    policy: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Print a read-only inventory and planned actions (no files are changed)."""
    from ursa.integration.harbor.retention import (
        load_retention_config,
        plan_job_retention,
    )

    typer.echo(
        plan_job_retention(
            run_dir, load_retention_config(policy)
        ).model_dump_json(indent=2)
    )


@artifacts_app.command()
def apply(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    policy: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Apply a policy, or resume its recorded cleanup after an interruption."""
    from ursa.integration.harbor.retention import (
        apply_job_retention,
        load_retention_config,
    )

    typer.echo(
        json.dumps(
            apply_job_retention(run_dir, load_retention_config(policy)),
            indent=2,
        )
    )


@artifacts_app.command()
def restore(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    destination: Annotated[Path, typer.Option()],
) -> None:
    """Extract verified archived payloads into a new directory."""
    from ursa.artifacts.retention import restore_artifacts

    deleted = restore_artifacts(manifest, destination)
    typer.echo(
        json.dumps({
            "destination": str(destination),
            "unrecoverable_deleted_files": deleted,
        })
    )


@app.command()
def validate(
    paths: TaskPaths,
) -> None:
    """Validate Harbor task environments supported by URSA."""
    from ursa.integration.harbor.validation import (
        discover_harbor_tasks,
        validate_harbor_task,
    )

    tasks = discover_harbor_tasks(paths)
    failures = 0
    for task in tasks:
        try:
            validate_harbor_task(task)
        except Exception as exc:
            failures += 1
            typer.echo(f"FAIL {task}: {type(exc).__name__}: {exc}", err=True)
        else:
            typer.echo(f"OK   {task}")
    if failures:
        raise typer.Exit(1)
    typer.echo(f"Validated {len(tasks)} Harbor task(s).")


@app.command("prebuild")
def prebuild_sifs(
    paths: TaskPaths,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild SIFs already present in cache"),
    ] = False,
    image_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--image-cache-dir",
            help="Override the URSA Harbor SIF cache directory",
        ),
    ] = None,
) -> None:
    """Pre-build Dockerfile-backed SIF files for Harbor tasks."""
    from ursa.integration.harbor.validation import (
        discover_harbor_tasks,
        prebuild_harbor_task,
    )

    tasks = discover_harbor_tasks(paths)

    async def build_all() -> tuple[int, int]:
        failures = 0
        images: set[Path] = set()
        for task in tasks:
            try:
                built = await prebuild_harbor_task(
                    task,
                    force_build=force,
                    image_cache_dir=image_cache_dir,
                )
            except Exception as exc:
                failures += 1
                typer.echo(
                    f"FAIL {task}: {type(exc).__name__}: {exc}", err=True
                )
            else:
                images.update(built)
                typer.echo(f"OK   {task}: {len(built)} SIF file(s)")
                for image in built:
                    typer.echo(f"SIF  {image}")
        return failures, len(images)

    failures, image_count = asyncio.run(build_all())
    if failures:
        raise typer.Exit(1)
    typer.echo(
        f"Pre-built {image_count} SIF file(s) for {len(tasks)} Harbor task(s)."
    )


@app.command("runner")
def run_runner(
    encoded: Annotated[
        str,
        typer.Argument(help="URL-safe base64-encoded runner configuration"),
    ],
) -> None:
    """Run the container-side Harbor agent process."""
    from ursa.integration.harbor.runner import main as runner_main

    runner_main(encoded)


def main() -> None:
    """Run the Harbor integration CLI."""
    app()


__all__ = ["UrsaHarborAgent", "app", "make_harbor_agent"]
