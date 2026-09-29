"""Run URSA agents as installed agents in the Harbor benchmark framework.

Harbor starts the adapter on the host.  The adapter then installs URSA and runs
the selected URSA agent *inside* the task container, so filesystem and shell
tools operate on the benchmark workspace rather than on the submit host.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import importlib.metadata
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from jsonargparse import Namespace
from pydantic import SecretStr

try:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.agents.model_connection import PROVIDERS, ModelConnectionSpec
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.config import NetworkMode
    from harbor.models.trial.paths import EnvironmentPaths
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The Harbor integration requires `uv add 'ursa-ai[harbor]'`."
    ) from exc

from ursa.agents import BaseAgent as UrsaBaseAgent
from ursa.cli.config import (
    ENV_SUB_REGEX,
    UrsaConfig,
    config_search_paths,
    load_config_file,
)
from ursa.util.secrets import externalize_secret_references


class UrsaHarborAgent(BaseInstalledAgent):
    """Generic Harbor binding for an importable URSA ``BaseAgent`` subclass.

    Args:
        agent_import_path: ``module:Class`` path for the URSA agent. Defaults
            to :class:`ursa.agents.ExecutionAgent`.
        config_file: URSA YAML or JSON configuration file.
        config_only: Merge only ``config_file`` before Harbor's settings. Set
            false to include the system and user config layers first.
        ursa_install_spec: Package requirement, Git URL, local Python project,
            or local wheel/sdist installed in each task container. Defaults to
            this URSA checkout or installed version.
        ursa_extras: URSA package extras to install, as a sequence or comma-separated
            string.
        extra_packages: Additional Python packages to install. Pass one requirement
            string or a sequence of requirement strings.
    """

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)
    URSA_PYTHON_VERSION = "3.13"
    URSA_RUNNER = "/usr/local/bin/ursa-harbor-runner"
    ENV_AUTH_PROVIDERS = frozenset({
        "amazon-bedrock",
        "sagemaker",
        "vertex_ai",
    })

    def __init__(
        self,
        *args: Any,
        agent_import_path: str = "ursa.agents:ExecutionAgent",
        config_file: str | Path,
        config_only: bool | str = True,
        ursa_install_spec: str | Path | None = None,
        ursa_extras: str | Sequence[str] | None = None,
        extra_packages: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent_import_path = agent_import_path
        self.config_file = Path(config_file).expanduser().resolve()
        self.config_only = self._parse_bool(config_only, "config_only")
        install_spec = (
            self._default_install_spec()
            if ursa_install_spec is None
            else ursa_install_spec
        )
        self.ursa_install_spec = self._parse_install_spec(install_spec)
        self.ursa_extras = self._parse_list(
            ursa_extras, name="ursa_extras", split_commas=True
        )
        self.extra_packages = self._parse_list(
            extra_packages, name="extra_packages"
        )
        self._secret_env: dict[str, str] = {}
        self._model_env: dict[str, str] = {}
        self._workspace = "/"

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
                try:
                    values = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{name} must be a requirement or JSON array of strings"
                    ) from exc
                if not isinstance(values, list):
                    raise ValueError(
                        f"{name} must be a requirement or JSON array of strings"
                    )
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
    def _parse_bool(value: bool | str, name: str) -> bool:
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"{name} must be true or false")

    @staticmethod
    def _default_install_spec() -> str | Path:
        checkout = Path(__file__).resolve().parents[3]
        if (checkout / "pyproject.toml").is_file():
            return checkout
        distribution = importlib.metadata.distribution("ursa-ai")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            provenance = json.loads(direct_url)
            url = provenance.get("url")
            if isinstance(url, str):
                parsed = urlparse(url)
                if parsed.scheme == "file":
                    source = Path(unquote(parsed.path))
                    if (source / "pyproject.toml").is_file():
                        return source.resolve()
                    if source.is_file() and UrsaHarborAgent._is_package_archive(
                        source
                    ):
                        return source.resolve()
                    raise ValueError(
                        "URSA was installed from a local source that is no longer "
                        "available; pass ursa_install_spec explicitly"
                    )
                vcs = provenance.get("vcs_info", {})
                if isinstance(vcs, dict) and isinstance(vcs.get("vcs"), str):
                    revision = vcs.get("commit_id") or vcs.get(
                        "requested_revision"
                    )
                    suffix = f"@{revision}" if revision else ""
                    return f"{vcs['vcs']}+{url}{suffix}"
                return url
        return f"ursa-ai=={distribution.version}"

    @staticmethod
    def _parse_install_spec(value: str | Path) -> str | Path:
        if isinstance(value, Path):
            path = value.expanduser()
        else:
            value = value.strip()
            if not value:
                raise ValueError("ursa_install_spec cannot be empty")
            candidate = Path(value).expanduser()
            if not (
                candidate.exists()
                or candidate.is_absolute()
                or value.startswith(("./", "../"))
            ):
                return value
            path = candidate

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

    def _install_target(self, target: str) -> str:
        if not self.ursa_extras:
            return target
        extras = ",".join(self.ursa_extras)
        if "[" in target.split("@", 1)[0]:
            raise ValueError(
                "ursa_install_spec must not include extras when ursa_extras is set"
            )
        direct_reference = re.match(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\s*@\s*.+)$", target
        )
        if direct_reference:
            name, reference = direct_reference.groups()
            return f"{name}[{extras}]{reference}"
        if "://" in target:
            return f"ursa-ai[{extras}] @ {target}"
        distribution = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", target)
        if distribution:
            end = distribution.end()
            return f"{target[:end]}[{extras}]{target[end:]}"
        return f"{target}[{extras}]"

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
            if connection.configured_base_url is not None:
                provider_config["base_url"] = connection.configured_base_url
            provider_access = PROVIDERS.get(connection.provider or "")
            projected_names = (
                set()
                if env_authenticated or provider_access is None
                else {
                    *provider_access.api_key_envs,
                    *provider_access.base_url_envs,
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
        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"URSA config file not found: {self.config_file}"
            )
        paths = (
            [self.config_file]
            if self.config_only
            else config_search_paths(
                Namespace(config=self.config_file, subcommand=None), "final"
            )
        )
        layers = [
            load_config_file(path, interpolate_environment=False)
            for path in paths
        ]
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

    def _runtime_config(
        self,
        *,
        use_web: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        config = UrsaConfig().model_merge(*self._config_layers())
        harbor_config = self._harbor_config(config, use_web=use_web)
        harbor_mcp = harbor_config.pop("mcp_servers", {})
        config = config.model_merge(harbor_config)

        # A Harbor MCP entry describes the whole named server. Replacing that
        # entry avoids retaining incompatible fields when its transport changes.
        if harbor_mcp:
            config.mcp_servers = {
                **config.mcp_servers,
                **harbor_mcp,
            }

        config_data = config.model_dump(mode="python", exclude_unset=True)
        self._qualify_tagged_models(config_data)
        projected, secret_env = externalize_secret_references(config_data)
        runtime_config = UrsaConfig.model_validate(projected).model_dump(
            mode="json", exclude_none=True, exclude_unset=True
        )
        self._qualify_tagged_models(runtime_config)
        return runtime_config, secret_env

    @staticmethod
    def _network_use_web(environment: BaseEnvironment) -> bool | None:
        policy = getattr(environment, "network_policy", None)
        if policy is None:
            return None
        return policy.network_mode != NetworkMode.NO_NETWORK

    async def _upload_runtime_config(
        self,
        environment: BaseEnvironment,
        runtime_config: dict[str, Any],
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as runtime_config_file:
            json.dump(runtime_config, runtime_config_file)
            runtime_config_file.flush()
            await environment.upload_file(
                Path(runtime_config_file.name), self._remote_config_file
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
        return (
            f"if [ -s {pid_file} ]; then "
            f"pid=$(cat {pid_file}); "
            "descendants() { for child in "
            '$(cat "/proc/$1/task/$1/children" 2>/dev/null); '
            'do descendants "$child"; echo "$child"; done; }; '
            'children=$(descendants "$pid"); '
            'kill -TERM $children "$pid" 2>/dev/null || true; '
            'i=0; while kill -0 "$pid" 2>/dev/null '
            '&& [ "$i" -lt 20 ]; do '
            "sleep 0.1; i=$((i + 1)); done; "
            'kill -KILL $children "$pid" 2>/dev/null || true; fi'
        )

    @staticmethod
    def name() -> str:
        return "ursa"

    def version(self) -> str | None:
        try:
            from ursa import __version__

            return __version__
        except (ImportError, AttributeError):
            return None

    async def install(self, environment: BaseEnvironment) -> None:
        # Reject host-side configuration errors before doing any work in the
        # benchmark container.
        self._runtime_use_web = self._network_use_web(environment)
        runtime_config, self._secret_env = self._runtime_config(
            use_web=self._runtime_use_web
        )
        # uv's glibc build can crash under QEMU user-mode emulation (for
        # example, amd64 Terminal-Bench images on an arm64 host). The musl
        # release is statically linked and works both natively and under QEMU.
        uv_version = "0.12.8"
        await self.exec_as_root(
            environment,
            command=(
                "missing_packages=; "
                'command -v curl >/dev/null 2>&1 || missing_packages="$missing_packages curl"; '
                'command -v tar >/dev/null 2>&1 || missing_packages="$missing_packages tar"; '
                'command -v sha256sum >/dev/null 2>&1 || missing_packages="$missing_packages coreutils"; '
                'if [ -n "$missing_packages" ]; then '
                "if command -v microdnf >/dev/null; then "
                "microdnf install -y ca-certificates $missing_packages && microdnf clean all; "
                "elif command -v dnf >/dev/null; then "
                "dnf install -y ca-certificates $missing_packages && dnf clean all; "
                "elif command -v yum >/dev/null; then "
                "yum install -y ca-certificates $missing_packages && yum clean all; "
                "elif command -v apk >/dev/null; then "
                "apk add --no-cache ca-certificates $missing_packages; "
                "elif command -v apt-get >/dev/null; then "
                "apt-get update && apt-get install -y ca-certificates $missing_packages; "
                "else echo 'curl, tar, and sha256sum are required to install uv' >&2; exit 1; fi; fi; "
                "if ! command -v /opt/uv/uv >/dev/null 2>&1; then "
                "case $(uname -m) in "
                "x86_64|amd64) uv_arch=x86_64; "
                "uv_sha256=6ca4597639c97e921fb915e113061ce8e4a14ead9e42a1ead521dbb0a6763795 ;; "
                "aarch64|arm64) uv_arch=aarch64; "
                "uv_sha256=975917badc8370163989e5bbe5a7c69bf922d19f8e57cb2652531bbffc935f84 ;; "
                "*) echo 'unsupported architecture for uv: '$(uname -m) >&2; exit 1 ;; "
                "esac; mkdir -p /opt/uv; uv_archive=/tmp/uv.tar.gz; "
                f"curl -LsSf https://github.com/astral-sh/uv/releases/download/{uv_version}/"
                'uv-${uv_arch}-unknown-linux-musl.tar.gz -o "$uv_archive"; '
                'echo "$uv_sha256  $uv_archive" | sha256sum -c -; '
                'tar -xzf "$uv_archive" --strip-components=1 -C /opt/uv; '
                'rm -f "$uv_archive"; fi; '
                f"/opt/uv/uv python install {self.URSA_PYTHON_VERSION}"
            ),
            timeout_sec=600,
        )
        working_directory = await self.exec_as_agent(
            environment, command="pwd", timeout_sec=30
        )
        self._workspace = (working_directory.stdout or "").strip()
        if not self._workspace.startswith("/") or "\n" in self._workspace:
            raise RuntimeError(
                f"Invalid task working directory: {self._workspace!r}"
            )
        install_target = self.ursa_install_spec
        if isinstance(install_target, Path):
            if install_target.is_file():
                remote_source = f"/tmp/{install_target.name}"
                await environment.upload_file(install_target, remote_source)
                install_target = f"file://{remote_source}"
            else:
                remote_source = "/tmp/ursa-source"
                with tempfile.TemporaryDirectory(
                    prefix="ursa-harbor-source-"
                ) as temp_dir:
                    staged_source = Path(temp_dir) / "ursa"
                    self._stage_source(install_target, staged_source)
                    await environment.upload_dir(staged_source, remote_source)
                install_target = remote_source
        install_target = self._install_target(install_target)
        extra_packages = " ".join(
            f"--with {shlex.quote(package)}" for package in self.extra_packages
        )
        await self.exec_as_root(
            environment,
            command=(
                "UV_TOOL_DIR=/opt/ursa-tools "
                "UV_TOOL_BIN_DIR=/usr/local/bin "
                "/opt/uv/uv tool install --force --python "
                f"{self.URSA_PYTHON_VERSION} "
                f"{extra_packages} {shlex.quote(install_target)} && "
                'test "$(command -v ursa)" = /usr/local/bin/ursa && '
                "test -x /usr/local/bin/ursa-harbor-runner"
            ),
            timeout_sec=900,
        )
        self._remote_config_file = "/tmp/ursa-config.json"
        await self._upload_runtime_config(environment, runtime_config)

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
            runtime_config, self._secret_env = self._runtime_config(
                use_web=use_web
            )
            await self._upload_runtime_config(environment, runtime_config)
            self._runtime_use_web = use_web
        payload = {
            "agent_import_path": self.agent_import_path,
            "config_file": self._remote_config_file,
            "instruction": instruction,
            "workspace": self._workspace,
            "metrics_path": f"{self.environment_logs_dir}/ursa-metrics.json",
            "log_path": f"{self.environment_logs_dir}/ursa.log",
            "artifacts_dir": str(EnvironmentPaths.artifacts_dir),
            "use_web": use_web,
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
        runner_pid_file = "/tmp/ursa-harbor-runner.pid"
        try:
            result = await self.exec_as_agent(
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
            try:
                await asyncio.shield(
                    self.exec_as_root(
                        environment,
                        command=self._terminate_runner_command(runner_pid_file),
                        timeout_sec=10,
                    )
                )
            except Exception:
                pass
            raise
        marker = "URSA_HARBOR_RESULT="
        line = next(
            (
                line
                for line in reversed((result.stdout or "").splitlines())
                if line.startswith(marker)
            ),
            None,
        )
        if line:
            data = json.loads(line.removeprefix(marker))
            context.metadata = {"ursa_result": data.get("result")}
            context.n_input_tokens = data.get("n_input_tokens")
            context.n_output_tokens = data.get("n_output_tokens")
            context.cost_usd = data.get("cost_usd")
            return
        raise RuntimeError(
            "URSA runner exited without an URSA_HARBOR_RESULT record: "
            + (result.stderr or "no output")
        )


def make_harbor_agent(
    agent_class: type[UrsaBaseAgent],
    config_file: str | Path,
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
                config_file=config_file,
                **kwargs,
            )

    BoundUrsaHarborAgent.__name__ = f"{agent_class.__name__}HarborAgent"
    BoundUrsaHarborAgent.__qualname__ = BoundUrsaHarborAgent.__name__
    return BoundUrsaHarborAgent


__all__ = ["UrsaHarborAgent", "make_harbor_agent"]
