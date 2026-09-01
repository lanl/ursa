"""Run URSA agents as installed agents in the Harbor benchmark framework.

Harbor starts the adapter on the host.  The adapter then installs URSA and runs
the selected URSA agent *inside* the task container, so filesystem and shell
tools operate on the benchmark workspace rather than on the submit host.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.agents.model_connection import ModelConnectionSpec
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.paths import EnvironmentPaths
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The Harbor integration requires `uv add 'ursa-ai[harbor]'`."
    ) from exc

from ursa.agents import BaseAgent as UrsaBaseAgent
from ursa.cli.config import ENV_SUB_REGEX, UrsaConfig, load_config_file
from ursa.util.secrets import externalize_secret_references


class UrsaHarborAgent(BaseInstalledAgent):
    """Generic Harbor binding for an importable URSA ``BaseAgent`` subclass.

    Args:
        agent_import_path: ``module:Class`` path for the URSA agent. Defaults
            to :class:`ursa.agents.ExecutionAgent`.
        config_file: URSA YAML or JSON configuration file.
        ursa_install_spec: Package spec installed in each task container.
        ursa_source_dir: Development-only local source tree to upload and install.
        ursa_extras: URSA package extras to install, as a sequence or comma-separated
            string.
        extra_packages: Additional Python packages to install, as a sequence or
            comma-separated string.
    """

    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)
    URSA_PYTHON = "/opt/ursa/bin/python3"
    URSA_PYTHON_VERSION = "3.13"

    def __init__(
        self,
        *args: Any,
        agent_import_path: str = "ursa.agents:ExecutionAgent",
        config_file: str | Path,
        ursa_install_spec: str = "ursa-ai",
        ursa_source_dir: str | Path | None = None,
        ursa_extras: str | Sequence[str] | None = None,
        extra_packages: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.agent_import_path = agent_import_path
        self.config_file = Path(config_file).expanduser().resolve()
        if not ursa_install_spec:
            raise ValueError("ursa_install_spec cannot be empty")
        self.ursa_install_spec = ursa_install_spec
        self.ursa_extras = self._parse_list(ursa_extras)
        self.extra_packages = self._parse_list(extra_packages)
        self._secret_env: dict[str, str] = {}
        self._workspace = "/"
        self.ursa_source_dir = (
            Path(ursa_source_dir).resolve() if ursa_source_dir else None
        )

    @staticmethod
    def _parse_list(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        values = value.split(",") if isinstance(value, str) else value
        return tuple(item.strip() for item in values if item.strip())

    def _install_target(self, target: str) -> str:
        if not self.ursa_extras:
            return target
        extras = ",".join(self.ursa_extras)
        if "[" in target.split("@", 1)[0]:
            raise ValueError(
                "ursa_install_spec must not include extras when ursa_extras is set"
            )
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

    def _runtime_config(self) -> tuple[dict[str, Any], dict[str, str]]:
        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"URSA config file not found: {self.config_file}"
            )
        raw_config = load_config_file(
            self.config_file, interpolate_environment=False
        )
        self._reject_environment_interpolation(raw_config)
        config = UrsaConfig.model_validate(raw_config)
        config_data = config.model_dump(
            mode="python", context={"include_defaults": True}
        )
        selected_provider = (
            self.model_name.partition("/")[0]
            if self.model_name and "/" in self.model_name
            else config.llm_model.inference_provider
        )
        configured_provider = (
            config.llm_model.inference_provider
            or config.llm_model.model_provider
        )
        provider_switched = selected_provider != configured_provider
        if provider_switched:
            model_config = {
                field: config_data["llm_model"][field]
                for field in ("model", "max_completion_tokens")
                if config_data["llm_model"].get(field) is not None
            }
            model_config["inference_provider"] = selected_provider
            config_data["llm_model"] = model_config
        required_providers = {selected_provider}
        if config.emb_model and config.emb_model.inference_provider:
            required_providers.add(config.emb_model.inference_provider)
        for name, provider in config_data["inference_providers"].items():
            if name not in required_providers:
                provider.pop("api_key", None)
        projected, secret_env = externalize_secret_references(config_data)
        runtime_config = UrsaConfig.model_validate(projected).model_dump(
            mode="json", context={"include_defaults": True}, exclude_none=True
        )
        if provider_switched:
            runtime_config["llm_model"].pop("model_provider", None)
            runtime_config["llm_model"].pop("ssl_verify", None)
        return runtime_config, secret_env

    @classmethod
    def _reject_environment_interpolation(
        cls, value: Any, path: tuple[str, ...] = ()
    ) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                cls._reject_environment_interpolation(
                    child, (*path, str(name))
                )
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
                    destination_file.symlink_to(source_file.readlink())
                elif source_file.is_file():
                    shutil.copy2(source_file, destination_file)
            return
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".env",
                ".env.*",
                ".netrc",
                ".pypirc",
                ".aws",
                "gcloud",
                "credentials",
                "credentials.json",
                "credentials.yaml",
                "credentials.yml",
                "*.key",
                "*.pem",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "jobs",
            ),
        )

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
        runtime_config, self._secret_env = self._runtime_config()
        # uv's glibc build can crash under QEMU user-mode emulation (for
        # example, amd64 Terminal-Bench images on an arm64 host). The musl
        # release is statically linked and works both natively and under QEMU.
        uv_version = "0.12.8"
        python_version_info = tuple(
            int(part) for part in self.URSA_PYTHON_VERSION.split(".")
        )
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
        if self.ursa_source_dir is not None:
            if not (self.ursa_source_dir / "pyproject.toml").is_file():
                raise ValueError(
                    f"ursa_source_dir is not a Python project: {self.ursa_source_dir}"
                )
            remote_source = "/tmp/ursa-source"
            with tempfile.TemporaryDirectory(
                prefix="ursa-harbor-source-"
            ) as temp_dir:
                staged_source = Path(temp_dir) / "ursa"
                self._stage_source(self.ursa_source_dir, staged_source)
                await environment.upload_dir(staged_source, remote_source)
            install_target = remote_source
        await self.exec_as_root(
            environment,
            command=(
                "/opt/uv/uv venv --managed-python --python "
                f"{self.URSA_PYTHON_VERSION} /opt/ursa && "
                f'{self.URSA_PYTHON} -c "import sys; '
                f'assert sys.version_info[:2] == {python_version_info!r}"'
            ),
            timeout_sec=600,
        )
        packages = [self._install_target(install_target), *self.extra_packages]
        await self.exec_as_root(
            environment,
            command=(
                "/opt/uv/uv pip install --python "
                f"{self.URSA_PYTHON} "
                + " ".join(shlex.quote(package) for package in packages)
            ),
            timeout_sec=900,
        )
        self._remote_config_file = "/tmp/ursa-config.json"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8"
        ) as runtime_config_file:
            json.dump(runtime_config, runtime_config_file)
            runtime_config_file.flush()
            await environment.upload_file(
                Path(runtime_config_file.name), self._remote_config_file
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        payload = {
            "agent_import_path": self.agent_import_path,
            "config_file": self._remote_config_file,
            "instruction": instruction,
            "model": self.model_name,
            "mcp_servers": self._mcp_config(),
            "workspace": self._workspace,
            "metrics_path": f"{self.environment_logs_dir}/ursa-metrics.json",
            "artifacts_dir": str(EnvironmentPaths.artifacts_dir),
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
                    f"exec {self.URSA_PYTHON} -m ursa.integrations.harbor_runner "
                    + shlex.quote(encoded)
                ),
                env={**(self.model_connection.env or {}), **self._secret_env},
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
