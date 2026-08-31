"""Run URSA agents as installed agents in the Harbor benchmark framework.

Harbor starts the adapter on the host.  The adapter then installs URSA and runs
the selected URSA agent *inside* the task container, so filesystem and shell
tools operate on the benchmark workspace rather than on the submit host.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import shutil
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
from ursa.cli.config import UrsaConfig, load_config_file


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

    def __init__(
        self,
        *args: Any,
        agent_import_path: str = "ursa.agents:ExecutionAgent",
        config_file: str | Path,
        ursa_install_spec: str = "ursa-ai",
        ursa_source_dir: str | Path | None = None,
        ursa_extras: str | Sequence[str] | None = None,
        extra_packages: str | Sequence[str] | None = None,
        command_timeout_sec: int = 3600,
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
        self.command_timeout_sec = command_timeout_sec
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

    def _validate_config(self) -> None:
        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"URSA config file not found: {self.config_file}"
            )
        UrsaConfig.model_validate(load_config_file(self.config_file))

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
        # uv's glibc build can crash under QEMU user-mode emulation (for
        # example, amd64 Terminal-Bench images on an arm64 host). The musl
        # release is statically linked and works both natively and under QEMU.
        uv_version = "0.12.8"
        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v curl >/dev/null 2>&1 || "
                "! command -v tar >/dev/null 2>&1; then "
                "if command -v microdnf >/dev/null; then "
                "microdnf install -y curl ca-certificates tar && microdnf clean all; "
                "elif command -v dnf >/dev/null; then "
                "dnf install -y curl ca-certificates tar && dnf clean all; "
                "elif command -v yum >/dev/null; then "
                "yum install -y curl ca-certificates tar && yum clean all; "
                "elif command -v apk >/dev/null; then "
                "apk add --no-cache curl ca-certificates tar; "
                "elif command -v apt-get >/dev/null; then "
                "apt-get update && apt-get install -y curl ca-certificates tar; "
                "else echo 'curl and tar are required to install uv' >&2; exit 1; fi; fi; "
                "if ! command -v /opt/uv/uv >/dev/null 2>&1; then "
                "case $(uname -m) in "
                "x86_64|amd64) uv_arch=x86_64 ;; "
                "aarch64|arm64) uv_arch=aarch64 ;; "
                "*) echo 'unsupported architecture for uv: '$(uname -m) >&2; exit 1 ;; "
                "esac; mkdir -p /opt/uv; "
                f"curl -LsSf https://github.com/astral-sh/uv/releases/download/{uv_version}/"
                'uv-${uv_arch}-unknown-linux-musl.tar.gz | '
                "tar -xz --strip-components=1 -C /opt/uv; fi; "
                "/opt/uv/uv python install 3.12"
            ),
            timeout_sec=600,
        )
        await self.exec_as_root(
            environment,
            command="mkdir -p /workspace && chmod 777 /workspace",
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
                shutil.copytree(
                    self.ursa_source_dir,
                    staged_source,
                    ignore=shutil.ignore_patterns(
                        ".git",
                        ".venv",
                        ".pytest_cache",
                        ".ruff_cache",
                        "__pycache__",
                        "jobs",
                    ),
                )
                await environment.upload_dir(staged_source, remote_source)
            install_target = remote_source
        await self.exec_as_root(
            environment,
            command="/opt/uv/uv venv --python 3.12 /opt/ursa",
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
        self._validate_config()
        self._remote_config_file = f"/tmp/ursa-config{self.config_file.suffix}"
        await environment.upload_file(
            self.config_file, self._remote_config_file
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
            "workspace": "/workspace",
            "metrics_path": f"{self.environment_logs_dir}/ursa-metrics.json",
            "artifacts_dir": str(EnvironmentPaths.artifacts_dir),
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode()
        result = await self.exec_as_agent(
            environment,
            command=(
                f"{self.URSA_PYTHON} -m ursa.integrations.harbor_runner "
                + shlex.quote(encoded)
            ),
            env=self.model_connection.env,
            cwd="/workspace",
            timeout_sec=self.command_timeout_sec,
        )
        marker = "URSA_HARBOR_RESULT="
        line = next(
            (
                line
                for line in reversed(result.stdout.splitlines())
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
            "URSA runner exited without an URSA_HARBOR_RESULT record"
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
