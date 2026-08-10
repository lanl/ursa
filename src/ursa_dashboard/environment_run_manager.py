from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from ursa.environments.config import (
    AgentSymposiumConfig,
    AgentTeamConfig,
    EnvironmentMemberConfig,
    save_symposium_config,
    save_team_config,
    symposium_cache_dir,
    team_cache_dir,
)
from ursa.environments.visualization import (
    EnvironmentEventRecorder,
    get_environment_run_paths,
    new_run_id,
    read_environment_run_manifest,
    utc_now_rfc3339,
)
from ursa.security import enforce_group_base_url_policy, validate_group_name

from .credentials import (
    CredentialConfigurationError,
    CredentialStore,
    assert_no_credential_metadata,
    assert_no_raw_api_key,
    resolve_api_key,
)

EnvironmentType = Literal["agent_team", "agent_symposium"]
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_ENV_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_SHORT_CLASSES = {
    "ArxivAgent",
    "ChatAgent",
    "DeepReviewAgent",
    "DSIAgent",
    "ExecutionAgent",
    "GitAgent",
    "GitGoAgent",
    "HypothesizerAgent",
    "LammpsAgent",
    "MaterialsProjectAgent",
    "OSTIAgent",
    "PlanningAgent",
    "PromptingAgent",
    "RAGAgent",
    "WebSearchAgent",
    "AgentTeamEnvironment",
}
_ALLOWED_CLASS_PATHS = {
    "ursa.agents.acquisition_agents.ArxivAgent",
    "ursa.agents.acquisition_agents.OSTIAgent",
    "ursa.agents.acquisition_agents.WebSearchAgent",
    "ursa.agents.chat_agent.ChatAgent",
    "ursa.agents.deep_review_agent.DeepReviewAgent",
    "ursa.agents.dsi_agent.DSIAgent",
    "ursa.agents.execution_agent.ExecutionAgent",
    "ursa.agents.git_agent.GitAgent",
    "ursa.agents.git_go_agent.GitGoAgent",
    "ursa.agents.hypothesizer_agent.HypothesizerAgent",
    "ursa.agents.lammps_agent.LammpsAgent",
    "ursa.agents.mp_agent.MaterialsProjectAgent",
    "ursa.agents.planning_agent.PlanningAgent",
    "ursa.agents.prompting_agent.PromptingAgent",
    "ursa.agents.rag_agent.RAGAgent",
    "ursa.environments.agent_team.AgentTeamEnvironment",
}


@dataclass(frozen=True)
class ValidatedEnvironmentLaunch:
    environment_type: EnvironmentType
    config: AgentTeamConfig | AgentSymposiumConfig
    config_mapping: dict[str, Any]


@dataclass
class EnvironmentRunManagerConfig:
    concurrency: int = 2
    stdout_cap_bytes: int = 25 * 1024 * 1024
    stderr_cap_bytes: int = 25 * 1024 * 1024


@dataclass
class _InFlight:
    proc: asyncio.subprocess.Process
    cancel_requested: bool = False


class EnvironmentDefinitionExistsError(FileExistsError):
    pass


class EnvironmentRunExistsError(FileExistsError):
    pass


def _allow_custom_classes() -> bool:
    return str(
        os.environ.get("URSA_DASHBOARD_ALLOW_CUSTOM_ENVIRONMENT_CLASSES", "")
    ).strip().lower() in {"1", "true", "yes", "on"}


def _validate_simple_name(value: Any, *, label: str) -> str:
    name = str(value or "").strip()
    if not _ENV_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{label} must start with a letter or number and contain only "
            "letters, numbers, '.', '_', or '-' (maximum 64 characters)."
        )
    return name


def _validate_model(model: Any, *, group: str, label: str) -> None:
    if model is None:
        return
    raw = (
        model.model_dump(exclude_none=True)
        if hasattr(model, "model_dump")
        else model
    )
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} model configuration must be a mapping.")
    assert_no_raw_api_key(raw, context=f"{label}.model")
    assert_no_credential_metadata(raw, context=f"{label}.model")
    enforce_group_base_url_policy(
        str(raw.get("base_url")) if raw.get("base_url") else None,
        group,
    )
    env_name = str(raw.get("api_key_env") or "").strip()
    if env_name and not _ENV_VAR_RE.fullmatch(env_name):
        raise ValueError(f"{label} has an invalid api_key_env name.")


def _validate_member(
    member: EnvironmentMemberConfig,
    *,
    group: str,
    label: str,
) -> None:
    _validate_simple_name(member.name, label=f"{label} name")
    agent = str(member.agent or "").strip()
    if not agent:
        raise ValueError(f"{label} must specify an agent class.")
    if (
        agent not in _ALLOWED_SHORT_CLASSES
        and agent not in _ALLOWED_CLASS_PATHS
        and not _allow_custom_classes()
    ):
        raise ValueError(
            f"{label} agent class {agent!r} is not available from the dashboard. "
            "Use a built-in URSA class or explicitly enable custom environment classes."
        )
    _validate_model(member.model, group=group, label=label)
    if agent.rsplit(".", 1)[-1] == "AgentTeamEnvironment":
        nested = (member.config or {}).get("config")
        if not isinstance(nested, Mapping):
            raise ValueError(
                f"{label} is a nested team and requires config.config mapping."
            )
        _validate_config_mapping(
            "agent_team", dict(nested), group=group, nested=True
        )


def _validate_config_mapping(
    environment_type: EnvironmentType,
    data: dict[str, Any],
    *,
    group: str,
    nested: bool = False,
) -> AgentTeamConfig | AgentSymposiumConfig:
    data["group"] = group
    assert_no_raw_api_key(data, context="environment config")
    try:
        if environment_type == "agent_team":
            config: AgentTeamConfig | AgentSymposiumConfig = (
                AgentTeamConfig.from_mapping(data)
            )
            lead = config.pi
            lead_label = "PI"
        else:
            config = AgentSymposiumConfig.from_mapping(data)
            lead = config.organizer
            lead_label = "Organizer"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid environment configuration: {exc}") from exc

    _validate_simple_name(config.name, label="Environment name")
    if not config.members:
        raise ValueError("An environment must contain at least one member.")
    _validate_member(lead, group=group, label=lead_label)
    seen = {lead.name}
    for index, member in enumerate(config.members, start=1):
        _validate_member(member, group=group, label=f"Member {index}")
        if member.name in seen:
            raise ValueError(
                f"Environment member name {member.name!r} is duplicated."
            )
        seen.add(member.name)
    if (
        nested and environment_type != "agent_team"
    ):  # pragma: no cover - defensive
        raise ValueError("Only teams can be nested in an environment.")
    return config


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _plain(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def validate_environment_launch(
    environment_type: EnvironmentType,
    config_yaml: str,
    *,
    group: str,
) -> ValidatedEnvironmentLaunch:
    effective_group = validate_group_name(group)
    try:
        data = yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Environment YAML must contain a top-level mapping.")
    raw = dict(data)
    config = _validate_config_mapping(
        environment_type, raw, group=effective_group
    )
    return ValidatedEnvironmentLaunch(
        environment_type=environment_type,
        config=config,
        config_mapping=_plain(config),
    )


class EnvironmentRunManager:
    """Queue and supervise dashboard-launched environment subprocesses."""

    def __init__(
        self,
        *,
        group: str,
        credential_store: CredentialStore,
        config: EnvironmentRunManagerConfig | None = None,
    ) -> None:
        self.group = validate_group_name(group)
        self.credential_store = credential_store
        self.config = config or EnvironmentRunManagerConfig()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._inflight: dict[str, _InFlight] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self._recover_runs()
        for _ in range(self.config.concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def shutdown(self) -> None:
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        for item in list(self._inflight.values()):
            with contextlib.suppress(ProcessLookupError, OSError):
                item.proc.terminate()
        self._workers.clear()
        self._started = False

    def validate_credentials(
        self,
        *,
        llm: Mapping[str, Any],
        config_mapping: Mapping[str, Any],
    ) -> None:
        self._resolve_credentials(llm=llm, config_mapping=config_mapping)

    def _resolve_credentials(
        self,
        *,
        llm: Mapping[str, Any],
        config_mapping: Mapping[str, Any],
    ) -> dict[str, Any]:
        if bool(llm.get("disabled")) or str(llm.get("model") or "").lower() in {
            "none",
            "disabled",
        }:
            raise CredentialConfigurationError(
                "Agent teams and symposia require an enabled dashboard LLM."
            )
        main_key = resolve_api_key(
            llm,
            group=self.group,
            kind="llm",
            store=self.credential_store,
        )
        names: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                if value.get("model") and value.get("api_key_env"):
                    name = str(value["api_key_env"]).strip()
                    if not _ENV_VAR_RE.fullmatch(name):
                        raise CredentialConfigurationError(
                            f"Invalid member API-key environment variable {name!r}."
                        )
                    names.add(name)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    collect(item)

        collect(config_mapping)
        member_keys: dict[str, str] = {}
        for name in names:
            value = os.environ.get(name)
            if not value:
                raise CredentialConfigurationError(
                    f"Member model API-key environment variable {name!r} is not set."
                )
            member_keys[name] = value
        return {"llm_api_key": main_key, "member_api_keys": member_keys}

    async def create_run(
        self,
        *,
        launch: ValidatedEnvironmentLaunch,
        prompt: str,
        llm: dict[str, Any],
        runner: dict[str, Any],
        run_id: str | None = None,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        name = launch.config.name
        definition_path = (
            team_cache_dir(self.group, name) / "team.yaml"
            if launch.environment_type == "agent_team"
            else symposium_cache_dir(self.group, name) / "symposium.yaml"
        )
        definition_matches = False
        if definition_path.exists():
            with contextlib.suppress(Exception):
                existing = yaml.safe_load(
                    definition_path.read_text(encoding="utf-8")
                )
                if isinstance(existing, Mapping):
                    existing_mapping = dict(existing)
                    existing_mapping["group"] = self.group
                    existing_config = (
                        AgentTeamConfig.from_mapping(existing_mapping)
                        if launch.environment_type == "agent_team"
                        else AgentSymposiumConfig.from_mapping(existing_mapping)
                    )
                    definition_matches = (
                        _plain(existing_config) == launch.config_mapping
                    )
            if not replace_existing and not definition_matches:
                raise EnvironmentDefinitionExistsError(definition_path)
        if launch.environment_type == "agent_team":
            if replace_existing or not definition_path.exists():
                save_team_config(launch.config, definition_path)
            class_name = "AgentTeamEnvironment"
        else:
            if replace_existing or not definition_path.exists():
                save_symposium_config(launch.config, definition_path)
            class_name = "AgentSymposiumEnvironment"

        effective_run_id = (
            _validate_simple_name(run_id, label="Run ID")
            if run_id is not None
            else new_run_id()
        )
        paths = get_environment_run_paths(self.group, effective_run_id)
        if paths.run_dir.exists():
            raise EnvironmentRunExistsError(paths.run_dir)
        paths.run_dir.mkdir(parents=True, exist_ok=False)
        paths.artifacts_dir.mkdir()
        paths.logs_dir.mkdir()
        config_path = paths.run_dir / "environment.yaml"
        config_path.write_text(
            yaml.safe_dump(launch.config_mapping, sort_keys=False),
            encoding="utf-8",
        )
        (paths.run_dir / "task.json").write_text(
            json.dumps({"prompt": prompt}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (paths.run_dir / "launch.json").write_text(
            json.dumps(
                {
                    "environment_type": launch.environment_type,
                    "environment_class": class_name,
                    "environment_name": name,
                    "group": self.group,
                    "prompt": prompt,
                    "llm": llm,
                    "runner": runner,
                    "definition_path": str(definition_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        recorder = EnvironmentEventRecorder(
            run_id=effective_run_id,
            group=self.group,
            environment_name=name,
            environment_type=class_name,
        )
        recorder.write_manifest(status="queued", task=prompt)
        self._update_manifest(
            effective_run_id,
            {
                "launch_source": "dashboard",
                "definition_path": str(definition_path),
                "config_path": "environment.yaml",
                "task_path": "task.json",
            },
        )
        await self._queue.put(effective_run_id)
        return read_environment_run_manifest(self.group, effective_run_id)

    async def cancel(self, run_id: str, *, reason: str) -> dict[str, Any]:
        _validate_simple_name(run_id, label="Run ID")
        manifest = read_environment_run_manifest(self.group, run_id)
        if manifest.get("launch_source") != "dashboard":
            raise PermissionError(
                "Only environment runs launched by this dashboard can be cancelled here."
            )
        if manifest.get("status") in TERMINAL_STATUSES:
            return manifest
        inflight = self._inflight.get(run_id)
        if inflight is None:
            self._update_manifest(
                run_id,
                {
                    "status": "cancelled",
                    "cancel_reason": reason,
                    "cancelled_at": utc_now_rfc3339(),
                },
            )
        else:
            inflight.cancel_requested = True
            self._update_manifest(
                run_id, {"status": "cancelling", "cancel_reason": reason}
            )
            with contextlib.suppress(ProcessLookupError, OSError):
                inflight.proc.send_signal(signal.SIGTERM)
            asyncio.create_task(self._kill_after(run_id))
        return read_environment_run_manifest(self.group, run_id)

    def _update_manifest(self, run_id: str, patch: Mapping[str, Any]) -> None:
        paths = get_environment_run_paths(self.group, run_id)
        current: dict[str, Any] = {}
        if paths.manifest_path.exists():
            with contextlib.suppress(Exception):
                current = json.loads(
                    paths.manifest_path.read_text(encoding="utf-8")
                )
        current.update(dict(patch))
        current["updated_at"] = utc_now_rfc3339()
        paths.manifest_path.write_text(
            json.dumps(current, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    async def _recover_runs(self) -> None:
        from ursa.environments.visualization import environment_runs_dir

        root = environment_runs_dir(self.group)
        if not root.exists():
            return
        for run_dir in root.iterdir():
            launch_path = run_dir / "launch.json"
            manifest_path = run_dir / "manifest.json"
            if not launch_path.is_file() or not manifest_path.is_file():
                continue
            with contextlib.suppress(Exception):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                status = manifest.get("status")
                if status == "queued":
                    await self._queue.put(run_dir.name)
                elif status in {"starting", "running", "cancelling"}:
                    self._update_manifest(
                        run_dir.name,
                        {
                            "status": "failed",
                            "error": "Dashboard restarted during environment run.",
                        },
                    )

    async def _worker_loop(self) -> None:
        while True:
            run_id = await self._queue.get()
            try:
                manifest = read_environment_run_manifest(self.group, run_id)
                if manifest.get("status") == "cancelled":
                    continue
                await self._execute(run_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                with contextlib.suppress(Exception):
                    self._update_manifest(
                        run_id,
                        {"status": "failed", "error": str(exc)},
                    )
            finally:
                self._queue.task_done()

    async def _execute(self, run_id: str) -> None:
        paths = get_environment_run_paths(self.group, run_id)
        launch = json.loads(
            (paths.run_dir / "launch.json").read_text(encoding="utf-8")
        )
        config_mapping = yaml.safe_load(
            (paths.run_dir / "environment.yaml").read_text(encoding="utf-8")
        )
        self._update_manifest(run_id, {"status": "starting"})
        try:
            secrets = await asyncio.to_thread(
                self._resolve_credentials,
                llm=launch.get("llm") or {},
                config_mapping=config_mapping,
            )
        except Exception as exc:
            self._update_manifest(
                run_id, {"status": "failed", "error": str(exc)}
            )
            return
        if (
            read_environment_run_manifest(self.group, run_id).get("status")
            == "cancelled"
        ):
            return

        output_path = paths.run_dir / "output.json"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "ursa_dashboard.environment_worker_main",
            "--run-id",
            run_id,
            "--group",
            self.group,
            "--environment-type",
            str(launch["environment_type"]),
            "--config-yaml",
            str(paths.run_dir / "environment.yaml"),
            "--task-json",
            str(paths.run_dir / "task.json"),
            "--llm-json",
            str(paths.run_dir / "launch.json"),
            "--output-json",
            str(output_path),
            "--secrets-stdin",
        ]
        env = self._worker_environment(launch)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(paths.run_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._inflight[run_id] = _InFlight(proc=proc)
        if proc.stdin is None:  # pragma: no cover - asyncio contract
            proc.kill()
            await proc.wait()
            self._update_manifest(
                run_id,
                {
                    "status": "failed",
                    "error": "Worker secret channel unavailable.",
                },
            )
            self._inflight.pop(run_id, None)
            return
        pipe_error: Exception | None = None
        try:
            proc.stdin.write(
                (json.dumps(secrets, ensure_ascii=False) + "\n").encode("utf-8")
            )
            await proc.stdin.drain()
        except Exception as exc:
            pipe_error = exc
        finally:
            proc.stdin.close()
            with contextlib.suppress(Exception):
                await proc.stdin.wait_closed()
            secrets = {}

        if pipe_error is not None:
            inflight = self._inflight.pop(run_id, None)
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            if inflight and inflight.cancel_requested:
                self._update_manifest(
                    run_id,
                    {
                        "status": "cancelled",
                        "cancelled_at": utc_now_rfc3339(),
                    },
                )
            else:
                self._update_manifest(
                    run_id,
                    {
                        "status": "failed",
                        "error": "Could not deliver credentials to worker.",
                    },
                )
            return

        inflight = self._inflight[run_id]
        if inflight.cancel_requested:
            self._update_manifest(
                run_id, {"status": "cancelling", "runtime": {"pid": proc.pid}}
            )
        else:
            self._update_manifest(
                run_id, {"status": "running", "runtime": {"pid": proc.pid}}
            )
        stdout_task = asyncio.create_task(
            self._drain(
                proc.stdout,
                paths.logs_dir / "stdout.log",
                self.config.stdout_cap_bytes,
            )
        )
        stderr_task = asyncio.create_task(
            self._drain(
                proc.stderr,
                paths.logs_dir / "stderr.log",
                self.config.stderr_cap_bytes,
            )
        )
        timeout = (launch.get("runner") or {}).get("timeout_seconds")
        timed_out = False
        try:
            if timeout:
                await asyncio.wait_for(proc.wait(), timeout=float(timeout))
            else:
                await proc.wait()
        except asyncio.TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        finally:
            await asyncio.gather(
                stdout_task, stderr_task, return_exceptions=True
            )
        inflight = self._inflight.pop(run_id, None)
        if inflight and inflight.cancel_requested:
            self._update_manifest(
                run_id,
                {"status": "cancelled", "cancelled_at": utc_now_rfc3339()},
            )
        elif timed_out:
            self._update_manifest(
                run_id,
                {
                    "status": "failed",
                    "error": f"Timed out after {timeout} seconds.",
                },
            )
        elif proc.returncode != 0:
            error = f"Environment worker exited with status {proc.returncode}."
            if output_path.exists():
                with contextlib.suppress(Exception):
                    output = json.loads(output_path.read_text(encoding="utf-8"))
                    error = str(output.get("message") or error)
            self._update_manifest(run_id, {"status": "failed", "error": error})
        else:
            current = read_environment_run_manifest(self.group, run_id)
            if current.get("status") not in TERMINAL_STATUSES:
                self._update_manifest(run_id, {"status": "succeeded"})

    @staticmethod
    def _worker_environment(launch: Mapping[str, Any]) -> dict[str, str]:
        env = dict(os.environ)
        secret_markers = (
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "secret",
            "password",
            "credential",
            "bearer",
        )
        for name in list(env):
            lowered = name.lower()
            if any(
                marker in lowered for marker in secret_markers
            ) or lowered.endswith("_token"):
                env.pop(name, None)
        env["PYTHONUNBUFFERED"] = "1"
        project_root = str(Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            project_root
            if not existing
            else project_root + os.pathsep + existing
        )
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    async def _kill_after(self, run_id: str) -> None:
        await asyncio.sleep(5)
        item = self._inflight.get(run_id)
        if item and item.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                item.proc.kill()

    @staticmethod
    async def _drain(
        stream: asyncio.StreamReader | None,
        path: Path,
        cap_bytes: int,
    ) -> None:
        if stream is None:
            return
        written = 0
        with path.open("ab") as handle:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                if written < cap_bytes:
                    payload = chunk[: cap_bytes - written]
                    handle.write(payload)
                    handle.flush()
                    written += len(payload)


TEAM_STARTER_YAML = """name: research_team
description: A coordinated team of specialists.
pi:
  name: pi
  role: Principal investigator and final synthesizer
  agent: ChatAgent
  config:
    use_web: false
members:
  - name: analyst
    role: Performs focused analysis and reports evidence
    agent: ExecutionAgent
    config:
      use_web: false
"""


SYMPOSIUM_STARTER_YAML = """name: research_symposium
description: Independent approaches followed by review and synthesis.
revision_rounds: 1
organizer:
  name: organizer
  role: Final synthesizer and judge of evidence quality
  agent: ChatAgent
  config:
    use_web: false
members:
  - name: primary_path
    role: Develops a concrete solution
    agent: ExecutionAgent
    config:
      use_web: false
  - name: critical_path
    role: Challenges assumptions and proposes alternatives
    agent: ChatAgent
    config:
      use_web: false
"""
