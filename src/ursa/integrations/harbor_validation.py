"""Static validation for Harbor task environments supported by URSA."""

from __future__ import annotations

from pathlib import Path

from harbor.models.task.config import StepConfig, VerifierEnvironmentMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    resolve_effective_verifier_env_config,
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.config import (
    EnvironmentConfig as TrialEnvironmentConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.trial.network_policy import (
    TrialNetworkPlan,
    resolve_trial_network_plan,
)

from ursa.integrations.harbor_singularity import (
    DockerfileSingularityEnvironment,
)

_IGNORED_SCAN_DIRECTORIES = {
    "__pycache__",
    "jobs",
    "node_modules",
    "venv",
}


def _is_ignored_task_config(config: Path, root: Path) -> bool:
    relative_parts = config.relative_to(root).parts[:-1]
    return any(
        part.startswith(".") or part in _IGNORED_SCAN_DIRECTORIES
        for part in relative_parts
    )


def _verifier_mode(
    task: Task, step: StepConfig | None
) -> VerifierEnvironmentMode:
    if step is None:
        return resolve_task_verifier_mode(task.config)
    return resolve_step_verifier_mode(task.config, step)


def _validate_environment(
    task: Task,
    *,
    environment_dir: Path,
    task_env_config,
    network_policy,
    phase_network_policies,
    suffix: str,
) -> None:
    DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name=task.short_name,
        session_id=f"{task.short_name}__validate__{suffix}",
        trial_paths=TrialPaths(task.paths.task_dir / ".ursa-validation"),
        task_env_config=task_env_config,
        network_policy=network_policy,
        phase_network_policies=phase_network_policies,
    )


def validate_harbor_task(task_dir: Path | str) -> None:
    """Validate one Harbor task against URSA's Singularity constraints."""
    task = Task(task_dir)
    trial_agent = AgentConfig()
    trial_environment = TrialEnvironmentConfig()
    steps: list[StepConfig | None] = list(task.config.steps or [None])
    plans: list[
        tuple[StepConfig | None, VerifierEnvironmentMode, TrialNetworkPlan]
    ] = []

    for step in steps:
        mode = _verifier_mode(task, step)
        verifier_environment = resolve_effective_verifier_env_config(
            task.config, step
        )
        plan = resolve_trial_network_plan(
            task.config,
            trial_agent,
            trial_environment,
            step,
            verifier_mode=mode,
            env_config=verifier_environment,
        )
        plans.append((step, mode, plan))

    agent_phases = [
        policy
        for _, mode, plan in plans
        for policy in (
            [plan.agent_phase, plan.verifier_phase]
            if mode == VerifierEnvironmentMode.SHARED
            else [plan.agent_phase]
        )
    ]
    _validate_environment(
        task,
        environment_dir=task.paths.environment_dir,
        task_env_config=task.config.environment,
        network_policy=plans[0][2].agent_env_baseline,
        phase_network_policies=agent_phases,
        suffix="agent",
    )

    for index, (step, mode, plan) in enumerate(plans):
        if mode != VerifierEnvironmentMode.SEPARATE:
            continue
        verifier_environment = resolve_effective_verifier_env_config(
            task.config, step
        )
        if verifier_environment is None or plan.verifier_env_baseline is None:
            raise RuntimeError("Separate verifier environment did not resolve")
        environment_dir = task.paths.tests_dir
        if step is not None:
            step_tests = task.paths.step_tests_dir(step.name)
            if step_tests.exists():
                environment_dir = step_tests
        _validate_environment(
            task,
            environment_dir=environment_dir,
            task_env_config=verifier_environment,
            network_policy=plan.verifier_env_baseline,
            phase_network_policies=[plan.verifier_phase],
            suffix=f"verifier-{index}",
        )


def discover_harbor_tasks(paths: list[Path]) -> list[Path]:
    """Resolve task directories from task files, task directories, or roots."""
    tasks: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Validation path does not exist: {raw_path}"
            )
        if path.is_file():
            candidates = [path.parent, *path.parents]
            task_dir = next(
                (
                    candidate
                    for candidate in candidates
                    if (candidate / "task.toml").is_file()
                ),
                None,
            )
            if task_dir is None:
                raise ValueError(
                    f"File is not inside a Harbor task: {raw_path}"
                )
            tasks.add(task_dir)
            continue
        if (path / "task.toml").is_file():
            tasks.add(path)
            continue
        tasks.update(
            config.parent
            for config in path.rglob("task.toml")
            if not _is_ignored_task_config(config, path)
        )

    if not tasks:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"No Harbor tasks found below: {joined}")
    return sorted(tasks)


__all__ = ["discover_harbor_tasks", "validate_harbor_task"]
