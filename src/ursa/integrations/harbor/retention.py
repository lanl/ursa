"""Post-job artifact retention for Harbor, also usable on completed job folders."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import yaml
from harbor.models.job.plugin import BaseJobPlugin
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ursa.artifacts.retention import (
    MANIFEST,
    RetentionPlan,
    RetentionPolicy,
    plan_retention,
    retain_artifacts,
)
from ursa.util.fileio import atomic_write_text

if TYPE_CHECKING:
    from harbor.job import Job
    from harbor.models.job.result import JobResult

STATUS = "retention-status.json"
_PROTECTED = {
    "reward.txt",
    "reward.json",
    "test-stdout.txt",
    "test-stderr.txt",
    "manifest.json",
}


class HarborRetentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    collections: list[Literal["artifacts", "verifier", "trial"]] = Field(
        default_factory=lambda: ["artifacts"]
    )
    policy: RetentionPolicy

    @model_validator(mode="after")
    def protect_status(self) -> Self:
        if "trial" in self.collections and set(self.collections) != {"trial"}:
            raise ValueError("The trial collection must be selected on its own")
        if (
            self.policy.archive is not None
            and self.policy.archive.path == STATUS
        ):
            raise ValueError("Archive path conflicts with retention status")
        return self


def load_retention_config(path: Path) -> HarborRetentionConfig:
    """Read YAML or JSON, with an optional top-level ``retention`` key."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Retention configuration must be a mapping")
    data = dict(data.get("retention", data))
    collections = data.pop("collections", ["artifacts"])
    return HarborRetentionConfig(collections=collections, policy=data)


def _scope(
    root: Path, config: HarborRetentionConfig, *, require_finished: bool = False
) -> tuple[list[str], list[str]]:
    roots, protected = [], []
    trials = {result.parent for result in root.glob("*/result.json")}
    removable_results = set()
    whole_trial = "trial" in config.collections
    manifest = root / MANIFEST
    if whole_trial and (manifest.exists() or manifest.is_symlink()):
        if manifest.is_symlink():
            raise ValueError("Retention manifest must not be a symlink")
        state = json.loads(manifest.read_text())
        previous = RetentionPlan.model_validate(state["plan"])
        if (
            previous.root != str(root.resolve())
            or previous.policy != config.policy
            or previous.protected
            or any(
                Path(name).name != name
                or name in {".", "..", ""}
                or "\\" in name
                for name in previous.roots
            )
        ):
            raise ValueError("Existing retention policy/inventory differs")
        # Cleanup can already have removed a trial's result. The recorded
        # roots keep resumed and completed runs discoverable in that case.
        trials.update(root / name for name in previous.roots)
        if state["cleanup_started"]:
            removable_results = {
                entry.path
                for entry in previous.entries
                if previous.policy.removes(entry.action)
            }
    for trial in sorted(trials):
        if trial.is_symlink():
            raise ValueError(f"Trial directory must not be a symlink: {trial}")
        result = trial / "result.json"
        if require_finished:
            if result.exists():
                if not json.loads(result.read_text()).get("finished_at"):
                    raise ValueError(
                        f"Retention requires a finished trial: {trial}"
                    )
            elif result.relative_to(root).as_posix() not in removable_results:
                raise ValueError(
                    f"Retention requires a finished trial: {trial}"
                )
        if whole_trial:
            roots.append(trial.relative_to(root).as_posix())
            continue
        bases = [trial, *sorted((trial / "steps").glob("*"))]
        for base in bases:
            if base.is_symlink() or (trial / "steps").is_symlink():
                raise ValueError("Step directories must not be symlinks")
            for collection in sorted(set(config.collections)):
                directory = base / collection
                if directory.is_symlink():
                    raise ValueError(
                        "Artifact collections must not be symlinks"
                    )
                if directory.is_dir():
                    relative = directory.relative_to(root).as_posix()
                    roots.append(relative)
                    protected.extend(
                        f"{relative}/{name}" for name in _PROTECTED
                    )
    return roots, protected


def plan_job_retention(
    root: Path, config: HarborRetentionConfig
) -> RetentionPlan:
    roots, protected = _scope(root, config)
    return plan_retention(root, config.policy, roots=roots, protected=protected)


def apply_job_retention(
    root: Path,
    config: HarborRetentionConfig,
    *,
    job_result: JobResult | None = None,
) -> dict:
    """Require completed job/trial metadata before applying or resuming cleanup."""
    if root.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    try:
        if job_result is not None and job_result.finished_at is None:
            raise ValueError("Retention requires a finished Harbor job")
        result = json.loads((root / "result.json").read_text())
        if not result.get("finished_at"):
            raise ValueError("Retention requires a finished Harbor job")
        roots, protected = _scope(root, config, require_finished=True)
        state = retain_artifacts(
            root, config.policy, roots=roots, protected=protected
        )
        _write_status(root, "complete", duration_sec=state["duration_sec"])
        return state
    except Exception as exc:
        _write_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _write_status(root: Path, status: str, **details) -> None:
    target = root / STATUS
    if target.is_symlink():
        raise ValueError("Retention status must not be a symlink")
    atomic_write_text(target, json.dumps({"status": status, **details}) + "\n")


class RetentionPlugin(BaseJobPlugin):
    """Run after Harbor has finished all trials and verifier transfers.

    Harbor logs plugin exceptions without failing the job. Consumers requiring
    retention must check ``retention-status.json`` for ``status: complete``.
    """

    def __init__(self, policy: str) -> None:
        self.config = load_retention_config(Path(policy))
        self.root: Path | None = None

    async def on_job_start(self, job: Job) -> None:
        self.root = job.job_dir
        _write_status(self.root, "pending")

    async def on_job_end(self, job_result: JobResult) -> None:
        if self.root is None:
            raise RuntimeError("Retention plugin has not started")
        await asyncio.to_thread(
            apply_job_retention, self.root, self.config, job_result=job_result
        )
