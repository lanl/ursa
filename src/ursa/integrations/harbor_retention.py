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
    collections: list[Literal["artifacts", "verifier"]] = Field(
        default_factory=lambda: ["artifacts"]
    )
    policy: RetentionPolicy

    @model_validator(mode="after")
    def protect_status(self) -> Self:
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
    root: Path, config: HarborRetentionConfig
) -> tuple[list[str], list[str]]:
    roots, protected = [], []
    # Only recognized trial payloads are eligible. Agent logs, results, lock
    # files and job metadata never enter the policy's matching scope.
    for result in sorted(root.glob("*/result.json")):
        trial = result.parent
        if trial.is_symlink():
            raise ValueError(f"Trial directory must not be a symlink: {trial}")
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


def apply_job_retention(root: Path, config: HarborRetentionConfig) -> dict:
    """Require completed job/trial metadata before applying or resuming cleanup."""
    if root.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    try:
        state = _apply_job_retention(root, config)
        _write_status(root, "complete", duration_sec=state["duration_sec"])
        return state
    except Exception as exc:
        _write_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _apply_job_retention(root: Path, config: HarborRetentionConfig) -> dict:
    result = json.loads((root / "result.json").read_text())
    if not result.get("finished_at"):
        raise ValueError("Retention requires a finished Harbor job")
    for trial in root.glob("*/result.json"):
        if not json.loads(trial.read_text()).get("finished_at"):
            raise ValueError(
                f"Retention requires a finished trial: {trial.parent}"
            )
    roots, protected = _scope(root, config)
    return retain_artifacts(
        root, config.policy, roots=roots, protected=protected
    )


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

    def _status(self, status: str, **details) -> None:
        if self.root is None:
            raise RuntimeError("Retention plugin has not started")
        _write_status(self.root, status, **details)

    async def on_job_start(self, job: Job) -> None:
        self.root = job.job_dir
        self._status("pending")

    async def on_job_end(self, job_result: JobResult) -> None:
        if self.root is None:
            raise RuntimeError("Retention plugin has not started")
        try:
            if job_result.finished_at is None:
                raise ValueError("Retention requires a finished Harbor job")
            state = await asyncio.to_thread(
                apply_job_retention, self.root, self.config
            )
            self._status("complete", duration_sec=state["duration_sec"])
        except Exception as exc:
            self._status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
