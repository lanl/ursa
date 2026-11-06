"""Filter and archive completed-run artifacts with resumable cleanup.

Callers supply an idle output directory and explicit artifact roots. No process
or verifier may still be writing those roots while retention is applied.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import tarfile
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from filelock import FileLock
from pathspec import PathSpec
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ursa.util.fileio import atomic_write_text

Action = Literal["keep", "delete", "archive"]
GlobPattern = Annotated[str, Field(min_length=1, pattern=r"^[^!#]")]
MANIFEST = "retention.json"


class RetentionRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    match: Annotated[list[GlobPattern], Field(min_length=1)]
    action: Action


class ArchiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["tar.xz"] = "tar.xz"
    path: str = "retained-artifacts.tar.xz"
    remove_originals: bool = True
    compression_level: Annotated[int, Field(ge=0, le=9)] = 6

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return _relative(value).as_posix()


class RetentionPolicy(BaseModel):
    """Ordered first-match rules; unmatched files remain loose by default."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    rules: list[RetentionRule] = Field(default_factory=list)
    default_action: Action = "keep"
    archive: ArchiveConfig | None = None

    def removes(self, action: Action) -> bool:
        return action == "delete" or (
            action == "archive" and self.archive.remove_originals
        )

    @model_validator(mode="after")
    def validate_archive(self) -> Self:
        if self.default_action == "archive" or any(
            rule.action == "archive" for rule in self.rules
        ):
            if self.archive is None:
                raise ValueError("Archive actions require archive settings")
        if self.archive is not None:
            _relative(self.archive.path)
            if self.archive.path in {MANIFEST, ".retention.lock"}:
                raise ValueError(
                    "Archive path conflicts with retention metadata"
                )
        return self


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    action: Action
    bytes: int
    identity: list[int]
    sha256: str | None = None
    link: str | None = None


class RetentionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    root: str
    roots: list[str]
    protected: list[str]
    policy: RetentionPolicy
    entries: list[ArtifactEntry]


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Expected a relative path within the run: {value!r}")
    return path


def _path(root: Path, value: str) -> Path:
    relative = _relative(value)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in artifact parent path: {value}")
    return root.joinpath(*relative.parts)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(path: Path) -> list[int]:
    info = path.lstat()
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
    ]


def _inventory(root: Path, roots: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in roots:
        directory = _path(root, name)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Artifact root must be a real directory: {name}")
        for parent, directories, names in os.walk(directory, followlinks=False):
            for child in list(directories):
                if (Path(parent) / child).is_symlink():
                    directories.remove(child)
                    names.append(child)
            for child in names:
                path = Path(parent) / child
                if not (path.is_symlink() or path.is_file()):
                    raise ValueError(f"Unsupported artifact file type: {path}")
                files[path.relative_to(root).as_posix()] = path
    return files


def plan_retention(
    root: Path,
    policy: RetentionPolicy,
    *,
    roots: Sequence[str],
    protected: Sequence[str] = (),
) -> RetentionPlan:
    """Inventory without changing files; globs match relative to each root.

    Patterns use git-wildmatch semantics (``**/*.dat`` includes root files).
    Negative patterns are not allowed; use an earlier keep rule instead.
    """
    if root.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    root = root.resolve(strict=True)
    names = sorted({_relative(name).as_posix() for name in roots})
    protected_names = sorted({_relative(name).as_posix() for name in protected})
    for name in names:
        if any(
            name.startswith(other + "/") for other in names if other != name
        ):
            raise ValueError("Artifact roots must not overlap")
    reserved = [MANIFEST, ".retention.lock"]
    if policy.archive is not None:
        reserved.append(policy.archive.path)
    for output in reserved:
        _path(root, output)
        if any(
            output == name or output.startswith(name + "/") for name in names
        ):
            raise ValueError("Retention outputs must be outside artifact roots")
    rules = [
        (
            PathSpec.from_lines(
                "gitignore", ["/" + p.lstrip("/") for p in rule.match]
            ),
            rule.action,
        )
        for rule in policy.rules
    ]
    entries = []
    for name, path in sorted(_inventory(root, names).items()):
        directory = next(n for n in names if name.startswith(n + "/"))
        relative = name[len(directory) + 1 :]
        action = next(
            (
                action
                for pattern, action in rules
                if pattern.match_file(relative)
            ),
            policy.default_action,
        )
        if name in protected_names:
            action = "keep"
        identity = _identity(path)
        link = str(path.readlink()) if path.is_symlink() else None
        digest = _hash(path) if action == "archive" and link is None else None
        if identity != _identity(path):
            raise ValueError(f"Artifact changed during planning: {name}")
        entries.append(
            ArtifactEntry(
                path=name,
                action=action,
                bytes=identity[3],
                identity=identity,
                link=link,
                sha256=digest,
            )
        )
    return RetentionPlan(
        root=str(root),
        roots=names,
        protected=protected_names,
        policy=policy,
        entries=entries,
    )


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write(root: Path, state: dict) -> None:
    path = root / MANIFEST
    if path.is_symlink():
        raise ValueError("Retention manifest must not be a symlink")
    atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    _sync_directory(root)


def _verify_archive(path: Path, entries: list[ArtifactEntry]) -> None:
    expected = {entry.path: entry for entry in entries}
    with tarfile.open(path, "r:xz") as archive:
        for member in archive:
            entry = expected.pop(member.name, None)
            if entry is None:
                raise ValueError(
                    "Archive inventory differs from retention plan"
                )
            if entry.link is not None:
                if not member.issym() or member.linkname != entry.link:
                    raise ValueError(f"Archive link mismatch: {entry.path}")
            else:
                if not member.isfile() or member.size != entry.bytes:
                    raise ValueError(f"Archive member mismatch: {entry.path}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Missing archive contents: {entry.path}")
                with stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != entry.sha256:
                    raise ValueError(f"Archive checksum mismatch: {entry.path}")
        if expected:
            raise ValueError("Archive is missing planned artifacts")


def _check_inputs(plan: RetentionPlan, *, cleanup: bool) -> None:
    root = Path(plan.root)
    current = _inventory(root, plan.roots)
    expected = {entry.path for entry in plan.entries}
    if set(current) - expected:
        raise ValueError("New artifacts appeared after retention was planned")
    for entry in plan.entries:
        path = current.get(entry.path)
        if path is None:
            if cleanup and plan.policy.removes(entry.action):
                continue
            raise ValueError(f"Artifact disappeared: {entry.path}")
        if _identity(path) != entry.identity:
            raise ValueError(f"Artifact changed after planning: {entry.path}")


def _publish_archive(root: Path, plan: RetentionPlan, state: dict) -> None:
    entries = [entry for entry in plan.entries if entry.action == "archive"]
    if not entries:
        return
    config = plan.policy.archive
    target = _path(root, config.path)
    if target.is_symlink():
        raise ValueError("Archive destination must not be a symlink")
    if target.exists():
        # Recover publication that completed before the journal update.
        _verify_archive(target, entries)
    else:
        temporary = _path(root, state["temporary_archive"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if temporary.is_symlink():
            raise ValueError("Temporary archive must not be a symlink")
        try:
            with tarfile.open(
                temporary,
                "w:xz",
                preset=config.compression_level,
                dereference=False,
            ) as archive:
                for entry in entries:
                    path = _path(root, entry.path)
                    if _identity(path) != entry.identity:
                        raise ValueError(f"Artifact changed: {entry.path}")
                    # Preserve contents of hard-linked files independently.
                    archive.inodes.clear()
                    archive.add(path, arcname=entry.path, recursive=False)
            _verify_archive(temporary, entries)
            _check_inputs(plan, cleanup=False)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            # A hard link publishes atomically without overwriting another file.
            os.link(temporary, target)
            _sync_directory(target.parent)
        finally:
            temporary.unlink(missing_ok=True)
    state["archive"] = {
        "path": config.path,
        "bytes": target.stat().st_size,
        "sha256": _hash(target),
    }


def _link_warnings(plan: RetentionPlan) -> list[dict[str, str]]:
    archived = {
        entry.path for entry in plan.entries if entry.action == "archive"
    }
    warnings = []
    for entry in plan.entries:
        if entry.action != "archive" or entry.link is None:
            continue
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(entry.path), entry.link)
        )
        if target.startswith(("/", "../")) or not any(
            name == target or name.startswith(target + "/") for name in archived
        ):
            warnings.append({
                "path": entry.path,
                "target": entry.link,
                "reason": "target is outside the archive",
            })
    return warnings


def apply_retention(plan: RetentionPlan) -> dict:
    """Apply or resume this exact plan, preserving sources until verification."""
    root = Path(plan.root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Run directory is no longer available")
    manifest = root / MANIFEST
    if manifest.is_symlink() or (root / ".retention.lock").is_symlink():
        raise ValueError("Retention metadata must not be symlinks")
    with FileLock(str(root / ".retention.lock")):
        state = json.loads(manifest.read_text()) if manifest.exists() else None
        if state is not None:
            if state["plan"] != plan.model_dump(mode="json"):
                raise ValueError(
                    "Existing retention manifest describes another plan"
                )
        else:
            if plan.policy.archive is not None:
                target = _path(root, plan.policy.archive.path)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
            _check_inputs(plan, cleanup=False)
            state = {
                "version": 1,
                "status": "pending",
                "cleanup_started": False,
                "plan": plan.model_dump(mode="json"),
                "removed": [],
                "archive": None,
                "duration_sec": 0.0,
                "input_bytes": {
                    action: sum(
                        entry.bytes
                        for entry in plan.entries
                        if entry.action == action
                    )
                    for action in ("keep", "delete", "archive")
                },
                "link_warnings": _link_warnings(plan),
                "temporary_archive": (
                    plan.policy.archive.path + f".{uuid.uuid4().hex}.partial"
                    if plan.policy.archive is not None
                    else None
                ),
            }
            _write(root, state)
        completed = state["status"] == "complete"
        started = time.monotonic()
        try:
            _check_inputs(plan, cleanup=state["cleanup_started"])
            if state["archive"] is not None:
                archive = _path(root, state["archive"]["path"])
                if (
                    archive.is_symlink()
                    or _hash(archive) != state["archive"]["sha256"]
                ):
                    raise ValueError("Published archive changed")
            if state["status"] == "complete":
                return state
            if not state["cleanup_started"]:
                _publish_archive(root, plan, state)
                state["cleanup_started"] = True
                _write(root, state)
            for entry in plan.entries:
                if not plan.policy.removes(entry.action):
                    continue
                path = _path(root, entry.path)
                if path.exists() or path.is_symlink():
                    if _identity(path) != entry.identity:
                        raise ValueError(f"Artifact changed: {entry.path}")
                    path.unlink()
                    _sync_directory(path.parent)
                if entry.path not in state["removed"]:
                    state["removed"].append(entry.path)
                    _write(root, state)
            state["status"] = "complete"
            state.pop("error", None)
        except BaseException as exc:
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if not completed or state["status"] != "complete":
                state["duration_sec"] += time.monotonic() - started
                _write(root, state)
        return state


def retain_artifacts(
    root: Path,
    policy: RetentionPolicy,
    *,
    roots: Sequence[str],
    protected: Sequence[str] = (),
) -> dict:
    """Plan/apply, or resume a previously recorded identical policy."""
    if root.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    root = root.resolve(strict=True)
    manifest = root / MANIFEST
    if manifest.is_symlink():
        raise ValueError("Retention manifest must not be a symlink")
    if manifest.exists():
        plan = RetentionPlan.model_validate(
            json.loads(manifest.read_text())["plan"]
        )
        if (
            plan.root != str(root)
            or plan.policy != policy
            or plan.roots != sorted(set(roots))
            or plan.protected != sorted(set(protected))
        ):
            raise ValueError("Existing retention policy/inventory differs")
    else:
        plan = plan_retention(root, policy, roots=roots, protected=protected)
    return apply_retention(plan)


def restore_artifacts(manifest: Path, destination: Path) -> list[str]:
    """Restore archived files to a new directory; reject escaping links."""
    state = json.loads(manifest.read_text())
    plan = RetentionPlan.model_validate(state["plan"])
    info = state.get("archive")
    if info is None:
        raise ValueError("This retention manifest has no published archive")
    archive_path = _path(manifest.resolve().parent, info["path"])
    if archive_path.is_symlink() or _hash(archive_path) != info["sha256"]:
        raise ValueError("Archive checksum mismatch")
    entries = [e for e in plan.entries if e.action == "archive"]
    _verify_archive(archive_path, entries)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "restored"
        staged.mkdir()
        with tarfile.open(archive_path, "r:xz") as archive:
            # data_filter rejects escaping links, devices and unsafe names.
            archive.extractall(staged, filter="data")
        staged.rename(destination)
    return [e.path for e in plan.entries if e.action == "delete"]
