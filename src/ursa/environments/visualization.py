from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic_ns
from typing import Any, Iterator, Mapping
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from ursa.security import group_root_dir, validate_group_name
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.workflows.base_workflow import InputLike

ENVIRONMENT_RUN_SCHEMA_VERSION = "ursa.environment_run.v1"
ENVIRONMENT_EVENT_SCHEMA_VERSION = "ursa.environment_event.v1"
DEFAULT_MAX_PAYLOAD_CHARS = 30_000
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_event_id() -> str:
    return uuid4().hex


def new_run_id() -> str:
    return uuid4().hex


def environment_runs_dir(group: str | None = None) -> Path:
    """Return the directory that stores recorded environment visualization runs."""
    return group_root_dir(validate_group_name(group)) / "environment_runs"


def environment_run_dir(group: str | None, run_id: str) -> Path:
    return environment_runs_dir(group) / run_id


def _preview(value: Any, *, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [truncated {len(text) - limit} chars]"


def _truncate_text(value: str, *, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + f"… [truncated {len(value) - limit} chars]"


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(tag) for tag in value]
    return [str(value)]


def _make_json_safe(value: Any, *, max_chars: int) -> Any:
    """Convert arbitrary event payload values into JSON-safe values.

    LangChain callback payloads can contain Path objects, UUIDs, messages, or
    other custom values. The recorder should never fail a run because a progress
    event contains a non-JSON value, so unsupported objects are stringified.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _truncate_text(value, limit=max_chars)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): _make_json_safe(v, max_chars=max_chars)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [_make_json_safe(v, max_chars=max_chars) for v in value]
    return _truncate_text(str(value), limit=max_chars)


def _infer_event_type(data: Mapping[str, Any]) -> str:
    event_type = data.get("event_type")
    if event_type:
        return str(event_type)
    if data.get("environment"):
        stage = str(data.get("stage") or "environment")
        phase = str(data.get("phase") or "update")
        return f"{stage}_{phase}" if phase else stage
    if data.get("tool"):
        stage = str(data.get("stage") or "progress")
        return f"tool_{stage}"
    if data.get("agent"):
        stage = str(data.get("stage") or "progress")
        return f"agent_{stage}"
    return str(data.get("stage") or "progress")


def _tool_target_from_payload(data: Mapping[str, Any]) -> dict[str, Any] | None:
    tool = data.get("tool")
    if not tool:
        return None
    return {
        "id": data.get("tool_call_id") or tool,
        "name": tool,
        "kind": "tool",
        "path": [tool],
    }


def _source_from_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    source = data.get("source")
    if isinstance(source, Mapping):
        return dict(source)
    if data.get("environment"):
        return {
            "id": data.get("environment_id") or data.get("environment"),
            "name": data.get("environment"),
            "kind": "environment",
            "path": data.get("path") or [data.get("environment")],
        }
    if data.get("agent"):
        return {
            "id": data.get("agent_id") or data.get("agent"),
            "name": data.get("agent"),
            "kind": "agent",
            "path": data.get("agent_path")
            or data.get("environment_member_path")
            or [data.get("agent")],
        }
    if data.get("tool"):
        return {
            "id": data.get("tool_call_id") or data.get("tool"),
            "name": data.get("tool"),
            "kind": "tool",
            "path": data.get("path") or [data.get("tool")],
        }
    return {"id": "ursa", "name": "ursa", "kind": "system", "path": []}


@dataclass(slots=True)
class EnvironmentRunPaths:
    run_dir: Path
    manifest_path: Path
    events_path: Path
    artifacts_dir: Path
    logs_dir: Path


def get_environment_run_paths(
    group: str | None, run_id: str
) -> EnvironmentRunPaths:
    run_dir = environment_run_dir(group, run_id)
    return EnvironmentRunPaths(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        events_path=run_dir / "events.jsonl",
        artifacts_dir=run_dir / "artifacts",
        logs_dir=run_dir / "logs",
    )


def ensure_environment_run_dirs(paths: EnvironmentRunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def read_environment_run_manifest(
    group: str | None, run_id: str
) -> dict[str, Any]:
    path = get_environment_run_paths(group, run_id).manifest_path
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_environment_run_manifests(
    group: str | None = None,
) -> list[dict[str, Any]]:
    root = environment_runs_dir(group)
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        manifest = run_dir / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            manifests.append(data)
    return manifests


def read_environment_run_events(
    group: str | None,
    run_id: str,
    *,
    after_seq: int = 0,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    path = get_environment_run_paths(group, run_id).events_path
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(events) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            try:
                seq = int(event.get("seq", 0))
            except Exception:
                seq = 0
            if seq > after_seq:
                events.append(event)
    return events


class EnvironmentEventRecorder(BaseCallbackHandler):
    """Record URSA structured progress events to a replayable JSONL file."""

    def __init__(
        self,
        *,
        run_id: str,
        group: str = "default",
        environment_name: str,
        environment_type: str,
        run_dir: Path | None = None,
        event_name: str = DEFAULT_EVENT_NAME,
        max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
    ) -> None:
        self.run_id = run_id
        self.group = validate_group_name(group)
        self.environment_name = environment_name
        self.environment_type = environment_type
        self.event_name = event_name
        self.max_payload_chars = max_payload_chars
        if run_dir is None:
            self.paths = get_environment_run_paths(self.group, run_id)
        else:
            self.paths = EnvironmentRunPaths(
                run_dir=run_dir,
                manifest_path=run_dir / "manifest.json",
                events_path=run_dir / "events.jsonl",
                artifacts_dir=run_dir / "artifacts",
                logs_dir=run_dir / "logs",
            )
        ensure_environment_run_dirs(self.paths)
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def config(self) -> RunnableConfig:
        return {
            "callbacks": [self],
            "metadata": {
                "environment_run_id": self.run_id,
                "environment_name": self.environment_name,
                "environment_type": self.environment_type,
                "group": self.group,
            },
            "tags": ["environment_run", self.environment_name],
        }

    def write_manifest(
        self,
        *,
        status: str,
        task: Any | None = None,
        error: str | None = None,
    ) -> None:
        existing: dict[str, Any] = {}
        if self.paths.manifest_path.exists():
            try:
                existing = json.loads(
                    self.paths.manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing = {}
        now = utc_now_rfc3339()
        manifest = {
            **existing,
            "schema_version": ENVIRONMENT_RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "group": self.group,
            "environment_name": self.environment_name,
            "environment_type": self.environment_type,
            "status": status,
            "updated_at": now,
            "events_path": "events.jsonl",
            "artifacts_path": "artifacts",
            "logs_path": "logs",
        }
        manifest.setdefault("created_at", now)
        if task is not None:
            manifest["task_preview"] = _preview(task)
        if error:
            manifest["error"] = error
        self.paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if name != self.event_name or not isinstance(data, Mapping):
            return
        event = self.normalize_event(data, tags=tags, metadata=metadata)
        self.append_event(event)

    def normalize_event(
        self,
        data: Mapping[str, Any],
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_payload_tags = data.get("tags")
        safe_data = _make_json_safe(data, max_chars=self.max_payload_chars)
        if not isinstance(safe_data, dict):
            safe_data = {"value": safe_data}
        event_type = _infer_event_type(safe_data)
        source = _source_from_payload(safe_data)
        target = safe_data.get("target")
        if isinstance(target, str):
            target = {"id": target, "name": target}
        elif isinstance(target, Mapping):
            target = dict(target)
        else:
            target = None
        if (
            target is None
            and safe_data.get("tool")
            and source.get("kind") != "tool"
        ):
            target = _tool_target_from_payload(safe_data)
        environment_name = str(
            safe_data.get("environment") or self.environment_name
        )
        return {
            "schema_version": ENVIRONMENT_EVENT_SCHEMA_VERSION,
            "event_id": new_event_id(),
            "seq": 0,
            "ts": utc_now_rfc3339(),
            "monotonic_timestamp_ns": safe_data.get(
                "monotonic_timestamp_ns", monotonic_ns()
            ),
            "run_id": self.run_id,
            "environment_id": safe_data.get("environment_id")
            or environment_name,
            "environment_name": environment_name,
            "environment_type": safe_data.get("environment_type")
            or self.environment_type,
            "event_type": event_type,
            "stage": safe_data.get("stage"),
            "phase": safe_data.get("phase"),
            "level": safe_data.get("level", "info"),
            "source": source,
            "target": target,
            "message": safe_data.get("message") or event_type,
            "payload": safe_data,
            "tags": _normalize_tags(tags) + _normalize_tags(raw_payload_tags),
            "metadata": _make_json_safe(
                metadata or {}, max_chars=self.max_payload_chars
            ),
        }

    def append_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = dict(event)
            record["seq"] = self._seq
            with self.paths.events_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                f.flush()
            return record


def visualization_config(
    recorder: EnvironmentEventRecorder,
    config: RunnableConfig | None = None,
) -> RunnableConfig:
    """Merge an optional runnable config with the recorder callback config."""
    if config is None:
        return recorder.config
    return merge_configs(recorder.config, config)


@contextmanager
def environment_run_recorder(
    environment: Any,
    *,
    task: Any | None = None,
    config: RunnableConfig | None = None,
    run_id: str | None = None,
    max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
) -> Iterator[tuple[EnvironmentEventRecorder, RunnableConfig]]:
    """Create a recorder and runnable config for an environment run."""
    recorder = EnvironmentEventRecorder(
        run_id=run_id or new_run_id(),
        group=getattr(environment, "group", "default"),
        environment_name=getattr(
            environment, "name", type(environment).__name__
        ),
        environment_type=type(environment).__name__,
        max_payload_chars=max_payload_chars,
    )
    recorder.write_manifest(status="running", task=task)
    try:
        yield recorder, visualization_config(recorder, config)
    except BaseException as exc:
        recorder.write_manifest(status="failed", task=task, error=str(exc))
        raise
    else:
        recorder.write_manifest(status="succeeded", task=task)


@contextmanager
def record_environment_run(
    environment: Any,
    *,
    task: Any | None = None,
    config: RunnableConfig | None = None,
    run_id: str | None = None,
    max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
) -> Iterator[tuple[EnvironmentEventRecorder, RunnableConfig]]:
    """Alias for ``environment_run_recorder`` for readable user code."""
    with environment_run_recorder(
        environment,
        task=task,
        config=config,
        run_id=run_id,
        max_payload_chars=max_payload_chars,
    ) as value:
        yield value


async def arun_with_visualization(
    environment: Any,
    inputs: InputLike,
    *,
    config: RunnableConfig | None = None,
    run_id: str | None = None,
    max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
    **kwargs: Any,
) -> Any:
    """Run an environment asynchronously while recording visualization events."""
    recorder = EnvironmentEventRecorder(
        run_id=run_id or new_run_id(),
        group=getattr(environment, "group", "default"),
        environment_name=getattr(
            environment, "name", type(environment).__name__
        ),
        environment_type=type(environment).__name__,
        max_payload_chars=max_payload_chars,
    )
    recorder.write_manifest(status="running", task=inputs)
    run_config = visualization_config(recorder, config)
    try:
        result = await environment.ainvoke(inputs, config=run_config, **kwargs)
    except BaseException as exc:
        recorder.write_manifest(status="failed", task=inputs, error=str(exc))
        raise
    recorder.write_manifest(status="succeeded", task=inputs)
    return result


def run_with_visualization(
    environment: Any,
    inputs: InputLike,
    *,
    config: RunnableConfig | None = None,
    run_id: str | None = None,
    max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
    **kwargs: Any,
) -> Any:
    """Run an environment synchronously while recording visualization events."""
    recorder = EnvironmentEventRecorder(
        run_id=run_id or new_run_id(),
        group=getattr(environment, "group", "default"),
        environment_name=getattr(
            environment, "name", type(environment).__name__
        ),
        environment_type=type(environment).__name__,
        max_payload_chars=max_payload_chars,
    )
    recorder.write_manifest(status="running", task=inputs)
    run_config = visualization_config(recorder, config)
    try:
        result = environment.invoke(inputs, config=run_config, **kwargs)
    except BaseException as exc:
        recorder.write_manifest(status="failed", task=inputs, error=str(exc))
        raise
    recorder.write_manifest(status="succeeded", task=inputs)
    return result


__all__ = [
    "DEFAULT_MAX_PAYLOAD_CHARS",
    "ENVIRONMENT_EVENT_SCHEMA_VERSION",
    "ENVIRONMENT_RUN_SCHEMA_VERSION",
    "EnvironmentEventRecorder",
    "EnvironmentRunPaths",
    "arun_with_visualization",
    "environment_run_dir",
    "environment_run_recorder",
    "environment_runs_dir",
    "get_environment_run_paths",
    "list_environment_run_manifests",
    "read_environment_run_events",
    "read_environment_run_manifest",
    "record_environment_run",
    "run_with_visualization",
    "visualization_config",
]
