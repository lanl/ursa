# ruff: noqa: TID251

"""Rich rendering and MIME-typed artifacts for URSA events."""

from __future__ import annotations

import json
import logging
import mimetypes
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Mapping, NotRequired, TypedDict

from pygments.util import ClassNotFound
from rich.console import Console, RenderableType
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

FILE_REFERENCE_MIME_TYPE = "application/vnd.ursa.file-reference"
MAX_INLINE_FILE_BYTES = 1_000_000


class EventArtifact(TypedDict):
    """Renderer-neutral content attached to an URSA progress event."""

    content: Any
    mime_type: str
    metadata: NotRequired[dict[str, str | float | int]]


ArtifactMetadata = Mapping[str, str | float | int]
ArtifactRenderer = Callable[[Any, ArtifactMetadata], RenderableType]


def event_artifact(
    content: Any,
    mime_type: str,
    *,
    metadata: Mapping[str, str | float | int] | None = None,
) -> EventArtifact:
    """Build an artifact payload for a progress event."""
    artifact: EventArtifact = {"content": content, "mime_type": mime_type}
    if metadata:
        artifact["metadata"] = dict(metadata)
    return artifact


def file_content_mime_type(path: str | Path) -> str:
    """Return the MIME type of a referenced file without reading its content."""
    suffix = Path(path).suffix.lower()
    overrides = {
        ".diff": "text/x-diff",
        ".json": "application/json",
        ".md": "text/markdown",
        ".patch": "text/x-diff",
        ".py": "text/x-python",
    }
    if suffix in overrides:
        return overrides[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def file_artifact(path: str | Path, *, title: str = "File") -> EventArtifact:
    """Build a lightweight artifact that references a file by path."""
    path = str(path)
    return event_artifact(
        path,
        FILE_REFERENCE_MIME_TYPE,
        metadata={
            "title": title,
            "path": path,
            "content_mime_type": file_content_mime_type(path),
        },
    )


def event_artifacts(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return singular and plural artifacts from an event payload."""
    artifacts: list[Mapping[str, Any]] = []
    artifact = payload.get("artifact")
    if isinstance(artifact, Mapping):
        artifacts.append(artifact)
    plural = payload.get("artifacts")
    if isinstance(plural, list):
        artifacts.extend(item for item in plural if isinstance(item, Mapping))
    return artifacts


def _render_diff(content: Any, metadata: ArtifactMetadata) -> RenderableType:
    return Syntax(str(content), "diff", word_wrap=True)


def _render_json(content: Any, metadata: ArtifactMetadata) -> RenderableType:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return Syntax(content, "json", word_wrap=True)
    return JSON.from_data(content)


def _render_markdown(
    content: Any, metadata: ArtifactMetadata
) -> RenderableType:
    return Markdown(str(content))


def _render_file_reference(
    content: Any, metadata: ArtifactMetadata
) -> RenderableType:
    path = Path(str(content))
    mime_type = str(
        metadata.get("content_mime_type") or file_content_mime_type(path)
    )
    if (
        not path.is_file()
        or path.stat().st_size > MAX_INLINE_FILE_BYTES
        or not (
            mime_type.startswith("text/")
            or mime_type in {"application/json", "application/xml"}
        )
    ):
        return Text(str(path), style="cyan")

    try:
        file_content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return Text(str(path), style="cyan")

    lexer = {
        "application/json": "json",
        "application/xml": "xml",
        "text/markdown": "markdown",
        "text/x-diff": "diff",
        "text/x-python": "python",
    }.get(mime_type)
    if lexer is None:
        try:
            lexer = str(Syntax.guess_lexer(path.name, file_content))
        except (ClassNotFound, TypeError, ValueError):
            lexer = "text"
    return Syntax(file_content, lexer, line_numbers=True, word_wrap=False)


ARTIFACT_RENDERERS: dict[str, ArtifactRenderer] = {
    FILE_REFERENCE_MIME_TYPE: _render_file_reference,
    "application/json": _render_json,
    "text/markdown": _render_markdown,
    "text/x-diff": _render_diff,
}


def register_artifact_renderer(
    mime_type: str, renderer: ArtifactRenderer
) -> None:
    """Register or replace the Rich renderer for an artifact MIME type."""
    ARTIFACT_RENDERERS[mime_type] = renderer


def render_event_artifact(artifact: Mapping[str, Any]) -> RenderableType:
    """Render an artifact according to its MIME type using Rich primitives."""
    raw_content = artifact.get("content", "")
    mime_type = str(artifact.get("mime_type") or "text/plain").split(";", 1)[0]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    path = str(metadata.get("path") or "")
    title = str(metadata.get("title") or path or mime_type)

    if renderer := ARTIFACT_RENDERERS.get(mime_type):
        body = renderer(raw_content, metadata)
    else:
        content = str(raw_content)
        lexer = {"text/x-diff": "diff", "text/x-python": "python"}.get(
            mime_type
        )
        if lexer is None:
            try:
                lexer = str(Syntax.guess_lexer(path or "artifact.txt", content))
            except (ClassNotFound, TypeError, ValueError):
                lexer = "text"
        body = Syntax(content, lexer, line_numbers=False, word_wrap=True)
    return Panel(body, title=title, border_style="green", expand=False)


def render_event_artifacts(
    artifacts: list[Mapping[str, Any]],
) -> RenderableType:
    """Render one artifact or a fixed side-by-side row of artifacts."""
    rendered = [render_event_artifact(artifact) for artifact in artifacts]
    if len(rendered) == 1:
        return rendered[0]

    artifact_table = Table.grid(expand=True, padding=(0, 1))
    for _ in rendered:
        artifact_table.add_column(ratio=1, overflow="fold")
    artifact_table.add_row(*rendered)
    return artifact_table


class EventConsoleFormatter(logging.Formatter):
    """Render URSA progress events as compact console messages."""

    DETAIL_KEYS = (
        "path",
        "filename",
        "query",
        "output_path",
        "returncode",
        "stdout_chars",
        "stderr_chars",
        "result_chars",
        "error",
    )

    def __init__(
        self,
        *,
        force_terminal: bool | None = None,
        render_artifacts: bool = True,
    ) -> None:
        super().__init__()
        self.force_terminal = force_terminal
        self.render_artifacts = render_artifacts

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "ursa_event_payload", None)
        if not isinstance(payload, dict):
            return super().format(record)

        source = (
            payload.get("agent")
            or payload.get("tool")
            or payload.get("name")
            or "ursa"
        )
        stage = payload.get("stage")
        phase = payload.get("phase")
        message = payload.get("message") or record.getMessage()
        label = str(source)
        if stage:
            label += f" {stage}"
        if phase:
            label += f"/{phase}"
        details = [
            f"{key}={payload[key]}"
            for key in self.DETAIL_KEYS
            if payload.get(key) not in (None, "")
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        summary = f"[ursa] {label}: {message}{suffix}"
        artifacts = event_artifacts(payload)
        if not self.render_artifacts or not artifacts:
            return summary

        output = StringIO()
        console = Console(
            file=output,
            color_system="auto",
            force_terminal=self.force_terminal,
        )
        console.print(render_event_artifacts(artifacts))
        return f"{summary}\n{output.getvalue().rstrip()}"
