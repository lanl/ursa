from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactEntry:
    rel_path: str
    kind: str
    mime: str
    size_bytes: int
    mtime: float
    sha256: str | None = None


def _guess_kind(rel_path: str, mime: str) -> str:
    lp = rel_path.lower()
    if mime.startswith("image/"):
        return "image"
    if mime in ("text/plain", "text/markdown") or lp.endswith((
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
    )):
        return "text"
    if lp.endswith((".pdf",)):
        return "pdf"
    return "file"


def _sha256(path: Path, *, max_bytes: int) -> str | None:
    size = path.stat().st_size
    if size > max_bytes:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_artifacts(
    base_dir: Path,
    *,
    exclude_dirs: set[str] | None = None,
    max_sha256_bytes: int = 50 * 1024 * 1024,
) -> list[dict]:
    """Scan base_dir recursively and return manifest entries.

    By default, this is used to populate the dashboard's "Workspace / Artifacts"
    panel. Many current URSA agents write outputs into their workspace root
    (not necessarily an `artifacts/` subdir), so this scan can include the whole
    run directory.

    `exclude_dirs` should contain directory names (not paths) to skip.
    """
    if not base_dir.exists():
        return []

    exclude_dirs = exclude_dirs or set()

    entries: list[ArtifactEntry] = []
    for root, dirs, files in os.walk(base_dir):
        # prune excluded dirs
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for fn in files:
            p = Path(root) / fn
            if not p.is_file():
                continue
            rel = str(p.relative_to(base_dir)).replace("\\", "/")
            mime, _ = mimetypes.guess_type(str(p))
            mime = mime or "application/octet-stream"
            st = p.stat()
            sha = _sha256(p, max_bytes=max_sha256_bytes)
            kind = _guess_kind(rel, mime)
            entries.append(
                ArtifactEntry(
                    rel_path=rel,
                    kind=kind,
                    mime=mime,
                    size_bytes=st.st_size,
                    mtime=st.st_mtime,
                    sha256=sha,
                )
            )

    entries.sort(key=lambda e: e.rel_path)
    return [e.__dict__ for e in entries]
