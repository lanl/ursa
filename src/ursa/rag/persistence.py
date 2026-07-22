"""Persistence helpers for named URSA RAG collections."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings

from ursa.security import GROUP_CONFIG_FILENAME, URSA_CACHE_DIR

_RAG_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def normalize_rag_tool_names(names: str | Sequence[str] | None) -> list[str]:
    """Normalize CLI/Python RAG tool declarations into validated names."""
    if names is None:
        return []
    if isinstance(names, str):
        raw = names.split(",")
    else:
        raw = []
        for item in names:
            raw.extend(str(item).split(","))
    return [validate_rag_agent_name(item) for item in raw if item.strip()]


def validate_rag_agent_name(name: str) -> str:
    """Validate a RAG collection name using the same policy as named agents."""
    if not name or not name.strip():
        raise ValueError("RAG agent name must not be empty")
    name = name.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("RAG agent name must be a simple directory name")
    if not _RAG_AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            "RAG agent name may only contain letters, numbers, dot, underscore, and hyphen"
        )
    return name


def validate_rag_group_name(group_name: str | None) -> str:
    group = (group_name or "default").strip()
    if not group:
        raise ValueError("RAG group name must not be empty")
    if Path(group).name != group or group in {".", ".."}:
        raise ValueError("RAG group name must be a simple directory name")
    return group


def rag_group_dir(group_name: str = "default") -> Path:
    return URSA_CACHE_DIR / validate_rag_group_name(group_name) / "rag"


def _missing_group_error(group_name: str) -> ValueError:
    return ValueError(
        (
            f"Group '{group_name}' does not exist. "
            f"Please use `ursa create-group {group_name} <group_config_file>` to create"
        )
    )


def regular_agent_group_dir(group_name: str = "default") -> Path:
    return URSA_CACHE_DIR / validate_rag_group_name(group_name)


def sync_rag_group_from_agent_group(group_name: str = "default") -> Path:
    """Ensure the group's RAG directory exists.

    In the hierarchical cache layout, a group has one shared root at
    ``~/.cache/ursa/<group>``. Named agents live below ``agents/`` and RAG
    collections live below ``rag/``. Non-default RAG groups are only valid if the
    group root exists and contains the shared ``group.yaml`` policy file.
    """
    group = validate_rag_group_name(group_name)
    group_dir = regular_agent_group_dir(group)
    target_dir = rag_group_dir(group)

    if group == "default":
        group_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    if not group_dir.exists() or not group_dir.is_dir():
        raise _missing_group_error(group)

    source_config = group_dir / GROUP_CONFIG_FILENAME
    if not source_config.exists() or not source_config.is_file():
        raise FileNotFoundError(
            f"Group '{group}' is missing required config file '{GROUP_CONFIG_FILENAME}'."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def rag_agent_dir(group_name: str, rag_agent_name: str) -> Path:
    return rag_group_dir(group_name) / validate_rag_agent_name(rag_agent_name)


def ensure_rag_agent_dir(group_name: str, rag_agent_name: str) -> Path:
    sync_rag_group_from_agent_group(group_name)
    path = rag_agent_dir(group_name, rag_agent_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "database").mkdir(parents=True, exist_ok=True)
    (path / "summaries").mkdir(parents=True, exist_ok=True)
    (path / "vectorstore").mkdir(parents=True, exist_ok=True)
    return path


def require_rag_agent_dir(group_name: str, rag_agent_name: str) -> Path:
    path = rag_agent_dir(group_name, rag_agent_name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"RAG agent does not exist: {rag_agent_name} in group {group_name}"
        )
    return path


def list_rag_agent_names(group_name: str = "default") -> list[str]:
    group_dir = rag_group_dir(group_name)
    if not group_dir.exists():
        return []
    return sorted(p.name for p in group_dir.iterdir() if p.is_dir())


def resolve_ingest_source(source: str | Path) -> Path:
    """Resolve and validate a RAG ingest source without copying it.

    Persistent RAG collections store embeddings/summaries under URSA's cache root,
    but raw source documents remain at their original file-system location.
    """
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"RAG ingest source not found: {src}")
    if not (src.is_file() or src.is_dir()):
        raise ValueError(
            f"RAG ingest source must be a file or directory: {src}"
        )
    return src


def build_persistent_rag_agent(
    *,
    name: str,
    group: str = "default",
    llm: BaseChatModel,
    embedding: Embeddings | None = None,
    create: bool = False,
    return_k: int = 10,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    thread_id: str | None = None,
    database_path: str | Path | None = None,
    **kwargs: Any,
):
    """Instantiate a RAGAgent backed by ~/.cache/ursa/<group>/rag/<name>.

    Args:
        database_path: Optional source file/directory to scan for ingestion.
            If omitted, the collection's empty ``database`` directory is used for
            backwards compatibility. Passing an external path avoids copying raw
            documents into URSA's persistent cache.
    """
    from ursa.agents import RAGAgent

    root = (
        ensure_rag_agent_dir(group, name)
        if create
        else require_rag_agent_dir(group, name)
    )
    source_path = (
        resolve_ingest_source(database_path)
        if database_path is not None
        else root / "database"
    )
    return RAGAgent(
        llm=llm,
        embedding=embedding,
        workspace=root,
        database_path=str(source_path),
        summaries_path="summaries",
        vectorstore_path="vectorstore",
        group=group,
        return_k=return_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        thread_id=thread_id or "ursa",
        **kwargs,
    )


def show_rag_agent(name: str, group_name: str = "default") -> None:
    path = require_rag_agent_dir(group_name, name)
    print(f"name: {path.name}")  # noqa: T201
    print(f"group: {validate_rag_group_name(group_name)}")  # noqa: T201
    print(f"path: {path}")  # noqa: T201
    for subdir in ("database", "summaries", "vectorstore"):
        p = path / subdir
        if p.exists():
            entries = list(p.rglob("*")) if p.is_dir() else []
            files = sum(1 for item in entries if item.is_file())
            print(f"{subdir}: {p} ({files} files)")  # noqa: T201


def delete_rag_agent(name: str, group_name: str = "default") -> None:
    path = require_rag_agent_dir(group_name, name)
    shutil.rmtree(path)
    print(f"Deleted RAG agent '{name}' from group '{group_name}'")  # noqa: T201
    print(f"Path: {path}")  # noqa: T201


def save_rag_agent(name: str, group_name: str = "default") -> None:
    src = require_rag_agent_dir(group_name, name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{validate_rag_agent_name(name)}.{timestamp}"
    dst = src.parent / checkpoint_name
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(src, dst)
    print(f"Saved RAG agent checkpoint: {checkpoint_name}")  # noqa: T201
    print(f"Source: {src}")  # noqa: T201
    print(f"Checkpoint: {dst}")  # noqa: T201
