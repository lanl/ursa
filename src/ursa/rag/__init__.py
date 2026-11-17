"""Persistent RAG collection helpers for URSA."""

from typing import TYPE_CHECKING

from .persistence import (
    URSA_CACHE_DIR,
    build_persistent_rag_agent,
    delete_rag_agent,
    ensure_rag_agent_dir,
    list_rag_agent_names,
    normalize_rag_tool_names,
    rag_agent_dir,
    rag_group_dir,
    regular_agent_group_dir,
    require_rag_agent_dir,
    resolve_ingest_source,
    save_rag_agent,
    show_rag_agent,
    sync_rag_group_from_agent_group,
    validate_rag_agent_name,
    validate_rag_group_name,
)

if TYPE_CHECKING:
    from .tools import build_rag_tool, build_rag_tools, rag_tool_name


def __getattr__(name: str):
    if name in {"build_rag_tool", "build_rag_tools", "rag_tool_name"}:
        from . import tools

        value = getattr(tools, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "URSA_CACHE_DIR",
    "build_persistent_rag_agent",
    "build_rag_tool",
    "build_rag_tools",
    "resolve_ingest_source",
    "delete_rag_agent",
    "ensure_rag_agent_dir",
    "list_rag_agent_names",
    "normalize_rag_tool_names",
    "rag_agent_dir",
    "rag_group_dir",
    "regular_agent_group_dir",
    "rag_tool_name",
    "require_rag_agent_dir",
    "save_rag_agent",
    "show_rag_agent",
    "sync_rag_group_from_agent_group",
    "validate_rag_agent_name",
    "validate_rag_group_name",
]
