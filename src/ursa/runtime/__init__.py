"""Shared process-isolated runtime for CLI and dashboard hosts."""

from typing import TYPE_CHECKING

from .config import RuntimeConfig
from .manager import (
    AgentContext,
    AgentInvocationError,
    AgentManager,
    AgentMessage,
    AgentNotFoundError,
    AgentProcess,
    AgentProcessExitedError,
    AgentRuntimeError,
    AgentSnapshot,
    AgentSpec,
    AgentStatus,
    SyncAgentManager,
)

if TYPE_CHECKING:
    from .agent_runtime import (
        AgentRuntime,
        RuntimeAgent,
        RuntimeHierarchyStatus,
    )

__all__ = [
    "AgentContext",
    "AgentRuntime",
    "AgentInvocationError",
    "AgentManager",
    "AgentMessage",
    "AgentNotFoundError",
    "AgentProcess",
    "AgentProcessExitedError",
    "AgentRuntimeError",
    "AgentSnapshot",
    "AgentSpec",
    "AgentStatus",
    "RuntimeAgent",
    "RuntimeHierarchyStatus",
    "RuntimeConfig",
    "SyncAgentManager",
]


def __getattr__(name):
    # Configuration and worker entry points do not need the CLI/agent catalog.
    if name in {"AgentRuntime", "RuntimeAgent", "RuntimeHierarchyStatus"}:
        from . import agent_runtime

        return getattr(agent_runtime, name)
    raise AttributeError(name)
