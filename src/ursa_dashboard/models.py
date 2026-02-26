from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ParamSource(str, Enum):
    """Where a parameter is applied."""

    run_input = "run_input"  # goes into agent.invoke(...)
    agent_init = "agent_init"  # passed to the agent constructor
    llm = "llm"  # used to build the LLM object / client
    runner = "runner"  # affects dashboard runner behavior (timeouts, etc.)


JsonType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
]


class ParamConstraint(BaseModel):
    """JSON-schema-like constraints (subset)."""

    model_config = ConfigDict(populate_by_name=True)

    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, alias="minLength")
    max_length: int | None = Field(default=None, alias="maxLength")
    pattern: str | None = None
    enum: list[Any] | None = None


class AgentParam(BaseModel):
    """A UI-renderable parameter definition."""

    name: str = Field(..., description="Machine name (stable).")
    title: str = Field(..., description="Human-friendly label.")
    description: str = ""

    type: JsonType
    required: bool = False
    default: Any | None = None

    constraints: ParamConstraint | None = None

    advanced: bool = False
    hidden: bool = False

    # Mapping to runtime configuration
    source: ParamSource
    target: str = Field(
        ..., description="Dot-path target within the destination (source)"
    )


class AgentCapabilities(BaseModel):
    """Flags that the UI can use to enable/disable features."""

    supports_streaming: bool
    # True if the agent itself cooperates with cancellation.
    # (The runner can always force-kill; that's separate.)
    supports_cancellation: bool
    produces_artifacts: bool


class AgentSpec(BaseModel):
    agent_id: str
    display_name: str
    description: str

    capabilities: AgentCapabilities
    parameters: list[AgentParam]

    # Helpful for grouping and UI selection
    tags: list[str] = []


class AgentListResponse(BaseModel):
    agents: list[AgentSpec]
