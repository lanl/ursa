from tempfile import TemporaryDirectory
from typing import Literal
from dataclasses import dataclass, field, fields
from pathlib import Path

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical"
]


@dataclass
class ModelConfig:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    max_completion_tokens: int | None = None
    ssl_verify: bool = True


@dataclass
class UrsaConfig:
    workspace: Path = field(
        default_factory=lambda: Path("ursa_workspace"),
    )
    """Directory to store URSA's output."""

    llm_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model="openai:gpt-5",
            max_completion_tokens=5000,
        )
    )
    """Default LLM"""

    emb_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            model="openai:text-embedding-3-small",
        )
    )
    """Default Embedding model"""

    mcp_servers: dict[str, dict] = field(default_factory=dict)
    """MCP Servers to connect to Ursa."""

    def __post_init__(self):
        # If workspace is tmp, create a temp directory for this session only
        if str(self.workspace) == "tmp" and not self.workspace.exists():
            self.__temp_workspace = TemporaryDirectory(prefix="ursa")
            self.workspace = Path(self.__temp_workspace.name)

    @classmethod
    def from_namespace(cls, cfg):
        return cls(**{f.name: cfg[f.name] for f in fields(cls)})


@dataclass
class MCPServerConfig:
    """MCP Server Options"""

    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "localhost"
    """Host to bind for network transports (ignored for stdio)"""
    port: int = 8000
    """Port to bind for network transports (ignored for stdio)"""
    log_level: LoggingLevel = "info"
