import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import httpx
import yaml
from jsonargparse import Namespace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ursa.util.mcp import ServerParameters

LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical"
]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    """Model provider and model name.
    Use the format <provider>:<model-name>
    """
    base_url: str | None = None
    """Base URL for model API access"""

    max_completion_tokens: int | None = None
    """Maximum tokens for LLM to output"""

    ssl_verify: bool = True
    """Flag for verifying SSL certs. during API access"""

    @property
    def kwargs(self) -> dict:
        """Return a dict suitable for init_chat_model/init_embedding_model
        Removes parameters set to `None`
        """
        kwargs = {k: v for k, v in self.model_dump().items() if v is not None}
        if kwargs.pop("ssl_verify", None) is False:
            kwargs["http_client"] = httpx.Client(verify=False)
        return kwargs


class UrsaConfig(BaseModel):
    _temp_workspace: TemporaryDirectory | None = PrivateAttr(default=None)

    workspace: Path = Field(
        default_factory=lambda: Path("ursa_workspace"),
    )
    """Directory to store URSA's output."""

    llm_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="openai:gpt-5.2",
            max_completion_tokens=5000,
        )
    )
    """Default LLM"""

    emb_model: ModelConfig | None = None
    """Default Embedding model"""

    mcp_servers: dict[str, ServerParameters] = Field(default_factory=dict)
    """MCP Servers to connect to Ursa."""

    def model_post_init(self, __context):
        """Handle temporary workspace creation post validation."""
        if str(self.workspace) == "tmp" and not self.workspace.exists():
            temp_workspace = TemporaryDirectory(prefix="ursa")
            self.workspace = Path(temp_workspace.name)
            self._temp_workspace = temp_workspace

    @classmethod
    def from_namespace(cls, cfg: Namespace):
        """Instantiate from a jsonargparse namespace."""
        return cls.model_validate(cfg.as_dict())

    @classmethod
    def from_file(cls, path: Path):
        loader = (
            yaml.safe_load if path.suffix in [".yaml", ".yml"] else json.load
        )
        with open(path, "r") as fid:
            data = loader(fid)

        return cls.model_validate(data)


def dict_diff(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return the subset of candidate entries that differ from the reference."""
    missing = object()
    diff: dict[str, Any] = {}
    for key, value in candidate.items():
        ref_value = reference.get(key, missing)
        if isinstance(value, dict) and isinstance(ref_value, dict):
            nested = dict_diff(ref_value, value)
            if nested:
                diff[key] = nested
        elif isinstance(value, list) and isinstance(ref_value, list):
            if value != ref_value:
                diff[key] = value
        elif isinstance(value, tuple) and isinstance(ref_value, tuple):
            if value != ref_value:
                diff[key] = value
        elif ref_value is missing or value != ref_value:
            diff[key] = value
    return diff


def deep_merge_dicts(
    base: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge updates into base without mutating inputs."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)  # type: ignore[index]
        else:
            merged[key] = value
    return merged


@dataclass
class MCPServerConfig:
    """MCP Server Options"""

    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "localhost"
    """Host to bind for network transports (ignored for stdio)"""
    port: int = 8000
    """Port to bind for network transports (ignored for stdio)"""
    log_level: LoggingLevel = "info"
