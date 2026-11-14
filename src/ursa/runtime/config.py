"""Execution placement, independent of model and host configuration."""

import shlex
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class RuntimeConfig(BaseModel):
    """Processes isolate each agent; pools share processes using agent threads."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    backend: Literal["process", "thread", "pool"] = "process"
    min_workers: int = Field(default=1, ge=1, strict=True)
    max_workers: int = Field(default=4, ge=1, strict=True)
    agents_per_worker: int = Field(default=8, ge=1, strict=True)
    rebalance_interval: float = Field(default=10, gt=0, allow_inf_nan=False)
    idle_timeout: float = Field(default=60, ge=0, allow_inf_nan=False)
    launch_command: list[str] | None = None
    bind_host: str = "127.0.0.1"
    advertise_host: str | None = None
    rendezvous_dir: Path | None = None
    tls_certificate: Path | None = None
    tls_key: Path | None = None

    @field_validator("launch_command", mode="before")
    @classmethod
    def _command(cls, value):
        if isinstance(value, str):
            value = shlex.split(value)
        if value is not None and (not value or any(not part for part in value)):
            raise ValueError("launch_command must contain a command")
        return value

    @model_validator(mode="after")
    def _limits(self) -> Self:
        if self.min_workers > self.max_workers:
            raise ValueError("min_workers must not exceed max_workers")
        if self.launch_command and self.backend != "pool":
            raise ValueError("launch_command requires the pool backend")
        if bool(self.tls_certificate) != bool(self.tls_key):
            raise ValueError(
                "tls_certificate and tls_key must be supplied together"
            )
        if (
            self.bind_host not in {"127.0.0.1", "localhost", "::1"}
            and not self.tls_certificate
        ):
            raise ValueError(
                "Non-loopback worker connections require a TLS certificate and key"
            )
        return self

    def model_merge(self, other: Self | dict) -> Self:
        updates = (
            other.model_dump(exclude_unset=True)
            if isinstance(other, type(self))
            else other
        )
        return type(self).model_validate(self.model_dump() | updates)
