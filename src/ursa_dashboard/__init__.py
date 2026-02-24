"""URSA Dashboard package.

This package provides web-dashboard plumbing around URSA agents:

- Agent registry with parameter schemas + capability flags.
- Adapters to execute heterogeneous agents behind a common interface.

Other dashboard subsystems (run orchestration, SSE streaming, artifact serving)
should build on the contracts defined in prior steps.
"""

from .registry import REGISTRY

__all__ = ["REGISTRY"]
