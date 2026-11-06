"""Helpers for APIs that accept synchronous or asynchronous implementations."""

from inspect import isawaitable
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Resolve an awaitable, preserving ordinary values and raised exceptions."""
    return await value if isawaitable(value) else value
