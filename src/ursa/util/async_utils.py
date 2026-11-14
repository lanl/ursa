"""Shared awaitable and task lifecycle helpers."""

import asyncio
from inspect import isawaitable
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Resolve an awaitable, preserving ordinary values and raised exceptions."""
    return await value if isawaitable(value) else value


async def cancel_tasks(*tasks: asyncio.Task[Any]) -> None:
    """Cancel a task snapshot and wait for cleanup, observing every failure."""
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
