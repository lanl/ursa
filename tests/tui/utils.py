"""Semantic synchronization primitives for Textual tests."""

import asyncio
from collections.abc import Callable
from typing import Any

from textual.app import App
from textual.pilot import Pilot

UI_TIMEOUT = 15.0


async def eventually(
    pilot: Pilot[Any],
    condition: Callable[[], bool],
    *,
    description: str = "UI condition",
    timeout: float = UI_TIMEOUT,
) -> bool:
    """Pump Textual until a semantic condition is true or fail clearly."""
    try:
        async with asyncio.timeout(timeout):
            while not condition():
                await pilot.pause()
    except TimeoutError as exc:
        raise AssertionError(
            f"Timed out after {timeout:g}s waiting for {description}"
        ) from exc
    return True


async def await_workers(
    app: App[Any],
    pilot: Pilot[Any],
    *,
    timeout: float = UI_TIMEOUT,
) -> None:
    """Wait for workers, then drain the UI messages they produced."""
    try:
        async with asyncio.timeout(timeout):
            await app.workers.wait_for_complete()
            await pilot.pause()
    except TimeoutError as exc:
        raise AssertionError(
            f"Timed out after {timeout:g}s waiting for workers"
        ) from exc


async def await_event(pilot: Pilot[Any], event: Any) -> None:
    """Pump Textual while waiting for a threading or asyncio event."""
    await eventually(
        pilot,
        event.is_set,
        description="event to be set",
    )


async def await_screen(pilot: Pilot[Any], screen_type: type[Any]) -> Any:
    """Pump Textual until the requested screen becomes active."""
    await eventually(
        pilot,
        lambda: isinstance(pilot.app.screen, screen_type),
        description=f"{screen_type.__name__} screen",
    )
    return pilot.app.screen
