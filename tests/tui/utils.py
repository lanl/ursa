"""Semantic synchronization primitives for Textual tests."""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar, cast

from textual.app import App
from textual.dom import DOMNode
from textual.pilot import Pilot
from textual.screen import Screen

UI_TIMEOUT = 15.0

ScreenType = TypeVar("ScreenType", bound=Screen)


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


async def await_event(
    pilot: Pilot[Any],
    event: Any,
    *,
    description: str = "event",
    timeout: float = UI_TIMEOUT,
) -> None:
    """Wait for a threading or asyncio event while pumping Textual."""
    await eventually(
        pilot,
        event.is_set,
        description=description,
        timeout=timeout,
    )


async def await_screen(
    pilot: Pilot[Any],
    screen_type: type[ScreenType],
    *,
    timeout: float = UI_TIMEOUT,
) -> ScreenType:
    """Wait for a particular screen type to become current."""
    await eventually(
        pilot,
        lambda: isinstance(pilot.app.screen, screen_type),
        description=f"{screen_type.__name__} to become current",
        timeout=timeout,
    )
    return cast(ScreenType, pilot.app.screen)


async def await_dom(
    pilot: Pilot[Any],
    root: DOMNode,
    selector: str,
    *,
    count: int = 1,
    timeout: float = UI_TIMEOUT,
) -> Any:
    """Wait for an exact number of nodes matching a selector."""
    await eventually(
        pilot,
        lambda: len(root.query(selector)) == count,
        description=f"{count} node(s) matching {selector!r}",
        timeout=timeout,
    )
    return root.query(selector)


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


async def await_animations(
    pilot: Pilot[Any],
    *,
    timeout: float = UI_TIMEOUT,
) -> None:
    """Wait for current and scheduled Textual animations to complete."""
    try:
        async with asyncio.timeout(timeout):
            await pilot.wait_for_scheduled_animations()
    except TimeoutError as exc:
        raise AssertionError(
            f"Timed out after {timeout:g}s waiting for animations"
        ) from exc
