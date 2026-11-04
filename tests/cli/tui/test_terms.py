# ruff: noqa: TID251

"""Focused integration coverage for the live terminal browser."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rich.cells import cell_len
from rich.text import Text
from textual.widgets import Static, Tab, TabbedContent

from tests.cli._app_fakes import FakeHITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.terminal_view import TerminalView, snapshot_text
from ursa.cli.tui.widgets import PromptArea, TermsScreen
from ursa.tools.terminal import (
    TerminalRenderSnapshot,
    TerminalSpan,
    TerminalStyle,
    TermInfo,
)


def info(term_id: str, order: int, *, screen: bool = False) -> TermInfo:
    return TermInfo(
        term_id=term_id,
        backend="GhosttyTerm" if screen else "ProcessTerm",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        creation_order=order,
        capabilities=frozenset({"read"}),
        supports_screen=screen,
    )


def snapshot(term_id: str, text: str, *, screen=False, style=TerminalStyle()):
    return TerminalRenderSnapshot(
        term_id,
        (TerminalSpan(text, style),),
        rows=7 if screen else None,
        cols=31 if screen else None,
        screen=screen,
    )


class Manager:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []

    async def render_snapshot(self, term_id):
        self.calls.append(term_id)
        if term_id not in self.snapshots:
            raise KeyError(term_id)
        return self.snapshots[term_id]


async def test_terms_screen_handles_empty_and_ordered_live_sessions(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(90, 28)) as pilot:
        app.push_screen(TermsScreen([], manager=Manager({})))
        await pilot.pause()
        assert (
            str(app.screen.query_one(".terminals-empty", Static).content)
            == "No managed terminal sessions."
        )
        await app.pop_screen()

        terminals = [info("newest00", 30), info("oldest00", 10)]
        manager = Manager({
            item.term_id: snapshot(item.term_id, item.term_id)
            for item in terminals
        })
        app.push_screen(TermsScreen(terminals, manager=manager))
        await pilot.pause()
        assert [str(tab.label) for tab in app.screen.query(Tab)] == [
            "oldest00",
            "newest00",
        ]
        assert (
            app.screen.query_one("#terminal-tabs", TabbedContent).active
            == "terminal-tab-newest00"
        )


async def test_terminal_views_render_process_and_ghostty_and_refresh(tmp_path):
    style = TerminalStyle(
        foreground=(12, 34, 56), background=(65, 43, 21), bold=True
    )
    manager = Manager({
        "process0": snapshot("process0", "first"),
        "ghost000": snapshot("ghost000", "styled", screen=True, style=style),
    })
    app = UrsaTextualApp(FakeHITL(tmp_path))
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            TermsScreen(
                [info("process0", 1), info("ghost000", 2, screen=True)],
                manager=manager,
            )
        )
        await pilot.pause()
        await pilot.pause()
        view = app.screen.query_one("#terminal-view-ghost000", TerminalView)
        assert isinstance(view.content, Text) and view.content.no_wrap
        assert view.content.get_style_at_offset(app.console, 0).bold

        manager.snapshots["ghost000"] = snapshot(
            "ghost000", "second", screen=True, style=style
        )
        await asyncio.sleep(view.refresh_interval + 0.05)
        await pilot.pause()
        assert "second" in str(view.content)
        await pilot.press("escape")
        assert app.focused is app.query_one(PromptArea)


def test_snapshot_text_keeps_terminal_style_flags_and_grid_width():
    rendered = snapshot_text(
        TerminalRenderSnapshot(
            "ghost000",
            (
                TerminalSpan(
                    "👩\u200d💻👍🏽",
                    TerminalStyle(strike=True, overline=True),
                    cells=8,
                ),
            ),
            rows=1,
            cols=8,
            screen=True,
        )
    )
    assert cell_len(rendered.plain) == 8
    assert rendered.spans[0].style.strike and rendered.spans[0].style.overline
