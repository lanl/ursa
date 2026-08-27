# ruff: noqa: TID251

"""Terminal-browser tests for the Textual application."""

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


def terminal_info(
    term_id: str, order: int, *, screen: bool = False
) -> TermInfo:
    return TermInfo(
        term_id=term_id,
        backend="GhosttyTerm" if screen else "ProcessTerm",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        creation_order=order,
        capabilities=frozenset({"read"}),
        supports_screen=screen,
    )


def snapshot(
    term_id: str,
    text: str,
    *,
    screen: bool = False,
    rows: int | None = None,
    cols: int | None = None,
    style: TerminalStyle = TerminalStyle(),
) -> TerminalRenderSnapshot:
    return TerminalRenderSnapshot(
        term_id=term_id,
        spans=(TerminalSpan(text, style),),
        rows=rows,
        cols=cols,
        screen=screen,
    )


class SnapshotManager:
    def __init__(self, snapshots: dict[str, TerminalRenderSnapshot]) -> None:
        self.snapshots = snapshots
        self.calls: list[str] = []

    async def render_snapshot(self, term_id: str) -> TerminalRenderSnapshot:
        self.calls.append(term_id)
        value = self.snapshots.get(term_id)
        if value is None:
            raise KeyError(term_id)
        return value


async def test_terms_screen_explains_when_there_are_no_terminals(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(TermsScreen([], manager=SnapshotManager({})))
        await pilot.pause()

        empty = app.screen.query_one(".terminals-empty", Static)
        assert str(empty.content) == "No managed terminal sessions."
        assert not app.screen.query(TabbedContent)


async def test_terms_are_oldest_to_newest_with_newest_active_on_right(tmp_path):
    infos = [terminal_info("newest00", 30), terminal_info("oldest00", 10)]
    manager = SnapshotManager({
        info.term_id: snapshot(info.term_id, info.term_id) for info in infos
    })
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(90, 28)) as pilot:
        app.push_screen(TermsScreen(infos, manager=manager))
        await pilot.pause()

        tabs = list(app.screen.query(Tab))
        assert [str(tab.label) for tab in tabs] == ["oldest00", "newest00"]
        assert app.screen.query_one("#terminal-tabs", TabbedContent).active == (
            "terminal-tab-newest00"
        )
        assert tabs[-1].region.x > tabs[0].region.x


async def test_process_terminal_fills_viewport_and_soft_wraps(tmp_path):
    line = "0123456789" * 20
    manager = SnapshotManager({"process0": snapshot("process0", line)})
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(60, 20)) as pilot:
        app.push_screen(
            TermsScreen([terminal_info("process0", 1)], manager=manager)
        )
        await pilot.pause()

        view = app.screen.query_one(TerminalView)
        rendered = view.content
        assert isinstance(rendered, Text)
        assert rendered.overflow == "fold"
        assert not rendered.no_wrap
        assert view.region.width == view.parent.content_region.width
        assert view.region.height == view.parent.content_region.height
        assert view.virtual_size.height > 1


async def test_process_terminal_view_is_anchored_to_newest_output(tmp_path):
    content = "\n".join(f"output-line-{line:03}" for line in range(60))
    manager = SnapshotManager({"process0": snapshot("process0", content)})
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(60, 20)) as pilot:
        app.push_screen(
            TermsScreen([terminal_info("process0", 1)], manager=manager)
        )
        await pilot.pause()
        await pilot.pause()

        screenshot = app.export_screenshot(simplify=False)
        assert "output-line-059" in screenshot
        assert "output-line-000" not in screenshot


async def test_ghostty_terminal_keeps_dimensions_and_rich_styling(tmp_path):
    style = TerminalStyle(
        foreground=(12, 34, 56),
        background=(65, 43, 21),
        bold=True,
        italic=True,
        underline=True,
    )
    manager = SnapshotManager({
        "ghost000": snapshot(
            "ghost000", "styled", screen=True, rows=7, cols=31, style=style
        )
    })
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            TermsScreen(
                [terminal_info("ghost000", 1, screen=True)], manager=manager
            )
        )
        await pilot.pause()

        view = app.screen.query_one(TerminalView)
        rendered = view.content
        assert isinstance(rendered, Text)
        assert rendered.no_wrap
        assert rendered.overflow == "crop"
        assert view.region.size == (31, 7)
        rich_style = rendered.get_style_at_offset(app.console, 0)
        assert rich_style.color.triplet == style.foreground
        assert rich_style.bgcolor.triplet == style.background
        assert rich_style.bold and rich_style.italic and rich_style.underline


def test_snapshot_text_preserves_all_terminal_style_flags():
    style = TerminalStyle(
        faint=True,
        blink=True,
        reverse=True,
        conceal=True,
        strike=True,
        overline=True,
    )
    rendered = snapshot_text(snapshot("term0000", "x", style=style))
    rich_style = rendered.spans[0].style

    assert rich_style.dim and rich_style.blink and rich_style.reverse
    assert rich_style.conceal and rich_style.strike and rich_style.overline


def test_snapshot_text_preserves_ghostty_grid_width_for_joined_emoji():
    rendered = snapshot_text(
        TerminalRenderSnapshot(
            term_id="ghost000",
            spans=(TerminalSpan("👩\u200d💻👍🏽", cells=8),),
            rows=1,
            cols=8,
            screen=True,
        )
    )

    assert rendered.plain.startswith("👩\u200d💻👍🏽")
    assert cell_len(rendered.plain) == 8


async def test_terminal_view_is_view_only_and_refreshes_live(tmp_path):
    manager = SnapshotManager({"process0": snapshot("process0", "first")})
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(70, 22)) as pilot:
        app.push_screen(
            TermsScreen([terminal_info("process0", 1)], manager=manager)
        )
        await pilot.pause()
        view = app.screen.query_one(TerminalView)
        assert not view.can_focus
        assert str(view.content) == "first"

        manager.snapshots["process0"] = snapshot("process0", "second")
        await asyncio.sleep(view.refresh_interval + 0.05)
        await pilot.pause()
        assert str(view.content) == "second"
        assert manager.calls.count("process0") >= 2

        prompt_text = app.query_one(PromptArea).text
        await pilot.press("x", "enter")
        assert app.screen.query_one(TerminalView) is view
        assert app.query_one(PromptArea).text == prompt_text


async def test_removed_terminal_and_modal_cleanup_are_safe(tmp_path):
    manager = SnapshotManager({"process0": snapshot("process0", "running")})
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(70, 22)) as pilot:
        app.push_screen(
            TermsScreen([terminal_info("process0", 1)], manager=manager)
        )
        await pilot.pause()
        view = app.screen.query_one(TerminalView)
        timer = view._refresh_timer
        assert timer is not None and timer._active

        manager.snapshots.clear()
        view.request_snapshot()
        await pilot.pause()
        assert str(view.content) == "Terminal session no longer exists."

        await pilot.press("escape")
        await pilot.pause()
        assert view._refresh_timer is None
        calls_after_close = len(manager.calls)
        await asyncio.sleep(view.refresh_interval + 0.05)
        await pilot.pause()
        assert len(manager.calls) == calls_after_close


async def test_slash_terms_opens_modal_and_restores_prompt_focus(
    tmp_path, monkeypatch
):
    import ursa.tools.terminal as terminal_module

    info = terminal_info("process0", 1)
    manager = SnapshotManager({"process0": snapshot("process0", "output")})
    manager.terminals = lambda: (info,)  # type: ignore[attr-defined]
    monkeypatch.setattr(terminal_module, "term_manager", manager)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "t", "e", "r", "m", "s", "enter")
        await pilot.pause()
        assert isinstance(app.screen, TermsScreen)
        assert str(app.screen.query_one(TerminalView).content) == "output"

        await pilot.press("q")
        await pilot.pause()
        assert app.focused is app.query_one(PromptArea)
