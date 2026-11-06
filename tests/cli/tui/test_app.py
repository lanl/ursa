import asyncio
import io
import logging
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

import ursa.cli.tui.app as app_module
import ursa.util.crossplatform as crossplatform
from tests.cli._app_helpers import emit_event, make_runtime, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import (
    EventCard,
    ExceptionCard,
    RunCommandCard,
    ToolCallCard,
)
from ursa.cli.tui.event_handler import TextualEventHandler
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import (
    ActivityIndicator,
    HierarchyMessageCard,
    HotlistScreen,
    MessageCard,
    MessagesScreen,
    PromptArea,
    QueuedMessage,
    WelcomeBanner,
)


def _activate_hierarchy(runtime, *paths):
    root_id = runtime.agents["chat"].runtime_id
    runtime._active_root_id = root_id
    runtime._active_agent_ready.set()
    runtime._hierarchies[root_id] = {path: f"runtime:{path}" for path in paths}
    return root_id


def test_copy_bindings_are_uniform_and_global():
    bindings = {
        binding.key: binding
        for binding in UrsaTextualApp._effective_bindings(UrsaTextualApp)
    }

    assert bindings["ctrl+c"].action != "copy_text"
    for key in ("ctrl+shift+c", "super+c"):
        assert bindings[key].action == "copy_text"
        assert bindings[key].priority


def test_hierarchy_activity_uses_uniform_labels():
    assert (
        UrsaTextualApp._event_activity({
            "tool": "inspect_code",
            "phase": "start",
            "message": "provider-specific input",
        })
        == "Running inspect code"
    )
    assert (
        UrsaTextualApp._event_activity({
            "tool": "inspect_code",
            "phase": "end",
        })
        == "Finished inspect code"
    )
    assert (
        UrsaTextualApp._event_activity({
            "reasoning_summary": "  checking\nimports  "
        })
        == "Thinking: checking imports"
    )


async def test_messages_command_shows_live_collapsible_colored_log(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))
    content = (
        "[OPENING round=1 root/p08] Review the first line.\n\n"
        "Include the complete supporting detail. [red]literal brackets[/red]"
    )
    await app.record_hierarchy_message(
        "root",
        ("root/researcher", "root/reviewer"),
        content,
    )

    async with app.run_test(size=(100, 32)) as pilot:
        await app._show_command("messages")
        assert isinstance(app.screen, MessagesScreen)
        await wait_for(
            pilot, lambda: len(app.screen.query(HierarchyMessageCard)) == 1
        )

        card = app.screen.query_one(HierarchyMessageCard)
        header = card.query_one(".hierarchy-message-header")
        route = card.query_one(".hierarchy-message-route", Static)
        timestamp = card.query_one(".hierarchy-message-time", Static)
        preview = card.query_one(".hierarchy-message-preview", Static)
        full = card.query_one(".hierarchy-message-full", Markdown)
        assert "root" in str(route.content)
        assert "root/researcher" in str(route.content)
        assert "root/reviewer" in str(route.content)
        assert str(timestamp.content) == "now"
        assert timestamp.region.right == header.content_region.right
        assert "\n" not in str(preview.content)
        assert str(preview.content) == " ".join(content.split())
        assert full.has_class("hidden")
        assert not card.query(".hierarchy-message-expand-hint")
        assert "click or Enter" in str(
            app.screen.query_one("#messages-footer", Static).content
        )
        assert len(set(app._agent_colors.values())) == 3
        assert card.sender_color == app._agent_color("root")

        thirty_seconds_later = card.message.timestamp + timedelta(seconds=30)
        card.refresh_timestamp(thirty_seconds_later)
        assert str(timestamp.content) == "30s ago"

        await pilot.click(card)
        await pilot.pause()
        assert card.expanded
        assert preview.has_class("hidden")
        assert not full.has_class("hidden")
        assert "complete supporting detail" in str(full.source)

        # Expanding a card pauses follow so the body remains readable. Resume
        # before asserting that subsequent live traffic appears automatically.
        await app.screen.action_latest()

        await app.record_hierarchy_message(
            "root/researcher",
            ("root",),
            "[DISCUSS round=1 root/p08] Research complete.",
        )
        assert await wait_for(
            pilot,
            lambda: len(app.screen.query(HierarchyMessageCard)) == 2,
        )
        assert not app.screen.query("#messages-body Input")


def test_tui_logging_never_writes_through_terminal_handler(capsys):
    root = logging.getLogger()
    terminal_handler = logging.StreamHandler(sys.stderr)
    root.addHandler(terminal_handler)
    try:
        with app_module._tui_logging():
            assert terminal_handler not in root.handlers
            assert any(
                isinstance(handler, app_module.TextualHandler)
                for handler in root.handlers
            )
            logging.getLogger("tests.tui.output").warning("tui logging marker")
        assert terminal_handler in root.handlers
    finally:
        root.removeHandler(terminal_handler)

    captured = capsys.readouterr()
    assert "tui logging marker" not in captured.out + captured.err


async def test_welcome_banner_starts_at_top_of_conversation(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", VerticalScroll)
        banner = app.query_one(WelcomeBanner)

        assert banner.region.y == conversation.content_region.y

        turn = Turn("short conversation", tmp_path)
        await conversation.mount(turn)
        await turn.add_response("Short response")
        await pilot.pause()

        await wait_for(
            pilot, lambda: banner.region.y == conversation.content_region.y
        )
        assert banner.region.y == conversation.content_region.y


async def test_prompt_submission_events_and_history(tmp_path):
    runtime = make_runtime(tmp_path)
    _activate_hierarchy(runtime, "root")
    calls = []

    async def run_agent(name, prompt, callbacks=None):
        calls.append((name, prompt))
        handler = callbacks[0]
        await emit_event(
            handler,
            tool="read_file",
            stage="read",
            message="Reading file",
            path="src/example.py",
        )
        await handler.on_llm_end(
            SimpleNamespace(
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 7,
                        "total_tokens": 37,
                        "prompt_tokens_details": {"cached_tokens": 12},
                    }
                }
            )
        )
        return "**Finished**"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("h", "e", "l", "l", "o", "enter")
        await pilot.pause()

        await wait_for(pilot, lambda: calls == [("chat", "hello")])
        assert calls == [("chat", "hello")]
        messages = list(app.query(MessageCard))
        assert len(messages) == 2
        assert messages[0].styles.background != messages[1].styles.background
        assert messages[1].content == "**Finished**"
        assert len(app.query(EventCard)) == 1
        assert app.total_tokens == 37
        assert app.input_tokens == 30
        assert app.output_tokens == 7
        assert app.cached_tokens == 12
        assert app._agent_token_usage["root"].input_tokens == 30
        assert app._agent_token_usage["root"].output_tokens == 7
        assert app._agent_token_usage["root"].cached_tokens == 12
        assert app._agent_token_usage["root"].total_tokens == 37
        assert app._agent_last_activity["root"] == "Running read file"
        status = str(app.query_one("#status", Static).content)
        assert "37 tokens" in status
        assert "input" not in status
        assert "output" not in status
        assert "cached" not in status
        turn = app.query_one(Turn)
        assert turn.token_usage == 37
        assert isinstance(list(turn.children)[-3], ActivityIndicator)
        assert isinstance(list(turn.children)[-2], MessageCard)
        assert list(turn.children)[-1].has_class("turn-end-marker")
        roles = list(app.query(".message-role"))
        assert len(roles) == 1
        assert str(roles[0].content) == "URSA"
        markdown = messages[1].query_one(Markdown)
        assert list(markdown.children)[-1].styles.margin.bottom == 0
        assert isinstance(messages[0].query_one(".message-body"), Static)

        prompt = app.query_one(PromptArea)
        await pilot.press("up")
        assert prompt.text == "hello"


async def test_long_submitted_prompt_is_not_clipped(tmp_path):
    runtime = make_runtime(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)
    prompt_text = (
        "Act as GM and run a complete compact game of Werewolf with exactly "
        "five player subagents. Privately assign roles, collect every reply "
        "with wait_for, and report whether all expected messages were "
        "received exactly once."
    )

    async with app.run_test(size=(140, 48)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text(prompt_text)
        await pilot.press("enter")
        await started.wait()
        await pilot.pause()

        user = app.query_one(".message-card.user", MessageCard)
        body = user.query_one(".message-body", Static)
        rendered = " ".join(
            body.render_line(line).text.strip()
            for line in range(body.region.height)
        )
        assert body.region.height >= 2
        assert "received exactly once." in rendered

        release.set()
        await app.workers.wait_for_complete()


async def test_root_view_hides_duplicate_child_progress_during_initial_run(
    tmp_path,
):
    runtime = make_runtime(tmp_path)
    child_started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        await callbacks[0].on_chat_model_start(
            metadata={"ursa_message_recipient": "root"}
        )
        await callbacks[0].on_agent_message_start("root/player1")
        child_started.set()
        await release.wait()
        await callbacks[0].on_agent_message_end("root/player1", succeeded=True)
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await child_started.wait()
        await pilot.pause()

        root_turn = app._agent_turns["root"]
        child_turn = app._agent_turns["root/player1"]
        assert app._active_agent_path == "root"
        assert app._view_agent_path == "root"
        assert not root_turn.has_class("hidden")
        assert child_turn.has_class("hidden")

        app._active_agent_path = "root/player1"
        app._view_agent_path = "root/player1"
        app._apply_agent_turn_visibility()
        assert root_turn.has_class("hidden")
        assert not child_turn.has_class("hidden")

        release.set()
        await app.workers.wait_for_complete()


async def test_agent_exception_card_expands_to_full_traceback(tmp_path):
    runtime = make_runtime(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        raise RuntimeError("provider disconnected")

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("h", "e", "l", "l", "o", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        card = app.query_one(ExceptionCard)
        assert card.lines == ["RuntimeError: provider disconnected"]
        assert len(card.details) == 1
        assert "Traceback (most recent call last):" in card.details[0]
        assert "in run_agent" in card.details[0]
        assert 'raise RuntimeError("provider disconnected")' in card.details[0]
        assert card.details[0].endswith("RuntimeError: provider disconnected\n")
        assert not card.expanded

        class Click:
            def stop(self):
                pass

        card.on_click(Click())
        await pilot.pause()
        await wait_for(pilot, lambda: card.expanded)
        assert card.expanded
        rich_traceback = card.query_one(".exception-traceback", Static)
        assert not rich_traceback.has_class("hidden")
        assert type(rich_traceback.content).__name__ == "Traceback"
        assert rich_traceback.content.trace.stacks[-1].exc_value == (
            "provider disconnected"
        )


async def test_command_events_from_a_worker_thread_update_the_ui(tmp_path):
    runtime = make_runtime(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        def emit_command_event():
            asyncio.run(
                emit_event(
                    handler,
                    {
                        "tool": "run_command",
                        "stage": "execute",
                        "phase": "start",
                        "message": "Running command",
                        "query": "pwd",
                    },
                )
            )

        await asyncio.to_thread(emit_command_event)
        return "Command finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        card = app.query_one(RunCommandCard)
        assert card.command == "pwd"
        assert len(app.query(MessageCard)) == 2


async def test_turn_spinner_animates_and_shows_reasoning_while_agent_runs(
    tmp_path,
):
    runtime = make_runtime(tmp_path)
    release_agent = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        await callbacks[0].on_llm_new_token(
            "",
            chunk=SimpleNamespace(
                message=SimpleNamespace(
                    additional_kwargs={
                        "reasoning_content": "Inspecting the request"
                    }
                )
            ),
        )
        await release_agent.wait()
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("w", "a", "i", "t", "enter")
        await pilot.pause()
        activity = app.query_one(ActivityIndicator)
        spinner = activity.query_one(".activity-spinner", Static)
        label = activity.query_one(".activity-text", Static)
        done_mark = activity.query_one(".activity-done-mark", Static)
        first_frame = str(spinner.content)
        assert first_frame in ActivityIndicator.FRAMES
        assert str(label.content) == "Thinking: Inspecting the request"

        assert await wait_for(
            pilot, lambda: str(spinner.content) != first_frame
        )
        assert str(spinner.content) in ActivityIndicator.FRAMES

        release_agent.set()
        assert await wait_for(pilot, lambda: not app.workers)
        assert str(spinner.content) == ""
        assert str(label.content) == ""
        assert str(done_mark.content) == ""
        assert activity.has_class("hidden")
        assert not activity.has_class("done")

        activity.finish(elapsed=31, tokens=1234)
        assert str(label.content) == "Done in 31s and 1,234 tokens"
        assert str(done_mark.content)
        assert not activity.has_class("hidden")
        assert activity.has_class("done")
        assert label.styles.content_align == ("right", "middle")
        activity.finish(elapsed=30, tokens=1234)
        assert str(label.content) == ""
        assert str(done_mark.content) == ""
        assert activity.has_class("hidden")


async def test_ctrl_c_stops_managed_agent_at_committed_state(
    tmp_path, monkeypatch
):
    runtime = make_runtime(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)
    notifications = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("w", "a", "i", "t", "enter")
        await started.wait()
        prompt = app.query_one(PromptArea)
        assert not prompt.disabled
        assert "Message active agent" in prompt.placeholder

        await pilot.press("ctrl+c")
        assert await wait_for(pilot, lambda: len(notifications) == 1)

        assert not prompt.disabled
        assert any(worker.group == "agent" for worker in app.workers)
        assert notifications[0][0] == (
            "Stopped chat at its last committed state."
        )
        assert notifications[0][1]["severity"] == "warning"

        release.set()
        await app.workers.wait_for_complete()
        assert not prompt.disabled


async def test_message_stays_dimmed_until_active_agent_accepts_it(tmp_path):
    runtime = make_runtime(tmp_path)
    _activate_hierarchy(runtime, "root")
    started = asyncio.Event()
    release_root = asyncio.Event()
    accept_message = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release_root.wait()
        return "Original finished"

    async def run_active_agent(
        path,
        prompt,
        *,
        callbacks=None,
        on_start=None,
    ):
        await accept_message.wait()
        await on_start(None)
        release_root.set()
        return f"Processed {prompt}"

    runtime.run_agent = run_agent
    runtime.run_active_agent = run_active_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("s", "t", "a", "r", "t", "enter")
        await started.wait()
        await pilot.press("u", "p", "d", "a", "t", "e", "enter")

        assert await wait_for(pilot, lambda: bool(app.query(QueuedMessage)))
        queued = app.query_one(QueuedMessage)
        prompt = app.query_one(PromptArea)
        assert queued.region.y < prompt.region.y
        assert all(card.content != "update" for card in app.query(MessageCard))

        await pilot.press("escape")
        assert queued.is_mounted
        assert not release_root.is_set()
        prompt.load_text("unsent draft")
        await pilot.press("escape")
        assert prompt.text == "unsent draft"
        assert len(app.query(QueuedMessage)) == 1
        assert not release_root.is_set()

        accept_message.set()
        assert await wait_for(pilot, lambda: not app.query(QueuedMessage))
        assert any(card.content == "update" for card in app.query(MessageCard))
        await app.workers.wait_for_complete()


async def test_in_band_message_promotes_without_a_duplicate_empty_reply(
    tmp_path,
):
    runtime = make_runtime(tmp_path)
    _activate_hierarchy(runtime, "root")
    started = asyncio.Event()
    release_root = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release_root.wait()
        return "Original response"

    async def run_active_agent(
        _path,
        _prompt,
        *,
        callbacks=None,
        on_start=None,
    ):
        del callbacks
        await on_start(None)
        release_root.set()
        # The message reached a wait_for tool result in the existing turn.
        return None

    runtime.run_agent = run_agent
    runtime.run_active_agent = run_active_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("s", "t", "a", "r", "t", "enter")
        await started.wait()
        await pilot.press("u", "p", "d", "a", "t", "e", "enter")

        assert await wait_for(pilot, lambda: not app.query(QueuedMessage))
        assert any(card.content == "update" for card in app.query(MessageCard))
        assert all(card.content != "None" for card in app.query(MessageCard))
        await app.workers.wait_for_complete()


async def test_subagents_command_switches_message_and_progress_target(tmp_path):
    runtime = make_runtime(tmp_path)
    _activate_hierarchy(runtime, "root", "root/researcher")
    messages = []

    async def run_active_agent(
        path,
        prompt,
        *,
        callbacks=None,
        on_start=None,
    ):
        messages.append((path, prompt))
        await on_start(None)
        return "Research complete"

    runtime.run_active_agent = run_active_agent
    app = UrsaTextualApp(runtime)
    app._active_agent_path = "root"
    app._message_mode = True

    async with app.run_test(size=(100, 36)) as pilot:
        root_turn = Turn("delegate", tmp_path, agent_path="root")
        await app._mount_existing_turn(root_turn)
        handler = TextualEventHandler(app, root_turn)
        await handler.on_agent_message_start("root/researcher")
        await handler.on_tool_start(
            {"name": "inspect_code"},
            '{"path": "src/example.py"}',
            run_id="child-tool",
            metadata={"ursa_message_recipient": "root/researcher"},
        )
        assert (
            app._agent_last_activity["root/researcher"]
            == "Running inspect code"
        )
        await handler.on_agent_message_end("root/researcher", succeeded=True)
        assert (
            app._agent_last_activity["root/researcher"]
            == "Running inspect code"
        )

        await app._show_command("subagents")
        assert isinstance(app.screen, HotlistScreen)
        await pilot.press("r", "e", "s", "e", "a", "r", "c", "h", "enter")
        assert await wait_for(
            pilot, lambda: app._active_agent_path == "root/researcher"
        )
        child_turn = app._agent_turns["root/researcher"]
        assert not child_turn.has_class("hidden")
        assert root_turn.has_class("hidden")
        assert len(child_turn.query(ToolCallCard)) == 1

        await pilot.press("r", "e", "v", "i", "e", "w", "enter")
        assert await wait_for(pilot, lambda: bool(messages))
        assert messages == [("root/researcher", "review")]
        await app.workers.wait_for_complete()


async def test_clear_conversation_is_refused_during_active_turn(
    tmp_path, monkeypatch
):
    runtime = make_runtime(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)
    notifications = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("w", "a", "i", "t", "enter")
        await started.wait()
        turn = app.query_one(Turn)

        await pilot.press("ctrl+l")
        await pilot.pause()

        await wait_for(pilot, lambda: turn.is_mounted)
        assert turn.is_mounted
        assert "not allowed" in notifications[0][0]
        assert "Ctrl+D" in notifications[0][0]
        release.set()
        await pilot.pause()


async def test_quitting_waits_for_active_agent_then_exits(
    tmp_path, monkeypatch
):
    runtime = make_runtime(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        finished.set()
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)
    notifications = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("w", "a", "i", "t", "enter")
        await started.wait()
        await pilot.press("ctrl+q")
        await pilot.pause()

        assert not app._exit
        assert notifications
        assert "active turn finishes" in notifications[0][0]
        assert "Ctrl+D" in notifications[0][0]

        release.set()
        await finished.wait()
        await pilot.pause()

    await wait_for(pilot, lambda: app._exit)
    assert app._exit


async def test_command_arrows_navigate_turn_markers_and_end_anchor(tmp_path):
    runtime = make_runtime(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("o", "n", "e", "enter")
        await app.workers.wait_for_complete()
        await pilot.press("t", "w", "o", "enter")
        await app.workers.wait_for_complete()

        markers = app._turn_markers()
        assert len(markers) == 8
        assert [
            marker.role
            if isinstance(marker, MessageCard)
            else "end"
            if marker.has_class("turn-end-marker")
            else "activity"
            for marker in markers
        ] == [
            "user",
            "activity",
            "assistant",
            "end",
            "user",
            "activity",
            "assistant",
            "end",
        ]

        await pilot.press("alt+down")
        assert app._turn_navigation_marker is markers[7]
        await pilot.press("alt+up")
        assert app._turn_navigation_marker is markers[6]
        await pilot.press("alt+up")
        assert app._turn_navigation_marker is markers[5]
        await pilot.press("alt+up")
        assert app._turn_navigation_marker is markers[4]
        await pilot.press("alt+down")
        assert app._turn_navigation_marker is markers[5]


async def test_turn_navigation_changes_real_scroll_position(tmp_path):
    runtime = make_runtime(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        return "Finished"

    runtime.run_agent = run_agent
    app = UrsaTextualApp(runtime)

    async with app.run_test(size=(80, 20)) as pilot:
        for index in range(6):
            await pilot.press(*str(index), "enter")
            await app.workers.wait_for_complete()
        conversation = app.query_one("#conversation", VerticalScroll)
        bottom = conversation.scroll_y
        assert bottom > 0

        await pilot.press("alt+down")
        await pilot.pause()
        await wait_for(
            pilot,
            lambda: app._turn_navigation_marker is app._turn_markers()[-1],
        )
        assert app._turn_navigation_marker is app._turn_markers()[-1]
        assert conversation.scroll_y == conversation.max_scroll_y
        assert conversation.is_anchored

        # Several markers from the latest turns may already be visible at the
        # maximum scroll offset. Cross into an earlier turn before asserting
        # that the viewport moved.
        await pilot.press(*(["alt+up"] * 20))
        await pilot.pause()
        assert conversation.scroll_y < bottom


async def test_new_cards_follow_bottom_without_moving_scrolled_view(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        conversation = app.query_one("#conversation", VerticalScroll)
        turn = Turn("test", tmp_path)
        await conversation.mount(turn)

        for index in range(12):
            await app.add_turn_event(
                turn,
                {
                    "type": "custom",
                    "tool": f"tool-{index}",
                    "phase": "start",
                },
            )
            await pilot.pause()

        assert await wait_for(
            pilot,
            lambda: conversation.scroll_y == conversation.max_scroll_y,
        )

        conversation.scroll_to(
            y=max(0, conversation.scroll_y - 3),
            animate=False,
            immediate=True,
        )
        await pilot.pause()
        scrolled_position = conversation.scroll_y
        assert scrolled_position < conversation.max_scroll_y

        await app.add_turn_event(
            turn,
            {
                "type": "custom",
                "tool": "one-more-tool",
                "phase": "start",
            },
        )
        await pilot.pause()

        assert conversation.scroll_y == pytest.approx(scrolled_position)


async def test_user_scroll_cancels_initial_anchor_transition(tmp_path):
    app = UrsaTextualApp(make_runtime(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        conversation = app.query_one("#conversation", VerticalScroll)
        turn = Turn("test", tmp_path)
        await conversation.mount(turn)

        for index in range(12):
            await app.add_turn_event(
                turn,
                {
                    "type": "custom",
                    "tool": f"tool-{index}",
                    "phase": "start",
                },
            )
            await pilot.pause(0.01)
            if app._conversation_anchor_transition:
                break

        assert app._conversation_anchor_transition
        conversation.scroll_home(animate=False, immediate=True)
        await pilot.pause(0.2)

        assert not conversation.is_anchored
        scrolled_position = conversation.scroll_y
        await app.add_turn_event(
            turn,
            {
                "type": "custom",
                "tool": "after-interruption",
                "phase": "start",
            },
        )
        await pilot.pause()

        assert conversation.scroll_y == pytest.approx(scrolled_position)
        assert conversation.scroll_y < conversation.max_scroll_y

        await app.submit_prompt(PromptArea.Submitted("next prompt"))
        await app.workers.wait_for_complete()
        for _ in range(50):
            await pilot.pause(0.02)
            if conversation.is_anchored:
                break

        assert conversation.is_anchored
        assert conversation.scroll_y == conversation.max_scroll_y


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("#plan\tinspect this", ("plan", "inspect this")),
        ("#plan\ninspect this\ncarefully", ("plan", "inspect this\ncarefully")),
        ("#plan", ("plan", "")),
        ("#missing inspect this", ("chat", "#missing inspect this")),
    ],
)
def test_hash_agent_routing_accepts_whitespace_and_multiline_prompts(
    tmp_path, prompt, expected
):
    app = UrsaTextualApp(make_runtime(tmp_path))

    assert app._route_prompt(prompt) == expected


def test_copy_to_clipboard_prefers_platform_tool_when_available(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    fallback = []
    monkeypatch.setattr(crossplatform, "copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(
        app_module.App,
        "copy_to_clipboard",
        lambda self, text: fallback.append(text),
    )

    app.copy_to_clipboard("hello")

    assert fallback == []
    assert app.clipboard == "hello"


def test_copy_to_clipboard_uses_osc52_when_platform_copy_unavailable(
    tmp_path, monkeypatch
):
    app = UrsaTextualApp(make_runtime(tmp_path))
    fallback = []
    monkeypatch.setattr(crossplatform, "copy_to_clipboard", lambda text: False)
    monkeypatch.setattr(
        app_module.App,
        "copy_to_clipboard",
        lambda self, text: fallback.append(text),
    )

    app.copy_to_clipboard("hello")

    assert fallback == ["hello"]
    assert app.clipboard == "hello"


def test_one_shot_routes_hash_agent_and_writes_response(tmp_path):
    runtime = make_runtime(tmp_path)
    calls = []

    async def run_agent(name, prompt, callbacks=None):
        calls.append((name, prompt))
        return "One-shot result"

    runtime.run_agent = run_agent
    output = io.StringIO()

    result = app_module.prompt_agent_once(
        runtime, "#plan inspect this", stdout=output
    )

    assert result == "One-shot result"
    assert calls == [("plan", "inspect this")]
    assert runtime._closed
    assert "One-shot result" in output.getvalue()


async def test_user_scroll_during_anchor_start_gap_is_not_overridden(tmp_path):
    # The anchor transition must start its animation synchronously with
    # its flag; a user scroll in the gap before a deferred start could
    # not stop an animation that had not begun, and the late start then
    # drove the viewport away from the user's position.
    app = UrsaTextualApp(make_runtime(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        conversation = app.query_one("#conversation", VerticalScroll)
        turn = Turn("test", tmp_path)
        await conversation.mount(turn)

        for index in range(12):
            await app.add_turn_event(
                turn,
                {"type": "custom", "tool": f"tool-{index}", "phase": "start"},
            )
            await pilot.pause(0.01)
            if app._conversation_anchor_transition:
                break

        assert app._conversation_anchor_transition
        assert app.animator.is_being_animated(conversation, "scroll_y")

        conversation.scroll_home(animate=False, immediate=True)
        for _ in range(8):
            await asyncio.sleep(0.03)
            assert conversation.scroll_y == 0


async def test_update_status_survives_absent_status_widget(tmp_path):
    # The agent worker's tail calls _update_status; when teardown has
    # already pruned the status widget, the bare query crashed the
    # worker and surfaced as WorkerFailed at run_test exit, after every
    # test assertion had passed (the recurring Windows CI signature).
    app = UrsaTextualApp(make_runtime(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await app.query_one("#status").remove()
        await pilot.pause()

        app._update_status("ready")
