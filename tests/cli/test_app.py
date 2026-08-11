import asyncio
import io
from types import SimpleNamespace

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Markdown, RichLog, Static

import ursa.cli.app as app_module
from tests.cli._app_fakes import FakeHITL, emit_event
from ursa.cli.app import UrsaTextualApp
from ursa.cli.event_cards import EventCard, RunCommandCard
from ursa.cli.turn import Turn
from ursa.cli.widgets import ActivityIndicator, MessageCard, PromptArea


async def test_prompt_submission_events_history_and_transcript(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(name, prompt, callbacks=None):
        hitl.calls.append((name, prompt))
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

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("h", "e", "l", "l", "o", "enter")
        await pilot.pause()

        assert hitl.calls == [("chat", "hello")]
        messages = list(app.query(MessageCard))
        assert len(messages) == 2
        assert messages[0].styles.background != messages[1].styles.background
        assert len(app.query(EventCard)) == 1
        assert app.total_tokens == 37
        assert app.input_tokens == 30
        assert app.output_tokens == 7
        assert app.cached_tokens == 12
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
        for message in app.query(MessageCard):
            markdown = message.query_one(Markdown)
            assert list(markdown.children)[-1].styles.margin.bottom == 0

        prompt = app.query_one(PromptArea)
        await pilot.press("up")
        assert prompt.text == "hello"

        await pilot.press("ctrl+t")
        assert not turn.query_one(".transcript").has_class("hidden")
        assert turn.query_one(".events").has_class("hidden")
        assert messages[-1].has_class("hidden")
        assert turn.transcript[-1] == "**Finished**"
        assert turn.query_one(RichLog).lines[-1].text == "**Finished**"

        await pilot.press("ctrl+t")
        assert not messages[-1].has_class("hidden")


async def test_command_events_from_a_worker_thread_update_the_ui(tmp_path):
    hitl = FakeHITL(tmp_path)

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

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        card = app.query_one(RunCommandCard)
        assert card.command == "pwd"
        assert len(app.query(MessageCard)) == 2


async def test_turn_spinner_animates_and_shows_reasoning_while_agent_runs(
    tmp_path,
):
    hitl = FakeHITL(tmp_path)
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

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("w", "a", "i", "t", "enter")
        await pilot.pause()
        activity = app.query_one(ActivityIndicator)
        spinner = activity.query_one(".activity-spinner", Static)
        label = activity.query_one(".activity-text", Static)
        done_mark = activity.query_one(".activity-done-mark", Static)
        first_frame = str(spinner.content)
        assert first_frame in ActivityIndicator.FRAMES
        assert str(label.content) == "Inspecting the request"

        await asyncio.sleep(0.1)
        await pilot.pause()
        assert str(spinner.content) in ActivityIndicator.FRAMES
        assert str(spinner.content) != first_frame

        release_agent.set()
        await pilot.pause()
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


async def test_ctrl_c_reports_that_running_agent_cannot_be_cancelled(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)
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
        assert prompt.disabled

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert prompt.disabled
        assert any(worker.group == "agent" for worker in app.workers)
        assert len(notifications) == 1
        assert "not supported" in notifications[0][0]
        assert "Ctrl+D" in notifications[0][0]
        assert notifications[0][1]["severity"] == "warning"

        release.set()
        await pilot.pause()
        assert not prompt.disabled


async def test_clear_conversation_is_refused_during_active_turn(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)
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

        assert turn.is_mounted
        assert "not allowed" in notifications[0][0]
        assert "Ctrl+D" in notifications[0][0]
        release.set()
        await pilot.pause()


async def test_quitting_waits_for_active_agent_then_exits(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        started.set()
        await release.wait()
        finished.set()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)
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

    assert app._exit


def test_ctrl_d_uses_abrupt_process_exit(tmp_path, monkeypatch):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    exit_codes = []
    monkeypatch.setattr("ursa.cli.app.os._exit", exit_codes.append)

    app.action_hard_quit()

    assert exit_codes == [130]


async def test_transcript_retains_reasoning_and_filtered_file_completions(
    tmp_path,
):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_chat_model_start()
        await handler.on_llm_new_token(
            "",
            chunk=SimpleNamespace(
                message=SimpleNamespace(
                    additional_kwargs={
                        "reasoning_content": "Checking the requested file"
                    }
                )
            ),
        )
        await handler.on_tool_start(
            {"name": "read_file"},
            "",
            run_id="read-complete",
            inputs={"path": "notes.md"},
        )
        await emit_event(
            handler,
            {
                "tool": "read_file",
                "phase": "end",
                "path": "notes.md",
                "message": "File read",
            },
        )
        await handler.on_tool_end("contents", run_id="read-complete")
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "e", "a", "d", "enter")
        await pilot.pause()
        turn = app.query_one(Turn)
        transcript = "\n".join(turn.transcript)
        assert "Checking the requested file" in transcript
        assert '"phase": "end"' in transcript
        assert '"result": "contents"' in transcript

        await pilot.press("ctrl+t")
        assert not turn.query_one(".transcript").has_class("hidden")


async def test_command_arrows_navigate_turn_markers_and_end_anchor(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("o", "n", "e", "enter")
        await pilot.pause()
        await pilot.press("t", "w", "o", "enter")
        await pilot.pause()

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
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        for index in range(6):
            await pilot.press(*str(index), "enter")
            await pilot.pause()
        conversation = app.query_one("#conversation", VerticalScroll)
        bottom = conversation.scroll_y
        assert bottom > 0

        await pilot.press("alt+down")
        await pilot.pause()
        assert app._turn_navigation_marker is app._turn_markers()[-1]
        assert conversation.scroll_y == conversation.max_scroll_y

        # Several markers from the latest turns may already be visible at the
        # maximum scroll offset. Cross into an earlier turn before asserting
        # that the viewport moved.
        await pilot.press(*(["alt+up"] * 20))
        await pilot.pause()
        assert conversation.scroll_y < bottom


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
    app = UrsaTextualApp(FakeHITL(tmp_path))

    assert app._route_prompt(prompt) == expected


def test_one_shot_routes_hash_agent_and_writes_response(tmp_path):
    class OneShotHITL(FakeHITL):
        async def run_agent(self, name, prompt, callbacks=None):
            self.calls.append((name, prompt))
            return "One-shot result"

    hitl = OneShotHITL(tmp_path)
    output = io.StringIO()

    result = app_module.run_textual_once(
        hitl, "#plan inspect this", stdout=output
    )

    assert result == "One-shot result"
    assert hitl.calls == [("plan", "inspect this")]
    assert hitl.closed
    assert "One-shot result" in output.getvalue()
