import asyncio
import io
from types import SimpleNamespace

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

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
            SimpleNamespace(llm_output={"token_usage": {"total_tokens": 37}})
        )
        return "Finished"

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
        turn = app.query_one(Turn)
        assert turn.token_usage == 37
        assert isinstance(list(turn.children)[-2], ActivityIndicator)
        assert isinstance(list(turn.children)[-1], MessageCard)
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


async def test_command_arrows_navigate_three_markers_per_turn(
    tmp_path, monkeypatch
):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)
    scrolled_to = []

    def record_scroll(self, widget, **kwargs):
        scrolled_to.append((widget, kwargs))
        return True

    monkeypatch.setattr(VerticalScroll, "scroll_to_widget", record_scroll)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("o", "n", "e", "enter")
        await pilot.pause()
        await pilot.press("t", "w", "o", "enter")
        await pilot.pause()

        markers = app._turn_markers()
        assert len(markers) == 6
        assert [
            marker.role if isinstance(marker, MessageCard) else "activity"
            for marker in markers
        ] == [
            "user",
            "activity",
            "assistant",
            "user",
            "activity",
            "assistant",
        ]

        await pilot.press("super+up")
        assert app._turn_navigation_marker is markers[4]
        await pilot.press("super+up")
        assert app._turn_navigation_marker is markers[3]
        await pilot.press("super+up")
        assert app._turn_navigation_marker is markers[2]
        await pilot.press("super+down")
        assert app._turn_navigation_marker is markers[3]
        assert [widget for widget, _ in scrolled_to] == [
            markers[4],
            markers[3],
            markers[2],
            markers[3],
        ]
        assert all(
            kwargs
            == {
                "top": True,
                "animate": False,
                "immediate": True,
                "force": True,
                "origin_visible": False,
            }
            for _, kwargs in scrolled_to
        )


async def test_turn_navigation_changes_real_scroll_position(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 20)) as pilot:
        for index in range(6):
            await pilot.press(*str(index), "enter")
            await pilot.pause()
        conversation = app.query_one("#conversation", VerticalScroll)
        bottom = conversation.scroll_y
        assert bottom > 0

        # The final response and activity markers may both already be visible
        # at the maximum scroll offset. Cross into the previous marker before
        # asserting the viewport moved.
        await pilot.press("super+up", "super+up", "super+up")
        await pilot.pause()
        assert conversation.scroll_y < bottom


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
    assert "One-shot result" in output.getvalue()
