import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

import ursa.cli.app as app_module
import ursa.cli.event_handler as event_handler_module
import ursa.cli.tips as tips
import ursa.cli.turn as turn_module
import ursa.cli.widgets as widgets_module
from ursa.cli.app import UrsaTextualApp
from ursa.cli.event_cards import (
    AgentEventCard,
    ArtifactCard,
    CommandSafetyIndicator,
    EventCard,
    FileActivityCard,
    PlanCard,
    RunCommandCard,
    SearchEventCard,
)
from ursa.cli.helpers import _fuzzy_match, _reasoning_trace, _token_usage
from ursa.cli.tips import TIPS
from ursa.cli.turn import Turn
from ursa.cli.widgets import (
    ActivityIndicator,
    HotlistScreen,
    InformationScreen,
    MessageCard,
    PromptArea,
    WelcomeBanner,
)
from ursa.util.events import DEFAULT_EVENT_NAME


class FakeAgent:
    description = "A configured test agent."
    config = {"mode": "test"}


def test_random_tip_chooses_from_tip_catalog(monkeypatch):
    received = None

    def choose(candidates):
        nonlocal received
        received = candidates
        return candidates[0]

    monkeypatch.setattr(tips.random, "choice", choose)

    assert len(TIPS) > 1
    assert tips.random_tip() == TIPS[0]
    assert received is TIPS


class FakeHITL:
    model = SimpleNamespace(model_name="test-model")
    embedding = None
    group = "default"
    config = SimpleNamespace(mcp_servers={})
    agents = {"chat": FakeAgent(), "plan": FakeAgent()}

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.calls = []

    async def run_agent(self, name, prompt, callbacks=None):
        self.calls.append((name, prompt))
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "read_file",
                "stage": "read",
                "message": "Reading file",
                "path": "src/example.py",
            },
        )
        await handler.on_llm_end(
            SimpleNamespace(llm_output={"token_usage": {"total_tokens": 37}})
        )
        return "Finished"


def test_fuzzy_match_and_token_usage_support_common_shapes():
    assert _fuzzy_match("sre", "src/example.py")
    assert not _fuzzy_match("xyz", "src/example.py")
    assert _token_usage({"usage": {"total_tokens": 42}}) == 42
    chunk = SimpleNamespace(
        message=SimpleNamespace(
            additional_kwargs={"reasoning_content": "Checking command safety"}
        )
    )
    assert _reasoning_trace(chunk) == "Checking command safety"


async def test_prompt_submission_events_history_and_transcript(tmp_path):
    hitl = FakeHITL(tmp_path)
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


async def test_agent_hotlist_routes_selected_agent(tmp_path):
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("#")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)

        assert app.screen.query_one("#hotlist").region.width == 100
        assert (
            app.screen.query_one("#hotlist-query").placeholder == "fzf search…"
        )
        options = app.screen.query_one("#hotlist-options")
        assert options.highlighted == 0
        await pilot.press("p", "l")
        assert app.screen.matches == ["plan"]
        assert options.highlighted == 0
        await pilot.press("enter")
        await pilot.pause()
        prompt = app.query_one(PromptArea)
        assert prompt.text == "#plan "

        prompt.insert("make a plan")
        await pilot.press("enter")
        await pilot.pause()
        assert hitl.calls == [("plan", "make a plan")]


async def test_agent_selection_moves_to_front_replaces_and_preserves_cursor(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs carefully")
        prompt.move_cursor((0, 6))

        await pilot.press("#", "p", "l", "enter")
        await pilot.pause()
        assert prompt.text == "#plan Review docs carefully"
        assert prompt.cursor_location == (0, 12)

        await pilot.press("#", "c", "h", "enter")
        await pilot.pause()
        assert prompt.text == "#chat Review docs carefully"
        assert prompt.cursor_location == (0, 12)


async def test_escape_from_agent_selector_restores_prompt_and_cursor(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("Review docs carefully")
        prompt.move_cursor((0, 6))

        await pilot.press("#")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        await pilot.press("escape")
        await pilot.pause()

        assert prompt.text == "Review docs carefully"
        assert prompt.cursor_location == (0, 6)
        assert prompt.has_focus


async def test_file_hotlist_uses_at_trigger(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("@")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        assert app.screen.query_one("#hotlist").region.width == 100
        assert app.screen.candidates == [
            "docs/",
            "docs/guide.md",
            "notes.md",
        ]
        options = app.screen.query_one("#hotlist-options")
        assert options.highlighted == 0

        await pilot.press("n", "o")
        assert app.screen.matches == ["notes.md"]
        assert options.highlighted == 0
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "@notes.md "


async def test_file_hotlist_inserts_directories_with_trailing_slash(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide")
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("@")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)

        await pilot.press("d", "o", "c", "s", "enter")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "@docs/ "


async def test_macro_hotlists_close_with_escape_and_can_be_reopened(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        main_screen = app.screen
        prompt = app.query_one(PromptArea)

        for trigger in ("@", "#"):
            await pilot.press(trigger)
            await pilot.pause()
            assert isinstance(app.screen, HotlistScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is main_screen
            assert not app._hotlist_open
            assert prompt.has_focus

            prompt.load_text("")
            await pilot.pause()


async def test_shift_enter_adds_a_prompt_newline(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("a", "shift+enter", "b")
        assert app.query_one(PromptArea).text == "a\nb"


async def test_prompt_has_markdown_highlighting_paste_undo_and_redo(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        assert prompt.language == "markdown"

        app.post_message(events.Paste("# heading\nbody"))
        await pilot.pause()
        assert prompt.text == "# heading\nbody"

        await pilot.press("ctrl+z")
        assert prompt.text == ""
        await pilot.press("ctrl+y")
        assert prompt.text == "# heading\nbody"


async def test_prompt_supports_option_arrow_word_navigation(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        prompt.load_text("alpha beta")
        prompt.move_cursor((0, len(prompt.text)))

        await pilot.press("alt+left")
        assert prompt.cursor_location == (0, 6)
        await pilot.press("alt+right")
        assert prompt.cursor_location == (0, 10)


async def test_ctrl_c_clears_prompt_and_adds_it_to_history(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test() as pilot:
        await pilot.press("o", "l", "d", "enter")
        await pilot.pause()
        await pilot.press("d", "r", "a", "f", "t", "ctrl+c")
        prompt = app.query_one(PromptArea)
        assert prompt.text == ""

        await pilot.press("up")
        assert prompt.text == "draft"
        await pilot.press("up")
        assert prompt.text == "old"
        await pilot.press("down")
        assert prompt.text == "draft"


async def test_prompt_grows_from_one_to_ten_content_lines(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        prompt = app.query_one(PromptArea)
        assert prompt.region.height == 3  # One content row plus the border.

        prompt.load_text("\n".join(str(index) for index in range(12)))
        await pilot.pause()
        assert prompt.region.height == 12  # Ten content rows plus the border.


async def test_command_events_from_a_worker_thread_update_the_ui(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        def emit_command_event():
            asyncio.run(
                handler.on_custom_event(
                    DEFAULT_EVENT_NAME,
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
        assert str(done_mark.content) == "✓"
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
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
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


async def test_files_are_grouped_by_read_write_and_edit_operations(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        events = [
            {"tool": "read_file", "path": "src/read.py", "message": "Reading"},
            {
                "tool": "read_file",
                "path": "src/other.py",
                "message": "Reading",
            },
            {
                "tool": "write_code",
                "path": "src/new.py",
                "code": "new file\n",
                "message": "Writing",
            },
            {
                "tool": "edit_code",
                "path": "src/edit.py",
                "old_code": "old\n",
                "new_code": "new\nmore\n",
                "message": "Editing",
            },
            {
                "tool": "edit_code",
                "path": str(tmp_path / "src" / "edit.py"),
                "message": "Editing",
            },
        ]
        for event in events:
            await handler.on_custom_event(DEFAULT_EVENT_NAME, event)
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        file_groups = list(app.query(FileActivityCard))
        assert len(file_groups) == 2
        reading, editing = file_groups
        assert reading.files == {
            "Reading": {
                "src/read.py": (None, None),
                "src/other.py": (None, None),
            },
            "Editing": {},
        }
        assert editing.files == {
            "Reading": {},
            "Editing": {
                "src/new.py": (1, 0),
                "src/edit.py": (2, 1),
            },
        }
        reading_summary = reading.query_one(".file-summary", Static).content
        editing_summary = editing.query_one(".file-summary", Static).content
        assert "📖 Reading: src/read.py, src/other.py" in reading_summary.plain
        assert "✍️ Editing" in editing_summary.plain
        assert "`" not in reading_summary.plain + editing_summary.plain
        assert "+1 -0" in editing_summary.plain
        assert "+2 -1" in editing_summary.plain
        assert {str(span.style) for span in editing_summary.spans} >= {
            "green",
            "red",
        }


async def test_event_summary_groups_follow_activity_order(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        for path in ("fileA", "fileB", "fileC"):
            await handler.on_custom_event(
                DEFAULT_EVENT_NAME,
                {"tool": "read_file", "path": path, "message": "Reading"},
            )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "path": "fileB",
                "message": "Editing",
                "additions": 2,
                "deletions": 1,
            },
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="ordered-command",
            inputs={"query": "pwd"},
        )
        await handler.on_tool_end("", run_id="ordered-command")
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await pilot.pause()
        events = list(app.query_one(Turn).query_one(".events").children)
        assert [type(event) for event in events] == [
            FileActivityCard,
            FileActivityCard,
            RunCommandCard,
        ]
        assert all(event.styles.margin.bottom == 1 for event in events)
        assert list(events[0].files["Reading"]) == [
            "fileA",
            "fileB",
            "fileC",
        ]
        assert list(events[1].files["Editing"]) == ["fileB"]


async def test_activity_kinds_keep_independent_cards_open(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        turn_module,
        "SUMMARY_GROUP_GRACE_SECONDS",
        1.0,
    )
    hitl = FakeHITL(tmp_path)
    reading_started = asyncio.Event()
    switch_to_editing = asyncio.Event()
    editing_started = asyncio.Event()
    return_to_reading = asyncio.Event()
    reading_updated = asyncio.Event()
    finish_agent = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        reading_started.set()
        await switch_to_editing.wait()
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "path": "fileB",
                "message": "Editing",
                "additions": 2,
                "deletions": 1,
            },
        )
        editing_started.set()
        await return_to_reading.wait()
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        reading_updated.set()
        await finish_agent.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await reading_started.wait()
        await pilot.pause()
        reading = list(app.query(FileActivityCard))[0]

        switch_to_editing.set()
        await editing_started.wait()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        editing = groups[1]
        assert not reading.done
        assert not editing.done

        await asyncio.sleep(0.6)
        return_to_reading.set()
        await reading_updated.wait()
        await pilot.pause()
        assert list(reading.files["Reading"]) == ["fileA", "fileC"]
        assert len(app.query(FileActivityCard)) == 2
        assert not reading.done
        assert not editing.done

        await asyncio.sleep(0.45)
        await pilot.pause()
        assert not reading.done
        assert editing.done

        await asyncio.sleep(0.6)
        await pilot.pause()
        assert reading.done

        finish_agent.set()
        await pilot.pause()


async def test_same_event_kind_stays_grouped_within_three_second_grace(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        clock[0] += 2.99
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileB", "message": "Reading"},
        )
        clock[0] += 2.99
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        assert list(groups[0].files["Reading"]) == [
            "fileA",
            "fileB",
            "fileC",
        ]


async def test_same_event_kind_starts_new_group_after_three_seconds(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(turn_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(event_handler_module, "monotonic", lambda: clock[0])
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        clock[0] += 3.001
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileB", "message": "Reading"},
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        assert list(groups[0].files["Reading"]) == ["fileA"]
        assert list(groups[1].files["Reading"]) == ["fileB"]


async def test_summary_card_mounts_immediately_updates_and_finalizes_after_idle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        turn_module,
        "SUMMARY_GROUP_GRACE_SECONDS",
        0.5,
    )
    hitl = FakeHITL(tmp_path)
    first_emitted = asyncio.Event()
    emit_second = asyncio.Event()
    second_emitted = asyncio.Event()
    emit_third = asyncio.Event()
    third_emitted = asyncio.Event()
    finish_agent = asyncio.Event()

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileA", "message": "Reading"},
        )
        first_emitted.set()
        await emit_second.wait()
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileB", "message": "Reading"},
        )
        second_emitted.set()
        await emit_third.wait()
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {"tool": "read_file", "path": "fileC", "message": "Reading"},
        )
        third_emitted.set()
        await finish_agent.wait()
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("g", "o", "enter")
        await first_emitted.wait()
        await pilot.pause()

        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        first = groups[0]
        assert list(first.files["Reading"]) == ["fileA"]
        assert not first.done

        # Stay comfortably inside the 500 ms production grace period even on
        # a busy test runner.
        await asyncio.sleep(0.1)
        emit_second.set()
        await second_emitted.wait()
        await pilot.pause()
        assert list(first.files["Reading"]) == ["fileA", "fileB"]
        assert not first.done

        await asyncio.sleep(0.3)
        await pilot.pause()
        assert not first.done
        await asyncio.sleep(0.25)
        await pilot.pause()
        assert first.done
        assert str(first.query_one(".event-card-done", Static).content) == "✓"

        emit_third.set()
        await third_emitted.wait()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 2
        assert groups[0] is first
        assert list(groups[1].files["Reading"]) == ["fileC"]
        assert not groups[1].done

        finish_agent.set()
        await pilot.pause()
        assert groups[1].done
        response = list(app.query(MessageCard))[-1]
        assert response.has_class("assistant")
        assert response.styles.margin.top == 1


async def test_parallel_read_callbacks_render_as_one_group(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        async def read(run_id, path):
            await handler.on_tool_start(
                {"name": "read_file"},
                "",
                run_id=run_id,
                inputs={"filename": path},
            )
            await handler.on_custom_event(
                DEFAULT_EVENT_NAME,
                {
                    "tool": "read_file",
                    "phase": "start",
                    "path": path,
                    "message": "Reading file",
                },
            )

        await asyncio.gather(
            read("read-a", "README.md"),
            read("read-b", "tests/cli/test_config.py"),
            read("read-c", "pyproject.toml"),
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-after-reads",
            inputs={"query": "pwd"},
        )
        await handler.on_tool_end("output", run_id="command-after-reads")
        # These delayed completions belong to the original Reading group and
        # must not create another Reading group after the command.
        for path in (
            "README.md",
            "tests/cli/test_config.py",
            "pyproject.toml",
        ):
            await handler.on_custom_event(
                DEFAULT_EVENT_NAME,
                {
                    "tool": "read_file",
                    "phase": "end",
                    "path": path,
                    "message": "File read",
                },
            )
        await asyncio.gather(
            handler.on_tool_end("a", run_id="read-a"),
            handler.on_tool_end("b", run_id="read-b"),
            handler.on_tool_end("c", run_id="read-c"),
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "e", "a", "d", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        groups = list(app.query(FileActivityCard))
        assert len(groups) == 1
        events = list(app.query_one(Turn).query_one(".events").children)
        assert [type(event) for event in events] == [
            FileActivityCard,
            RunCommandCard,
        ]
        assert list(groups[0].files["Reading"]) == [
            "README.md",
            "tests/cli/test_config.py",
            "pyproject.toml",
        ]


def test_long_command_preview_keeps_top_and_bottom_eight_lines():
    command = "\n".join(f"line {index}" for index in range(1, 26))

    preview = RunCommandCard._preview_command(command)

    assert preview.splitlines() == [
        *(f"line {index}" for index in range(1, 9)),
        "… 9 lines omitted …",
        *(f"line {index}" for index in range(18, 26)),
    ]


async def test_overlapping_commands_stay_compact_and_update_their_icons(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("run both", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        for command_id, query in (("one", "sleep 1"), ("two", "sleep 2")):
            await turn.event({
                "tool": "run_command",
                "phase": "start",
                "query": query,
                "_command_id": command_id,
            })
        await pilot.pause()

        cards = list(turn.query(RunCommandCard))
        assert len(cards) == 2
        assert all(card.multi_command for card in cards)
        assert all(
            not card.query_one(".command-compact").has_class("hidden")
            for card in cards
        )

        await turn.event({
            "tool": "run_command",
            "phase": "end",
            "query": "sleep 1",
            "_command_id": "one",
            "returncode": 0,
            "result": "",
        })
        await pilot.pause()
        assert (
            str(cards[0].query_one(".command-compact-state", Static).content)
            == "✓"
        )
        assert (
            str(cards[1].query_one(".command-compact-state", Static).content)
            in ActivityIndicator.FRAMES
        )


async def test_run_command_card_tracks_safety_and_collapses_on_result(tmp_path):
    hitl = FakeHITL(tmp_path)
    pass_safety = asyncio.Event()
    return_result = asyncio.Event()
    return_second = asyncio.Event()
    command = "\n".join(f"echo line-{index}" for index in range(1, 7))

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-1",
            inputs={"query": command},
        )
        await pass_safety.wait()
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "message": "Command passed safety check",
                "query": command,
                "safe": True,
            },
        )
        await return_result.wait()
        await handler.on_tool_end(
            "STDOUT:\ncommand output\nSTDERR:\n", run_id="command-1"
        )
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="command-2",
            inputs={"query": "echo second"},
        )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "message": "Command passed safety check",
                "query": "echo second",
                "safe": True,
            },
        )
        await return_second.wait()
        await handler.on_tool_end(
            "STDOUT:\nsecond output\nSTDERR:\n", run_id="command-2"
        )
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        card = app.query_one(RunCommandCard)
        source = card.query_one(".command-source", Static)
        safety = card.query_one(CommandSafetyIndicator)
        assert len(source.content.code.splitlines()) == 6
        assert (
            str(safety.query_one(".activity-text", Static).content)
            == "Running safety check"
        )

        pass_safety.set()
        await pilot.pause()
        assert str(safety.query_one(".activity-spinner", Static).content) == "✓"

        return_result.set()
        await pilot.pause()
        assert source.content.code == "echo line-1 …"
        output = card.query_one(".command-output", Static)
        assert output.content.code == "command output"
        cards = list(app.query(RunCommandCard))
        assert len(cards) == 2
        assert [item.command for item in cards] == [command, "echo second"]
        assert not output.has_class("hidden")
        assert card.query_one(".command-compact").has_class("hidden")

        return_second.set()
        await pilot.pause()
        assert output.has_class("hidden")
        assert not card.query_one(".command-compact").has_class("hidden")
        assert (
            str(card.query_one(".command-compact-state", Static).content) == "✓"
        )
        newest = cards[-1]
        assert newest.query_one(".command-compact").has_class("hidden")
        assert not newest.query_one(".command-output").has_class("hidden")

        await pilot.press("ctrl+o")
        assert not output.has_class("hidden")


async def test_single_command_output_preserves_top_and_bottom_until_expanded(
    tmp_path,
):
    hitl = FakeHITL(tmp_path)
    full_output = "\n".join(f"line {index}" for index in range(1, 31))

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_tool_start(
            {"name": "run_command"},
            "",
            run_id="long-output",
            inputs={"query": "generate output"},
        )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "run_command",
                "stage": "safety_check",
                "query": "generate output",
                "safe": True,
            },
        )
        await handler.on_tool_end(full_output, run_id="long-output")
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        output = app.query_one(".command-output", Static)
        preview = output.content.code
        assert len(preview.splitlines()) == 9
        assert "line 1\n" in preview
        assert "line 4\n… 22 lines omitted …\nline 27" in preview
        assert preview.endswith("line 30")

        await pilot.press("ctrl+o")
        assert output.content.code == full_output


async def test_collapsed_command_icons_reflect_outcome(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]

        async def command(
            run_id,
            query,
            *,
            safe=True,
            returncode=None,
            output="",
        ):
            await handler.on_tool_start(
                {"name": "run_command"},
                "",
                run_id=run_id,
                inputs={"query": query},
            )
            await handler.on_custom_event(
                DEFAULT_EVENT_NAME,
                {
                    "tool": "run_command",
                    "stage": "safety_check",
                    "query": query,
                    "safe": safe,
                    "reason": "Rejected" if not safe else "Allowed",
                },
            )
            if returncode is not None:
                await handler.on_custom_event(
                    DEFAULT_EVENT_NAME,
                    {
                        "tool": "run_command",
                        "stage": "execute",
                        "phase": "end",
                        "query": query,
                        "returncode": returncode,
                    },
                )
            await handler.on_tool_end(output, run_id=run_id)

        await command("empty", "true", returncode=0)
        await command("failed", "false", returncode=2, output="failed")
        await command("unsafe", "dangerous", safe=False, output="Rejected")
        return "Finished"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("r", "u", "n", "enter")
        await pilot.pause()
        cards = list(app.query(RunCommandCard))
        assert [
            str(card.query_one(".command-compact-state", Static).content)
            for card in cards
        ] == ["✓", "✗", "⚔️"]
        assert all(
            not card.query_one(".command-compact").has_class("hidden")
            for card in cards
        )
        assert (
            str(
                cards[-1]
                .query_one(CommandSafetyIndicator)
                .query_one(".activity-spinner", Static)
                .content
            )
            == "⚔️"
        )


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


async def test_welcome_banner_and_endpoint_status_are_visible(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.model = SimpleNamespace(
        model_name="test-model", base_url="https://llm.test/v1"
    )
    hitl.embedding = SimpleNamespace(
        model="embed-model", base_url="https://embed.test/v1"
    )
    hitl.group = "research"
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        banner = app.query_one(WelcomeBanner)
        snapshot = str(
            banner.query_one("#welcome-config-values", Static).content
        )
        workspace = banner.query_one("#welcome-workspace", Static)
        assert str(workspace.content).endswith(tmp_path.name[-12:])
        assert "test-model (https://llm.test/v1)" in snapshot
        assert "embed-model (https://embed.test/v1)" in snapshot
        assert "research" in snapshot
        assert "test-model (https://llm.test/v1)" in str(
            app.query_one("#status", Static).content
        )


async def test_welcome_tip_is_selected_once_from_tip_catalog(
    tmp_path, monkeypatch
):
    selected = TIPS[-1]
    monkeypatch.setattr(widgets_module, "random_tip", lambda: selected)
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)):
        banner = app.query_one(WelcomeBanner)
        assert banner.tip == selected
        assert str(app.query_one("#welcome-tip", Static).content) == (
            f"Tip: {selected}"
        )


async def test_welcome_version_and_workspace_align_at_narrow_width(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        logo = app.query_one("#welcome-logo")
        config = app.query_one("#welcome-config")
        art = app.query_one("#welcome-logo-art", Static)
        version = app.query_one("#welcome-version", Static)
        workspace_row = app.query_one("#welcome-workspace-row")
        label = app.query_one("#welcome-workspace-label", Static)
        workspace = app.query_one("#welcome-workspace", Static)

        assert version.styles.content_align_horizontal == "right"
        assert version.region.right == logo.content_region.right
        assert version.region.y == art.region.bottom
        assert len(str(version.content)) <= version.content_region.width
        assert workspace_row.has_class("workspace-stacked")
        assert workspace.styles.content_align_horizontal == "right"
        assert workspace.styles.text_overflow == "ellipsis"
        assert workspace.region.right == config.content_region.right
        assert workspace.region.y > label.region.y
        assert len(str(workspace.content)) <= workspace.content_region.width
        assert str(workspace.content).endswith(tmp_path.name[-12:])


async def test_workspace_uses_one_borderless_row_when_it_fits(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.workspace = Path("/tmp/ursa")
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        row = app.query_one("#welcome-workspace-row")
        label = app.query_one("#welcome-workspace-label", Static)
        workspace = app.query_one("#welcome-workspace", Static)
        values = app.query_one("#welcome-config-values", Static)

        assert row.has_class("workspace-inline")
        assert label.region.y == workspace.region.y
        assert values.region.y == row.region.bottom
        assert str(workspace.content) == str(Path("/tmp/ursa").resolve())


async def test_compact_terminal_renders_every_welcome_section(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)):
        screenshot = app.export_screenshot()
        for expected in (
            "Workspace",
            "LLM",
            "Embedding",
            "Group",
            "Tip:",
        ):
            assert expected in screenshot


async def test_slash_picker_opens_status_inside_textual(tmp_path):
    hitl = FakeHITL(tmp_path)
    hitl.config = SimpleNamespace(
        mcp_servers={
            "local": {"transport": "stdio", "command": "ursa-mcp"},
            "remote": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
            },
        }
    )
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/")
        await pilot.pause()
        assert isinstance(app.screen, HotlistScreen)
        hotlist = app.screen.query_one("#hotlist")
        options = app.screen.query_one("#hotlist-options")
        assert hotlist.region.width == 80
        assert options.region.height >= 3
        assert options.region.bottom <= hotlist.region.bottom
        screenshot = app.export_screenshot()
        assert all(
            choice in screenshot for choice in ("agents", "status", "keymap")
        )
        assert app.screen.candidates == [
            "agents — Configured agents, descriptions, and options",
            "status — Tokens, models, endpoints, group, and MCP servers",
            "keymap — Complete keyboard map",
        ]

        await pilot.press("s", "t", "a", "t", "u", "s", "enter")
        await pilot.pause()
        assert isinstance(app.screen, InformationScreen)
        assert "LLM endpoint" in app.screen.content
        assert "MCP servers" in app.screen.content
        assert "ursa-mcp" in app.screen.content
        assert "https://example.test/mcp" in app.screen.content


async def test_command_picker_prioritizes_command_name_over_description(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("/", "k", "e")
        await pilot.pause()

        assert isinstance(app.screen, HotlistScreen)
        assert app.screen.matches[0].startswith("keymap —")
        assert any(match.startswith("status —") for match in app.screen.matches)
        assert app.screen.query_one("#hotlist-options").highlighted == 0


def test_command_details_cover_agents_and_the_full_app_keymap(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))
    app.total_tokens = 1234

    agents = app._agents_markdown()
    status = app._status_markdown()
    keymap = app._keymap_markdown()

    assert "## #chat" in agents
    assert "A configured test agent." in agents
    assert "`mode`" in agents and "`test`" in agents
    assert "1,234" in status
    assert "test-model" in status
    for expected in (
        "Shift+Enter",
        "Ctrl+C",
        "Ctrl/Alt/Option+Left / Right",
        "Ctrl+X / Ctrl+V",
        "Picker Up / Down",
        "Ctrl+T",
        "Cmd+Up / Cmd+Down",
        "Info Q / Esc",
    ):
        assert expected in keymap


async def test_plan_card_tracks_drafting_review_and_revisions(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("make a plan", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        steps = [
            {
                "name": f"Step {index}",
                "description": (
                    "The quick brown fox jumped over the detailed implementation "
                    "notes and continued all the way to the lazy river."
                ),
            }
            for index in range(1, 8)
        ]
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate",
            "message": "Drafting plan",
        })
        await pilot.pause()
        plan = turn.query_one(PlanCard)
        source = str(plan.query_one(Markdown).source)
        assert "🗺️ Planning" in source
        assert "✍️ Drafting Plan" in source
        assert any(
            f"✍️ Drafting Plan{frame}" in source
            for frame in PlanCard.SPINNER_FRAMES
        )

        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate_result",
            "message": "Drafted plan",
            "steps": steps,
        })
        await pilot.pause()
        markdown = plan.query_one(Markdown)
        source = str(markdown.source)
        assert len(turn.query(PlanCard)) == 1
        assert "📄 **Initial Plan**" in source
        assert "1. Step 1" in source
        assert "2. Step 2" in source
        assert "… 3 middle steps hidden …" in source
        assert "\n\n _… 3 middle steps hidden …_\n\n" in source
        assert "6. Step 6" in source
        assert "7. Step 7" in source
        assert "_… truncated …_" in source
        assert any(
            f"📋 Reviewing{frame}" in source
            for frame in PlanCard.SPINNER_FRAMES
        )
        hint = plan.query_one(".plan-expand-hint", Static)
        assert not hint.has_class("hidden")
        assert hint.styles.content_align_horizontal == "right"
        assert hint.region.right == plan.content_region.right
        assert (
            sum(
                type(node).__name__ == "MarkdownOrderedList"
                for node in markdown.query("*")
            )
            == 2
        )
        assert all(
            node.region.height == 1
            for node in markdown.query("*")
            if type(node).__name__ == "MarkdownListItem"
        )
        assert markdown.virtual_size.height >= 10

        await pilot.resize_terminal(160, 36)
        await pilot.pause()
        wide_source = str(markdown.source)
        wide_first_step = next(
            line for line in wide_source.splitlines() if "1. Step 1" in line
        )
        assert "truncated" not in wide_first_step
        assert "lazy river" in wide_first_step

        await turn.event({
            "agent": "PlanningAgent",
            "stage": "reflect_result",
            "message": "Plan needs another pass",
            "approved": False,
            "reason": "Add a concrete validation step before implementation.",
        })
        collapsed_source = str(plan.query_one(Markdown).source)
        assert "❌ Plan needs another revision" in collapsed_source
        assert "concrete validation step" not in collapsed_source

        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate",
            "message": "Drafting plan",
        })
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate_result",
            "message": "Revised plan",
            "steps": steps[:4],
        })
        revised = list(turn.query(PlanCard))[-1]
        assert "📄 **Revised Plan**" in str(revised.query_one(Markdown).source)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "reflect",
            "message": "Reviewing plan",
        })
        assert "📋 Reviewing" in str(revised.query_one(Markdown).source)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "reflect_result",
            "message": "Plan approved",
            "approved": True,
        })
        await pilot.pause()

        plans = list(turn.query(PlanCard))
        assert len(plans) == 2
        assert not plans[0].expanded
        assert not plans[1].expanded
        assert "✅ 📋 Plan is complete" in str(
            plans[1].query_one(Markdown).source
        )

        await pilot.click(PlanCard)
        assert plans[0].expanded
        assert plans[0].query_one(".plan-expand-hint").has_class("hidden")
        assert "middle steps hidden" not in str(
            plans[0].query_one(Markdown).source
        )
        expanded_source = str(plans[0].query_one(Markdown).source)
        assert "**Revision feedback**" in expanded_source
        assert "> Add a concrete validation step before implementation." in (
            expanded_source
        )
        assert any(
            type(node).__name__ == "MarkdownBlockQuote"
            for node in plans[0].query_one(Markdown).query("*")
        )


async def test_agent_completion_stops_pending_plan_review_spinner(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "PlanningAgent",
                "stage": "generate_result",
                "message": "Drafted final plan",
                "steps": [{"name": "Finish", "description": "Ship it"}],
            },
        )
        return "Final plan"

    hitl.run_agent = run_agent
    app = UrsaTextualApp(hitl)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.press("p", "l", "a", "n", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        plan = app.query_one(PlanCard)
        source = str(plan.query_one(Markdown).source)
        assert plan.state == "complete"
        assert "✅ 📋 Plan is complete" in source
        assert "📋 Reviewing" not in source

        await asyncio.sleep(0.7)
        await pilot.pause()
        assert str(plan.query_one(Markdown).source) == source


async def test_file_failure_is_retained_on_its_activity_card(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("read it", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "tool": "read_file",
            "phase": "start",
            "path": "broken.txt",
        })
        await turn.event({
            "tool": "read_file",
            "phase": "error",
            "path": "broken.txt",
            "error": "permission denied",
        })
        await pilot.pause()

        card = turn.query_one(FileActivityCard)
        assert card.outcomes[("Reading", "broken.txt")] == (
            "failed",
            "permission denied",
        )
        assert "permission denied" in str(
            card.query_one(".file-summary", Static).content
        )


async def test_specialized_agent_events_and_artifacts_update_live(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("investigate", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "agent": "HypothesizerAgent",
            "stage": "generate",
            "message": "Generating hypotheses",
        })
        await turn.event({
            "agent": "HypothesizerAgent",
            "stage": "critique_result",
            "message": "Critiqued hypotheses",
            "preview": "The second hypothesis survives.",
        })
        await turn.event({
            "agent": "HypothesizerAgent",
            "stage": "finalize_result",
            "message": "Finalized hypotheses",
            "artifact": {
                "content": "# Final hypothesis",
                "mime_type": "text/markdown",
                "metadata": {"title": "Hypothesis"},
            },
        })
        await pilot.pause()

        agent_cards = list(turn.query(AgentEventCard))
        assert len(agent_cards) == 1
        assert agent_cards[0].lines == [
            "✨ Generating hypotheses",
            "🧪 Critiqued hypotheses",
            "⭐ Finalized hypotheses",
        ]
        assert agent_cards[0].details == ["The second hypothesis survives."]
        assert len(turn.query(ArtifactCard)) == 1


async def test_search_and_lammps_events_render_specialized_details(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("search then simulate", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "tool": "run_web_search",
            "stage": "search_result",
            "phase": "end",
            "message": "Web search complete",
            "query": "ursa events",
            "result_chars": 2048,
        })
        await turn.event({
            "agent": "LammpsAgent",
            "stage": "choose_potential",
            "phase": "end",
            "message": "Selected potential",
            "potential_id": "Ni_u3.eam",
            "chosen_index": 2,
            "rationale": "Best match for nickel.",
            "output_path": "runs/ni",
        })
        await pilot.pause()

        search = turn.query_one(SearchEventCard)
        assert search.lines == ["✓ Web search complete: ursa events"]
        assert search.details == ["2,048 result characters"]
        lammps = turn.query_one(AgentEventCard)
        assert lammps.lines == ["🧲 Selected potential"]
        assert "Ni_u3.eam" in lammps.details[0]
        assert "Best match for nickel." in lammps.details[0]
        assert "Output: runs/ni" in lammps.details[0]


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
