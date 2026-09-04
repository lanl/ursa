from textual.containers import VerticalScroll
from textual.widgets import Static

from tests.cli._app_fakes import FakeHITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import (
    AgentEventCard,
    ArtifactCard,
    EditCard,
    SearchEventCard,
)
from ursa.cli.tui.turn import Turn


async def test_multiple_edit_rows_expand_independently_under_one_heading(
    tmp_path,
):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 60)) as pilot:
        turn = Turn("edit two files", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        for event in (
            {
                "tool": "edit_code",
                "path": "one.py",
                "old_code": "value = 1",
                "new_code": "value = 2",
            },
            {
                "tool": "write_code",
                "path": "two.py",
                "code": "value = 2",
            },
        ):
            await turn.event(event)
        await pilot.pause()

        first, second = turn.query(EditCard)
        assert len(turn.query(".edit-group-title")) == 1
        assert "one.py" in str(first.query_one(".edit-title", Static).content)
        assert "two.py" in str(second.query_one(".edit-title", Static).content)
        first.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(first.query_one(".edit-header"))
        assert not first.query_one(".edit-diff").has_class("hidden")
        assert second.query_one(".edit-diff").has_class("hidden")
        expanded = first.query_one(".edit-diff", Static).content.code
        assert "-value = 1" in expanded
        assert "+value = 2" in expanded


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
        assert len(agent_cards[0].lines) == 3
        assert agent_cards[0].details == ["The second hypothesis survives."]
        assert len(turn.query(ArtifactCard)) == 1

        artifact = turn.query_one(ArtifactCard)
        for card in turn.query(".event-card"):
            assert (
                str(card.query_one(".event-expand-hint", Static).content)
                == "Click to expand"
            )
        artifact.mark_done()

        class Click:
            stopped = False

            def stop(self):
                self.stopped = True

        click = Click()
        artifact.on_click(click)
        assert click.stopped
        assert artifact.done
        assert artifact.expanded
        assert (
            str(artifact.query_one(".event-expand-hint", Static).content)
            == "Click to collapse"
        )


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
        assert len(search.lines) == 1
        assert "ursa events" in search.lines[0]
        assert search.details == ["2,048 result characters"]
        lammps = turn.query_one(AgentEventCard)
        assert len(lammps.lines) == 1
        assert "Ni_u3.eam" in lammps.details[0]
        assert "Best match for nickel." in lammps.details[0]
        assert "Output: runs/ni" in lammps.details[0]
