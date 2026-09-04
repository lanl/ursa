import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from tests.cli._app_fakes import FakeHITL, emit_event, wait_for
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import PlanCard
from ursa.cli.tui.turn import Turn


def plan_steps(count=7):
    return [
        {
            "name": f"Step {index}",
            "description": (
                "The quick brown fox jumped over the detailed implementation "
                "notes and continued all the way to the lazy river."
            ),
        }
        for index in range(1, count + 1)
    ]


async def test_plan_card_renders_drafting_and_collapsed_steps(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("make a plan", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate",
            "message": "Drafting plan",
        })
        await pilot.pause()
        plan = turn.query_one(PlanCard)
        assert plan.state == "drafting"
        assert plan.steps == []

        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate_result",
            "message": "Drafted plan",
            "steps": plan_steps(),
        })
        await pilot.pause()
        markdown = plan.query_one(Markdown)
        source = str(markdown.source)
        assert len(turn.query(PlanCard)) == 1
        assert plan.revision == 1
        assert plan.state == "reviewing"
        assert len(plan.steps) == 7
        assert "1. Step 1" in source
        assert "2. Step 2" in source
        assert "… 3 middle steps hidden …" in source
        assert "6. Step 6" in source
        assert "7. Step 7" in source
        assert "_… truncated …_" in source
        hint = plan.query_one(".event-expand-hint", Static)
        assert str(hint.content) == "Click to expand"
        assert all(
            node.region.height == 1
            for node in markdown.query("*")
            if type(node).__name__ == "MarkdownListItem"
        )

        await pilot.resize_terminal(160, 36)
        await pilot.pause()
        wide_first_step = next(
            line
            for line in str(markdown.source).splitlines()
            if "1. Step 1" in line
        )
        assert "truncated" not in wide_first_step
        assert "lazy river" in wide_first_step


async def test_plan_card_tracks_revisions_approval_and_expansion(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("make a plan", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate_result",
            "message": "Drafted plan",
            "steps": plan_steps(),
        })
        plan = turn.query_one(PlanCard)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "reflect_result",
            "message": "Plan needs another pass",
            "approved": False,
            "reason": "Add a concrete validation step before implementation.",
        })
        collapsed_source = str(plan.query_one(Markdown).source)
        assert plan.state == "revision_needed"
        assert plan.review_reason == (
            "Add a concrete validation step before implementation."
        )
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
            "steps": plan_steps(4),
        })
        revised = list(turn.query(PlanCard))[-1]
        assert revised.revision == 2
        assert len(revised.steps) == 4
        assert revised.state == "reviewing"
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "reflect",
            "message": "Reviewing plan",
        })
        assert revised.state == "reviewing"
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
        assert plans[1].state == "complete"

        await pilot.press("ctrl+o")
        assert all(plan.expanded for plan in plans)
        assert (
            str(plans[0].query_one(".event-expand-hint", Static).content)
            == "Click to collapse"
        )
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

        await pilot.press("ctrl+o")
        assert all(not plan.expanded for plan in plans)
        assert (
            str(plans[0].query_one(".event-expand-hint", Static).content)
            == "Click to expand"
        )


async def test_agent_completion_stops_pending_plan_review_spinner(tmp_path):
    hitl = FakeHITL(tmp_path)

    async def run_agent(_name, _prompt, callbacks=None):
        handler = callbacks[0]
        await emit_event(
            handler,
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
        assert await wait_for(pilot, lambda: plan.state == "complete")
        frame = plan._frame

        await asyncio.sleep(0.7)
        await pilot.pause()
        assert plan._frame == frame


async def test_failed_agent_stops_drafting_plan_spinner(tmp_path):
    app = UrsaTextualApp(FakeHITL(tmp_path))

    async with app.run_test(size=(100, 36)) as pilot:
        turn = Turn("make a plan", tmp_path)
        await app.query_one("#conversation", VerticalScroll).mount(turn)
        await turn.event({
            "agent": "PlanningAgent",
            "stage": "generate",
            "message": "Drafting plan",
        })
        plan = turn.query_one(PlanCard)

        turn.finish_activity(succeeded=False)
        assert await wait_for(pilot, lambda: plan.state == "revision_needed")
        assert "draft completed" in plan.review_reason
        source = str(plan.query_one(Markdown).source)
        assert "Plan drafting failed" in source
        assert "Drafting Plan" not in source
        await asyncio.sleep(0.7)
        await pilot.pause()
        assert str(plan.query_one(Markdown).source) == source
