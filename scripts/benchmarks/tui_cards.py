"""Benchmark the real Textual layout/compositor against a growing card journal.

Run from the repository: uv run python -m scripts.benchmarks.tui_cards
No agents or network calls. Fixture setup is excluded from measurements.
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

from textual import on
from textual.events import Key
from textual.widgets import TextArea

from tests.cli._app_fakes import FakeHITL
from tests.cli.tui.test_card_store import seed_cards
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.event_cards import EventCard
from ursa.cli.tui.event_cards.card_store import CardStore
from ursa.cli.tui.event_cards.card_stream import CardStream
from ursa.cli.tui.turn import Turn
from ursa.cli.tui.widgets import MessageCard, PromptArea


class BenchmarkApp(UrsaTextualApp):
    def __init__(self, hitl):
        super().__init__(hitl)
        self.input_changed = asyncio.Event()

    @on(TextArea.Changed)
    def input_measured(self):
        self.input_changed.set()


def summarize(samples):
    ordered = sorted(samples)
    return {
        "p50_ms": round(statistics.median(samples) * 1000, 3),
        "p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)] * 1000, 3),
    }


def frame(app):
    # Measure actual layout and composition, not merely the SQLite query.
    app.screen._compositor.reflow(app.screen, app.screen.size)
    app.screen._compositor.render_full_update()


async def measure(app, pilot, samples):
    await pilot.pause()
    for _ in range(5):
        frame(app)
    frames = []
    for _ in range(samples):
        start = perf_counter()
        frame(app)
        frames.append(perf_counter() - start)
    keys = []
    prompt = app.query_one(PromptArea)
    prompt.focus()
    for _ in range(20):
        await pilot.pause()
        app.input_changed.clear()
        start = perf_counter()
        prompt.post_message(Key("x", "x"))
        await asyncio.wait_for(app.input_changed.wait(), 5)
        keys.append(perf_counter() - start)
    return {
        "mounted_cards": len(app.query(EventCard)),
        "dom_nodes": len(app.query("*")),
        "layout_and_render": summarize(frames),
        "key_to_input_changed": summarize(keys),
    }


async def measure_live(app, pilot, workspace):
    stream = CardStream(app.card_store, workspace, "live benchmark")
    await stream.start()
    app.current_turn = stream
    await app.card_viewport.end()
    await pilot.pause()
    event_count = 200
    lag = []
    keys = []

    async def produce():
        for i in range(event_count):
            await app.add_turn_event(
                stream,
                {
                    "tool": "benchmark",
                    "_run_id": str(i),
                    "phase": "end",
                    "result": {"item": i},
                },
            )
        await stream.finish("finished")
        await app.card_viewport.refresh()

    async def heartbeat(writer):
        while not writer.done():
            start = perf_counter()
            await asyncio.sleep(0.005)
            lag.append(max(0, perf_counter() - start - 0.005))

    async def type_during_stream(writer):
        prompt = app.query_one(PromptArea)
        while not writer.done():
            app.input_changed.clear()
            start = perf_counter()
            prompt.post_message(Key("x", "x"))
            await asyncio.wait_for(app.input_changed.wait(), 5)
            keys.append(perf_counter() - start)
            await asyncio.sleep(0.01)

    start = perf_counter()
    writer = asyncio.create_task(produce())
    await asyncio.gather(writer, heartbeat(writer), type_during_stream(writer))
    return {
        "events": event_count,
        "elapsed_s": round(perf_counter() - start, 3),
        "event_loop_lag": summarize(lag),
        "input_during_events": summarize(keys),
        "mounted_cards": len(app.query(EventCard)),
    }


async def main(args):
    results = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "terminal": [100, 36],
        "samples": args.samples,
        "persisted": [],
    }
    with tempfile.TemporaryDirectory(prefix="ursa-tui-benchmark-") as directory:
        workspace = Path(directory)
        path = workspace / "session.db"
        previous = 0
        for count in args.sizes:
            store = CardStore(path)
            seed_cards(store, count - previous, start=previous + 1)
            previous = count
            query_samples = []
            for _ in range(args.samples):
                start = perf_counter()
                store.window(50)
                query_samples.append(perf_counter() - start)
            store.close()
            hitl = FakeHITL(workspace)
            hitl.card_database_path = path
            app = BenchmarkApp(hitl)
            async with app.run_test(size=(100, 36)) as pilot:
                result = await measure(app, pilot, args.samples)
                result.update(
                    cards_in_database=count,
                    window_read=summarize(query_samples),
                )
                results["persisted"].append(result)
                if count == args.sizes[-1]:
                    results["live_at_largest_history"] = await measure_live(
                        app, pilot, workspace
                    )
            print(json.dumps(result), flush=True)
        # Retained-widget baseline mounts the full original layout.
        # Identical card contents and terminal; no viewport pruning.
        if args.baseline:
            app = BenchmarkApp(FakeHITL(workspace))
            async with app.run_test(size=(100, 36)) as pilot:
                turn = Turn()
                await app.query_one("#conversation").mount(turn)
                await turn.mount(MessageCard("user", "baseline"), before=0)
                cards = []
                for i in range(args.baseline):
                    card = EventCard(f"card:{i}", f"Event {i}")
                    card.add(f"Item {i}")
                    card.mark_done()
                    cards.append(card)
                events = turn.query_one(".events")
                events.remove_class("hidden")
                await events.mount(*cards)
                for card in cards:
                    card.refresh_content()
                await pilot.pause()
                result = await measure(app, pilot, args.samples)
                result["cards_in_conversation"] = args.baseline
                results["retained_widget_baseline"] = result
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1000, 10000, 100000, 1000000]
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--baseline", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks/tui-cards.json")
    )
    asyncio.run(main(parser.parse_args()))
