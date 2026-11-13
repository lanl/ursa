# Persisted TUI cards

The TUI converts callbacks to versioned JSON presentation state before returning
from the callback. The journal lives in the session agent's
`db/checkpointer.db`, alongside its LangGraph checkpoint. Named agents keep their
transcript across restarts; unnamed agents use the existing temporary-agent
lifecycle. Threads within an agent have separate transcripts.

`event_cards` is an append-only journal with an `INTEGER PRIMARY KEY
AUTOINCREMENT` ID. Its other columns describe ordering, session, turn, and record
state; card-specific content stays in `card_json`. No pickle or widget internals
are persisted. `event_card_heads` is a transactional projection from stable
conversation positions to the latest journal record. `event_card_links` provides
indexed correlation for late tool results. `event_card_state` holds the small
set of UI preferences and resume state; `event_card_view_state` keeps individual
card expansion preferences without retaining those cards in memory.

Magnetic file/progress groups persist each individual card immediately, then a
bounded aggregate snapshot. A final `commit` record identifies its member IDs
and carries its final renderable state. Groups close after the existing idle
grace period, at turn completion, or after 32 members. Command commits identify
their journal position and final member ID, so an arbitrarily long command stream
can use an indexed range without a growing list of member IDs. A late result
updates the original position even if that card is outside the viewport.

`content_ref` is an optional opaque JSON reference (for example,
`{"kind":"terminal","id":"..."}`). A referenced command buffer is not copied
into the card journal. The renderer for that resource owns lookup and live
state; this branch does not introduce a terminal backend. Existing ordinary
command output is saved as static content.

The viewport estimates card height from its mounted window and retains roughly
five screens of cards. It uses indexed ID cursors, including when navigating
backward, jumping to either end, or resizing. It does not use `OFFSET`, count
all cards, replay the journal, or build invisible placeholder widgets for the
rest of the transcript. The scrollbar describes the loaded window; Home/End and
turn-marker navigation can fetch other parts of the transcript. Updates are
coalesced at 30 Hz in a widget-owned worker, so mounting a window does not hold
the app's input message queue. SQLite I/O and event conversion run outside the
UI thread. The editor's quick prompt recall is capped
at 100 entries; the transcript itself is retained on disk.

Rendering/layout cost and retained widgets are independent of total history:
O(W) for the mounted window, with additional event processing proportional to
currently active groups/commands. SQLite cursor seeks are O(log N + W), rather
than claiming constant-time database access. Reading a card's own content still
costs proportional to its size. Storage grows with the journal; no automatic
history deletion is performed.

Resume loads only the current window and restores token totals, individual and
global detail expansion, theme, reading position, and recent prompt recall. Unfinished command,
tool, and plan cards from an interrupted process display an interrupted state;
opening history does not re-execute tools. Conversations created before this
feature have no card journal and are not backfilled from agent checkpoints.
Clearing the view records a position boundary rather than deleting agent history.

## Validation and benchmark

Run the focused tests with:

```sh
uv run pytest -n 0 tests/cli/tui tests/cli/test_runtime.py
```

The scaling regression test measures SQLite VM instructions for windows at the
start, middle, and end of a 100,000-card journal, so timing noise cannot hide a
history scan. UI tests exercise large histories, long single turns, resize,
paging, concurrent callback correlation, restart, interrupted commands, and
preserving a reading position during new activity.

Run the benchmark with:

```sh
uv run python -m scripts.benchmarks.tui_cards
```

The benchmark seeds 1,000 through 1,000,000 persisted cards, then measures actual
Textual layout and full-screen composition at 100 × 36, database window reads,
and key-post-to-editor-change latency. At the largest history, it also emits 200
new tool events while measuring input latency and event-loop lag. It measures
the original retained widget approach using the same event cards. Fixture generation, startup, and
mounting are excluded from frame timings. Results are machine-specific; raw
measurements and environment details are in [tui-cards.json](tui-cards.json).

## Recorded results

Linux aarch64, Python 3.11.16, 100 × 36 terminal; 100 frame samples.

| Stored cards | Mounted cards | Layout + render median | Layout + render p95 | Input p95 |
|---:|---:|---:|---:|---:|
| 1,000 | 40 | 4.69 ms | 5.43 ms | 9.55 ms |
| 10,000 | 40 | 4.64 ms | 5.37 ms | 9.48 ms |
| 100,000 | 40 | 4.74 ms | 5.46 ms | 9.07 ms |
| 1,000,000 | 40 | 4.73 ms | 5.53 ms | 13.25 ms |

The retained-widget baseline with 1,000 cards measured 102.34 ms median layout/render time and 904.53 ms p95 input latency.

During a 200-event burst on top of 1,000,000 stored cards, p95 input latency was 7.69 ms and p95 event-loop lag was 28.30 ms. The window held 73 tool cards as their measured heights changed.

The complete TUI/runtime suite passed 241 tests. The final viewport lifecycle and app checks passed 31 tests. Ruff lint/format checks and `git diff --check` passed.
