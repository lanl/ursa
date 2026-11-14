# Async messages and subagents

Every `BaseAgent` has a serialized asynchronous inbox and can own a hierarchy of
concurrently running subagents. `ursa.messaging.AsyncMessageMixin` manages inbox
and hierarchy lifecycles, validates limits, and serializes async invocations
while allowing reentrant calls. Shared rules live in `ursa.messaging.coordination`.

The host can place agents in dedicated processes, local threads, or a shared
local/MPI/Slurm worker pool; see [Runtime workers](runtime.md).

## User messages

Use `request` to await a result, `post_message` for one-way delivery, and
`wait_for_messages` to drain the inbox:

```python
result = await agent.request("Analyze the latest measurements")
await agent.post_message("Also check the calibration notes")
await agent.wait_for_messages()
```

User and agent messages share a FIFO queue. Arrivals wait while the agent is
busy; they do not cancel model or tool calls. On delivery, the receiving model
gets a trusted system envelope naming the sender and recipient, followed by the
exact, unchanged human message. A message can also reach a running `wait_for`,
as described below.

## Dynamic hierarchies

Construct children on demand or attach existing agents:

```python
researcher = lead.spawn_subagent("researcher", ResearchAgent)
reviewer = lead.attach_subagent("reviewer", reviewer_agent)
specialist = researcher.spawn_subagent("specialist", SpecialistAgent)
```

Constructed children inherit the parent's model, workspace, and group unless
explicitly overridden. Use `await lead.aspawn_subagent(...)` for async factories.
The top-level `subagents` configuration defaults to 32 subagents and three levels.
Either `max_agents: 0` or `max_depth: 0` disables creation; `null` removes that
limit. Optional `llm_model` and `embedding_model` settings apply to all descendants;
`null` inherits the root model. Reaching a limit produces a recoverable
`spawn_agent` tool error, allowing the parent to continue.

Slash-delimited paths start at `root` and can address children, parents, or siblings:

```python
await lead.send_message("researcher", "Gather supporting evidence")
review = await researcher.ask_agent("reviewer", "Check these conclusions")
root = specialist.resolve_agent("../..")
assert reviewer.agent_path == "root/reviewer"
```

Every `ChatAgent` exposes these five tools to its model:

- `spawn_agent(name, message, agent_type=None)` creates a child at the caller's
  path plus `name`, queues its initial task, and returns without waiting for its
  reply. In `AgentRuntime`, the host validates an optional configured URSA
  `agent_type`; omission inherits the caller's type. Use `wait_for` to synchronize.
- `send_message(target, message)` queues one-way delivery to a path, group, or
  mixed list. The whole list is expanded and validated before delivery; each
  concrete recipient is acknowledged once. Replies require an explicit
  `send_message`: ordinary final responses are not automatically routed back.
- `wait_for(target, timeout_seconds=120, wait_all=False)` accepts one path or a
  non-empty list and waits at most two minutes. A list returns when any target
  is ready; `wait_all=True` requires all targets. Timeouts are normal tool results.
  A working target counts as ready only if the wait would close a dependency
  cycle, preventing reciprocal deadlocks without releasing unrelated waits.
- `stop_agent(target)` marks a descendant subtree as stopping and returns
  immediately. The default process backend hard-stops its processes in the
  background without waiting for responses or completion; threaded backends
  cancel async work as described in [Runtime workers](runtime.md).
  New messages and spawns into that subtree
  are rejected. Agents cannot stop themselves, ancestors, siblings, or unrelated
  agents.
- `list_agents()` lists the hierarchy and marks the caller.

### Group targets

`send_message` uses supervisor/peer/report terminology:

| Target | Recipients |
|---|---|
| `@everyone` | Every other agent in the caller's hierarchy |
| `@supervisor` | The agent that directly launched the caller |
| `@peers` | Other agents launched by the same supervisor |
| `@reports` | Agents directly launched by the caller, excluding deeper descendants |

For example, `send_message(target="@reports", message="Report your progress.")`
addresses immediate reports. A mixed list such as `["@peers", "root/reviewer"]`
deduplicates recipients in first-occurrence order. Groups are case-insensitive;
agent paths remain case-sensitive. Group broadcasts always exclude the sender,
even when its explicit path appears in the list.

Every group must contain another agent. An empty or unknown group, invalid path,
or stopping recipient rejects the entire send, including lists with otherwise
valid paths. These are recoverable tool errors. The message log records concrete
recipients. Groups apply only to `send_message`; `wait_for` and `stop_agent` retain
path-based targets.

### Waiting and parallel work

Agents can coordinate directly with peers without routing through the root.
Chat prompts encourage independent spawns, different messages, and identical
announcements to be batched in one model response. Use a target list for an
identical announcement and one list-target `wait_for` to synchronize several
agents. Dependent actions and conflicting writes stay sequential. Tools use
normal `bind_tools` without overriding provider settings. OpenAI allows
[parallel function calls by default](https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling);
the model chooses the calls and LangGraph executes each batch.

A message received during `wait_for` ends the wait immediately and returns its
sender, recipient, and exact content. That delivery is consumed, never replayed
as another turn. The host also buffers arrivals to busy agents before they enter
`wait_for`; a managed wait returns all messages already buffered at its start.
If the invocation finishes instead, those messages arrive together in the next
invocation, each with its own trusted envelope. The model can consider newer
instructions alongside earlier discussion without one turn per queued message.

Concurrent waits share an inbox: a message may come from any sender, and another
wait may only report that it was awakened. Count replies by their actual sender
and task identifier, not by the wait target or completion flag.

## Managed runtime

`AgentRuntime`, including the CLI and TUI, registers each dynamic subagent with
its manager and runs it in a supervised process. Independent starts run
concurrently; the host reserves names and capacity first, releases reservations
on failure or cancellation, and reports pending launches as `Starting`.

The root discovers MCP tools once. Children with identical resolved server
configuration reuse those schemas and metadata, opening their own tool sessions.
Root restart or reconfiguration invalidates the cache; initial root startup still
includes discovery. The host owns limits, routing, status, shutdown, and secret
resolution before configuration crosses process boundaries. Models and agent
state stay inside their processes. With durable `agent_name` persistence, all
members share the root checkpoint database and use distinct hierarchy-derived
thread IDs.

Direct library hierarchies run in-process. Their application owner should call
`await lead.stop_message_loop(descendants=True)` to shut them down.

## TUI controls

- `/subagents` switches the message target and conversation to any active agent,
  including `root`, and shows its live progress.
- `/coordination` opens **Messages** and **Subagents** tabs. The live tree shows
  per-agent tokens, state, and latest activity. **Enter** switches to an agent;
  **Space** expands or collapses its details. States include Starting, Working,
  Waiting, Stopping, Queued, and Ready. `/status` totals usage across all agents.
- In the editor, **Enter** queues a message without stopping current work. It
  stays dimmed above the editor until accepted. Accepted follow-ups appear before
  subsequent replies, including when delivered through `wait_for` without a
  separate model turn.

The read-only message log includes initial `spawn_agent` tasks and explicit
replies. Cards show sender, recipients, stable agent colors, and live relative
timestamps. Bodies collapse to one line; **Tab** focuses a card and **Enter**,
**Space**, or a click expands it. Shared footer hints describe these controls.

Messages live in the `hierarchy_messages` SQLite table, scoped to the root thread:
`db/checkpointer.db` for named agents or the workspace's `graph_store.sqlite`
otherwise. The UI loads at most 50 previews, fetching full bodies on expansion
and releasing them on collapse. **Older**/**Newer** or left/right keys page through
history; **Home** goes to the beginning. Scrolling back, expanding a card, or
reading history pauses live following so incoming traffic cannot move the page.
The footer shows Live, Paused, or newer messages available. **Latest** or **End**
resumes live traffic.

Within each turn, `send_message`, `wait_for`, `spawn_agent`, and `stop_agent` share
one **Coordination** card. It shows action, running, and failure counts plus two
recent actions. Expansion reveals individual tool cards, with 20 actions per
page; incoming actions do not displace a historical page. Full inputs and results
live in `coordination_actions`, load only when an individual card expands, and
are released when that card or its group collapses. Failed actions have an
explicit error state. **Shift+Tab** from the editor reaches cards; **Enter** or
**Space** toggles them. `/keymap` lists every shortcut.

**Esc** closes pickers and dialogs; it neither submits messages nor cancels work.
Interrupt-and-send support remains deferred pending follow-up work based on
[PR #404](https://github.com/lanl/ursa/pull/404).
