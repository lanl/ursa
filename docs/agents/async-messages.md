# Async messages and subagents

Every `BaseAgent` has an asynchronous inbox and can own a dynamic hierarchy of
subagents. Inbox processing is serialized per agent, while different agents may
run concurrently. The inbox and hierarchy lifecycle live in
`ursa.messaging.AsyncMessageMixin`; shared coordination rules live in
`ursa.messaging.coordination`. `BaseAgent` uses the mixin to validate subagent
limits and serialize async invocations, including reentrant calls.

## User messages

Use `request` when the caller needs the result, or `post_message` for one-way
delivery:

```python
result = await agent.request("Analyze the latest measurements")
await agent.post_message("Also check the calibration notes")
await agent.wait_for_messages()
```

A message arriving while that agent is already running waits in its inbox.
Messages do not cancel ongoing model or tool calls.

## Dynamic hierarchies

Attach an existing agent or construct one on demand:

```python
researcher = lead.spawn_subagent("researcher", ResearchAgent)
reviewer = lead.attach_subagent("reviewer", reviewer_agent)
specialist = researcher.spawn_subagent("specialist", SpecialistAgent)
```

Constructed children inherit the parent's model, workspace, and group unless
those arguments are supplied explicitly. Async factories use
`await lead.aspawn_subagent(...)`.

The top-level `subagents` configuration limits the hierarchy to 32 agents and
three levels by default. Setting either `max_agents` or `max_depth` to `0`
disables subagent creation; setting one to `null` removes that limit. Optional
`llm_model` and `embedding_model` entries
select models for every child and descendant; `null` inherits the root model.
If a model calls `spawn_agent` after reaching a configured limit, the tool
returns a recoverable error to that model; the parent agent continues running.

Paths are slash-delimited and the hierarchy root is always named `root`.
Agents can address children, parents, and siblings:

```python
await lead.send_message("researcher", "Gather supporting evidence")
review = await researcher.ask_agent("reviewer", "Check these conclusions")
root = specialist.resolve_agent("../..")
assert reviewer.agent_path == "root/reviewer"
```

Every `ChatAgent` also exposes five hierarchy tools to its model:

- `spawn_agent(name, message, agent_type=None)` creates a child whose path is
  the caller's path joined with `name`, immediately sends its initial task,
  and returns as soon as that task is queued. In `AgentRuntime`, `agent_type`
  selects a configured URSA agent implementation; omitting it inherits the
  caller's type. The host validates the requested type before launch. Use
  `wait_for` when the caller needs to synchronize with the child's completion.
- `send_message(target, message)` queues a one-way message and returns without
  waiting for the receiving agent to reply. Agents send replies explicitly
  with their own `send_message` calls. `target` accepts paths, message groups,
  or a list mixing both. URSA expands and validates the entire list before
  delivering anything, then acknowledges each concrete recipient once.
- `wait_for(target, timeout_seconds=120, wait_all=False)` waits for one
  agent, or a non-empty list of agents, for at most two minutes. A list returns
  when any target is ready by default; pass `wait_all=True` to wait for every
  target. Prefer a single list-target wait when synchronizing several agents,
  rather than several independent waits. A timeout is returned as a normal
  tool result. A still-working target is considered ready only when that wait
  would close a dependency cycle, so reciprocal waits cannot deadlock while
  unrelated agents still wait for real work to finish. If the waiting agent
  receives a new agent-to-agent message, the wait ends immediately and returns
  its sender,
  recipient, and exact content to the waiting model. That delivery is consumed
  by the wait and is not replayed as a duplicate turn. Replies that reach the
  host immediately before the model enters `wait_for` are buffered for that
  wait instead of starting a second activity turn for the busy recipient.
  A managed wait returns all messages already buffered when it starts. If an
  agent finishes its invocation instead, those messages are delivered together
  in its next invocation, retaining a separate trusted sender/recipient envelope
  for each message. This avoids one model turn per queued peer message and lets
  the model consider newer instructions alongside earlier discussion.
- `stop_agent(target)` atomically marks a child, grandchild, or deeper subtree
  as stopping and returns immediately. The host then hard-stops every process
  in that subtree in the background, without waiting for agents to finish or
  respond. New messages and spawns into a stopping subtree are rejected. An
  agent cannot stop itself, an ancestor, a sibling, or an unrelated agent.
- `list_agents()` lists the current hierarchy and marks the calling agent.

`send_message` supports these group targets, using supervisor/peer/report
terminology for the delegation hierarchy:

| Target | Recipients |
|---|---|
| `@everyone` | Every other agent in the caller's hierarchy |
| `@supervisor` | The agent that directly launched the caller |
| `@peers` | Other agents launched by the same supervisor |
| `@reports` | Agents directly launched by the caller, excluding deeper descendants |

For example, `send_message(target="@reports", message="Report your progress.")`
sends to the caller's immediate reports. A mixed target list such as
`["@peers", "root/reviewer"]` sends only once to each resolved agent, preserving
first-occurrence order. Group names are case-insensitive; agent paths retain
their normal case-sensitive spelling.

Group expansions always exclude the sender, including an explicit sender path
mixed into a group broadcast. Each group must resolve to at
least one other agent: an empty group, unknown group, invalid path, or stopping
recipient rejects the whole send before delivery. This also applies when an
empty group is mixed with valid paths. These are recoverable tool errors, so
the agent can revise the target and continue. The message log records the
concrete recipients. Group targets are specific to `send_message`; `wait_for`
and `stop_agent` continue to accept their existing path-based targets.

Multiple `send_message` tool calls targeting different agents can be queued in
one tool-calling round while the sending agent continues its own work. Chat
agents are instructed to batch independent spawns, different messages, and
identical group announcements in the same model response. Use one list-target
`wait_for` call to synchronize several outstanding agents. Dependent actions
and conflicting writes remain sequential. Tools use the model's normal
`bind_tools` method without overriding provider settings. OpenAI allows
[parallel function calls by default](https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling);
the model chooses which calls to emit, and LangGraph executes each batch.

Peers can send directly to siblings or broadcast to explicit target lists;
communication does not have to pass through the root. Exclude the sender from
group recipients and explicitly request `send_message` replies. An ordinary
agent final response is not automatically routed to its caller. Concurrent
waits share the caller's inbox: a returned message may come from any sender,
and another wait may only report that a message woke it. Count replies by the
returned sender and task identifier, not by the wait target or completion flag.

When agents run through `AgentRuntime` (including the CLI and TUI), each
dynamically spawned subagent is registered with the runtime manager and runs in
its own supervised process. Independent subprocess starts run concurrently.
The host reserves names and capacity before starting them, releases reservations
on failure or cancellation, and reports unfinished startup as `Starting`.
The root discovers MCP tools once; children with the same resolved server
configuration reuse those tool schemas and metadata, opening their own sessions
when invoking tools. Restarting the root or reconfiguring invalidates that cache.
The first root startup still includes MCP discovery.
The host owns hierarchy limits, routing, status,
and shutdown, and resolves secrets before configuration crosses a process
boundary; model objects and agent state remain inside their processes.
When durable persistence is enabled with `agent_name`, all members share the
root agent's checkpoint database while using distinct hierarchy-derived thread
IDs.
Direct library-created hierarchies retain the lightweight in-process behavior
shown above.

Call `await lead.stop_message_loop(descendants=True)` when an application owns
the hierarchy outside a managed URSA runtime.

## TUI controls

The Textual interface exposes `/subagents` once an agent hierarchy is active.
Choose any listed path to make it the target of subsequent messages and show
that agent's live progress. Switch back to the root path to restore the root
conversation view.

The **Subagents** tab in `/coordination` shows the hierarchy as a tree with
each agent's current Working, Waiting, Stopping, Queued, or Ready state. The
status summary's total token count includes model calls made by every agent in
the hierarchy. Select an agent with **Enter** to switch to it; use **Space** to
expand or collapse its details.

Use `/coordination` for a live, read-only log of messages across the hierarchy.
Each card identifies the sender and recipients, assigns every agent a stable
color, shows a live relative timestamp, and collapses the message body to one
line until expanded. The shared footer describes the click and close controls.
Initial `spawn_agent` tasks and explicit agent-to-agent replies are included.

Messages are stored in the `hierarchy_messages` SQLite table, scoped to the
root thread. Named agents use their shared `db/checkpointer.db`; unnamed
sessions use the workspace's `graph_store.sqlite`. The UI reads at most 50
message previews at a time and fetches a complete body only when expanded.
Collapsing the message releases that body. Use **Older**, **Newer**, or the
left/right keys to page through history, **Home** for the beginning, and
**Latest** or **End** to return to live traffic. Incoming traffic does not
move the page while you are reading older messages.
Scrolling back or expanding a message pauses live following; the header shows
whether you are live, paused, or reading history with newer messages available.
Press **End** or select **Latest** to resume. Message cards support **Tab** focus
and **Enter** or **Space** expansion, with their full text released on collapse.

Within each conversation turn, `send_message`, `wait_for`, `spawn_agent`, and
`stop_agent` share one **Coordination** card. Its summary shows the action,
running, and failure counts, followed by the two latest actions. Expand it
to inspect the individual tool cards, with 20 actions per page. Full tool
inputs and results live in the `coordination_actions` table and are loaded
only when an individual tool card is expanded. Collapsing that card or the
group releases those details.
The group and its action cards also support **Enter** or **Space** expansion;
use **Shift+Tab** from the prompt editor to reach them. Failed actions have an
explicit error state, and new actions do not displace a historical action page
being read. All card shortcuts are included in `/keymap`.

On delivery, the runtime gives the receiving model a trusted system envelope
with the sender and recipient paths. The original payload remains the exact
human message that follows that envelope, so an agent can report who sent it,
who received it, and what was sent without relying on prompt conventions.

While an agent is working, press `Enter` in the message editor to queue a draft
without stopping current work. User and agent messages share the same FIFO
delivery queue. A queued user message stays dimmed above the editor until it
reaches the selected agent; if that agent is in `wait_for`, the message becomes
that wait's exact tool result instead of starting a duplicate model turn.

`Esc` closes open pickers and dialogs; it does not submit messages or cancel
agent work. Interrupt-and-send support is deferred until the work in
[PR #404](https://github.com/lanl/ursa/pull/404) is available.

