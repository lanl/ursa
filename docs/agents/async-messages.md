# Async messages and subagents

Every `BaseAgent` has an asynchronous inbox and can own a dynamic hierarchy of
subagents. Inbox processing is serialized per agent, while different agents may
run concurrently.

## User messages

Use `request` when the caller needs the result, or `post_message` for one-way
delivery:

```python
result = await agent.request("Analyze the latest measurements")
await agent.post_message("Also check the calibration notes")
await agent.wait_for_messages()
```

A message arriving while that agent is already running waits in its inbox. Use
`interrupt` when the new direction should replace the active invocation:

```python
result = await agent.interrupt("Use the corrected dataset instead")
```

The replacement input receives a leading system instruction explaining that the
user interrupted the prior work. If the input already has a leading system
message, URSA prepends the interruption instruction to its content so the model
still receives one provider-valid system message.

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

Every `ChatAgent` also exposes three hierarchy tools to its model:

- `spawn_agent(name)` creates a child whose path is the caller's path joined
  with `name`, such as `root/researcher`.
- `send_message(target, message)` asks a child or sibling agent to handle work
  and returns its response.
- `list_agents()` lists the current hierarchy and marks the calling agent.

Multiple `send_message` tool calls targeting different agents can execute
concurrently in one tool-calling round.

Call `await lead.stop_message_loop(descendants=True)` when an application owns
the hierarchy outside a managed URSA runtime.

## TUI controls

The Textual interface exposes `/subagents` once an agent hierarchy is active.
Choose any listed path to make it the target of subsequent messages and show
that agent's live progress. Switch back to the root path to restore the root
conversation view.

The **Subagents** tab in `/status` shows the hierarchy as a tree with each
agent's current Working, Queued, or Ready state. The status summary's total
token count includes model calls made by every agent in the hierarchy.

While an agent is working:

1. Type in the live-message editor and press `Enter` to queue the draft
   without stopping current work.
2. Press `Esc` to interrupt current work and send all queued messages in their
   original order. A current draft is appended to that queue before it is sent.
3. Pressing `Esc` with no queued messages or draft leaves current work unchanged.

