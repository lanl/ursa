# Werewolf communication exercise

Paste [prompt.md](prompt.md) into a fresh URSA chat with subagents enabled
(`max_agents >= 20`, `max_depth >= 1`). It runs twenty player agents with private
alliances, wolf coordination, public discussion, and voting. The GM resolves
phases; players communicate directly using `send_message` and explicit groups.

For a recording, launch from the repository root with a unique agent name and
thread ID, paste the prompt, then open `/messages`:

```sh
asciinema rec --stdin --cols 140 --rows 48 --idle-time-limit 2 \
  --command 'uv run ursa --name werewolf-social --thread_id werewolf-social-1' \
  werewolf-social.cast
```

Keep the pane open through the final `GAME OVER` broadcast and expand the
GM's final summary. Preserve setup and any corrections in the recording.

## Verify parallel calls

Several message cards appearing close together do not prove parallel tool
calling. Audit the saved model responses: a parallel tool turn contains more
than one tool call in the **same AI message**. A list-target broadcast counts
as one call with several recipients, not as several model tool calls.

```sh
uv run examples/combine_agents_examples/werewolf/audit_trace.py \
  /path/to/agents/werewolf-social/db/checkpointer.db \
  werewolf-social-1 --output werewolf-social-audit.json
```

The audit reads all available checkpoints without modifying them and deduplicates
calls across snapshots. During a live game, repeat the command with `--merge` to
retain already-observed evidence if older checkpoints are pruned. It reports
batch widths, batching by agent, batches beyond
initial spawning, and successful peer/group sends. Full batch details include
timestamps and arguments for comparison with `/messages`. The route resolver
in this exercise's audit assumes the game's flat `root/pNN` roster.

Look for repeated batches throughout the game, including player-originated
calls, plus actual peer-to-peer messages and public group messages. Model
batching and runtime overlap are distinct: LangGraph runs a tool batch
concurrently. The host reserves child names and capacity atomically, then starts
their processes concurrently; a ready child can receive its initial task while
other children are still starting.
