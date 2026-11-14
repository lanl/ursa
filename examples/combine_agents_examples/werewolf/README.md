# Play Werewolf with a team of agents

Run a twenty-player social deduction game in URSA. The root agent acts as the
Game Master (GM); twenty subagents hold private roles, form alliances, discuss
suspicions, and vote. This example demonstrates asynchronous messages,
independent tool calls, and coordination across an agent hierarchy.

## 1. Start a game

From the repository root, configure a working chat model and credentials with
`uv run ursa config`. See [model configuration](../../../docs/configuration/models.md)
if you have not used URSA before. Players inherit the root model unless you set
`subagents.llm_model`.

Launch a fresh named session with capacity for twenty immediate subagents:

```sh
uv run ursa --name werewolf --thread_id werewolf-1 \
  --subagents.max_agents 20 --subagents.max_depth 1
```

Use a new name and thread ID for each game so earlier messages do not enter the
new game. Twenty players can make many model calls. `/status` shows total token
usage; the Subagents tab in `/coordination` shows each player's usage.

## 2. Submit the game prompt

Copy the full [prompt.md](prompt.md) into the prompt entry and submit it once.
The assignments are fixed: four wolves, one seer, one doctor, and fourteen
villagers. Model decisions and the winner can differ between runs.

??? note "Full game prompt — expand to copy"

    ```text
    --8<-- "examples/combine_agents_examples/werewolf/prompt.md"
    ```

The GM spawns `p01` through `p20`, with each role and initial instructions in
that player's `spawn_agent` message. Players acknowledge setup before the GM
opens discussion, so no one sends to an agent that has not been created yet.

## 3. Follow the conversation

Open `/coordination` to see traffic across the whole team, including spawn
messages. Expand a message to read its full content. Use `/subagents` to
select a player and follow that player's progress, or use Coordination's
Subagents tab to inspect the live agent tree.

The game proceeds through these stages:

| Stage | Coordination | What to inspect |
| --- | --- | --- |
| Setup | Twenty independent `spawn_agent` calls; acknowledgments to root. | Each player receives a different identity and role. |
| Opening | Each player contacts the next player and independently reports that contact to root. | Direct peer messages and two-call model turns. |
| Night | Wolves discuss a target privately; seer and doctor choose actions independently. | Private messages stay within their intended recipients. |
| Discussion | Opening speakers address the living group; anyone can question peers. | Group sends, replies, accusations, and alliances. |
| Vote | Each survivor sends one vote to root; GM resolves the result. | Sender identities, phase numbers, and published vote totals. |
| Game over | GM announces a winner and reveals roles to all twenty players. | Every player receives the final message and stops acting. |

For example, one completed run began with this direct message:

```text
root/p06 → root/p07
OPENING Round 1 — I am root/p06, the SEER. I seek hidden truths and would
value your trust. What is your role, and are you open to sharing information
as the game unfolds?
```

The peer receives the sender, recipient, and content together. Replies require
an explicit `send_message`; a player's final response is not automatically
forwarded. The GM uses `wait_for` with a list of outstanding players, but a wait
can also deliver a message from another sender. It must count actual replies
by sender and phase rather than treating a finished turn as a vote.

That run ended with a Village win in round eight. The GM made bookkeeping
mistakes, including an omitted survivor and later role disclosures, so the
outcome illustrates agent coordination rather than a mechanically adjudicated
game. Check the roster and vote totals when evaluating your own run.

## 4. Inspect parallel tool calls

Several messages appearing together do not establish that the model emitted a
parallel tool batch. [audit_trace.py](audit_trace.py) reads saved model responses
and counts turns containing multiple tool calls in the **same AI message**.
A list-target broadcast is one tool call, however many recipients it has.

With the default cache directory and group, run:

```sh
uv run examples/combine_agents_examples/werewolf/audit_trace.py \
  ~/.cache/ursa/default/agents/werewolf/db/checkpointer.db \
  werewolf-1 --output werewolf-audit.json
```

Adjust the database path if you use a different cache directory or group.
The audit reads without modifying the checkpoint database. Run it during play
and repeat with `--merge` to retain evidence before older checkpoints are pruned.
The game's root and players share this file, with separate thread IDs.

An excerpt from the completed run's audit:

```json
{
  "tool_turns": 785,
  "parallel_tool_turns": 86,
  "non_spawn_parallel_turns": 85,
  "successful_peer_sends": 334,
  "successful_peer_group_sends": 135
}
```

All twenty players emitted parallel tool calls in that run. These counts will
vary: the prompt encourages batching, the model chooses the calls, and LangGraph
executes each batch concurrently. Successful sends mean the runtime accepted
the messages; inspect recipient state to verify receipt. This audit resolves
explicit paths in the example's flat `root/pNN` roster.

To adapt the game, change roles and phase rules in `prompt.md`. Keep private
role assignments separate from public announcements, and update the roster and
subagent limit together if you change the player count.
