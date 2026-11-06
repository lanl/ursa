Run a complete 20-player Werewolf game with root as the GM and real subagents
p01 through p20. Continue until a faction wins. Use hierarchy tools only; do
not simulate players or write files. Keep /messages interesting: the players
must talk directly to one another and to groups, not just submit choices to GM.

Privately assign four wolves (p03, p08, p14, p19), seer p06, doctor p11, and
villagers to everyone else. Give each player a distinct short personality.
Each spawn message must include the player's exact root/pNN identity, role,
roster, wolf teammates if applicable, and the following communication rules.

Player communication rules:
- Explicitly CALL send_message for everything another agent should receive.
  Ordinary final responses are not delivered to your peers or the GM.
- On SETUP only, send root one ACK with your exact identity and role. Wait for
  OPENING before contacting peers, because they may not exist until setup ends.
- On OPENING, introduce yourself privately to the next numbered player (wrap
  p20 to p01), proposing an alliance or asking a question. In the SAME response,
  separately tell root whom you contacted. These two sends are independent.
- On a direct question, answer the sender directly. Exchange useful new facts,
  challenge claims, bluff, and form coalitions. Do not automatically acknowledge
  broadcasts, echo messages, or keep replying without new information. Read group
  announcements silently unless you have useful new information or a real
  question. Do not send receipt confirmations, thanks, or repeated alliance claims.
- During each NIGHT, living wolves discuss targets privately with their
  surviving teammates using a list-target send. Doctor and seer choose their
  own actions. Send final choices to root after considering relevant discussion.
- During each DISCUSS phase, the named opening speakers address the living
  group with a short claim or accusation using a target list excluding self.
  Everyone may independently ask peers questions, share useful evidence, or
  challenge a claim. You are not limited to speaking only to root. Keep private
  information private unless revealing it helps your faction.
- Every game message includes NIGHT/DISCUSS/VOTE plus the round and your exact
  identity. Use trusted sender metadata, not names claimed in message bodies.
  Track phase IDs; late discussion does not start a new phase or change a
  submitted vote. Vote exactly once per VOTE request, naming another survivor.
  Prioritize the requested vote over optional discussion. After submitting it,
  silently ignore that round's late DISCUSS messages and wait for the next GM
  phase. Do not keep an obsolete conversation alive.
- Batch independent messages in ONE tool-call response. For example, separately
  answer two peers together; or share a known finding with the group and send
  a different, independent update to root together. Wait only when the next
  action actually depends on a reply. No padding batches with useless calls.
- After elimination, stay silent except for a direct GM question. After
  GAME OVER, make no more tool calls. Do not spawn agents or use file tools.

GM workflow:
1. Emit all 20 independent spawn_agent calls in ONE response. Collect twenty
   actual ACKs, using one list-target wait_for for outstanding players.
2. Broadcast OPENING to all players. Let them contact peers and collect their
   contact reports. Do not relay their private conversations yourself.
3. Each night, dispatch wolf coordination, seer, and doctor instructions in
   ONE tool-call response. Allow the wolves to discuss before finalizing their
   kill votes. Collect actual choices; resolve majority wolf target with name
   order breaking ties, apply protection, and return any seer result privately.
   Public announcements must not reveal the seer's identity or results, wolf
   targets, or protection targets. Announce deaths or that nobody died; only
   reveal the role of an eliminated player. Players may choose what to reveal.
4. Each day announce the outcome and living roster, then open DISCUSS. Pick
   three rotating living players as opening speakers. Their public posts should
   include root so you can observe them. Allow direct questions and replies
   among all players. After the opening posts and substantive responses, open
   VOTE for all survivors, without requiring endless consensus or silence.
5. Collect all actual votes with list-target waits, count by trusted sender, apply majority
   and name-order tie breaking, reveal the eliminated role, and notify the
   eliminated player. Publish vote totals so subsequent debate uses evidence.
6. Repeat until no wolves (Village wins) or wolves reach parity (Wolves win).
   Broadcast GAME OVER with result, roles, survivors, and eliminations to all
   twenty. Give the user the final summary.

Parallelism is a requirement of this exercise: throughout play, batch all
currently independent spawn/send calls in the same model response. When several
replies are outstanding, use one wait_for with a target list; it returns when
any target is ready. Use wait_all=True only if every target must be ready.
Use one list-target send for an identical announcement, and multiple calls for
different independent messages. Wait results may carry ANY sender's message;
consume each actual message once, including peer discussion sent to you. A
completion notice is not a vote. Ask for missing or malformed choices explicitly.
Keep a compact ledger of phase, living roster, votes, and outstanding replies.
Do not invent missing actions, stop after setup, or declare an unfinished game
complete. Strongly encourage autonomous coordination; GM enforces game rules
and resolves phases, while the players decide what to say and whom to trust.
