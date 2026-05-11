# Notes on the changes in progress

## Separated Agent history/presistence from workspace
- User can set an agent's name which will point to a directory in `~/.cache/ursa_agents/<group_name>/<agent_name>`
  - Agent checkpointing and "module" like loading of checkpointed agents with data/time logging
- Future use of that agent name will load the history of the agent and allow it to read in contextual information 
  like skills that it saves.

### TODO:
- Ensure this did not break the YAML workflow
- Ensure that the agents do a good job of reading/writing these changes. Right now, agents need to be told to pretty explicitly
- There needs to be something for handling the case where a user wants to have two sessions going with the same persistent agent. Right now, their checkpoint writes may get in the way of each other, leading to a bad checkpoint file or at the very least, a messy, disjointed history.

## Settable agent "group" for information control
- The user can set the "group" for the agents. Agents in a particular group will have whitelisted endpoints 
  and not be able to be loaded from persistence in another group. The goal is to ensure that the user can 
  persist agents but not have to worry about accidentially using an agent that has information in one context 
  (like Triad proprietary data or CUI) and pointing it an endpoint that it shouldnt (like the public OpenAI endpoint)
- `group_name` above is the name of the group. Need to make this user settable and users should use intuitive names
  - Implemented the baseline interaction of this with the CLI. 
- Tested for dashboard.

### TODO:
- Implement for plan_execute_from_yaml

## Chat has tools now
- The efforts to do ChatWithTools naturally just became giving the basic Chat functionality tools and renaming the
  ChatAgent as BasicChatAgent. The goal here is that the ChatAgent becomes the natural way to interact downstream with
  a persisted agent or to combine agents together.
- Moved the summarization and dangling tool handling to the base agent class. This will allow it to be pulled into 
  other agents more easily and be more stable/structured. Really this was needed eventually with the move to persistent 
  agents because others can pick up a message history that could have problems. 
  - Gave the execution agent a review/kickback step to differentiate from this ChatWithTools agent.

## Added a use_web argument to the execution agent and chat agent
- Added a flag so a user can easily turn on and off using web-based search tools. 
- Propagated into arguments/settings for the CLI and Dashboard

## URSA and URSA dev bot
- Did a lot of dev with an "ursa_cli_bot" which basically can act as an URSA dev bot that can be packaged
  with URSA going forward to allow others to have an agent easily pick up the dev notes

### TODO:
- Make an "ursa" agent that has reviewed the code usage and can act as an oracle for Q/A
- Need to actually package into the git repo (and figure the best way to do so)

# Things to implement in a future PR:

# Choosing the right agents:
- A BotBearian:
  - A RAG agent that gets access to each persisted agent's memory in a group and can help the user select which persisted agent
    would be the best for a given task.

## RAG integration:
- Persistent agents should be able to use RAG on their history in some way
  - Maybe theres a way to resuscitate the old AgentMemory structure here
- Easier to couple in RAG as a tool 
  - Maybe be able to give the agents a list of paths to data to RAG?
  - There is probably a clever way to do this
- A RAGbearian:
  - An agent that can search documentation on what each RAG database has and can pipe user requests to the most
    relevant RAG Agent
  - Like the BotBearian, but of the RAG databases. 
    - Perhaps centralizing RAG agents like other agents is all that needs to happen here?

## Agent interaction environments
- Maybe this moves out to a separate one but seems useful here.
- Agent Symposium
- Agent Teams
  - Agent teams should be able to be treated like a single agent for user interaction sake
