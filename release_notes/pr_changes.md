# Notes on the changes in progress

## Separated Agent history/presistence from workspace
- User can set an agent's name which will point to a directory in `~/.cache/ursa_agents/<group_name>/<agent_name>`
  - Agent checkpointing and "module" like loading of checkpointed agents with data/time logging
- Future use of that agent name will load the history of the agent and allow it to read in contextual information 
  like skills that it saves.

### TODO:
- ~~Agent checkpointing and "module" like loading of checkpointed agents with data/time logging~~
- Tools for writing/reading self documentation for context offloading
- Implement interface for web dashboard
- Ensure this did not break the YAML workflow

## Settable agent "group" for information control
- The user can set the "group" for the agents. Agents in a particular group will have whitelisted endpoints 
  and not be able to be loaded from persistence in another group. The goal is to ensure that the user can 
  persist agents but not have to worry about accidentially using an agent that has information in one context 
  (like Triad proprietary data or CUI) and pointing it an endpoint that it shouldnt (like the public OpenAI endpoint)
- `group_name` above is the name of the group. Need to make this user settable and users should use intuitive names
  - Implemented the baseline interaction of this with the CLI. 

### TODO:
- ~~Propogate this to the white-list checking.~~
- Test for python scripting interface
- Implement for plan_execute_from_yaml
- Test for dashboard

## Chat has tools now
- The efforts to do ChatWithTools naturally just became giving the basic Chat functionality tools and renaming the
  ChatAgent as BasicChatAgent. The goal here is that the ChatAgent becomes the natural way to interact downstream with
  a persisted agent or to combine agents together.

### TODO:
- Test

## Added a use_web argument to the execution agent and chat agent
- Added a flag so a user can easily turn on and off using web-based search tools. 

### TODO:
- Make this propagate into arguments/settings for the CLI and Dashboard
- Make this default to False, so that web tools are opt-in. Right now this defaults to true for compatibility
  and because until it is settable in the CLI/Dashboard, I want it to still be True there.



# Things to implement in this PR still:

## RAG integration:
- Persistent agents should be able to use RAG on their history in some way
  - Maybe theres a way to resuscitate the old AgentMemory structure here
- Easier to couple in RAG as a tool 
  - Maybe be able to give the agents a list of paths to data to RAG?
  - There is probably a clever way to do this

# Choosing the right agents:
- A RAGbearian:
  - An agent that can search documentation on what each RAG database has and can pipe user requests to the most
    relevant RAG Agent
- A BotBearian:
  - A RAG agent that gets access to each persisted agent's memory in a group and can help the user select which persisted agent
    would be the best for a given task.

# URSA and URSA dev bots
- Once the above is working, making an URSA bot and URSA dev bot that can be packaged
  with URSA in the public repo, both as examples of these sorts of persistent agents
  but also to share the ability to have these sort of bots shared for everyone working on
  or with URSA.

## Agent interaction environments
- Maybe this moves out to a separate one but seems useful here.
- Agent Symposium
- Agent Teams
  - Agent teams should be able to be treated like a single agent for user interaction sake
